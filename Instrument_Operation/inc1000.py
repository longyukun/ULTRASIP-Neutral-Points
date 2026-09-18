"""RS232/RS485 protocol driver for the SkyMEMS INC1000 inclinometer.

Only ``read_measurement`` is needed for normal data acquisition.  The
``set_*`` and ``save_settings`` methods change device state and are never
called automatically.  The wire protocol comes from the INC1000 Series
manual; this computer's RS232 unit was verified on COM1 at 115200 baud.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Callable


PREAMBLE = b"NJ"  # 4E 4A
MAX_COMMAND_LENGTH = 64
READ_COMMAND = 0x41
READ_RESPONSE = 0xC1
BAUD_CODES = {9600: 0x03, 19200: 0x04, 35400: 0x05, 57600: 0x06, 115200: 0x07}


class INC1000Error(Exception):
    """Base class for INC1000 communication errors."""


class INC1000TimeoutError(INC1000Error):
    """No complete valid response arrived before the deadline."""


class INC1000ProtocolError(INC1000Error):
    """A response had an invalid checksum, address, code, or payload."""


@dataclass(frozen=True)
class Measurement:
    x_deg: float
    y_deg: float
    temperature_c: float


@dataclass(frozen=True)
class Frame:
    address: int
    code: int
    data: bytes


def build_frame(address: int, code: int, data: bytes = b"") -> bytes:
    """Encode one INC1000 command (16-bit additive checksum, big-endian)."""
    if not 0 <= address <= 0xFFFF:
        raise ValueError("address must be 0..65535")
    if not 0 <= code <= 0xFF:
        raise ValueError("command code must be 0..255")
    if len(data) + 1 > MAX_COMMAND_LENGTH:
        raise ValueError("data payload is too long")
    body = PREAMBLE + address.to_bytes(2, "big") + bytes((len(data) + 1, code)) + data
    return body + (sum(body) & 0xFFFF).to_bytes(2, "big")


def decode_frame(raw: bytes) -> Frame:
    """Validate and decode one complete INC1000 response frame."""
    if len(raw) < 8 or raw[:2] != PREAMBLE:
        raise INC1000ProtocolError("missing INC1000 preamble")
    command_length = raw[4]
    if not 1 <= command_length <= MAX_COMMAND_LENGTH or len(raw) != 5 + command_length + 2:
        raise INC1000ProtocolError("invalid frame length")
    expected = sum(raw[:-2]) & 0xFFFF
    received = int.from_bytes(raw[-2:], "big")
    if expected != received:
        raise INC1000ProtocolError(
            f"checksum mismatch: calculated {expected:04X}, received {received:04X}"
        )
    return Frame(int.from_bytes(raw[2:4], "big"), raw[5], raw[6:-2])


class INC1000:
    """Connect to one INC1000 over an 8N1 serial port.

    ``serial_factory`` is primarily for tests; the default uses pyserial.
    Use as a context manager so the COM port is always released.
    """

    def __init__(
        self,
        port: str = "COM1",
        *,
        baudrate: int = 115200,
        address: int = 0,
        timeout: float = 1.0,
        serial_factory: Callable[..., object] | None = None,
    ) -> None:
        if not 0 <= address <= 0xFFFF:
            raise ValueError("address must be 0..65535")
        if baudrate <= 0 or timeout <= 0:
            raise ValueError("baudrate and timeout must be positive")
        self.port = port
        self.baudrate = baudrate
        self.address = address
        self.timeout = timeout
        self._serial_factory = serial_factory
        self._serial = None

    def open(self) -> INC1000:
        if self._serial is not None:
            return self
        factory = self._serial_factory
        if factory is None:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError("Install pyserial to use INC1000 (pip install pyserial)") from exc
            factory = serial.Serial
        self._serial = factory(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=self.timeout,
            write_timeout=self.timeout,
        )
        return self

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def __enter__(self) -> INC1000:
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _require_open(self):
        if self._serial is None:
            raise RuntimeError("INC1000 port is not open; call open() or use 'with'")
        return self._serial

    def _read_frame(self) -> Frame:
        serial_port = self._require_open()
        deadline = time.monotonic() + self.timeout
        buffer = bytearray()
        last_error = None
        while time.monotonic() < deadline:
            next_byte = serial_port.read(1)
            if not next_byte:
                continue
            buffer.extend(next_byte)
            while buffer and buffer[0] != PREAMBLE[0]:
                del buffer[0]
            if len(buffer) >= 2 and buffer[1] != PREAMBLE[1]:
                del buffer[0]
                continue
            if len(buffer) < 5:
                continue
            command_length = buffer[4]
            if not 1 <= command_length <= MAX_COMMAND_LENGTH:
                last_error = INC1000ProtocolError("invalid response length")
                del buffer[0]
                continue
            total_length = 5 + command_length + 2
            if len(buffer) < total_length:
                continue
            raw = bytes(buffer[:total_length])
            del buffer[:total_length]
            try:
                return decode_frame(raw)
            except INC1000ProtocolError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise INC1000TimeoutError(f"no INC1000 response from {self.port} within {self.timeout}s")

    def _exchange(self, code: int, data: bytes, response_code: int, response_size: int) -> bytes:
        serial_port = self._require_open()
        serial_port.reset_input_buffer()
        request = build_frame(self.address, code, data)
        written = serial_port.write(request)
        if written != len(request):
            raise INC1000Error(f"short serial write: {written}/{len(request)} bytes")
        serial_port.flush()
        response = self._read_frame()
        if response.address != self.address:
            raise INC1000ProtocolError(
                f"unexpected address {response.address:04X}; expected {self.address:04X}"
            )
        if response.code != response_code or len(response.data) != response_size:
            raise INC1000ProtocolError(
                f"unexpected response code/data: {response.code:02X}/{len(response.data)}"
            )
        return response.data

    def read_measurement(self) -> Measurement:
        """Read X angle, Y angle (degrees), and temperature (°C)."""
        data = self._exchange(READ_COMMAND, b"\x00", READ_RESPONSE, 12)
        x, y, temperature = struct.unpack(">fff", data)
        return Measurement(x, y, temperature)

    def read_samples(self, count: int, *, interval: float = 0.0) -> list[Measurement]:
        """Acquire multiple readings; ``interval`` is seconds between queries."""
        if count < 1 or interval < 0:
            raise ValueError("count must be positive and interval must be nonnegative")
        samples = []
        for index in range(count):
            if index:
                time.sleep(interval)
            samples.append(self.read_measurement())
        return samples

    def set_relative_zero(self) -> None:
        """Set the current orientation as the device's relative zero."""
        ack = self._exchange(0x91, b"\x00", 0x11, 1)
        if ack != b"\x55":
            raise INC1000ProtocolError(f"relative-zero command rejected: {ack.hex()}")

    def set_absolute_zero(self) -> None:
        """Restore the factory absolute-zero reference."""
        ack = self._exchange(0x92, b"\x00", 0x12, 1)
        if ack != b"\x55":
            raise INC1000ProtocolError(f"absolute-zero command rejected: {ack.hex()}")

    def set_baud_rate(self, baudrate: int) -> None:
        """Change the live baud rate; call ``save_settings`` to persist it.

        The manual lists 35400 (not 38400) for code 05; do not assume that
        entry is a typo without checking the particular device.
        """
        if baudrate not in BAUD_CODES:
            raise ValueError(f"documented baud rates: {tuple(BAUD_CODES)}")
        code = BAUD_CODES[baudrate]
        ack = self._exchange(0x21, bytes((code,)), 0xA1, 1)
        if ack != bytes((code,)):
            raise INC1000ProtocolError(f"baud-rate command rejected: {ack.hex()}")
        self._require_open().baudrate = baudrate
        self.baudrate = baudrate

    def set_address(self, new_address: int) -> None:
        """Change the live address; call ``save_settings`` to persist it.

        The manual's only example uses data ``00 00`` followed by the new
        two-byte address.  That documented format is used here.
        """
        if not 0 <= new_address <= 0xFFFF:
            raise ValueError("new_address must be 0..65535")
        new_bytes = new_address.to_bytes(2, "big")
        ack = self._exchange(0x13, b"\x00\x00" + new_bytes, 0x93, 2)
        if ack != new_bytes:
            raise INC1000ProtocolError(f"address command rejected: {ack.hex()}")
        self.address = new_address

    def save_settings(self) -> None:
        """Write current settings to device Flash (not done automatically)."""
        ack = self._exchange(0xF1, b"\x00", 0x71, 1)
        if ack != b"\x55":
            raise INC1000ProtocolError(f"save-settings command rejected: {ack.hex()}")
