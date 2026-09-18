"""Protocol tests that do not open a real COM port or alter a device."""

import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inc1000 import (  # noqa: E402
    INC1000,
    INC1000ProtocolError,
    INC1000TimeoutError,
    build_frame,
    decode_frame,
)


class FakeSerial:
    def __init__(self, *, responder, **settings):
        self.responder = responder
        self.settings = settings
        self.baudrate = settings["baudrate"]
        self.writes = []
        self.buffer = bytearray()
        self.closed = False

    def reset_input_buffer(self):
        self.buffer.clear()

    def write(self, data):
        self.writes.append(bytes(data))
        self.buffer.extend(self.responder(bytes(data)))
        return len(data)

    def flush(self):
        pass

    def read(self, size):
        value = bytes(self.buffer[:size])
        del self.buffer[:size]
        return value

    def close(self):
        self.closed = True


class FrameTests(unittest.TestCase):
    def test_documented_query_and_reply(self):
        self.assertEqual(
            build_frame(0, 0x41, b"\x00"),
            bytes.fromhex("4E 4A 00 00 02 41 00 00 DB"),
        )
        response = bytes.fromhex(
            "4E 4A 00 00 0D C1 BF 9A 16 68 BF A9 96 28 41 A3 17 07 06 65"
        )
        decoded = decode_frame(response)
        self.assertEqual(decoded.address, 0)
        self.assertEqual(decoded.code, 0xC1)
        self.assertEqual(len(decoded.data), 12)

    def test_checksum_rejection(self):
        damaged = bytearray(build_frame(0, 0xC1, struct.pack(">fff", 1, 2, 3)))
        damaged[-1] ^= 1
        with self.assertRaises(INC1000ProtocolError):
            decode_frame(bytes(damaged))


class INC1000Tests(unittest.TestCase):
    def make_device(self, responder, *, timeout=0.01):
        created = []

        def factory(**kwargs):
            port = FakeSerial(responder=responder, **kwargs)
            created.append(port)
            return port

        return INC1000(timeout=timeout, serial_factory=factory), created

    def test_read_measurement_skips_noise(self):
        response = build_frame(0, 0xC1, struct.pack(">fff", -1.25, -52.5, 20.5))
        device, ports = self.make_device(lambda request: b"\x0A\x0D****" + response)
        with device:
            measured = device.read_measurement()
            self.assertEqual((measured.x_deg, measured.y_deg, measured.temperature_c), (-1.25, -52.5, 20.5))
            self.assertEqual(ports[0].writes, [bytes.fromhex("4E 4A 00 00 02 41 00 00 DB")])
        self.assertTrue(ports[0].closed)

    def test_read_samples(self):
        response = build_frame(0, 0xC1, struct.pack(">fff", 1, 2, 3))
        device, ports = self.make_device(lambda request: response)
        with device:
            values = device.read_samples(3)
        self.assertEqual(len(values), 3)
        self.assertEqual(len(ports[0].writes), 3)

    def test_timeout(self):
        device, _ = self.make_device(lambda request: b"")
        with device, self.assertRaises(INC1000TimeoutError):
            device.read_measurement()

    def test_wrong_address_is_rejected(self):
        response = build_frame(1, 0xC1, struct.pack(">fff", 1, 2, 3))
        device, _ = self.make_device(lambda request: response)
        with device, self.assertRaises(INC1000ProtocolError):
            device.read_measurement()

    def test_zero_commands_and_save(self):
        response_by_code = {
            0x91: build_frame(0, 0x11, b"\x55"),
            0x92: build_frame(0, 0x12, b"\x55"),
            0xF1: build_frame(0, 0x71, b"\x55"),
        }
        device, ports = self.make_device(lambda request: response_by_code[request[5]])
        with device:
            device.set_relative_zero()
            device.set_absolute_zero()
            device.save_settings()
        self.assertEqual(
            ports[0].writes,
            [
                bytes.fromhex("4E 4A 00 00 02 91 00 01 2B"),
                bytes.fromhex("4E 4A 00 00 02 92 00 01 2C"),
                bytes.fromhex("4E 4A 00 00 02 F1 00 01 8B"),
            ],
        )

    def test_baud_change_requires_explicit_save(self):
        device, ports = self.make_device(
            lambda request: build_frame(0, 0xA1, request[6:7])
        )
        with device:
            device.set_baud_rate(9600)
            self.assertEqual(device.baudrate, 9600)
            self.assertEqual(ports[0].baudrate, 9600)
        self.assertEqual(ports[0].writes, [bytes.fromhex("4E 4A 00 00 02 21 03 00 BE")])

    def test_address_change_uses_documented_layout(self):
        device, ports = self.make_device(
            lambda request: build_frame(0, 0x93, request[8:10])
        )
        with device:
            device.set_address(0x0101)
        self.assertEqual(device.address, 0x0101)
        self.assertEqual(
            ports[0].writes,
            [bytes.fromhex("4E 4A 00 00 05 13 00 00 01 01 00 B2")],
        )

    def test_rejected_setting_does_not_change_local_state(self):
        device, _ = self.make_device(lambda request: build_frame(0, 0xA1, b"\x00"))
        with device, self.assertRaises(INC1000ProtocolError):
            device.set_baud_rate(9600)
        self.assertEqual(device.baudrate, 115200)


if __name__ == "__main__":
    unittest.main()
