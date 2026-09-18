# INC1000 inclinometer module

`inc1000.py` wraps the SkyMEMS INC1000 binary serial protocol. It reads X/Y
angles and temperature and exposes the documented configuration commands.
The RS232 unit used here has been verified on **COM1, 115200 baud, 8N1**.
Other units may use RS485, another COM port, address, or baud rate.

Install the only runtime dependency if needed:

```powershell
py -m pip install pyserial
```

From the repository root:

```python
from Instrument_Operation.inc1000 import INC1000

with INC1000(port="COM1", baudrate=115200, address=0) as sensor:
    reading = sensor.read_measurement()
    print(reading.x_deg, reading.y_deg, reading.temperature_c)

    # Five independent queries, 0.2 seconds apart:
    samples = sensor.read_samples(5, interval=0.2)
```

`Measurement` contains `x_deg`, `y_deg`, and `temperature_c`. Invalid frames,
wrong addresses, and checksum errors raise `INC1000ProtocolError`; a missing
response raises `INC1000TimeoutError`. Serial-port errors come from pyserial.
The context manager closes the port even if a read fails.

In the present Moog installation, the INC1000 **Y** value decreases as the
platform is raised. To measure physical lift, read the `tilt = 0` baseline at
**the same pan angle**, then calculate:

```python
lift_deg = baseline.y_deg - raised.y_deg
```

Do not use one pan angle's baseline for another pan angle. The sensor measures
inclination relative to gravity, not the Moog pan/azimuth angle directly.

## Device settings (explicitly mutating)

These calls change the connected device; none is made during normal reads:

```python
with INC1000(port="COM1") as sensor:
    sensor.set_relative_zero()  # Current pose becomes zero
    sensor.set_absolute_zero()  # Restore factory absolute-zero reference
    # sensor.set_baud_rate(9600)
    # sensor.set_address(0x0101)
    # sensor.save_settings()     # Persist baud/address in device Flash
```

Changing baud rate immediately changes the host serial connection too. A baud
or address change is **not** saved to Flash automatically; call
`save_settings()` only if the change should survive a power cycle. Record a
new address before disconnecting the device. The manual lists 35400 baud for
code 05 (not 38400); verify that unusual rate with the manufacturer before
using it. Its address-change example shows only the `00 00` prefix followed
by a new two-byte address, which is the format used by this module. The
settings commands have unit tests but have **not** been tried on this physical
device.

Protocol reference: *INC1000 Series 0.01 Degree High Accuracy Single / Dual
Axis Inclinometer*, "Communication protocol" (pp. 5-8). The confirmed read
query for the default address is `4E 4A 00 00 02 41 00 00 DB`.

Run the hardware-free tests from the repository root:

```powershell
py -m unittest discover -s Instrument_Operation/tests -p test_inc1000.py -v
```
