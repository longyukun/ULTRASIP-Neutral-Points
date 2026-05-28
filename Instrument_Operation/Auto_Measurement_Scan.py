# Auto Measurement Scan with Adaptive Exposure
import numpy as np
import serial
import time
import h5py
import pytz
import os
import matplotlib.pyplot as plt
from zaber_motion import Units
from zaber_motion.ascii import Connection
from datetime import datetime
from suncalc import get_position
import moog_functions as mf
import uv_cam_functions as uv
import tkinter as tk
from hex_stop_button import HexStopButton

# Constants and Metadata
uv_wavelength = '355 FWHM 10nm'
angles = [0, 45, 90, 135]
Location = 'MeinelRoof'
#Get from Garmin GPS
latitude = 32.23134 #46.892995#45.66487 #32.23134;
longitude = -110.94712 #-113.449814#-111.04800 #-110.94712;

# Offsets
tilt_offset = 0
pan_offset = -9

start_tilt = 0
step_tilt = 2
# Exposure settings
uv_exp_initial = 1e3
#outpath = "D:/Data"
outpath = "C:/Users/deleo/Documents/Data"
def auto_exposure_all_angles(camera, axis, angles, 
                              target_median=2600, 
                              initial_exp=1e5, 
                              max_exp=1e6,
                              min_exp=500,
                              saturation_thresh=0.97, 
                              bit_depth=12):
    from numpy import median, clip, array

    max_pixel_value = 2**bit_depth - 1
    saturation_limit = max_pixel_value * saturation_thresh
    test_exp = initial_exp
    uv.setup_camera(camera, test_exp)

    medians = []
    saturated = []

    for angle in angles:
        axis.move_absolute(angle, Units.ANGLE_DEGREES)
        axis.wait_until_idle()
        time.sleep(0.5)

        frame = camera.get_frame()
        data = np.frombuffer(frame.get_buffer(), dtype=np.uint16)
        med = median(data)
        medians.append(med)
        saturated.append(med >= saturation_limit)

    medians = array(medians)
    avg_median = np.mean(medians)

    if all(saturated):
        print("[WARNING] All test angles are saturated. Falling back to minimum exposure.")
        return min_exp

    if avg_median == 0:
        new_exp = min_exp
    else:
        scale = target_median / avg_median
        new_exp = clip(test_exp * scale, min_exp, max_exp)

    print(f"Angle medians: {medians.astype(int)} → Target: {target_median}, Adaptive Exp: {int(new_exp)} µs")
    return new_exp

class DataCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Acquisition Stop")

        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack()

        # --- Control Variables ---
        self.running = True
        self.sun_targets = list(np.arange(4, 90, 0.2))  # Zenith angles to trigger at
        self.tolerance = 0.1  # degrees
        self.completed_targets = set()
        self.aq_num = 0

        # --- Stop Button ---
        self.stop_button = HexStopButton(
            canvas=self.canvas,
            cx=100, cy=100, size=80,
            command=self.stop_loop
            )

        # --- Start monitoring loop ---
        self.check_sun_angle_loop()

    def stop_loop(self):
        print("STOP button pressed!")
        self.running = False

    def check_sun_angle_loop(self):
        """Continuously check the Sun’s zenith angle and trigger acquisitions."""
        if not self.running:
            print("Loop stopped.")
            return

        dt = datetime.now()
        sun_pos = get_position(dt, longitude, latitude)
        sun_alt = np.degrees(sun_pos['altitude'])
        sun_zenith = sun_alt  # Convert altitude to zenith

        # Check if within tolerance of any target
        for target in self.sun_targets:
            if abs(sun_zenith - target) < self.tolerance and target not in self.completed_targets:
                print(f"Hit target zenith {target:.1f}° at {dt}, actual = {sun_zenith:.2f}°")
                self.acquire_data()
                self.completed_targets.add(target)
                break  # Avoid multiple triggers at once

        # Continue checking every 10 seconds
        self.root.after(10_000, self.check_sun_angle_loop)

    def acquire_data(self):
        # ----------------------------#Connect to motors#-------------------------#
        #Make new folder for today's date to save the data (if it doesn't exist)
        #Set filename for measurement using date/time
        aq_num = -1 #index for later
        dt = datetime.now()
        date_time = str(dt)
        date = date_time[0:10].replace('-','')
        timestamp = date_time[11:19].replace(':','_')
        filename = Location+'_'+date+'_'+timestamp+'.h5'
        datapath = os.path.join(outpath,str(date_time[0:10].replace('-','_')))
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        filename=os.path.join(datapath,filename)
        # #Connect to Moog
        # #Configure port connection
        moog = serial.Serial()
        moog.baudrate = 9600
        moog.port = 'COM7'
        moog.open()
        mf.init_autobaud(moog);
        mf.get_status_jog(moog)

        #Connect to Rotation Motor
        connection = Connection.open_serial_port("COM6")
        device_list = connection.detect_devices()
        device = device_list[0]
        axis = device.get_axis(1)
        if not axis.is_homed():
            axis.home()
        axis.settings.set('maxspeed',100, Units.ANGULAR_VELOCITY_DEGREES_PER_SECOND)
        #NOTE: Connection to DoFP and UV Cam done inline during image acqusition

        #Desired Structure 
        #File Metadata: Date, Time, Lat, Long
        #Group_# = Aquisition_#
        #Aquisition_#_Metadata: Time, Pan/Tilt, Sun Position, UV_Exposure, DoFP_Exposure
        #Aquisition_Dataset(s): UV_ImageData, DoFP_ImageData

        hdf5_file = h5py.File(filename,"w")
        meas = hdf5_file.create_group("Measurement_Metadata")
        meas.attrs['Latitude'] = str(latitude)
        meas.attrs['Longitude'] = str(longitude)
        meas.attrs['Pan_Offset'] = str(pan_offset)
        meas.attrs['Tilt_Offset'] = str(tilt_offset)

        #Get sun position and set to initial pan and tilt value for moog
        dt = datetime.now()
        sun_pos = get_position(dt, longitude, latitude)
        #Set initial position based on sun location
        pan = np.degrees(sun_pos['azimuth'])
        tilt = np.degrees(sun_pos['altitude']) 
        
        end_tilt = int(88-tilt)
        print(end_tilt)

        #Move Moog to Sun Position
        mf.mv_to_coord(moog,int((pan-pan_offset)*10),int((tilt-tilt_offset)*10))
        mf.get_status_jog(moog)
        time.sleep(1)

        measstart = time.time()

        for dtilt in range(start_tilt,end_tilt, step_tilt):
            aq_num = aq_num +  1
            mf.get_status_jog(moog)

            dt = datetime.now()
            sun_pos = get_position(dt, longitude, latitude)
            pan = np.degrees(sun_pos['azimuth'])
            tilt = np.degrees(sun_pos['altitude']) + dtilt

            moog_requested_pan = pan - pan_offset
            moog_requested_tilt = tilt - tilt_offset
            moog_pan_command = int(moog_requested_pan * 10)
            moog_tilt_command = int(moog_requested_tilt * 10)
            moog_target_pan = moog_pan_command / 10.0
            moog_target_tilt = moog_tilt_command / 10.0

            mf.mv_to_coord(moog, moog_pan_command, moog_tilt_command)
            time.sleep(0.4)
            moog_status = mf.get_status_jog(moog)

            uvimage_data = []
            cam_id = uv.parse_args()
            with uv.VmbSystem.get_instance():
                with uv.get_camera(cam_id) as uvcam:
                    uv.setup_camera(uvcam, uv_exp_initial)
                    if dtilt <3:
                        uv_exp = 8e2
                    else:
                        uv_exp = auto_exposure_all_angles(uvcam, axis, angles)
                    uv.setup_camera(uvcam, uv_exp)
                    handler = uv.Handler()
                    date_time = str(dt)
                    timestamp = date_time[11:19].replace(':', '_')
                    time1 = time.time()
                    for angle in angles:
                        axis.move_absolute(angle, Units.ANGLE_DEGREES)
                        axis.wait_until_idle()
                        time.sleep(0.5)
                        frame = uvcam.get_frame()
                        data = np.frombuffer(frame.get_buffer(), dtype=np.uint16)
                        uvimage_data = np.append(uvimage_data, data)
                    uvmeastime = time.time() - time1
                    axis.home()

            aq = hdf5_file.create_group(f"Aquistion_{aq_num}")
            aq.attrs['Timestamp AZ'] = timestamp
            utc_time = str(dt.astimezone(pytz.utc))
            utc_timestamp = utc_time[11:19].replace(':','_')
            aq.attrs['Timestamp UTC'] = utc_timestamp
            aq.attrs['Pan'] = pan
            aq.attrs['Tilt'] = tilt
            aq.attrs['Pan Offset'] = pan_offset
            aq.attrs['Tilt Offset'] = tilt_offset
            aq.attrs['Sun Position Azimuth'] = np.degrees(sun_pos['azimuth'])
            aq.attrs['Sun Position Altitude'] = np.degrees(sun_pos['altitude'])
            aq.attrs['Moog Requested Pan [deg]'] = moog_requested_pan
            aq.attrs['Moog Requested Tilt [deg]'] = moog_requested_tilt
            mf.write_moog_status_attrs(aq.attrs, moog_status, moog_target_pan, moog_target_tilt)

            uvimg = aq.create_group('UV Image Data')
            uvimg.create_dataset('UV Raw Images', data=uvimage_data)
            print(uv_exp)
            uvimg.attrs['UV Exposure Time'] = uv_exp
            uvimg.attrs['UV Bandpass'] = uv_wavelength
            uvimg.attrs['UV Image Capture Time'] = uvmeastime
            uvimg.attrs['UV Polarizer Angles'] = str(angles)

        measend = time.time()
        print('Measurement Completed', ((measend - measstart)))
        meas.attrs['Total Measurement Time'] = ((measend - measstart) / 60)

        sun_pos = get_position(dt, longitude, latitude)
        pan = np.degrees(sun_pos['azimuth'])
        tilt = np.degrees(sun_pos['altitude'])

        mf.mv_to_coord(moog, int((pan - pan_offset) * 10), int(tilt * 10))
        time.sleep(4)
        axis.home()
        moog.close()
        connection.close()
        hdf5_file.close()
        print(f"Acquisition {self.aq_num} complete at {timestamp}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectorApp(root)
    root.mainloop()
