# -*- coding: utf-8 -*-
"""
@author: C. M. DeLeon
ULTRASIP User Manual: 
    https://www.overleaf.com/read/hkkghcvgrrdt#c1c7d8

Code Description: 
    This code performs a sequence of "stop-and-stare" measurements relative to the sun.
    Such that polarization measurements are taken within the solar principal plane to find the neurtal point(s).
    
    
Pseudocode: 
"""

#Import libraries 
import numpy as np
import serial
import time
import h5py,pytz
import os
import matplotlib.pyplot as plt
from zaber_motion import Units
from zaber_motion.ascii import Connection
from datetime import datetime
from suncalc import get_position
import moog_functions as mf
import uv_cam_functions as uv
#import vis_cam_functions as vis

def auto_exposure_all_angles(camera, axis, angles, 
                              target_median=2600, 
                              initial_exp=1e5, 
                              max_exp=1e6,
                              min_exp=100,
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

#----------------Constants and Metadata: CHANGE AS NEEDED----------------#
uv_wavelength = '355 FWHM 10nm'
#vis_wavelengths = '470,525,635 nm; Bayer Fitler'
aq_num = -1 #index for later
angles = [0,45,90,135]
#angles = [135,90,45,0]
Location = 'MeinelRoof_'
#Get from Garmin GPS
latitude = 32.2314#46.892995#45.66487;
longitude = -110.94712#-113.449814#-111.04800
uv_exp_initial = 1e3

#Offsets needed -- from homing procedure (subtracted from calculated sun position)
tilt_offset = 0
pan_offset = -9.#to set origin to sun azimuth
#Scan range from sun, 0 is sun
start_tilt = 0
end_tilt = 4
step_tilt = 2
calibration_dtilts = [-0.5, 0.5, -1.0, 1.0, -1.5, 1.5]


def calibration_group_name(dtilt):
    direction = 'up' if dtilt > 0 else 'down' if dtilt < 0 else 'center'
    magnitude = f"{abs(float(dtilt)):g}".replace('.', 'p')
    return f"calibration_acqui_{direction}_{magnitude}"



#Data main directory 
#outpath = 'D:/Data'
outpath = "C:/Users/deleo/Documents/Data"

#Make new folder for today's date to save the data (if it doesn't exist)
#Set filename for measurement using date/time
dt = datetime.now()
date_time = str(dt)
date = date_time[0:10].replace('-','')
timestamp = date_time[11:19].replace(':','_')
filename = Location+'_'+date+'_'+timestamp+'.h5'
datapath = os.path.join(outpath,str(date_time[0:10].replace('-','_')))
if not os.path.exists(datapath):
    os.makedirs(datapath)
filename=os.path.join(datapath,filename)

# ----------------------------#Connect to motors#-------------------------#
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

#Move Moog to Sun Position
initial_pan_command = int((pan - pan_offset) * 10)
initial_tilt_command = int((tilt - tilt_offset) * 10)
mf.move_to_coord_and_wait(moog, initial_pan_command, initial_tilt_command)

measstart=time.time()

def capture_acquisition(group_name, dtilt, acquisition_type, normal_index=None, calibration_index=None):
    #Get sun position and set to initial pan and tilt value for moog
    dt = datetime.now()
    sun_pos = get_position(dt, longitude, latitude)
    
    #Set initial position based on sun location
    pan = np.degrees(sun_pos['azimuth']) 
    tilt = np.degrees(sun_pos['altitude']) + dtilt
    
    moog_requested_pan = pan - pan_offset
    moog_requested_tilt = tilt - tilt_offset
    moog_pan_command = int(moog_requested_pan * 10)
    moog_tilt_command = int(moog_requested_tilt * 10)
    moog_target_pan = moog_pan_command / 10.0
    moog_target_tilt = moog_tilt_command / 10.0
    moog_status = mf.move_to_coord_and_wait(moog, moog_pan_command, moog_tilt_command)
    
    uvimage_data = []
    cam_id = uv.parse_args()
    with uv.VmbSystem.get_instance():
        with uv.get_camera(cam_id) as uvcam:
            uv.setup_camera(uvcam, uv_exp_initial)
            if dtilt < 3:
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
                time.sleep(1)
                frame = uvcam.get_frame()
                data = np.frombuffer(frame.get_buffer(), dtype=np.uint16)
                uvimage_data = np.append(uvimage_data, data)
                uvmeastime = time.time() - time1
                axis.home()
            
        
    aq = hdf5_file.create_group(group_name)
    aq.attrs['Acquisition Type'] = acquisition_type
    if normal_index is not None:
        aq.attrs['Normal Acquisition Index'] = normal_index
    if calibration_index is not None:
        aq.attrs['Calibration Acquisition Index'] = calibration_index
        aq.attrs['Calibration Direction'] = 'up' if dtilt > 0 else 'down' if dtilt < 0 else 'center'
        aq.attrs['Calibration Step Magnitude [deg]'] = abs(float(dtilt))
    aq.attrs['Timestamp MDT'] = timestamp
    utc_time = str(dt.astimezone(pytz.utc))
    utc_timestamp = utc_time[11:19].replace(':','_')
    aq.attrs['Timestamp UTC'] = utc_timestamp
    aq.attrs['Pan'] = pan
    aq.attrs['Tilt'] = tilt
    aq.attrs['Sun Position Azimuth'] = np.degrees(sun_pos['azimuth'])
    aq.attrs['Sun Position Altitude'] = np.degrees(sun_pos['altitude'])
    aq.attrs['Moog Requested Pan [deg]'] = moog_requested_pan
    aq.attrs['Moog Requested Tilt [deg]'] = moog_requested_tilt
    mf.write_moog_status_attrs(aq.attrs, moog_status, moog_target_pan, moog_target_tilt)


    uvimg = aq.create_group('UV Image Data')
    uvimg.create_dataset('UV Raw Images', data = uvimage_data)
    uvimg.attrs['UV Exposure Time'] = uv_exp
    uvimg.attrs['UV Bandpass'] = uv_wavelength
    uvimg.attrs['UV Image Capture Time'] = uvmeastime
    uvimg.attrs['UV Polarizer Angles'] = str(angles)


normal_dtilts = list(range(start_tilt, end_tilt, step_tilt))
if 0 not in normal_dtilts:
    normal_dtilts.insert(0, 0)

for normal_index, dtilt in enumerate(normal_dtilts):
    capture_acquisition(f"Aquistion_{normal_index}", dtilt, 'normal', normal_index=normal_index)
    if normal_index == 0:
        for calibration_index, calibration_dtilt in enumerate(calibration_dtilts):
            capture_acquisition(
                calibration_group_name(calibration_dtilt),
                calibration_dtilt,
                'calibration',
                calibration_index=calibration_index,
            )


measend=time.time()
print('Measurment Completed',((measend-measstart)))
meas.attrs['Total Measurement Time'] = ((measend-measstart)/60)
meas.attrs['Calibration Delta Tilts [deg]'] = np.array(calibration_dtilts, dtype=float)

sun_pos = get_position(dt, longitude, latitude)
#Set initial position based on sun location
pan = np.degrees(sun_pos['azimuth']) 
tilt = np.degrees(sun_pos['altitude'])

#Home and disconnect everything
mf.get_status_jog(moog)
mf.mv_to_coord(moog,int((pan- pan_offset)*10),int(tilt*10)) 
time.sleep(2)
axis.home()
moog.close()
connection.close()
# Close the HDF5 file
hdf5_file.close()
