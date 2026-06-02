# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:33:35 2024

@author: ULTRASIP_1
"""

import numpy as np

    

def pixel_geometry(sza0,HFOV,VFOV,img_x,img_y,x_center, y_center, Pan, Tilt, Sun_Position_Azimuth, Sun_Position_Altitude):
    
        
        delta_zen = (Sun_Position_Altitude-sza0)
       # print(delta_zen)

        # OpenCV reports centroids as x=column and y=row. With indexing='ij',
        # rr is the image row index and cc is the image column index.
        rows = np.arange(img_y)
        cols = np.arange(img_x)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')

        row_center = y_center
        col_center = x_center

        # Compute deviations
        row_dev = (rr - row_center) * VFOV
        col_dev = (col_center - cc) * HFOV

        # Compute view angles
        view_az = np.radians(Pan - col_dev)
        view_zen = np.radians((Tilt-delta_zen) - row_dev)
        sun_az = np.radians(Sun_Position_Azimuth)
        sun_zen = np.radians(Sun_Position_Altitude-delta_zen)
        
        
        return view_az, view_zen, sun_az, sun_zen
