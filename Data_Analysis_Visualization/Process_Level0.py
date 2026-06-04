# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 21:47:55 2025

@author: C.M.DeLeon

Process 0 - output corrected polarized data products
"""

# Import Libraries
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import cmocean.cm as cmo
import numpy as np
import h5py 
import glob
import os, time

#Functions
def pseudoinverse(matrix):
    
    # # --- Pixelwise pseudoinverse via SVD ---
    # Compute SVD of each 4x3 matrix
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)   # shapes:
    # U:  (H, W, 4, 3)
    # S:  (H, W, 3)
    # Vt: (H, W, 3, 3)

    # Invert singular values 
    S_inv = np.zeros_like(S)
    S_inv = np.where(S > 1e-12, 1/S, 0)                # avoids division by zero

    # Reconstruct pseudoinverse:  pinv(W) = V * S^-1 * U^T
    pseudoinv = np.matmul(Vt.transpose(0,1,3,2) * S_inv[...,None,:], 
                      U.transpose(0,1,3,2))            # shape = (H,W,3,4)
    return pseudoinv
#Correct image
def correct_img(Pij,Rij,Bij):
    R_avg = np.mean(Rij)
    B_avg = np.mean(Bij)
    Cij = (R_avg / Rij) * (Pij - Bij) + B_avg
    
    return Cij

#Load NUC and W-matrix Files 

nuc_files = np.load('D:/NUC_0813.npz', allow_pickle=True)
Rij = nuc_files['arr1'].item()   # convert array-object → Python dict
Bij = nuc_files['arr2'].item()
W_ULTRASIP = np.load('D:/ULTRASIP_AvgWmatrix_15.npy')
pinvW_ULTRASIP = pseudoinverse(W_ULTRASIP)

#Timer
start_time = time.time()

#Load Observations 
date = '2026_03_17'
#basepath = 'C:/Users/ULTRASIP_1/OneDrive/Desktop/'
basepath = 'D:/Data/'
folderdate = os.path.join(basepath, date)
files = glob.glob(f'{folderdate}/*.h5')
idx = len(files) 
idx_array = np.arange(0,idx)

def process0(idx):
    print(f'Processing file {idx}: {files[idx]}')
    
    try:
        f = h5py.File(files[idx], 'r+')
        
        for aqnum in range(0, len(f.keys()) - 1):
            try:
                # Set acquisition to view
                aq = f[f'Aquistion_{aqnum}']

                # Reconstruct the 4 flux images
                uv_data = aq['UV Image Data']
                uv_imgs = uv_data['UV Raw Images'][:].reshape(4, 2848, 2848)
                P0, P45, P90, P135 = uv_imgs
                
                # --- apply correction ---
                C0   = correct_img(P0,   Rij[0], Bij[0])
                C90  = correct_img(P90,  Rij[90], Bij[90])
                C45  = correct_img(P45,  Rij[45], Bij[45])
                C135 = correct_img(P135, Rij[135], Bij[135])
                
                P_corrected = np.stack([C0, C90, C45, C135], axis=-1)
                Stokes_meas = np.matmul(pinvW_ULTRASIP, P_corrected[...,None])[...,0]       # shape = (H, W, 3)
               # Stokes_meas = Stokes_meas.reshape(3, 2848, 2848)

                I_corrected = Stokes_meas[:,:,0]
                Q_corrected = Stokes_meas[:,:,1]
                U_corrected = Stokes_meas[:,:,2]
                
                # Overwrite or create datasets
                for name in ['I_corrected', 'Q_corrected', 'U_corrected']:
                    if name in uv_data:
                        del uv_data[name]

                uv_data.create_dataset('I_corrected', data=I_corrected)
                uv_data.create_dataset('Q_corrected', data=Q_corrected)
                uv_data.create_dataset('U_corrected', data=U_corrected)
                
         
            except KeyError as e:
                    print(f'Skipping Acquisition {aqnum} in file {idx} — {e}')
                    continue
    finally: 
            
            f['Measurement_Metadata'].attrs['Processed Level'] = 'Level 0'
            f.close()
        
threads = cpu_count()

with ThreadPool(threads) as p:
        p.map(process0, idx_array)
           
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.4f} seconds")
