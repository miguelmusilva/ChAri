from general_segmentation_functions.image_handling import get_image, save_image, Image, ims_channels_to_h5
import numpy as np
import h5py

# Convert ims file to h5 file
imaris_path = '28cycles_2(10)_27t_teg001.ims'
h5_path = 'test_original.h5'

ims_channels_to_h5(
    ims_path=imaris_path, 
    out_name=h5_path
)

# General data
test = h5py.File(h5_path, 'r')
G_raw = test.get('raw_channels')
RS1_original = np.array(G_raw.get('RS1'))
RS2_original = np.array(G_raw.get('RS2'))
RS3_original = np.array(G_raw.get('RS3'))

# Threshold stablish  and obtaining the ratio between intensities
RS1_flat = RS1_original.flatten()
RS1_flat[RS1_flat < 5] = 0
mean1 = np.mean(RS1_flat)

RS2_flat = RS2_original.flatten()
RS2_flat[RS2_flat < 5] = 0
mean2 = np.mean(RS2_flat)

ratio = mean2 / mean1
ratio = 5.162160953404561

# Multiply the channel to get similar intensities
RS1_ratio = RS1_original * ratio
RS_only_organoids = np.subtract(RS2_original, RS1_ratio)

# Create new h5 file with the subtract -- 1 : T-cell, 2 : only organoids (T-cell have to be "delected", 3: dying die)
# TODO: corroborar que en el canal 2 se hayan eliminado correctamente todas las t-cell con la resta de canales realizada. 
with h5py.File('test_subtract', 'w') as hdf:
    hdf.create_dataset('RS1_original', data = RS1_original)
    hdf.create_dataset('RS2_subtract', data = RS_only_organoids)
    hdf.create_dataset('RS3_original', data = RS3_original)

# Proyection along time period
RS1_mod = np.sum(RS1_original, axis = 0)
RS2_mod = np.sum(RS_only_organoids, axis = 0)
RS3_mod = np.sum(RS3_original, axis = 0)

# Supongo que no se vera absolutamente nada. No esta normalizado.
with h5py.File('test_mod.h5', 'w') as hdf:
    hdf.create_dataset('RS1_mod', data = RS1_mod)
    hdf.create_dataset('RS2_mod', data = RS2_mod)
    hdf.create_dataset('RS3_mod', data = RS3_mod)

# Normalize all values.
norm_RS1 = (RS1_mod - np.min(RS1_mod)) / (np.max(RS1_mod) - np.min(RS1_mod))
norm_RS2 = (RS2_mod - np.min(RS2_mod)) / (np.max(RS2_mod) - np.min(RS2_mod))
norm_RS3 = (RS3_mod - np.min(RS3_mod)) / (np.max(RS3_mod) - np.min(RS3_mod))

with h5py.File('test_norm.h5', 'w') as hdf:
    hdf.create_dataset('norm_RS1', data = norm_RS1)
    hdf.create_dataset('norm_RS2', data = norm_RS2)
    hdf.create_dataset('norm_RS3', data = norm_RS3)