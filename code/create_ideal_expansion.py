import os
import sys
import math
import numpy as np
import sv
sys.path.append("C:/Users/Faiza/Desktop/PHM_continue2024/code")
import helper_functions
sys.path.pop()


model_name = "r1l5g0f125"
exp_degree = 0
x0 = 0
xf = 5
nx = 31 
L = xf-x0
x = np.linspace(x0, xf, nx)
r_inlet = 1



A_inlet = np.pi * (r_inlet) ** 2
A_outlet = ((100+exp_degree) / 100) * A_inlet
r_outlet = np.sqrt(A_outlet / np.pi)
m = (r_outlet - r_inlet) / (xf - x0)
r = m * x + r_inlet


print('m: ', m)
print('ro: ', r_outlet)
print('r: ', r)

path_name = model_name + "_path"
segmentations_name = model_name + "_segmentations"
############################################################
############################################################
############################################################
############################################################
############################################################


path_points_array = np.zeros((len(x), 3))
path_points_array[:, 2] = x # make tube along z-axis
path_points_list = path_points_array.tolist()

# make radii list
radii_list = r.tolist()

if len(radii_list) != len(path_points_list):
    print("Error. Number of points in radius list does not match number of points in path list.")
#
# # create path and segmnetation objects
path = helper_functions.create_path_from_points_list(path_points_list)
segmentations = helper_functions.create_segmentations_from_path_and_radii_list(path, radii_list)
#
# # add path and segmentations to the SimVascular Data Manager (SV DMG) for visualization in GUI
sv.dmg.add_path(name = path_name, path = path)
sv.dmg.add_segmentation(name = segmentations_name, path = path_name, segmentations = segmentations)
