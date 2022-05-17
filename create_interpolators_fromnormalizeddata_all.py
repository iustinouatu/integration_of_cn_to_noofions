import scipy
import numpy as np
import os
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
import matplotlib
from matplotlib import pyplot as plt



Intensities_Xe = np.loadtxt("results_10_10_10_25_wcm2_RRon_ne_10minus9_Nr_Tsim22_correctBSI_18000samples_Normalized.txt", usecols=(55,), skiprows=1)
len_of_Intensities = len(Intensities_Xe)
print(len_of_Intensities)
print(np.amax(Intensities_Xe)) # 10^25
print(np.amin(Intensities_Xe)) # 10^20

available_data = dict()
for i in range(55):
    available_data[i] = np.loadtxt("results_10_10_10_25_wcm2_RRon_ne_10minus9_Nr_Tsim22_correctBSI_18000samples_Normalized.txt", usecols=(i,), skiprows=1)

linear_objects = dict()
for i in range(53): # last interpolator c54 needs special attention, so 54-1=53 is in range() there
    linear_objects[i+1] = interp1d(Intensities_Xe, available_data[i+1], kind='linear', bounds_error=False, fill_value=(0.0, 0.0))

# c_0 special attention at the fill value of LHS
linear_objects[0] = interp1d(Intensities_Xe, available_data[0], kind='linear', bounds_error=False, fill_value=(1.0, 0.0))
# c_54 special attention at the fill value of RHS
linear_objects[54] = interp1d(Intensities_Xe, available_data[54], kind='linear', bounds_error=False, fill_value=(0.0, 1.0))

for i in range(55):
    np.save("interpolator_Xenon_c{}_1dlinear_RRon_new_ne_10_minus_9_Nr_10_10_10_25_18000samples_Tsim22_newBSIrate".format(i), np.array(linear_objects[i]))
