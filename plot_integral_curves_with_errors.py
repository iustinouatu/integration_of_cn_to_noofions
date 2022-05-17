import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt

threeDimTensor = np.load("results_integration_3DimTensor_1000samples_newBSIrate.npy")

new_threeDimTensor = np.zeros((threeDimTensor.shape[0], threeDimTensor.shape[1], threeDimTensor.shape[2]))
indices = np.argsort(threeDimTensor[0, 1, :], axis=-1) # sorting all the simulations based on the Intensity value
new_threeDimTensor = threeDimTensor[:, :, indices] # keep the 2D tableaus intact, but arrange the sims in increasing order wrt Intensity

n0_cm3 = 1.98 * 10**(12)
n0_m3 = n0_cm3 * 10**6

# new_threeDimTensor[1,:,:] = n0_m3 * new_threeDimTensor[1,:,:] # results
# new_threeDimTensor[3,:,:] = n0_m3 * new_threeDimTensor[3,:,:] # error AITKEN
# new_threeDimTensor[4,:,:] = n0_m3 * new_threeDimTensor[4,:,:] # error RICHARDSON
# Stack Overflow SO method to eliminate the NaN's.
# -------------------------------------------------------------------------------------------
# is_element_nan = np.isnan(new_threeDimTensor[1, :, :]) # Across your 2nd row
# print(is_element_nan.shape) # shape (8, 14998)
# any_nan = np.any(is_element_nan, axis=0) # np.any flattens the 2d matrix.
# print(any_nan.shape) # (14998, )
# print(any_nan)
# print(~any_nan)
# clean_3Dtensor = new_threeDimTensor[:, :, ~any_nan] # ~is bitwise not'=
# print("Shape of clean_3Dtensor is:")
# print(clean_3Dtensor.shape) # (5, 8, 9632)

# My method to eliminate NaN's which at the moment doesn't work
# -------------------------------------------------------------------------------------------
# for idx in range(new_threeDimTensor.shape[2]): # for all simulations
#     boolean_array = np.isnan(new_threeDimTensor[1, :, idx]) # check if the results for any of the c_n's is NaN
#     # index = -1
#     condition = False
#     for element in boolean_array:
#         # index += 1
#         if element == True and condition == False:
#             condition = True
#             np.delete(new_threeDimTensor, idx, axis=2) # along last axis
###############################################################################################

# # Try to interpolate the NaN's appearing in the y-axis
# def interpolate_gaps(values, limit=None):
#     """
#     Fill gaps using linear interpolation, optionally only fill gaps up to a
#     size of `limit`.
#     """
#     i = np.arange(values.size)
#     valid = np.isfinite(values)
#     filled = np.interp(i, i[valid], values[valid])

#     # if limit is not None:
#     #     invalid = ~valid
#     #     for n in range(1, limit+1):
#     #         invalid[:-n] &= invalid[n:]
#     #     filled[invalid] = np.nan
#     return filled

##########################################################################################################################
# fig, ax = plt.subplots()
# for i in range(new_threeDimTensor.shape[1]): # for all c_n's
#     # ax.errorbar( new_threeDimTensor[0, i, :], new_threeDimTensor[1, i, :], yerr=new_threeDimTensor[3, i, :],
#     # markersize=0.2, linewidth=0.9,label='c{} result with AITKEN err'.format(i+47) )
#     # filled = interpolate_gaps(new_threeDimTensor[1,i,:])
#     # plt.errorbar(new_threeDimTensor[0,i,:], new_threeDimTensor[1,i,:], yerr=new_threeDimTensor[3,i,:],
#     #                markersize=0.2, linewidth=0.9, `, label='c{} with AITKEN error'.format(i+47))
#     plt.errorbar(new_threeDimTensor[0,i,:], new_threeDimTensor[1,i,:], yerr=new_threeDimTensor[3,i,:],
#                     markersize=0.2, linewidth=0.9, ecolor='black', label='c{} with AITKEN error'.format(i+47))
#     #ax.errorbar(threeDimTensor[0, i, :], threeDimTensor[1, i, :], yerr=threeDimTensor[4, i, :], label='c{} result with RICHARDSON err'.format(i+47))
# ax.set_ylabel("Integral result without n0 multiplication")
# ax.set_xlabel("Intensity [W/cm2]")
# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_xlim(left=10**20+1., right=10**25-1.)
# ax.legend()
# plt.savefig("integral_curves_c47_c54_AITKENerror.pdf")


# ##########################################################################################################################
# fig, ax = plt.subplots()
# for i in range(threeDimTensor.shape[1]):
#     #ax.errorbar(threeDimTensor[0, i, :], threeDimTensor[1, i, :], yerr=threeDimTensor[3, i, :], label='c{} result with AITKEN err'.format(i+47))
#     #ax.bar(threeDimTensor[0, i, :], threeDimTensor[3, i, :], label='c{} AITKEN error'.format(i+47))
#     ax.errorbar(threeDimTensor[0, i, :], threeDimTensor[1, i, :], yerr=threeDimTensor[4, i, :], label='c{} result with RICHARDSON err'.format(i+47))
# ax.set_ylabel("Integral result without n_0 multiplication")
# ax.set_xlabel("Intensity [W/cm2]")
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.legend()
# plt.savefig("integral_curves_c47_c54_RICHARDSONerror.pdf")

################################################################'
# s1mask = np.isfinite(new_threeDimTensor[1,:,:]) # results
# # print("Shape of s1mask is:")
# # print(s1mask.shape) # shape (8, 14998)
# s3mask = np.isfinite(new_threeDimTensor[3,:,:]) # error AITKEN
# s4mask = np.isfinite(new_threeDimTensor[4,:,:]) # error RICHARDSON

# plt.figure()
# for i in range(threeDimTensor.shape[1]-4):
#     # plt.scatter(new_threeDimTensor[0,i,:], new_threeDimTensor[1, i, :], s=0.5, linewidths=0.5, label="c{}".format(i+47))
#     plt.errorbar(new_threeDimTensor[0,i,:], new_threeDimTensor[1,i,:], yerr=new_threeDimTensor[3,i,:], linewidth=0.3, elinewidth=0.1,
#          ecolor='black', capsize=0.01, label='c_{} with AITKEN errs'.format(i+47))
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim([10**20 + 1., 10**25 - 1.])
# plt.ylim([10.**(-28),10.**(-10)])
# plt.xlabel("Intensity [W/cm2]")
# #plt.ylabel("Number of ions produced in the focus")
# plt.ylabel("Integral estimate")
# plt.title("integral estimates, versus peak Intensity")
# plt.legend()
# plt.savefig('presentation_graph1.pdf', bbox_inches='tight')



# plt.figure()
# plt.errorbar(new_threeDimTensor[0,-1,:], new_threeDimTensor[1,-1,:], yerr=new_threeDimTensor[3,-1,:], linewidth=0.3, elinewidth=0.1,
#          ecolor='black', capsize=0.01, label='c_54 with AITKEN errs')
# plt.errorbar(new_threeDimTensor[0,-2,:], new_threeDimTensor[1,-2,:], yerr=new_threeDimTensor[3,-2,:], linewidth=0.3, elinewidth=0.1,
#          ecolor='black', capsize=0.01, label='c_53 with AITKEN errs')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim([10**20 + 1., 10**25 - 1.])
# plt.ylim([10.**(-28),10.**(-10)])
# plt.xlabel("Intensity [W/cm2]")
# #plt.ylabel("Number of ions produced in the focus")
# plt.ylabel("Integral estimate")
# plt.title("integral estimates versus peak I")
# plt.legend()
# plt.savefig('presentation_graph2.pdf', bbox_inches='tight')

# plt.figure()
# plt.errorbar(new_threeDimTensor[0,-1,:], new_threeDimTensor[1,-1,:], yerr=new_threeDimTensor[3,-1,:], linewidth=0.3, elinewidth=0.2,
#          ecolor='black', capsize=0.1, label='c_54 with AITKEN errs')
# plt.errorbar(new_threeDimTensor[0,-2,:], new_threeDimTensor[1,-2,:], yerr=new_threeDimTensor[3,-2,:], linewidth=0.3, elinewidth=0.2,
#          ecolor='black', capsize=0.1, label='c_53 with AITKEN errs')
# plt.errorbar(new_threeDimTensor[0,-3,:], new_threeDimTensor[1,-3,:], yerr=new_threeDimTensor[3,-3,:], linewidth=0.3, elinewidth=0.2,
#          ecolor='black', capsize=0.1, label='c_52 with AITKEN errs')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim([10**20 + 1., 10**25 - 1.])
# plt.ylim([10.**(-28),10.**(-10)])
# plt.xlabel("Intensity [W/cm2]")
# plt.ylabel("integral estimate")
# plt.title("integral estimates versus peak I")
# plt.legend()
# plt.savefig('presentation_graph3.pdf', bbox_inches='tight')

# plt.figure()
# plt.scatter(new_threeDimTensor[0,-1,:], new_threeDimTensor[2,-1,:], label='c54',s=0.2)
# plt.scatter(new_threeDimTensor[0,-1,:], new_threeDimTensor[2,-2,:], label='c53',s=0.2)
# plt.scatter(new_threeDimTensor[0,-1,:], new_threeDimTensor[2,-3,:], label='c52', s=0.2)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim([10**20 + 1., 10**25 - 1.])
# plt.ylim([10.**(-28),10.**(-10)])
# plt.xlabel("Intensity [W/cm2]")
# plt.ylabel("integral estimate")
# plt.title("integral estimates versus peak I")
# plt.legend()
# plt.savefig('presentation_graph4.pdf', bbox_inches='tight')


# plt.plot(new_threeDimTensor[0,-1,:], new_threeDimTensor[1,-1,:], label='c54') # shows spikes at some random values of Intensity
# plt.plot(new_threeDimTensor[0,-1,:], new_threeDimTensor[1,-2,:], label='c53') # shows spikes at some random values of Intensity
# plt.plot(new_threeDimTensor[0,-1,:], new_threeDimTensor[1,-3,:], label='c52') # shows spikes at some random values of Intensity
# plt.plot(new_threeDimTensor[0,-1,:], new_threeDimTensor[1,-4,:], label='c51') # shows spikes at some random values of Intensity
# plt.plot(new_threeDimTensor[0,-1,:], new_threeDimTensor[1,-5,:], label='c50') # shows spikes at some random values of Intensity
# plt.plot(new_threeDimTensor[0,-1,:], new_threeDimTensor[1,-6,:], label='c49') # shows spikes at some random values of Intensity
# plt.plot(new_threeDimTensor[0,-1,:], new_threeDimTensor[1,-7,:], label='c48') # shows spikes at some random values of Intensity
# plt.plot(new_threeDimTensor[0,-1,:], new_threeDimTensor[1,-8,:], label='c47') # shows spikes at some random values of Intensity

#############################################################################################################################
# print(np.abs(new_threeDimTensor[5,-2,:])*n0_m3) # (1000,)

plt.figure()
plt.errorbar(new_threeDimTensor[0,-1,:], new_threeDimTensor[2,-1,:]*n0_m3, yerr=np.abs(new_threeDimTensor[5,-1,:]*n0_m3), color='red', ecolor='black', elinewidth=0.1,   label=r'$Xe^{54+}$ ions, with error', markersize=0.01) 
plt.errorbar(new_threeDimTensor[0,-1,:], new_threeDimTensor[2,-2,:]*n0_m3, yerr=np.abs(new_threeDimTensor[5,-2,:]*n0_m3), color='green', ecolor='black', elinewidth=0.1,  label=r'$Xe^{53+}$ ions, with error', markersize=0.01) 
plt.errorbar(new_threeDimTensor[0,-1,:], new_threeDimTensor[2,-3,:]*n0_m3, yerr=np.abs(new_threeDimTensor[5,-3,:]*n0_m3), color='sienna', ecolor='black', elinewidth=0.1,  label=r'$Xe^{52+}$ ions, with error', markersize=0.01) 
plt.errorbar(new_threeDimTensor[0,-1,:], new_threeDimTensor[2,-4,:]*n0_m3, yerr=np.abs(new_threeDimTensor[5,-4,:]*n0_m3), color='cyan', ecolor='black', elinewidth=0.1, label=r'$Xe^{51+}$ ions, with error', markersize=0.01) 
plt.errorbar(new_threeDimTensor[0,-1,:], new_threeDimTensor[2,-5,:]*n0_m3, yerr=np.abs(new_threeDimTensor[5,-5,:]*n0_m3), color='purple', ecolor='black', elinewidth=0.1, label=r'$Xe^{50+}$ ions, with error', markersize=0.01) 
plt.errorbar(new_threeDimTensor[0,-1,:], new_threeDimTensor[2,-6,:]*n0_m3, yerr=np.abs(new_threeDimTensor[5,-6,:]*n0_m3), color='blue', ecolor='black', elinewidth=0.1, label=r'$Xe^{49+}$ ions, with error', markersize=0.01) 
plt.errorbar(new_threeDimTensor[0,-1,:], new_threeDimTensor[2,-7,:]*n0_m3, yerr=np.abs(new_threeDimTensor[5,-7,:]*n0_m3), color='orange', ecolor='black', elinewidth=0.1, label=r'$Xe^{48+}$ ions, with error', markersize=0.01) 
plt.errorbar(new_threeDimTensor[0,-1,:], new_threeDimTensor[2,-8,:]*n0_m3, yerr=np.abs(new_threeDimTensor[5,-8,:]*n0_m3), color='dimgray', ecolor='black', elinewidth=0.1, label=r'$Xe^{47+}$ ions, with error', markersize=0.01) 
plt.xscale('log', nonpositive='clip')
plt.yscale('log', nonpositive='clip')
plt.xlim([10**20 + 1., 10**25 - 1.])
plt.ylim([10.**(-3), 10**(9)])
plt.xlabel("Intensity [Wcm" + r'$^{-2}$' + "]", fontsize=14)
plt.ylabel("Number of ions produced in the focus", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
# plt.show()
plt.savefig("Integration_Xe_simple_estimates_simpleerror_1000samples_newBSIrate.pdf", bbox_inches='tight')
#############################################################################################################################

