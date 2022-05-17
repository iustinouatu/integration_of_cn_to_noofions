import numpy as np
import os

# this script goes through all the folders resulted form the integration using MIDPOINT rule in 2Dims of the c_ns across the volume space
# and collects integral values for each such charge state, for each simulation (for each peak Intensity value)
# shall be invoked in the top directory (root one, parent one)
top_folder_path = os.getcwd()
base_path = top_folder_path + "/a0_"

a0s_long = np.loadtxt("Intensity_Wcm2_versus_a0_10_20_10_25_range_1000samples.txt", usecols=(1,))
num_sims = a0s_long.shape[0]

a0s_short = np.loadtxt("Intensity_Wcm2_versus_a0_10_20_10_25_range_1000samples.txt", usecols=(1,), max_rows=1)
os.chdir("a0_" + str(a0s_short))
onefile = np.loadtxt("integration_res_doubleMIDPOINT_new_ne_10minus9_Nr_RRon_powerbase2_maxpower12_1000samples_newBSIrate.txt")
num_of_cs_used = onefile.shape[1] - 1 # last column is Intensity
c_s = np.zeros((num_sims, num_of_cs_used))

os.chdir(top_folder_path)
#####################################   ----------------  ################################################################## --------------

#estimated_res_AITKEN = np.zeros((num_sims, num_of_cs_used), dtype=np.float128)
estimated_res_I4N = np.zeros((num_sims, num_of_cs_used), dtype=np.float128)
#estimated_ps_AITKEN = np.zeros((num_sims, num_of_cs_used), dtype=np.float128)
estimated_errs_simple = np.zeros((num_sims, num_of_cs_used), dtype=np.float128)
#estimated_errs_AITKEN = np.zeros((num_sims, num_of_cs_used), dtype=np.float128)
#estimated_errs_RICHARDSON = np.zeros((num_sims, num_of_cs_used), dtype=np.float128)


# for each line, the values of the diagnostics for that particular simulation idx are laid across the line, i.e. horizontally
# each column represents the result for one particular charge state c_n. there are num_of_cs_used such columns

intensitiesWcm2 = np.zeros((num_sims), dtype=np.float128)
# for directory in [x[0] for x in os.walk(top_folder_path)]: # first entry in os.walk(directory) is the directory name
how_many_folders_not_existent = 0
a0_values_notexistent = []
indices_of_a0_values_notexistent = []

idx = -1
contor = 0 # for the nans
for p in range(a0s_long.shape[0]): #
    idx += 1 # idx runs from 0 (first iteration, after the increment on this line) to num_sims-1
    results_folder = base_path + "{:.13f}".format(a0s_long[p])
    if os.path.exists(results_folder):
        os.chdir(results_folder)

        if os.path.isfile("integration_res_doubleMIDPOINT_new_ne_10minus9_Nr_RRon_powerbase2_maxpower12_1000samples_newBSIrate.txt"):
            a = np.loadtxt("integration_res_doubleMIDPOINT_new_ne_10minus9_Nr_RRon_powerbase2_maxpower12_1000samples_newBSIrate.txt") # last column is intensity.
            is_element_nan = np.isnan(a)
            #print("Feedback for whether these results file contains NaN's or not:")
            #print(np.any(is_element_nan))
            if np.any(is_element_nan)==True:
                contor += 1
            try:
                intensitiesWcm2[idx] = a[0, -1] # only get the intensity in Wcm2 from the current opened file of results
                remI = a[:, :-1] # last column is intensity. number of lines is num_of_cs_used

                # AITKEN extrapolation assumes we don't know the convergence order p of the integral series I_Nover2, I_N, I_2N, I_4N etc
                I_4N = remI[-1, :] # the last row is the most precise one, the most integration samples have been sued for the results appearing on this last row
                I_2N = remI[-2, :] # the second to last row is the next-to-most-precise-one
            # I_N = remI [-3, :]
            # estimated_res_AITKEN [idx, :] = (remI[-3,:] * remI[-1,:] - remI[-2,:]**2) / (remI[-3,:] + remI[-1,:] - 2*remI[-2,:]) # AITKEN extrapolation for integral value
            # the below AITKEN extrapolation formula is to avoid loss-of-significance errors in floating point arithmetics. the above doesn't do this.

            # estimated_res_AITKEN [idx, :] = I_4N - np.true_divide( np.square(I_4N - I_2N), ((I_4N-I_2N)-(I_2N-I_N)) )
            # estimated_res_AITKEN [idx, :] = remI[-1,:] - ( ((I_4N - I_2N)**2) / ( (I_4N - I_2N)-(I_2N - I_N) )  )
            # estimated_res_AITKEN [idx, :] = remI[-1,:] - ( np.true_divide( (np.square((remI[-1,:] - remI[-2,:]))) , ( (remI[-1,:]-remI[-2,:])-(remI[-2,:]-remI[-3,:]) ) ) )

                estimated_res_I4N[idx, :] = I_4N

            # estimated_ps_AITKEN [idx, :] = np.log2 ( (I_2N - I_N) / (I_4N - I_2N) ) # my curiosity to see what p is for these c'n integrals


                estimated_errs_simple[idx, :] = np.abs(I_4N - I_2N) # Assumes ground thruth is I_4N
            # estimated_errs_AITKEN [idx, :] =  ( ((I_4N - I_2N)**2) / ( (I_4N - I_2N)-(I_2N - I_N) )  )
            # estimated_errs_AITKEN [idx, :] = np.true_divide( (np.square(I_4N - I_2N)) , ((I_4N - I_2N)-(I_2N - I_N)) )

            # use the p's from estimated_ps_AITKEN to calculate error according to RICHARDSON error formula
            # estimated_errs_RICHARDSON [idx, :] = (1. / (2**(estimated_ps_AITKEN[idx,:]) - 1)) * (remI[-1,:] - remI[-2,:]) # I - I_N = (1/(2**p-1)) * (I_N - I_N/2)
            # estimated_errs_RICHARDSON [idx, :] = ( 1. / ( ((I_2N - I_N) / (I_4N - I_2N))-1 ) ) * (I_4N - I_2N)
            except:
                print("We have an error with this folder!")
    else:
        how_many_folders_not_existent += 1
        a0_values_notexistent.append(a0s_long[p])
        indices_of_a0_values_notexistent.append(idx)
        continue

    print("We have done folder number " + str(idx+1) + " , out of " + str(num_sims) + " folders")
    os.chdir(top_folder_path) # not necessary?

print("These many folders not existed: ")
print(how_many_folders_not_existent)
print("And the a0 values not existent: ")
print(a0_values_notexistent)
print("The Indices of non-existent a0 values: ")
print(indices_of_a0_values_notexistent) # we'll get rid of these entries from the np.zeros(shape) matrices initiated above, because these sims don't change the 0's from the initialization

print("contor for how many simulations have at least 1 NaN across all c_n's is: ")
print(contor)

#intensitiesWcm2 = np.delete(intensitiesWcm2, indices_of_a0_values_notexistent)
# estimated_res_AITKEN = np.delete(estimated_res_AITKEN, indices_of_a0_values_notexistent, axis=0)
#estimated_res_I4N = np.delete(estimated_res_I4N, indices_of_a0_values_notexistent, axis=0)
#estimated_errs_AITKEN = np.delete(estimated_errs_AITKEN, indices_of_a0_values_notexistent, axis=0)
#estimated_errs_simple = np.delete(estimated_errs_simple, indices_of_a0_values_notexistent, axis=0)
#estimated_errs_RICHARDSON = np.delete(estimated_errs_RICHARDSON, indices_of_a0_values_notexistent, axis=0)
#estimated_ps_AITKEN = np.delete(estimated_ps_AITKEN, indices_of_a0_values_notexistent, axis=0)

# The aim is to have a tableau like this, for each simulation:
# ------ header: not existent in file: c47, c48, c49, c50, c51, c52, c53, c54
# Intensity[W/cm2]
# result_AITKEN ...................................
# result_I4N
# convergence p ............................
# err_AITKEN ...............................
# err_SIMPLE = np.abs (I_4N - I_2N)
# err_RICHARDSON ...........................
# (7 rows, 8 columns) matrix
# for each simulation (ID = idx, idx runs from 0 to num_sims-1-deleted_sims)

# We save the above format in a 3D tensor, the third dimension being the idx of the simulation.
# the first 2 dimensions form the template above which is of shape (diagnostics = 7 lines , num_of_cs_used = 8 columns)
new_num_sims = estimated_res_I4N.shape[0]

all_matrices =  np.zeros( (7, num_of_cs_used, new_num_sims) )

all_matrices[0, :, np.arange(new_num_sims)] = intensitiesWcm2[np.arange(new_num_sims)].reshape((new_num_sims,1))
#all_matrices[1, :, np.arange(new_num_sims)] = estimated_res_AITKEN [np.arange(new_num_sims), :]
all_matrices[2, :, np.arange(new_num_sims)] = estimated_res_I4N[np.arange(new_num_sims), :]
#all_matrices[3, :, np.arange(new_num_sims)] = estimated_ps_AITKEN[np.arange(new_num_sims), :]
#all_matrices[4, :, np.arange(new_num_sims)] = estimated_errs_AITKEN[np.arange(new_num_sims), :]
all_matrices[5, :, np.arange(new_num_sims)] = estimated_errs_simple[np.arange(new_num_sims), :]
#all_matrices[6, :, np.arange(new_num_sims)] = estimated_errs_RICHARDSON[np.arange(new_num_sims), :]

print(all_matrices[0, 6, : ])
print(all_matrices[2, 6, :])
print(all_matrices[5, 6, :])


os.chdir(top_folder_path)

# use it as follows:
# all_matrices[diagnostic, charge_state, simulation_idx]
# where diagnostic runs from 0 to 6: IntensityWcm2, result_AITKEN, result_SIMPLE, error_AITKEN, error_SIMPLE, error_RICHARDSON
# charge state runs from 0 to num_of_cs_used
# simulation_idx runs from 0 to num_sims-1
np.save("results_integration_3DimTensor_1000samples_newBSIrate.npy", all_matrices)

# for readibility, want to save num_of_cs_used .txt files which can be open and inspected by human eye.
# for 1 .txt file for 1 particular c_n:
# # want to save as: many columns glued together, each column representing 1 simulation for this particular c_n at hand:
# # ----- header: not existent in file: ----- c47 only ------ (say, will be similar for c58, c49, etc.)
# # Intesity[W/cm2], Intesity[W/cm2], Intesity[W/cm2], Intesity[W/cm2], ...
# # result, result, result, result, result, ...
# # convergence p, convergence p, convergence p, convergence p, ...
# # err_AITKEN, err_AITKEN, err_AITKEN, err_AITKEN, err_AITKEN, ...
# # err_RICHARDSON, err_RICHARDSON, err_RICHARDSON, err_RICHARDSON, ...
# # this file will be read across columns. so 1 column represents 1 simulation's results for this particular c_N at hand.
