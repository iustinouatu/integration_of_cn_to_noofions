import os, numpy as np
import math
from math import pi
from mpi4py import MPI







from scipy.constants import elementary_charge, epsilon_0, electron_mass, c, pi
m_e = electron_mass

# get number of processors and processor rank
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size() # will be 8?

nsteps = 128 # will help to partition the z axis across the MPI processes launched to perform this single integral. see below

if rank == 0: # if this MPI process is the ROOT one
# this messy block of code takes care to divide as it should the piece of data to the MPI processes each waiting for a bunch of the data.
# just try it in a python3 terminal with some random numbers and see that it can handle them as it should, irrespective to how random the nprocs and nsteps are.
    ave, res = divmod(nsteps, nprocs)
    counts = [ave+1 if p < res else ave for p in range(nprocs)]
    starts = [sum(counts[:p]) for p in range(nprocs)]
    ends = [sum(counts[:p+1]) for p in range(nprocs)]

    data = [(starts[p], ends[p]) for p in range(nprocs)]

else: # if this MPI process is a WORKER one
    data = None # workers will receive their data via a .scatter(root=0) command, see below.

data = comm.scatter(data, root=0)

print("I am process number: ", rank)
print("and I have received data[0], data[1]: ", data[0], data[1])
########## ----------------------------------------------------------------------------------------------- ########################################
#### The Peak Intensity I_m for this Guassian Beam we are integrating across is obtained here.
## This I_m is different for every invocation of this .py script by the BASH of LINUX.
## It is changed by inputting a different value of a0 in line 6 of this .py script, by using 'awk' command inside BASH.
wavelength = 800 * 10**(-9) # 800 nm
w_0 = 3 * 10**(-6) # 3 um focal spot size, but expressed in SI units
z_R = (pi * w_0**2) / wavelength # Rayleigh range in SI units
a0_input = a0 # a0 will be inputted by the bash script inside the .sh submission script in the cluster
omega0 = 2 * pi * c / wavelength # laser light frequency in rad/s
E0 = (a0_input * m_e * c * omega0) / (elementary_charge)
Intensity_Wm2_output = (E0**2 * c * epsilon_0) / 2
Intensity_Wcm2_output = Intensity_Wm2_output / (10**4)
I_m = Intensity_Wm2_output # what this current script integrates. the current a0
I_thr = 10**24  # W/m2, SI units. equivalent to 10^20 W/cm2
######################################################################################################################################################

num_of_cs_used = 8
interpolators = dict()
for i in range(num_of_cs_used):
    interpolators[i+47] = np.load('../interpolator_Xenon_c{}_1dlinear_RRon_new_ne_10_minus_9_Nr_10_10_10_25_18000samples_Tsim22_newBSIrate.npy'.format(i+47), allow_pickle=True).item()

################ ------------ HELPER FUNCTIONS (Physics) ------- ########################################################################################
def get_end_of_zrange(I_m, I_thr): # the upper value on z axis, up to which we integrate (if you go higher than this with the z values, it's useless).
    return (np.sqrt( (I_m/I_thr) - 1) * z_R) # returns a z value in SI units!

def get_Imax_at_z(z): # maximum intensity (on axis), for the current value of z (input needs to be in SI units)
    return (I_m / ( 1 + (z**2/z_R**2) )   ) # returned in SI units, i.e. in W/m^2.   does it know about z_R ?

def get_w_of_current_z(z): # returns w in SI units, i.e. in meters m!  careful at units!
    w_of_z = w_0 * np.sqrt(  1 + (z**2/z_R**2)  ) # it knows about z_R , it's defined above!
    return w_of_z # returned in SI units

def get_r_thr(z_i):
    w_of_zi = get_w_of_current_z(z_i)
    return (  np.sqrt( (-w_of_zi**2 * np.log( (1+ (z_i/z_R)**2)*(I_thr/I_m)  ) ) / 2  )     )

def get_I_at_this_z_and_r(z , r):
    wz = get_w_of_current_z(z)
    return   ( I_m / ( (1+(z**2)/(z_R**2)) ) * np.exp(-2*((r**2)/(wz**2)) ) )    # Gaussian beam Intensity in SI units returned.

def get_I_matrix(r_j, z_i): # inputted arguments to this function shall be as:
    # r_j is a matrix shape (N_r, N_z),
    # z_i is a vector shape (N_z,)
    I_matrix = get_I_at_this_z_and_r(z_i[np.arange(N_z)], r_j[:, np.arange(N_z)]) # returns a shape(N_r, N_z)
    return I_matrix # shape (N_r, N_z)
############################### --------- END HELPER FUNCTIONS -------- ###########################################################################


z_range = 2 * get_end_of_zrange(Intensity_Wm2_output, I_thr) # the whole integral range (the Physics, no reference to any MPI things)
H = (data[1] - data[0]) * (z_range / nsteps)  # H = (end - start) * (z_range / nsteps) # the chunk of the whole z-axis z_range received by this MPI process
# H represents between which values do we integrate across the z-range in this MPI process (the chunk received by this particular MPI process).
# For this particular MPI process, the chunk size in adimensional units is represented by (data[1] - data[0]) / nsteps.
# The chunk size in Physical Units from the whole z range of integration is then simply the above multiplied by z_range.

maxpower = 12 # this is what we aim for. 2 runs of the for loop below: one with 2^11 divisions, one with 2^12 divisions.
for bzz in range(2): # for endeavor of calculating delta_IN on I_N integral and its dynamics, calculate the integral with 2^bzz datapoints across r AND z
    powerbase = 2
    N_z = powerbase ** (11 + bzz)
    N_r = powerbase ** (11 + bzz)

    z_i = (-z_range/2. + data[0]*z_range/nsteps) + (np.arange(N_z) + 0.5) * (H / N_z) # a + (i+0.5) * h_z, where h_z = (b-a) / N_z, shape (N_z,)
    # print("z_i is: ")
    # print(z_i)
    # print("Shape of z_i is: ")
    # print(z_i.shape)
    w_zi = get_w_of_current_z(z_i) # an array shape (N_z, ) 
    r_thresh = get_r_thr(z_i) # an array shape (N_z, )
    # print("r_thresh is: ")
    # print(r_thresh)
    # print("Shape of r_thresh is: ")
    # print(r_thresh.shape)
    # print("Min element of r_thresh is: ")
    # print(np.amin(r_thresh))
    # print("Location of the min element of r_thresh is: ")
    # print(np.where(r_thresh == np.amin(r_thresh)))
    # r_j = 0 + (np.arange(N_r) + 0.5) * ( (r_thresh - 0)  / N_r ) # c + (j+0.5) * h_r === c + (j+0.5) * (rmax-rmin) / (steps_along_r), where rmin=0, rmax=r_thres
    r_j = ((np.arange(N_r) + 0.5)/N_r).reshape(N_r,1) @ r_thresh.reshape(1,N_z) # dot product works in v. > Python 3.7
    # r_j has shape (N_r, N_z)
    # r_j is a matrix holding across columns (top-->bottom) the r-intervals (r-interval for that particular z value for that column)
    # so r_j[:, j] are N_r floats representing the partitioning of the interval [0-->r_thres[j]] into N_r pieces.
    # across lines (left-->right) it's the movement along the z-axis. starts from -z_max ends at z_max, where z_max is where we integrate up to across z-axis.

    # # print("shape of r_j is: ") # (N_r, N_z) for: r_j = ((np.arange(N_r) + 0.5)/N_r).reshape(N_r,1) @ r_thresh.reshape(1,N_z)
    # each column j out of the N_z columns of the r_j matrix is obtained as:
    # r_j[:,j] = (np.arange(N_r) + 0.5) * r_thresh[j] # where r_thresh is already populated with values and is shape (N_z,)

    # print("Min element of r_j is: ")
    # print(np.amin(r_j))

    I_matrix_r_z = get_I_matrix(r_j, z_i) # I_matrix_r_z is a matrix, shape (N_r, N_z), expressed in W/m2, SI units
    # I_matrix_r_z shows across columns (top-->bottom) the I values at that z, across the r-range for that z
    # so at the first column, I[0, 0] is Int at z= -zmax and r=0, I[0,1] is Int at z= -zmax and r=0 + 1*(smthR), I[0,2] is Int at z= -zmax and r =0 + 2*(smthR) ...
    # at the second column, I[0,1] is Int at z=-zmax+ 1*(smthZ) and r=0, I[1,1] is Int at z=-zmax + 1*(smthZ) and r = 0 + 1*(smthR)
    # ..... etc
    # For DEBUGGING purposes the lines below shall be uncommented:
    # print("Min intensity is:")
    # print(np.amin(I_matrix_r_z))
    # print("Location of the minimum intensity:")
    # ress = np.where(I_matrix_r_z == np.amin(I_matrix_r_z))
    # listOfCordinates = list(zip(ress[0], ress[1]))
    # for cord in listOfCordinates:
        # print(cord)
    # print("Shape of Imatrix is:") # (N_r, N_z)
    # print(I_matrix_r_z.shape)

    c_s = dict()
    for i in range(num_of_cs_used):
        c_s[i+47] = interpolators[i+47](I_matrix_r_z/(10**4)) # I_matrix_r_z has shape (N_r, N_z)

    c_s_times_rj = dict()
    for i in range(num_of_cs_used):
        c_s_times_rj[i+47] = c_s[i+47] * r_j # r_j has shape (N_r, N_z)

    h_r = (r_thresh[np.arange(N_z)] - 0) / N_r    # h_r has size (N_z, )

    c_s_times_rj_times_h_r = dict()
    for i in range(num_of_cs_used):
        c_s_times_rj_times_h_r[i+47] = c_s_times_rj[i+47] * h_r

    result_for_this_process = dict() # this contains the bit of the TOTAL-2D-integral RESULT as worked by one MPI process (the one calling this .py script)
    for i in range(num_of_cs_used):  # contains num_of_cs_used such bits, because we integrated num_of_cs_used interpolators (c_n's).
        result_for_this_process[i+47] = np.sum(c_s_times_rj_times_h_r[i+47]) * (H / N_z) * 2*pi


    # if rank == 0: # if statement not needed if only 1 process is launched. that only process which is launched is the root one.
    results_for_this_script = dict() # this contains the gathered bits for the TOTAL-2D-integral RESULT needed to assemble the TOTAL result for this particular 2D-integral.
    for i in range(num_of_cs_used):  # contains num_of_cs_used such baskets of results because we intragted num_of_cs_used interpolators (c_n's)
        results_for_this_script[i+47] = comm.gather(result_for_this_process[i+47], root=0) # if launched on 1 MPI process, quite useless to gather them all, but leave this instr. here if you want to launch on 2-4 MPI procesess

    final_2Dintegral_results = dict() # this contains the TOTAL-2D-integral RESULT as done for this particular 2D-integral
    for i in range(num_of_cs_used):
        final_2Dintegral_results[i+47] = np.sum(results_for_this_script[i+47])


# ----- calculate ERROR (estimated) on integral result and how it decays ----- #
    dataa = [final_2Dintegral_results[i+47] for i in range(num_of_cs_used)]
    dataa.append(Intensity_Wcm2_output)
    dataa = np.array(dataa)
    dataa = dataa[..., np.newaxis]
    dataa = np.transpose(dataa)
    filee = open("integration_res_doubleMIDPOINT_new_ne_10minus9_Nr_RRon_powerbase{}_maxpower{}_1000samples_newBSIrate.txt".format(powerbase, maxpower), "ab")
    # "ab" means open for appending A new information in binary mode B. places the pointer at the end of the file. if file with this name doesn't exist, creates one.
    np.savetxt(filee, dataa) # writes horizontal lines of results. leftmost float is the integration result for lowest XX in c_XX
    filee.close()

# this file will be arranged at end of the for-loop for bzz in range() (wrt maxpower variable) as:

# header (non existent in the actual file, shown here for explanatory purposes): c47, c48, c49, c50, c51, c52, c53, c54, Intensity(W/cm2)
# ------------------------------------------------------------------------------------------------------------------------
# c47_maxpower=11, c48_maxpower=11, c49_maxpower=11, c50_maxpower=11, c51_maxpower=11, c52_naxpower=11, c53_maxpower=11, c54_maxpower=11, I(W/cm2)
# c47_maxpower=12, c48_maxpower=12, c49_maxpower=12, c50_maxpower=12, c51_maxpower=12, c52_maxpower=12, c53_maxpower=12, c54_maxpower=12, I(W/cm2)
# ------------------------------------------------------------------------------------------------------------------------
