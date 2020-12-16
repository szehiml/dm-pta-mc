"""
    Main program
"""

import numpy as np
from mpi4py import MPI
import sys

from src.os_util import my_mkdir
from src.input_parser import get_input_variables, get_c_list
import src.parallel_util as put
from src.snr import opt_pulsar_snr
import src.signals as signals
import src.snr as snr
import src.generate_sim_quants as gsq
import src.constants as const

#####

# initializing MPI
comm = MPI.COMM_WORLD

# get the total number of processors
n_proc = comm.Get_size()

# get the id of processor
proc_id = comm.Get_rank()

# ID of main processor
root_process = 0

if proc_id == root_process:

    print("--- DM - PTA - MC ---")
    print()
    print("    v1.0")
    print()
    print("    Running on " + str(n_proc) + " processors")
    print()
    print("---")
    print()
    print("Reading input file...")
    print()

in_filename = sys.argv[-1]
in_dict = get_input_variables(in_filename)

if proc_id == root_process:

    print("Done reading input file!")
    print()
    print("    Input variables:")

    for key in in_dict:

        print("        " + key + " : " + str(in_dict[key]))

    print()
    print("---")
    print()

# tell ehich processors what to compute for
job_list = None
job_list_recv = None

if proc_id == root_process:

    # number of jobs to do
    num_jobs = in_dict["NUM_UNIVERSE"]
    total_job_list = []

    for ui in range(num_jobs):
        total_job_list.append([ui])

    job_list = put.generate_job_list(n_proc, np.array(total_job_list))

job_list_recv = comm.scatter(job_list, root=root_process)

if proc_id == root_process:

    print("    Determining simulation variables...")
    print()

dt = const.week_to_s * in_dict["DT_WEEK"]
obs_T = const.yr_to_s * in_dict["T_YR"]

# number of time points
Nt = int(obs_T / dt)
t_grid = np.linspace(0, dt * Nt, num=Nt, endpoint=False)
t_grid_yr = t_grid / const.yr_to_s

v_bar = const.km_s_to_kpc_yr * in_dict["V_BAR_KM_PER_SEC"]
v_0 = const.km_s_to_kpc_yr * in_dict["V_0_KM_PER_SEC"]
v_E = const.km_s_to_kpc_yr * in_dict["V_E_KM_PER_SEC"]
v_Esc = const.km_s_to_kpc_yr * in_dict["V_ESC_KM_PER_SEC"]
max_R = in_dict["R_FACTOR"] * v_bar * t_grid_yr[-1]

if proc_id == root_process:
    verbose = True
else:
    verbose = False

[num_objects, max_R, log10_M_min] = gsq.set_num_objects(
    max_R,
    log10_f=in_dict["LOG10_F"],
    log10_M=in_dict["LOG10_M"],
    use_HMF=in_dict["USE_HMF"],
    HMF_path=in_dict["HMF_PATH"],
    log10_M_min=in_dict["LOG10_M_MIN"],
    min_num_object=in_dict["MIN_NUM_OBJECT"],
    verbose=verbose,
)

# generate positions of pulsars (same across all universes)
dhat_list = gsq.gen_dhats(in_dict["NUM_PULSAR"])

if proc_id == root_process:

    print("    Number of time points                  = " + str(Nt))
    print("    Number of subhalos per pulsar/earth    = " + str(num_objects))
    print("    Radius of simulation sphere            = " + str(max_R) + " kpc")
    print()

    if in_dict["USE_HMF"]:

        print("    Halo mass function M_min = " + str(10 ** log10_M_min) + " M_sol")
        print()

    print("---")
    print()

snr_list = []

for job in range(len(job_list_recv)):

    if job_list_recv[job, 0] != -1:

        uni_id = job_list_recv[job, 0]

        if in_dict["CALC_TYPE"] == "pulsar":

            if proc_id == root_process and job == 0:

                print("Starting PULSAR term calculation...")
                print()
                print("    Generating signals and computing optimal pulsar SNR...")

            for pul in range(in_dict["NUM_PULSAR"]):

                r0_list = gsq.gen_positions(max_R, num_objects)

                v_list = gsq.gen_velocities(v_0, v_Esc, v_E, num_objects)

                mass_list = gsq.gen_masses(
                    num_objects,
                    use_HMF=in_dict["USE_HMF"],
                    log10_M=in_dict["LOG10_M"],
                    HMF_path=in_dict["HMF_PATH"],
                    log10_M_min=log10_M_min,
                )

                conc_list = get_c_list(
                    mass_list,
                    in_dict["USE_FORM"],
                    in_dict["USE_CM"],
                    c=in_dict["C"],
                    cM_path=in_dict["CM_PATH"],
                )

                d_hat = dhat_list[pul]

                dphi = signals.dphi_dop_chunked(
                    t_grid_yr,
                    mass_list,
                    r0_list,
                    v_list,
                    d_hat,
                    use_form=in_dict["USE_FORM"],
                    conc=conc_list,
                    use_chunk=in_dict["USE_CHUNK"],
                    chunk_size=in_dict["CHUNK_SIZE"],
                )

                ht = signals.subtract_signal(t_grid, dphi)

                snr_val = snr.opt_pulsar_snr(
                    ht, in_dict["T_RMS_NS"], in_dict["DT_WEEK"]
                )

                snr_list.append([uni_id, pul, snr_val])

            if proc_id == root_process and job == len(job_list_recv) - 1:
                print("    Done computing SNR!")
                print()
                print("Returning data to main processor...")
                print()

        if in_dict["CALC_TYPE"] == "earth":

            if proc_id == root_process:

                print("Starting EARTH term calculation...")
                print()
                print("    Generating signals and computing optimal Earth SNR...")

            r0_list = gsq.gen_positions(max_R, num_objects)

            v_list = gsq.gen_velocities(v_0, v_Esc, v_E, num_objects)

            mass_list = gsq.gen_masses(
                num_objects,
                use_HMF=in_dict["USE_HMF"],
                log10_M=in_dict["LOG10_M"],
                HMF_path=in_dict["HMF_PATH"],
                log10_M_min=log10_M_min,
            )

            conc_list = get_c_list(
                mass_list,
                in_dict["USE_FORM"],
                in_dict["USE_CM"],
                c=in_dict["C"],
                cM_path=in_dict["CM_PATH"],
            )

            dphi_vec = signals.dphi_dop_chunked_vec(
                t_grid_yr,
                mass_list,
                r0_list,
                v_list,
                use_form=in_dict["USE_FORM"],
                conc=conc_list,
                use_chunk=in_dict["USE_CHUNK"],
                chunk_size=in_dict["CHUNK_SIZE"],
            )  # (Nt, 3)

            ht_list = np.zeros((in_dict["NUM_PULSAR"], Nt))

            for pul in range(in_dict["NUM_PULSAR"]):

                d_hat = dhat_list[pul]

                dphi = np.einsum("ij,j->i", dphi_vec, d_hat)

                ht = signals.subtract_signal(t_grid, dphi)
                ht_list[pul, :] = ht

            snr_val = snr.opt_earth_snr(
                ht_list, in_dict["T_RMS_NS"], in_dict["DT_WEEK"]
            )

            snr_list.append([uni_id, -1, snr_val])

            if proc_id == root_process and job == len(job_list_recv) - 1:
                print("    Done computing SNR!")
                print()
                print("Returning data to main processor...")
                print()

# return data back to root
all_snr_list = comm.gather(snr_list, root=root_process)

# write to output file
if proc_id == root_process:

    print("Done returning data!")
    print()
    print("Writing data to output file...")

    my_mkdir(in_dict["OUTPUT_DIR"])

    file = open(
        in_dict["OUTPUT_DIR"]
        + "snr_"
        + in_dict["CALC_TYPE"]
        + "_"
        + in_dict["RUN_DESCRIP"]
        + ".txt",
        "w",
    )

    for i in range(n_proc):
        for j in range(len(all_snr_list[i])):

            # universe_index = universe_index_list[int(all_A_stat_list[i][j][0])]
            snr_final = all_snr_list[i][j][1]

            file.write(
                str(int(all_snr_list[i][j][0]))
                + " , "
                + str(all_snr_list[i][j][1])
                + " , "
                + str(all_snr_list[i][j][2])
            )
            file.write("\n")

    file.close()

    print("Done writing data!")
    print("---")
    print()
