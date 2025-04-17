"""
This is the script that creates the s-grid for toys to be run on, used for job submission
"""
# import os module
import os

# turn off file locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# import python modules
import argparse

import h5py
import numpy as np

from freqfit.limit import SetLimit
from freqfit.utils import load_config

# Setup the argument parser
__pars__ = argparse.ArgumentParser()

# include the arguments
__pars__.add_argument("-i", type=str, default="", help="path to the input yaml file")
__pars__.add_argument("-e", type=str, default="majorana", help="name of the experiment")
__pars__.add_argument(
    "-o",
    type=str,
    default="/home/sjborden/freqfit/gerda_toys",
    help="output path to a folder to dump all the files in",
)

# parse the arguments
__args__ = __pars__.parse_args()

# interpret the arguments
# proc_count = __args__.c if __args__.c < mp.cpu_count() else mp.cpu_count()
proc_input = __args__.i
proc_output = __args__.o
proc_experiment = __args__.e


if __name__ == "__main__":
    print("Starting to profile the test statistic!", flush=True)  # noqa:T201
    # Create an s-grid to profile the test statistic on
    x = np.linspace(0.1e-9, 0.2, 100)[:]
    x = np.around(x, 3)
    x[
        0
    ] = 1e-9  # This is the first value that the Brazil toys will be scanned at in limit.py, need to ensure both evaluated at same point

    p = SetLimit(load_config(proc_input), out_path=proc_output, name=proc_experiment)

    p.set_var_to_profile("global_S")
    ts_fine = p.scan_ts(x)

    # Save to a file
    f = h5py.File(proc_output + f"/{proc_experiment}_ts_wilks_fine.h5", "a")
    dset = f.create_dataset("s", data=x)
    dset = f.create_dataset("t_s", data=ts_fine)
    f.close()

    # Create the text file used for toy submission
    with open(proc_output + "/s.txt", "a") as file_out:
        for s_value in x:
            file_out.write(f"{s_value}\n")

    print("exiting program!", flush=True)  # noqa:T201
