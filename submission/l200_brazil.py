"""
This is the script that actually runs toys at 0-signal against S-grid alternative hypothesis
"""
# import os module
import os

# turn off file locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# import python modules
import argparse

import numpy as np

from freqfit.limit import SetLimit
from freqfit.utils import load_config

# Setup the argument parser
__pars__ = argparse.ArgumentParser()

# include the arguments
__pars__.add_argument("-i", type=str, default="", help="path to the input yaml file")
__pars__.add_argument("-j", type=int, default=1, help="job id in the job array")
__pars__.add_argument(
    "-o",
    type=str,
    default="/data/eliza1/LEGEND/data/L200/limit/gerda_brazil/",
    help="output path to a folder to dump all the files in",
)
__pars__.add_argument("-n", type=int, default=1, help="number of toys to run")
__pars__.add_argument(
    "-s", type=str, default="0.055", help="signal rate to evaluate toys at"
)

# parse the arguments
__args__ = __pars__.parse_args()

# interpret the arguments
# proc_count = __args__.c if __args__.c < mp.cpu_count() else mp.cpu_count()
proc_input = __args__.i
proc_output = __args__.o
proc_job_id = __args__.j
proc_num_toy = __args__.n
proc_s = __args__.s


if __name__ == "__main__":
    print("Starting program!", flush=True)  # noqa:T201

    # S-grid to test 0-signal toys against
    x = np.linspace(0.1e-9, 0.2, 100)[
        1:
    ]  # remove the first element because limit.py inserts 1e-9 as first element in array
    s = np.around(x, 3)

    p = SetLimit(
        load_config(proc_input),
        proc_job_id,
        proc_num_toy,
        proc_output,
        name="l200_brazil",
    )

    p.set_var_to_profile("global_S")

    p.run_and_save_brazil(s)

    print("exiting program!", flush=True)  # noqa:T201
