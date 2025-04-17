"""
This is the script that actually runs toys for a given value of S
"""
# import os module
import os

# turn off file locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# import python modules
import argparse

from freqfit.limit import SetLimit
from freqfit.utils import load_config

# Setup the argument parser
__pars__ = argparse.ArgumentParser()

# include the arguments
__pars__.add_argument("-i", type=str, default="", help="path to the input yaml file")
__pars__.add_argument(
    "-s", type=str, default="0.055", help="signal rate to evaluate toys at"
)
__pars__.add_argument("-j", type=int, default=1, help="job id in the job array")
__pars__.add_argument(
    "-o",
    type=str,
    default="/home/sjborden/freqfit/gerda_toys",
    help="output path to a folder to dump all the files in",
)
__pars__.add_argument("-n", type=int, default=1, help="number of toys to run")

# parse the arguments
__args__ = __pars__.parse_args()

# interpret the arguments
# proc_count = __args__.c if __args__.c < mp.cpu_count() else mp.cpu_count()
proc_input = __args__.i
proc_s = __args__.s
proc_output = __args__.o
proc_job_id = __args__.j
proc_num_toy = __args__.n


if __name__ == "__main__":
    print("Starting program!", flush=True)  # noqa:T201

    p = SetLimit(
        load_config(proc_input),
        proc_job_id,
        proc_num_toy,
        proc_output,
        name="l200_toys",
    )

    p.set_var_to_profile("global_S")

    print(f"profiling {float(proc_s)}", flush=True)  # noqa:T201

    p.run_and_save_toys(float(proc_s))

    print("exiting program!", flush=True)  # noqa:T201
