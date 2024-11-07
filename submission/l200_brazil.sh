#!/bin/bash

#$ -S /bin/bash                      #use bash
#$ -m n                              # don't send mail when job starts or stops.
#$ -w e                              #verify syntax and give error if so
#$ -V                                #inherit environment variables
#$ -N l200_brazil                   #job name
#$ -e /data/eliza1/LEGEND/jobs/l200_brazil.e  #error output of script
#$ -o /data/eliza1/LEGEND/jobs/l200_brazil.o  #standard output of script
#$ -l h_rt=53:50:00                  #hard time limit, your job is killed if it uses this much cpu.
#$ -l s_rt=52:50:00                   #soft time limit, your job gets signaled when you use this much time. Maybe you can gracefully shut down?
#$ -cwd                              #execute from the current working directory
#$ -t 1-10                         #give me N identical jobs, labelled by variable SGE_TASK_ID
#$ -pe smp 10                         # Give me ten processors on this node!!!!

#execute the $SGE_TASK_ID'th sub-job
set -x # I think this makes everything beyond this call get saved to the log

echo ${NSLOTS}
echo ${SGE_TASK_ID}
echo $TMPDIR
echo $PYTHONUSERBASE
cd "/home/sjborden"

singularity exec --bind /data/eliza1/LEGEND/:/data/eliza1/LEGEND/,/home/sjborden:/home/sjborden /data/eliza1/LEGEND/sw/containers/python3-10.sif python3 /home/sjborden/freqfit/l200_brazil.py -j ${SGE_TASK_ID} -i "/home/sjborden/freqfit/legendfreqfit/analysis/legend/legend_neutrino2024_config.yaml" -o "/data/eliza1/LEGEND/data/L200/limit/l200_brazil" -n 10000 -s "0.05"
