singularity exec --bind /data/eliza1/LEGEND/:/data/eliza1/LEGEND/,/home/sjborden:/home/sjborden /data/eliza1/LEGEND/sw/containers/python3-10.sif python3 /home/sjborden/freqfit/l200_s_grid.py -i "/home/sjborden/freqfit/legendfreqfit/analysis/legend/legend_neutrino2024_config.yaml" -o "/data/eliza1/LEGEND/data/L200/limit/l200_toys" -e "l200"
cat /data/eliza1/LEGEND/data/L200/limit/l200_toys/s.txt | while read line
do
    export svalue=$line
    qsub run_l200.sh
done
