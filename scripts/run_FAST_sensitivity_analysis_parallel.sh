#!/bin/bash
#PBS -N daesim2analysis
#PBS -P xe2
#PBS -q express
#PBS -l mem=12GB
#PBS -l jobfs=3GB
#PBS -l ncpus=3
#PBS -l walltime=00:30:00
#PBS -l storage=gdata/xe2
#PBS -o /home/272/ya6227/daesim2-analysis/deployment/outputs/output.log
#PBS -e /home/272/ya6227/daesim2-analysis/deployment/outputs/error.log

wd=/home/y/daesim2-analysis/

n_processes=32
n_samples=300
xsite=Rutherglen_1971_test
dir_results=/home/y/daesim2-analysis-data/
path_df_forcing_1=/home/y/data-daesim2-analysis-data/DAESim_forcing_Milgadara_2018.csv
path_df_forcing_2=/home/y/data-daesim2-analysis-parallel/DAESim_forcing_Milgadara_2019.csv
path_df_forcing_Rutherglen_1971=/home/y/daesim2-analysis-data/DAESim_forcing_Rutherglen_1971.csv
paths_df_forcing=("$path_df_forcing_Rutherglen_1971")

python3.9 notebooks/FAST_sensitivity_analysis_parallel.py \
  --n_processes $n_processes \
  --n_samples $n_samples \
  --xsite $xsite \
  --dir_results $dir_results \
  --paths_df_forcing "${paths_df_forcing[@]}"

