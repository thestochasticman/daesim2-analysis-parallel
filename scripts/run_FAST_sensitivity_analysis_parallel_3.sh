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

wd=/home/y/daesim2-analysis
source /g/data/xe2/ya6227/daesim2-analysis-env/bin/activate
n_processes=32
n_samples=600
dir_results=/g/data/xe2/ya6227/daesim2-analysis-data/FAST_results
path_df_forcing_1=/g/data/xe2/ya6227/daesim2-analysis-data/DAESim_forcing_data/Rutherglen_1971.csv
paths_df_forcing=("$path_df_forcing_1")

python3.9 notebooks/FAST_sensitivity_analysis_parallel_3.py \
  --n_processes $n_processes \
  --n_samples $n_samples \
  --dir_results $dir_results \
  --paths_df_forcing "$(IFS=,; echo "${paths_df_forcing[*]}")"
  
