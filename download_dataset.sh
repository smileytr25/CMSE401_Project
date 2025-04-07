#!/bin/bash
#SBATCH --time 00:30:00 
#SBATCH --job-name "Download_Dataset"
pip install kagglehub 
python download_dataset.py
ls ./my_data
