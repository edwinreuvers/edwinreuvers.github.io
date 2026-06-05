# -*- coding: utf-8 -*-
"""
This script computes summary values used in the manuscript methods section.

It reports how far the experimentally imposed average MTC length was below the
estimated optimum MTC length and compares stimulation durations between the
preliminary simulations used for condition selection and the experimental SSC
trials.
"""

#%%
import os, glob, pickle, sys
import pandas as pd
import numpy as np
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

from _FuncGen import get_stimDur

#%% Compute how much average MTC length is below optimum MTC length
diffLmus = []
for mus in ['GMe1','GMe2','GMe3']:
    diffL = []
    for exp in ['SSC_PA', 'SSC_PB']:
        parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
        muspar,dataQR,dataSR,dataISOM = pickle.load(open(parFile, 'rb'))
        lmtcOpt = muspar['lce_opt'] + (muspar['fmax']/muspar['ksee'])**0.5 + muspar['lsee0']
        
        files = glob.glob(os.path.join(dataDir,mus,'dataExp',exp,'*.csv'))
        for iFile,filename in enumerate(files):
            # data = OpenSMRfile(filename)
            df = pd.read_csv(filename)
            data = df.to_numpy()
            time,lmtc,stim,fsee = data.T[0:4]
            
            diffL.append(lmtc.mean()-lmtcOpt)
        diffLmus.append(np.mean(diffL))

diffLmean = np.mean(diffLmus)*1e3
diffLstd = np.std(diffLmus)*1e3
print(f"Average MTC length - Opt MTC length = {diffLmean:.2f} mm")
print(f"Standard deviation thereof = {diffLstd:.2f} mm")

#%% Difference between experimental stimulation duration and simulations
# Get stimulation duration of simulations first
fts = np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.80, 0.65, 0.35, 0.20, 0.80, 0.65, 0.35, 0.20])
muscles = ['GMz1', 'GMz2', 'GMz3']
durStim_prelim = []

# Loop over muscles
for iMus, mus in enumerate(muscles):
    muscle_durStim = []  # Temporary list to hold stim durations for one muscle
    
    # Loop over experiments
    for exp, amp, cf in [('SSC_PA', 2, np.array([1, 2, 3, 4, 5, 3, 3, 3, 3, 5, 4, 2, 1])),
                         ('SSC_PB', 4, np.array([1, 1.5, 2, 2.5, 3, 2, 2, 2, 2, 3, 2.5, 1.5, 1]))]:
        
        # Collect filepaths for the conditions for this experiment
        filepaths = [os.path.join(baseDir, 'suppmat', 'sel-ssc-conditions', mus, 
                                 f'{mus}_amp{amp:0.1f}mm_cf{cf[iCond]:0.1f}Hz_fts{fts[iCond]:0.2f}.csv') 
                     for iCond in range(13)]
        
        # Get stim duration data for current experiment and muscle
        durStim = get_stimDur(filepaths)
        
        # Append the stim durations for this experiment (convert to ms)
        muscle_durStim.extend(durStim)  # Extend adds the 13 conditions from this experiment

    # Append the stim durations for this muscle to the overall list
    durStim_prelim.append(muscle_durStim)

# Convert the list to a numpy array and ensure it has the correct shape (3x26)
durStim_prelim = np.array(durStim_prelim)

# Get stimulation duration of exp second
muscles = ['GMe1', 'GMe2', 'GMe3']
durStim_exp = []

# Loop over muscles
for iMus, mus in enumerate(muscles):
    muscle_durStim = []  # Temporary list to hold stim durations for one muscle
    
    # Loop over experiments (SSC_PA and SSC_PB)
    for exp, amp, cf in [('SSC_PA', 2, np.array([1, 2, 3, 4, 5, 3, 3, 3, 3, 5, 4, 2, 1])),
                         ('SSC_PB', 4, np.array([1, 1.5, 2, 2.5, 3, 2, 2, 2, 2, 3, 2.5, 1.5, 1]))]:
        
        # Collect filepaths for the conditions for this experiment
        filepaths = [os.path.join(dataDir, mus, 'dataExp', exp, 
                                 f'{mus}_{exp}{iCond:02d}_1.csv') 
                     for iCond in range(1, 14)]
        
        # Get stim duration data for current experiment and muscle
        durStim = get_stimDur(filepaths)
        
        # Append the stim durations for this experiment (convert to ms)
        muscle_durStim.extend(durStim)  # Add 13 conditions from this experiment

    # Append the stim durations for this muscle to the overall list
    durStim_exp.append(muscle_durStim)

# Convert the list to a numpy array and ensure it has the correct shape (3x26)
durStim_exp = np.array(durStim_exp)

# Compute differences
d = durStim_prelim-durStim_exp
avg = np.nanmean(d)*1e3
std = np.nanstd(d)*1e3
print(f"Stimulation duration in exp. was {avg:.1f}±{std:.1f} ms shorter than simulations.")
# Round it of to 20±10 ms
