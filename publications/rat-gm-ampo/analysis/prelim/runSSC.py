# -*- coding: utf-8 -*-
"""
This script runs one preliminary constant-velocity SSC simulation in order
to select the experimental conditions..

Stimulation timing is optimised for various combinations of cycle frequency, 
FTS and MTC length excursion. Results are saved as a CSV file.
"""

#%% Load packages and set directories
import os, sys, pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import derive_predictions

#%% Load muscle parameters
mus = 'GMz3'
parFile = os.path.join(mus+'.pkl')
muspar = pickle.load(open(parFile, 'rb'))
dataDirSim = os.path.join(cwd,mus,'')
lmtcOpt = muspar['lce_opt'] + (muspar['fmax']/muspar['ksee'])**0.5 + muspar['lsee0']
lmtcAvg = lmtcOpt-3e-3

#%% Optimise stimulation timing for the selected SSC
cfSet = [1.5]
ftsSet = [0.35]
mleSet = [4e-3]

for cf in cfSet:
    for fts in ftsSet:
        for mle in mleSet:
            initialGuess = {}
            initialGuess['stimGuess'] = [0, 0.5*fts/cf]
            AMPO, y, = derive_predictions.opt_stim(None,1,cf,fts,mle,lmtcAvg,muspar,initialGuess)
            time, lmtc, stim, gamma, lcerel, q, lsee, lpee, fisomrel, fsee, fpee, fce, fcerel, vcerel = y[0:14]
            
            fileName = mus+f'_amp{mle/2*1e3:02.1f}mm_cf{cf:{"0.1f"}}Hz_fts{fts:{"0.2f"}}'
            
            data = np.vstack((time,lmtc,stim,fsee,gamma,lcerel,q)).T       
            pd.DataFrame(data).to_csv(dataDirSim+fileName+'.csv',index=False,header=['Time [s]','Lmtc [m]',
                'STIM [ ]', 'Fsee [N]', 'Gamma [ ]','Lcerel [ ]', 'q [ ]'])  
