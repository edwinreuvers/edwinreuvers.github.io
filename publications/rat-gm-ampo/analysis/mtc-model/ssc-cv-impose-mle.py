"""
This script predicts the maximally attainable AMPO for SSCs with imposed MTC
length excursion using constant MTC shortening/lengthening velocity.

For each imposed MTC length excursion, the script first loads the predictions
for various combinations of cycle frequency and FTS. These predictions are used
to make the initial guess and to define local bounds for the optimisation.
Then, cycle frequency, FTS and stimulation timing are optimised, and the result
is saved to `simsCV`.
"""

#%% Load packages & set directories
import os, sys, pickle
import numpy as np
import pandas as pd
from scipy import integrate
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import derive_predictions, helpers, stimulation

import matplotlib.pyplot as plt
plt.close('all')

#%% Load muscle parameters
mus = 'GMe3'
parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
muspar = pickle.load(open(parFile, 'rb'))[0]

# Now we change lsee0 to 3mm
muspar_lsee0_new = 3e-3
eseerelmax = (muspar['fmax']/muspar['ksee'])**0.5/muspar['lsee0']
muspar['ksee'] = muspar['ksee']*(muspar['lsee0']/muspar_lsee0_new)**2
muspar['lsee0'] = muspar_lsee0_new
lmtc_avg = muspar['lce_opt']+(1+eseerelmax)*muspar['lsee0']

#%% Optimise cycle frequency and FTS for imposed MTC length excursion
dataDirSim = os.path.join(dataDir,mus,'simsCV','')

mle_set = np.arange(2, 11.1, 1)*1e-3 # [m]
for mle in mle_set[-1:]:
    # First read out solution for imposed CF, FTS & MLE
    cf_set = np.arange(0.5, 6.1, 0.5) # [Hz]
    fts_set = np.arange(0.05, 0.96, 0.05) # []
    AMPO_set = helpers.load_sims(cf_set,fts_set,mle,mus,dataDirSim)
        
    # Find maximum
    max_idx = np.nanargmax(AMPO_set)
    iRow, iCol = np.unravel_index(max_idx, AMPO_set.shape)
    
    # Create bounds
    try:
        cfBounds = (cf_set[iRow-1], cf_set[iRow+1])
    except:
        cfBounds = (cf_set[iRow-1], 14)
    ftsBounds = (fts_set[iCol-1], fts_set[iCol+1])
    
    # Now create initial guess.
    fileName = mus+f'_cf{cf_set[iRow]:{"0.1f"}}Hz_fts{fts_set[iCol]:{"0.2f"}}_mle{mle*1e3:{"0.1f"}}mm'
    df = df = pd.read_csv(dataDirSim+fileName+'.csv')
    data = df.to_numpy()
    time,lmtc,stim,fsee,_,_ = data.T[0:6]
    t_stimOn, t_stimOff = stimulation.get_stim_timing(time,stim)
    t_stimOn = t_stimOn[-1]-time[-1]
    t_stimOff = t_stimOff[-1]
        
    initialGuess = {}
    initialGuess['cfGuess'] = cf_set[iRow]
    initialGuess['ftsGuess'] = fts_set[iCol]
    initialGuess['stimGuess'] = [t_stimOn, t_stimOff]
    
    # Perform optimisation
    AMPO, y, optPar = derive_predictions.opt_ssc_par(2,cfBounds,ftsBounds,mle,lmtc_avg,muspar,initialGuess)
    time, lmtc, stim, gamma, lcerel, q, lsee, lpee, fisomrel, fsee, fpee, fce, fcerel, vcerel = y[0:14]
    
    # Save results
    filename = mus+f'_cfOpt_ftsOpt_mle{mle*1e3:0.1f}mm'
    filepath = os.path.join(dataDir,mus,'simsCV',filename+'.csv')
    try:
        df = pd.read_csv(filepath)
        data = df.to_numpy().T
        Pmech = integrate.trapezoid(data[3],data[1])/data[0][-1]
    except:
        Pmech = 0
    
    if AMPO > Pmech:
        # Save data
        data = np.vstack((time,lmtc,stim,fsee,gamma,lcerel,q,fisomrel,fce)).T
        pd.DataFrame(data).to_csv(filepath,index=False,header=['Time [s]','Lmtc [m]',
            'STIM [ ]', 'Fsee [N]', 'Gamma [ ]','Lcerel [ ]','q [ ]',
            'fisomrel [ ]', 'fce [N]'])
