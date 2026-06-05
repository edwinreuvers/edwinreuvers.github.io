"""
This script predicts the maximally attainable AMPO for SSCs with imposed FTS
using constant MTC shortening/lengthening velocity.

For each imposed FTS value, the script first loads the predictions for various
combinations of cycle frequency and MTC length excursion. These predictions are
used to make the initial guess and to define local bounds for the optimisation.
Then, cycle frequency, MTC length excursion and stimulation timing are
optimised, and the result is saved to `simsCV`.
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
mus = 'GMe1'
parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
muspar = pickle.load(open(parFile, 'rb'))[0]

# Now we change lsee0 to 3mm
muspar_lsee0_new = 3e-3
eseerelmax = (muspar['fmax']/muspar['ksee'])**0.5/muspar['lsee0']
muspar['ksee'] = muspar['ksee']*(muspar['lsee0']/muspar_lsee0_new)**2
muspar['lsee0'] = muspar_lsee0_new
lmtc_avg = muspar['lce_opt']+(1+eseerelmax)*muspar['lsee0']

#%% Optimise cycle frequency and MTC length excursion for imposed FTS
dataDirSim = os.path.join(dataDir,mus,'simsCV','')

fts_set = np.arange(0.05, 0.96, 0.05) # []
for fts in fts_set[-1:]:
    # First read out solution for imposed CF, FTS & MLE
    cf_set = np.arange(0.5, 6.1, 0.5) # [Hz]
    mle_set = np.arange(2, 11.1, 1)*1e-3 # [m]
    AMPO_set = helpers.load_sims(cf_set,fts,mle_set,mus,dataDirSim)
        
    # Find maximum
    max_idx = np.nanargmax(AMPO_set)
    iRow, iCol = np.unravel_index(max_idx, AMPO_set.shape)
    
    # Create bounds
    if iRow == 0:
        cf_bounds = (0.1, cf_set[iRow+1])
    elif iRow == cf_set.shape[0]-1:
        cf_bounds = (cf_set[iRow-1], 14)
    else:
        cf_bounds = (cf_set[iRow-1], cf_set[iRow+1])
    if iCol == 0:
        mle_bounds = (1e-3, mle_set[iCol+1])
    elif iCol == mle_set.shape[0]-1:
        mle_bounds = (mle_set[iCol-1], 18e-3)
    else:
        mle_bounds = (mle_set[iCol-1], mle_set[iCol+1])
    
    # Now create initial guess.
    fileName = mus+f'_cf{cf_set[iRow]:0.1f}Hz_fts{fts:0.2f}_mle{mle_set[iCol]*1e3:0.1f}mm'
    df = df = pd.read_csv(dataDirSim+fileName+'.csv')
    data = df.to_numpy()
    time,lmtc,stim,fsee,_,_ = data.T[0:6]
    t_stimOn, t_stimOff = stimulation.get_stim_timing(time,stim)
    t_stimOn = t_stimOn[-1]-time[-1]
    t_stimOff = t_stimOff[-1]
        
    initialGuess = {}
    initialGuess['cfGuess'] = cf_set[iRow]
    initialGuess['mleGuess'] = mle_set[iCol]
    initialGuess['stimGuess'] = [t_stimOn, t_stimOff]
    
    # Perform optimisation
    AMPO, y, optPar = derive_predictions.opt_ssc_par(2,cf_bounds,fts,mle_bounds,lmtc_avg,muspar,initialGuess)
    time, lmtc, stim, gamma, lcerel, q, lsee, lpee, fisomrel, fsee, fpee, fce, fcerel, vcerel = y[0:14]
    
    # Save results
    filename = mus+f'_cfOpt_fts{fts:0.2f}_mleOpt'
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
