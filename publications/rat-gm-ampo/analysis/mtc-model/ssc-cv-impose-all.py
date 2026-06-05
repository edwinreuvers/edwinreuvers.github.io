"""
This script predicts AMPO for a full grid of constant-velocity SSCs.

For each combination of cycle frequency, FTS and MTC length excursion,
stimulation onset and offset are optimised while MTC shortening and lengthening
velocity remain constant within each phase. The resulting simulations are saved
to `simsCV2` and provide the coarse parameter grid used to initialise the
one-parameter optimisation scripts.
"""

#%% Load packages & set directories
import os, sys, pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import derive_predictions

#%% Load muscle parameters
mus = 'GMe3'
parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
muspar = pickle.load(open(parFile, 'rb'))[0]

# Now we change lsee0 to 3mm
muspar_lsee0_new = 3e-3
eseerelmax = (muspar['fmax']/muspar['ksee'])**0.5/muspar['lsee0']
muspar['ksee'] = muspar['ksee']*(muspar['lsee0']/muspar_lsee0_new)**2
muspar['lsee0'] = muspar_lsee0_new
lmtc_opt = muspar['lce_opt']+muspar['lsee0']+eseerelmax*muspar['lsee0']
lmtc_avg = lmtc_opt

#%% Compute AMPO across the SSC parameter grid
# problematic... GMe1: cf=5.0 Hz, fts=0.45 & mle=10mm
cf_set = np.arange(0.5,6.1,0.5) # [Hz] n = 12
fts_set = np.arange(0.05,0.96,0.05) # [] n = 19
mle_set = np.arange(2,11.1,1)*1e-3 # [m] n = 10
# thus 2280 per rat!

# spyder 1
# mle_set = mle_set # GMe3
# print(mle_set)

# spyder 2
# mle_set = mle_set[8:] # GMe2 - 3 t/m 7 mm
# fts_set = fts_set[9:]
# # cf_set = cf_set[9:10]
# print(mle_set)
# print(fts_set)
# print(cf_set)

# spyder 3
mle_set = mle_set[-1::-1] # GMe3
print(mle_set)

# spyder 4
# mle_set = mle_set[16:] # GMe2

i = 0
for mle in mle_set:
    for fts in fts_set:
        for cf in cf_set:
            # Perform optimisation
            initialGuess = {}
            initialGuess['stimGuess'] = [-1e-3, 0.5*fts/cf]
            #initialGuess['stimGuess'] = [-1e-3, 0.5*fts/cf+1e-3]
            
            AMPO, y = derive_predictions.opt_stim(None,2,cf,fts,mle,lmtc_avg,muspar,initialGuess)
            time, lmtc, stim, gamma, lcerel, q, lsee, lpee, fisomrel, fsee, fpee, fce, fcerel, vcerel = y[0:14]
            
            # Save results            
            filename = mus+f'_cf{cf:0.1f}Hz_fts{fts:0.2f}_mle{mle*1e3:01.1f}mm'
            filepath = os.path.join(dataDir,mus,'simsCV2',filename+'.csv')
            try:
                df = pd.read_csv(filepath)
                data = df.to_numpy().T
                Pmech = np.trapezoid(data[3],data[1])/data[0][-1]
            except:
                Pmech = -np.inf
            
            if AMPO > Pmech:
                data = np.vstack((time,lmtc,stim,fsee,gamma,lcerel,q,fisomrel,fce)).T       
                pd.DataFrame(data).to_csv(filepath,index=False,header=['Time [s]','Lmtc [m]',
                    'STIM [ ]', 'Fsee [N]', 'Gamma [ ]','Lcerel [ ]','q [ ]',
                    'fisomrel [ ]', 'fce [N]']) 
            i += 1
            print(i)
