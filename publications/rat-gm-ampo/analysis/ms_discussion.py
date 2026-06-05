# -*- coding: utf-8 -*-
"""
This script computes summary values used in the manuscript discussion.

It compares AMPO from experimentally tested SSCs with the corresponding
constant-velocity simulations for the 4 mm at 3 Hz and 8 mm at 2 Hz conditions.
The resulting ratios are used to discuss how SEE properties influence MTC
behaviour and attainable AMPO.
"""

#%%
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

cwd = Path.cwd()
baseDir = cwd.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import helpers

plt.close('all')

#%% Load experimental AMPO
ampoMean_PA, ampoMean_PB = [], []
for mus in ['GMe1','GMe2','GMe3']:
    dataDirMus = os.path.join(dataDir,mus,'')
    dataExp = pd.read_excel(dataDirMus+str(mus)+'_dataAMPO.xlsx').to_numpy()[:,5:].astype(float)
    simsExp = pd.read_excel(dataDirMus+str(mus)+'_simsAMPO.xlsx').to_numpy()[:,5:].astype(float)
    
    dataExp[dataExp<1] = np.nan
    dataSSC_PA_T1 = np.nanmean(dataExp[0:3,:],axis=0)
    dataSSC_PA_T2 = np.nanmean(dataExp[3:6,:],axis=0)
    dataSSC_PB_T1 = np.nanmean(dataExp[6:9,:],axis=0)
    dataSSC_PB_T2 = np.nanmean(dataExp[9:12,:],axis=0)
    
    ampoMean_PA.append(dataSSC_PA_T2[2]/1000) # @ CF = 3Hz, FTS = 0.5, MLE = 4mm
    ampoMean_PB.append(dataSSC_PB_T2[2]/1000) # @ CF = 2Hz, FTS = 0.5, MLE = 8mm
    
#%% Load simulated AMPO
ampoSims_PA, ampoSims_PB = [], []
for mus in ['GMe1','GMe2','GMe3']:        
    # Load and interpolate data
    dataDirSim = os.path.join(dataDir,mus,'simsCV','')
    
    cf = 3
    fts = 0.5
    mle = 4e-3
    AMPO = helpers.load_sims(cf,fts,mle,mus,dataDirSim)
    ampoSims_PA.append(AMPO)
    
    cf = 2
    fts = 0.5
    mle = 8e-3
    AMPO = helpers.load_sims(cf,fts,mle,mus,dataDirSim)
    ampoSims_PB.append(AMPO)
    
#%% Compare simulated and experimental AMPO
diff_PA = np.array(ampoSims_PA)/np.array(ampoMean_PA)
print(diff_PA.mean())

diff_PB = np.array(ampoSims_PB)/np.array(ampoMean_PB)
print(diff_PB.mean())

diff = np.hstack((diff_PA, diff_PB))
print(diff.mean())
