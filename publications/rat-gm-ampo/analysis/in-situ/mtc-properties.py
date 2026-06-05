"""
The analysis of this page corresponds to the sections 'MTC property
estimation'.

Specifically, the following steps were taken:

-   Load the estimated MTC properties for each rat.
-   Compute the experimental half-rise time from selected isometric contractions
    at the MTC length closest to optimum.
-   Compute the half-rise time expected at constant CE length from the estimated
    activation dynamics.
-   Compare both half-rise time estimates to quantify the influence of SEE
    compliance on force development.

Custom functions used:

-   `hillmodel.act_state(gamma,lcerel,muspar)`
    : Computes the active state based on intrafilament Ca2+ and CE length.
-   `stimulation.get_stim_timing(time,stim)`
    : Detect stimulation onset and offset time in a signal.
    
MTC properties were estimated based on the 'improved method' in the following 
paper:
    Reuvers, E.D.H.M. & Kistemaker, D.A. (2025)
    Accuracy of experimentally estimated muscle properties: Evaluation and 
    improvement using a newly developed toolbox.
    https://doi.org/10.1101/2025.09.29.678508 
"""

#%% Load packages & set directories
import os, glob, pickle, sys
import pandas as pd
import numpy as np
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import hillmodel, stimulation

#%% Experimental half rise time
tHalfRiseExp = []

exp = 'ISOM'
iSel = [[9,10,11], [1,7,11], [2,4,7]]
for iMus,mus in enumerate(['GMe1', 'GMe2', 'GMe3']):
    parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
    muspar,dataQR,dataSR,dataISOM = pickle.load(open(parFile, 'rb'))
    
    
    files = sorted(glob.glob(os.path.join(dataDir,mus,'dataExp',exp,'*.csv')))
    selFiles = [files[i] for i in iSel[iMus]] # select only at max. Lmtc (closest to LmtcOpt)
    for iFile,filename in enumerate(selFiles):
        df = pd.read_csv(filename)
        data = df.to_numpy()
        time,lmtc,stim,fsee = data.T[0:4]
        
        # Slice first, we want only force development
        tStimOn,tStimOff = stimulation.get_stim_timing(time,stim)
        iOn = int(tStimOn[0]/time[1])
        iMax = np.argmax(fsee)
        
        time = time[iOn:iMax]-time[iOn]
        lmtc = lmtc[iOn:iMax]
        stim = stim[iOn:iMax]
        fsee = fsee[iOn:iMax]
        
        # Now find where 50% Fcemax occurs
        iHalf = np.argmin(abs(fsee-0.5*muspar['fmax']))
        tHalfRiseExp.append(time[iHalf])

tHalfRiseExpMean = np.mean(tHalfRiseExp)*1e3 # avg + to ms
tHalfRiseExpStd = np.std(tHalfRiseExp) # std + to ms
print(f"Experimental half rise times are: {tHalfRiseExpMean:.1f}±{tHalfRiseExpStd:.1f} ms")

#%% Half rise time @ constant CE length
tH50F = []
for mus in ['GMe1', 'GMe2', 'GMe3']:
    parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
    muspar,dataQR,dataSR,dataISOM = pickle.load(open(parFile, 'rb'))
    
    _,_,gamma05 = hillmodel.act_state(np.nan,1,muspar) # [ ] [ ] value of gamma at which q=0.5
    muspar['tHRise'] = -muspar['tact']*np.log(1-gamma05) # [s] "half-rise time"
    tH50F.append(muspar['tHRise'])
tH50FAvg = np.mean(tH50F)*1e3 # avg + to ms
tH50FStd = np.std(tH50F) # std + to ms
print(f"Time to reach 50% isometric force: {tH50FAvg:.1f}±{tH50FStd:.1f} ms")
