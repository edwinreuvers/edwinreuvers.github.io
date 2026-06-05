# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:50:22 2024

@author: Edwin
"""

#%%
import os, sys, pickle
import numpy as np
import pandas as pd

cwd = os.path.dirname(os.path.abspath(__file__))
baseDir = os.path.join(cwd,'..','..')
dataDir = os.path.join(baseDir,'suppmat','sel-ssc-conditions','')
funcDir = os.path.join(baseDir,'functions','')
sys.path.append(funcDir)

from FuncNewFig6 import stimOpt

#%%
mus = 'GMz3'
parFile = os.path.join(dataDir,mus+'.pkl')
muspar = pickle.load(open(parFile, 'rb'))
dataDirSim = os.path.join(dataDir,mus,'')
lmtcOpt = muspar['lce_opt'] + (muspar['fmax']/muspar['ksee'])**0.5 + muspar['lsee0']
lmtcAvg = lmtcOpt-3e-3

#%% Imposed cycle frequency & fts & amp - optimize tStimOff
cfSet = [1.5]
ftsSet = [0.35]
ampSet = [4]

for cf in cfSet:
    for fts in ftsSet:
        for amp in ampSet:
            initialGuess = {}
            initialGuess['stimGuess'] = [0, 0.5*fts/cf]
            AMPO, y, optPar = stimOpt(None,1,cf,fts,amp/1e3,lmtcAvg,muspar,initialGuess)
            time, lmtc, stim, gamma, lcerel, q, lsee, lpee, fisomrel, fsee, fpee, fce, fcerel, vcerel = y[0:14]
            
            fileName = mus+f'_amp{amp:02.1f}mm_cf{cf:{"0.1f"}}Hz_fts{fts:{"0.2f"}}'
            
            data = np.vstack((time,lmtc,stim,fsee,gamma,lcerel,q)).T       
            pd.DataFrame(data).to_csv(dataDirSim+fileName+'.csv',index=False,header=['Time [s]','Lmtc [m]',
                'STIM [ ]', 'Fsee [N]', 'Gamma [ ]','Lcerel [ ]', 'q [ ]'])  
