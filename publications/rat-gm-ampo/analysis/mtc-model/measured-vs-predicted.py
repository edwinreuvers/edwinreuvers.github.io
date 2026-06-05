"""
The analysis of this page corresponds to the sections 'Evaluation of measured 
versus predicted maximally attainable AMPO'.

First, we simulate the experimental conditions using experimentally measured
MTC length and stimulation over time as inputs for the Hill-type MTC model. 
Second, we evaluate measurements against predictions.

Custom functions used:

-   `hillmodel.force_eq(lmtc,gamma,muspar)`
    : Finds relative CE length such that SEE force equals the sum of CE and 
    PEE force.
-   `hillmodel.solve_simu_mtc(gamma0, lcerel0, muspar, inputs, ode_opts)`
    : Forward simulation of a Hill-type muscle-tendon complex (MTC) model.
-   `stimulation.get_stim_timing(time, stim)`
    :   Detects stimulation pulse trains in a signal and returns their onset and offset time.
-   `stats.pdiff(x,y)`
    :   Compute the percentage difference with x w.r.t. y.
"""

#%% Load packages & set directories
import pickle, os, sys, glob
import numpy as np
import pandas as pd
import scipy.stats
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import hillmodel, stats, stimulation


#%% Simulate experimental conditions
muscles = ['GMe1', 'GMe2', 'GMe3']
for mus in muscles:
    for exp in ['QR','SR','ISOM','SSC_PA','SSC_PB']: 
        parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
        muspar = pickle.load(open(parFile, 'rb'))[0]
        
        dataDirExp = os.path.join(dataDir,mus,'dataExp',exp,'')
        dataDirSim = os.path.join(dataDir,mus,'simsExp',exp,'')
        
        # files = glob.glob(dataDirSim+r'\*')    
        # for iFile,filename in enumerate(files):
        #     os.remove(filename)
        
        files = glob.glob(os.path.join(dataDirExp, '*.csv')) 
        for iFile,filepath in enumerate(files):
            filename = filepath.rsplit('\\', 1)[-1][:-4]
            data = pd.read_csv(filepath).T.to_numpy()
            time, lmtc, stim, fseeData, *_ = data
            tStimOn,tStimOff = stimulation.get_stim_timing(time,stim)
            
            gamma0  = muspar['gamma_0']
            lcerel0 = hillmodel.force_eq(lmtc[0],gamma0,muspar)[1]
            
            solmat = {}
            solmat['time'] = time
            solmat['lmtc'] = lmtc
            solmat['t_stim'] = np.vstack((tStimOn,tStimOff)).T
            
            ode_opts = {'atol': 1e-9, 'rtol': 1e-6, 'max_step': 1e-3, 't_eval': time}
            solstr = hillmodel.solve_simu_mtc(gamma0,lcerel0,muspar,solmat,ode_opts)[1]
            fsee = solstr[9]
              
            # Store
            dataSim = np.vstack((time,lmtc,stim,fsee)).T
            filepath = os.path.join(dataDirSim,filename+'_IM.csv')
            pd.DataFrame(dataSim).to_csv(filepath,index=False, 
                                      header=['time [s]','Lmtc [m]','STIM [ ]','Fsee [N]'])

#%% Evaluate measurements vs. predictions
dataExps, simsExps, r2 = [], [], []
for iMus,mus in enumerate(['GMe1','GMe2','GMe3']):
    dataDirMus = os.path.join(dataDir,mus,'')
    dataExp = pd.read_excel(dataDirMus+str(mus)+'_dataAMPO.xlsx').to_numpy()[:,5:].astype(float)
    simsExp = pd.read_excel(dataDirMus+str(mus)+'_simsAMPO.xlsx').to_numpy()[:,5:].astype(float)
    
    dataExp[dataExp<1] = np.nan
    simsExp[simsExp<1] = np.nan
    
    dataExps.append(dataExp)
    simsExps.append(simsExp)
    
    iNan = np.isnan(dataExp)
    dataExp = dataExp[~iNan]
    simsExp = simsExp[~iNan]

    r2.append(scipy.stats.pearsonr(dataExp,simsExp)[0])   
    
dataExps = np.array(dataExps).flatten()
simsExps = np.array(simsExps).flatten()

procDiff = stats.pdiff(dataExps,simsExps) # procentual difference
diffMean = np.nanmean(procDiff)
diffStd = np.nanstd(procDiff)
print(f"Measured AMPO / Simulated AMPO is on average: {diffMean:.1f} +- {diffStd:.1f} %")

r2_formatted = [f"{x:.4f}" for x in r2]
print(f"r^2 values for rat 1,2,3 are: {', '.join(r2_formatted)}")
