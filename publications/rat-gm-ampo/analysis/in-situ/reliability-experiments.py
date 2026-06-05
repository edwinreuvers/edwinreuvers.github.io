"""
The analysis of this page corresponds to the sections 'Reliability and
robustness of experimental results'.

Specifically, the following steps were taken:

-   Quantify the maximum force measured from GL+PL to verify limited mechanical
    interaction with GM.
-   Compute the coefficient of variation of repeated SR measurements.
-   Analyse repeated QR trials to assess shifts in SEE force and MTC
    force-length behaviour over the experiment.
-   Analyse repeated SR trials to assess changes in isometric force over the
    experiment.
-   Quantify cycle-to-cycle and trial-to-trial variation in experimentally
    measured AMPO.

Custom functions used:

-   `hillmodel.force_eq(lmtc, gamma, muspar)`
    : Finds relative CE length such that SEE force equals the sum of CE and
    PEE force.
-   `stats.pdiff(value1, value2)`
    : Compute the percentage difference between two values.
"""

#%% Load packages & set directories
import os, glob, pickle, sys
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent.parent.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import hillmodel, stats
from mp_estimator import loaddata

#%% Maximum GL force
fGLmax, fGLmin, fGLavg, fGLstd = [], [], [], []
for mus in ['GMe1','GMe2','GMe3']:
    for exp in ['QR','SR','ISOM','SSC_PA', 'SSC_PB']:
        files = glob.glob(os.path.join(dataDir,mus,'dataExp',exp,'*.csv'))
        for iFile,filename in enumerate(files):
            data = pd.read_csv(filename).T.to_numpy()
            time,_,_,fGM,fGL = data
            
            fGLmax.append(fGL.max())
            fGLmin.append(fGL.min())
            fGLavg.append(fGL.mean())
            fGLstd.append(fGL.std())

maxGLforce = np.max(fGLmax)
maxGLstd = np.max(fGLstd)

maxGLforceRound = np.ceil(maxGLforce*10)/10 # N and round to upper 0.1
maxGLstdRound = np.ceil(maxGLstd*2000)/2 # to mN and round to upper 5 mN

print(f"Maximum GL force < {maxGLforceRound:.1f} N")
print(f"Maximum within-trial std of GL force < {maxGLstdRound:.0f} mN")

#%% Coefficient of variation of SR and ISOMs
exp = 'SR'
fGMstd, fGMcov = [], []
for mus in ['GMe1','GMe2','GMe3']:
    files = glob.glob(os.path.join(dataDir,mus,'dataExp',exp,'*.csv'))
    fseeMax = np.empty(len(files))*np.nan
    for iFile,filename in enumerate(files):
        data = pd.read_csv(filename).T.to_numpy()
        time,_,_,fGM,fGL = data
        fseeMax[iFile] = np.max(fGM)
        
    fGMstd.append(np.std(fseeMax)) # [N] # standard deviation of maximum isometric GM force of SR experiments
    fGMcov.append(np.std(fseeMax)/np.mean(fseeMax)*100) # [%]

maxGLcov = np.max(fGMcov)
fGMcovRound = np.ceil(maxGLcov*10)/10
    
print(f"Coefficient of variation of isometric GM force in SR exp < {fGMcovRound:.0f} %")

#%% Outcomes related to repeated QRs
exp = 'QR'

# Create the column names dynamically
columns = ['type','var'] + ['GMe1 - start','GMe1 - 1/2', 'GMe1 - end'] + ['GMe2 - start','GMe2 - 1/2', 'GMe2 - end'] + ['GMe3 - start','GMe3 - 1/2', 'GMe3 - end'] 

# Initialize the data dict with placeholder values
data = {col: [] for col in columns}

for iMus,mus in enumerate(['GMe1','GMe2','GMe3']):
    for sel in [0,1]:
        if mus == 'GMe1':
            iSel = [[2,11,13], [6,12,14]]
        elif mus == 'GMe2':
            iSel = [[5,13,15], [3,14,16]]
        elif mus == 'GMe3':
            iSel = [[5,13], [6,14]]
        iSel = iSel[sel]
        
        # 
        parFile = os.path.join(dataDir,mus,mus+'_IM.pkl')
        muspar = pickle.load(open(parFile, 'rb'))[0]
        lmtcOpt = muspar['lce_opt'] + (muspar['fmax']/muspar['ksee'])**0.5 + muspar['lsee0']
        
        dataDirExp = os.path.join(dataDir,mus,'dataExp',exp)
        files = sorted(glob.glob(os.path.join(dataDirExp,'*.csv'))) + \
                sorted(glob.glob(os.path.join(dataDirExp,'REPEAT', '*.csv')))
        files = [files[i] for i in iSel]
    
        optsQR = {
            'dataDir':          dataDirExp,
            'iCols':            (0,1,3),
            'idxQRmin':         'auto',
            'idxQRpre':         'auto',
            'idxQRpst':         'auto',
            'nQRsamp':          5,
            'dispFig':          False,
            'nFiles':           len(files)
            }
        dataQR, idxQRmin, idxQRpre, idxQRpst = loaddata.qr(optsQR,files)
    
        dataQRmean = {}
        dataQRmean['fseeQRpre'] = np.array([np.mean(dataQR[i]['fsee'][dataQR[i]['idxQRpre'][0]:dataQR[i]['idxQRpre'][1]]) for i in range(0,len(files))])
        dataQRmean['fseeQRpst'] = np.array([np.mean(dataQR[i]['fsee'][dataQR[i]['idxQRpst'][0]:dataQR[i]['idxQRpst'][1]]) for i in range(0,len(files))])
        dataQRmean['lmtcQRpre'] = np.array([np.mean(dataQR[i]['lmtc'][dataQR[i]['idxQRpre'][0]:dataQR[i]['idxQRpre'][1]]) for i in range(0,len(files))])
        dataQRmean['lmtcQRpst'] = np.array([np.mean(dataQR[i]['lmtc'][dataQR[i]['idxQRpst'][0]:dataQR[i]['idxQRpst'][1]]) for i in range(0,len(files))])
        dataQRmean['sseeEst']   = (dataQRmean['fseeQRpre']-dataQRmean['fseeQRpst'])/(dataQRmean['lmtcQRpre']-dataQRmean['lmtcQRpst'])
        
        def find_root(muspar, fseeData, lmtcOpt):
            """Find the root for a given index i."""
            fun = lambda lmtc: hillmodel.force_eq(lmtc, 1, muspar)[0] - fseeData
            root = least_squares(fun, lmtcOpt-3e-3, bounds=(0, lmtcOpt)).x[0]
            return root
        dataQRmean['lmtcQRpre'] = np.array([find_root(muspar, dataQRmean['fseeQRpre'][i], lmtcOpt) for i in range(len(iSel))])        
        dataQRmean['lmtcQRpst'] = np.array([find_root(muspar, dataQRmean['fseeQRpst'][i], lmtcOpt) for i in range(len(iSel))])
                        
        for var in ['fseeQRpre','fseeQRpst','lmtcQRpre','lmtcQRpst']:
            if iMus == 0:  # Only append var and type once per variable
                data['type'].append(sel+1)
                data['var'].append(var)
            data[mus+' - start'].append(dataQRmean[var][0])
            data[mus+' - 1/2'].append(dataQRmean[var][1])
            if len(iSel) > 2:
                data[mus+' - end'].append(dataQRmean[var][2])
            else:
                data[mus+' - end'].append(np.nan)
df = pd.DataFrame(data)

# Difference in SEE force
fseeQRpre1 = df.iloc[0][2:].to_numpy(dtype='float')
fseeQRpre2 = df.iloc[4][2:].to_numpy(dtype='float')

halfway = np.array([
    stats.pdiff(fseeQRpre1[1], fseeQRpre1[0]),
    stats.pdiff(fseeQRpre1[4], fseeQRpre1[3]),
    stats.pdiff(fseeQRpre1[7], fseeQRpre1[6]),
    stats.pdiff(fseeQRpre2[1], fseeQRpre2[0]),
    stats.pdiff(fseeQRpre2[4], fseeQRpre2[3]),
    stats.pdiff(fseeQRpre2[7], fseeQRpre2[6])
])

end = np.array([
    stats.pdiff(fseeQRpre1[2], fseeQRpre1[0]),
    stats.pdiff(fseeQRpre1[5], fseeQRpre1[3]),
    stats.pdiff(fseeQRpre1[8], fseeQRpre1[6]),
    stats.pdiff(fseeQRpre2[2], fseeQRpre2[0]),
    stats.pdiff(fseeQRpre2[5], fseeQRpre2[3]),
    stats.pdiff(fseeQRpre2[8], fseeQRpre2[6])
])
print("Decrease in SEE force")
print(f"Halfway vs. start: {np.nanmean(halfway):.1f} +- {np.nanstd(halfway):.1f} %")
print(f"End vs. start: {np.nanmean(end):.1f} +- {np.nanstd(end):.1f} %")

# SEE shift
lmtcQRpre1 = df.iloc[2][2:].to_numpy(dtype='float')
lmtcQRpre2 = df.iloc[6][2:].to_numpy(dtype='float')
lmtcQRpst1 = df.iloc[3][2:].to_numpy(dtype='float')
lmtcQRpst2 = df.iloc[7][2:].to_numpy(dtype='float')

halfway = np.array([
    lmtcQRpre1[0] - lmtcQRpre1[1],
    lmtcQRpre1[3] - lmtcQRpre1[4],
    lmtcQRpre1[6] - lmtcQRpre1[7],
    lmtcQRpst1[0] - lmtcQRpst1[1],
    lmtcQRpst1[3] - lmtcQRpst1[4],
    lmtcQRpst1[6] - lmtcQRpst1[7]
])

end = np.array([
    lmtcQRpre1[0] - lmtcQRpre1[2],
    lmtcQRpre1[3] - lmtcQRpre1[5],
    np.nan,
    lmtcQRpst1[0] - lmtcQRpst1[2],
    lmtcQRpst1[3] - lmtcQRpst1[5],
    np.nan
])

print("Shift in MTC force-length relationship")
print(f"Halfway vs. start: {np.nanmean(halfway*1e3):.1f} +- {np.nanstd(halfway*1e3):.1f} mm")
print(f"End vs. start: {np.nanmean(end*1e3):.1f} +- {np.nanstd(end*1e3):.1f} mm")

#%% Outcomes related to repeated SRs
exp = 'SR'

fseeDiff = np.empty((6,2))*np.nan

for iMus,mus in enumerate(['GMe1','GMe2','GMe3']):
    # Indices of the SR files that are repeated
    if mus == 'GMe1': 
        iSel = [7,5,11,12,13,14]
    elif mus == 'GMe2':
        iSel = [3,6,12,13,14,15]
    elif mus == 'GMe3':
        iSel = [5,8,10,11]
    
    # Load the datafiles
    dataDirExp = os.path.join(dataDir,mus,'dataExp',exp)
    files = sorted(glob.glob(os.path.join(dataDirExp,'*.csv'))) + \
            sorted(glob.glob(os.path.join(dataDirExp,'REPEAT', '*.csv')))
    files = [files[i] for i in iSel]
    
    # Get the SR data
    opts = {}
    opts['iCols'] = [0,1,3]
    opts['nFiles'] = len(files)
    opts['nSRsamp'] = 20
    opts['idxSRcon'] = 'auto'
    opts['dispFig'] = False
    dataSR, idxSRcon = loaddata.sr(opts,files)
    fseeCon = np.array([np.mean(dataSR[i]['fsee'][dataSR[i]['idxSRcon'][0]:dataSR[i]['idxSRcon'][1]]) for i in range(0,len(files))])
        
    # Compute differences between start, middle and end of experiment in SEE force
    fseeDiff[iMus*2,0] = stats.pdiff(fseeCon[0],fseeCon[2])
    fseeDiff[iMus*2+1,0] = stats.pdiff(fseeCon[1],fseeCon[3])
    if mus != 'GMe3':
        fseeDiff[iMus*2,1] = stats.pdiff(fseeCon[0],fseeCon[4])
        fseeDiff[iMus*2+1,1] = stats.pdiff(fseeCon[1],fseeCon[5])

print(f"Halfway vs. start: {np.nanmean(fseeDiff[:,0]):.0f} +- {np.nanstd(fseeDiff[:,0]):.0f} %")
# no you can observed that at the end it did at least not decrease.. see fseeDiff[0:4,0] vs fseeDiff[0:4,1]

#%% Difference in AMPO between cycles
AMPO = np.empty((3,0))
for mus in ['GMe1','GMe2','GMe3']:
    dataDirExp = os.path.join(dataDir,mus,'')
    dataExp = pd.read_excel(dataDirExp+str(mus)+'_dataAMPO.xlsx').to_numpy()[:,5:].astype(float)
    
    dataExp[dataExp<1] = np.nan
    AMPO = np.hstack((AMPO,dataExp[0:3,:]))
    AMPO = np.hstack((AMPO,dataExp[3:6,:]))
    AMPO = np.hstack((AMPO,dataExp[6:9,:]))
    AMPO = np.hstack((AMPO,dataExp[9:12,:]))
    
AMPOsort = -np.sort(-AMPO,axis=0)

d1 = stats.pdiff(AMPOsort[1,:],AMPOsort[0,:])
d2 = stats.pdiff(AMPOsort[2,:],AMPOsort[0,:])
d = np.hstack((d1,d2))
avg = np.nanmean(d)
std = np.nanstd(d)
print(f"AMPO of 2nd and 3rd highest are {avg:0.2f}±{std:0.2f} % lower than the highest.")

#%% Difference in AMPO between 1st and 2nd trial
AMPOt1 = np.empty((3,0))
AMPOt2 = np.empty((3,0))
for mus in ['GMe1','GMe2','GMe3']:
    dataDirExp = os.path.join(dataDir,mus,'')
    dataExp = pd.read_excel(dataDirExp+str(mus)+'_dataAMPO.xlsx').to_numpy()[:,5:].astype(float)
    
    dataExp[dataExp<1] = np.nan
    AMPOt1 = np.hstack((AMPOt1,dataExp[0:3,:]))
    AMPOt2 = np.hstack((AMPOt2,dataExp[3:6,:]))
    AMPOt1 = np.hstack((AMPOt1,dataExp[6:9,:]))
    AMPOt2 = np.hstack((AMPOt2,dataExp[9:12,:]))
    
d = stats.pdiff(AMPOt2,AMPOt1)
avg = np.nanmean(d)
std = np.nanstd(d)
print(f"AMPO of 2nd trial is {avg:0.2f}±{std:0.2f} % higher than 1st trial.")
