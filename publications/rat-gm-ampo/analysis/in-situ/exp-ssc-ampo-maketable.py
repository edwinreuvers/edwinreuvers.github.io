"""
This script computes experimentally measured AMPO for every stimulated SSC
cycle and stores the values in the per-specimen spreadsheet.

For each rat, condition and trial, stimulation timing is used to identify the
cycles. AMPO is computed from the work-loop area of each stimulated cycle.
Saving is disabled by default and controlled with `do_save`.
"""

#%% Load packages & set directories
import os, sys, scipy, openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import stimulation

plt.close('all')

#%% Compute AMPO per cycle and store in sheet
do_save = False # set to true to save
for mus in ['GMe1', 'GMe2', 'GMe3']:
    AMPOarray = np.nan*np.empty((12,13))
    for exp in ['SSC_PA', 'SSC_PB']:
        for iCond in range(1,14):
            if exp == 'SSC_PA':
                cf = [1,2,3,4,5,3,3,3,3,5,4,2,1][iCond-1]
            elif exp == 'SSC_PB':
                cf = [1,1.5,2,2.5,3,2,2,2,2,3,2.5,1.5,1][iCond-1]
            for iTrial in [1,2]:
                if exp == 'SSC_PA':
                    iStartRow = [0,3][iTrial-1]
                elif exp == 'SSC_PB':
                    iStartRow = [6,9][iTrial-1]
                    
                filepath = os.path.join(dataDir,mus,'dataExp',exp,f'{mus}_{exp}{iCond:02d}_{iTrial}.csv')
                try:
                    data = pd.read_csv(filepath).T.to_numpy()
                    time, lmtc, stim, fsee, *_ = data
                    t_stimOn, t_stimOff = stimulation.get_stim_timing(time, stim)
                    
                    # Determine samples per cycle do this as follows:
                    # 1) Determine time between stimulation onsets and stimulation offsets
                    tCycle = np.mean(np.diff(t_stimOn))/2 + np.mean(np.diff(t_stimOff))/2
                    # 2) From time to number of samples
                    nCycle = int(tCycle/np.mean(np.diff(time)))
                    
                    # Determine iMax:
                    iMax = scipy.signal.find_peaks(lmtc,distance=nCycle*0.95)[0]
                    
                    if len(iMax) != 6:
                        if mus == 'GMe3' and exp=='SSC_PB' and iCond==1 and iTrial==2:
                            pass # checked: iMax @ end missing 
                        elif mus == 'GMe3' and exp=='SSC_PB' and iCond==3 and iTrial==1:
                            pass # checked: iMax @ end missing 
                        elif mus == 'GMe3' and exp=='SSC_PB' and iCond==5 and iTrial==1:
                            pass # checked: iMax @ end missing 
                        elif mus == 'GMe3' and exp=='SSC_PB' and iCond==7 and iTrial==1:
                            pass # checked: iMax @ end missing 
                        else:
                            breakpoint()
                    
                    nSamples = np.diff(iMax)[1:4]
                    if abs(nSamples-2000/cf).max() > 2:
                        breakpoint() # something goes wrong with findpeaks!
                    
                    for i in range(1,4):
                        iSel = slice(iMax[i],iMax[i+1])
                        AMPO = -scipy.integrate.trapezoid(fsee[iSel], lmtc[iSel])/(time[iSel][-1]-time[iSel][0])
                        if AMPO < 0:
                            # This should only print:
                                # GMe1, SSC_PB, iCond=2, iTrial=2, iCycle=3
                                # For some reason the 3rd cycle did not get stimulation here..
                            print(f'AMPO<0 for mus={mus}, exp={exp}, iCond={iCond}, iTrial={iTrial}, iCycle={i}')
                        AMPOarray[iStartRow+i-1,iCond-1] = AMPO*1e3 # to mW
                except Exception:
                    pass
    
    df = pd.DataFrame(AMPOarray)
    
    if do_save is True:
        # Path to existing Excel file
        filepath = os.path.join(dataDir,mus,f'{mus}_dataAMPO.xlsx')
        
        # Open the existing Excel file using openpyxl
        wb = openpyxl.load_workbook(filepath)
        
        # Select the sheet (you can also use wb[sheet_name] if you know the name)
        ws = wb.active  # or wb['Sheet1'] to select a specific sheet by name
        
        # Step 1: Define the starting position (row, column)
        start_row = 2  # Starting row where you want to insert data
        start_col = 6  # Starting column where you want to insert data (1 = 'A', 2 = 'B', etc.)
        
        # Step 2: Write the DataFrame to specific rows and columns
        for r_idx, row in enumerate(df.itertuples(index=False), start=start_row):
            for c_idx, value in enumerate(row, start=start_col):
                ws.cell(row=r_idx, column=c_idx, value=value)
        
        # Step 3: Save the modified workbook
        wb.save(filepath)
        wb.close()
