# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:22:52 2024

@author: Edwin
"""
import os, sys, pickle
import numpy as np
import pandas as pd
cwd = os.getcwd()
baseDir = os.path.join(cwd,'..')
dataDir = os.path.join(baseDir,'0_data')
funcDir = os.path.join(baseDir,'0_functions')
sys.path.append(funcDir)
from FuncMus import ActState
from FuncGen import roundValue

#%% Initialize dataframe
# Define the variable names and types
# varnames,z = [['Hill constant', 'a'], ['Hill constant', 'b']].T

varnames = [['a', 'Hill constant', '$a$'], 
            ['b', 'Hill constant', '$b$'], 
            ['epeerelmax', 'relative PEE lengthening at $F_{CE}^{max}$', '$E_{PEE}^{rel,max}$'], 
            ['eseerelmax', 'relative SEE lengthening at $F_{CE}^{max}$', '$E_{SEE}^{rel,max}$'], 
            ['fmax', 'maximal isometric CE force', '$F_{CE}^{max}$'],
            ['lce_opt', 'CE optimum length', '$L_{CE}^{opt}$'],
            ['lpee0', 'PEE slack length', '$L_{PEE}^0$'],
            ['lsee0', 'SEE slack length', '$L_{SEE}^0$'],
            ['tact', 'Calcium dynamics activation time constant', r'$\tau_{act}$'],
            ['tdeact', 'Calcium dynamics deactivation time constant', r'$\tau_{deact}$'],
            ['lmtcOpt', 'MTC length yielding maximal isometric SEE force', '$L_{MTC}^{opt}$'],
            ['pmax', 'maximal instantaneous CE power', '$P_{CE}^{max}$'],
            ['tHRise', r"'half-rise time'", '$t_{HRT}$'],
            ['vmax', 'maximal CE shortening velocity', '$v_{CE}^{max}$'],
            ['vceopt', 'CE velocity at $P_{CE}^{max}$', '$v_{CE}^{opt}$']]

units = ['N', 'mm/s', '-', '-', 'N','mm','mm','mm','ms','ms','mm','mW','ms','mm/s','mm/s']
variables,descriptions,symbols = list(zip(*varnames))  # This transposes the list of lists

types = ['MP'] * 10 +['DM']*5
muscles = ['GMe1', 'GMe2', 'GMe3']

# Create the column names dynamically
columns = ['type','description','symbol','unit'] + muscles

# Initialize the data dict with placeholder values
data = {col: [] for col in columns}

#%%  Fill the DataFrame
num_vars = len(variables)
num_gmes = len(muscles)

for iMus,mus in enumerate(muscles):
    parFile = os.path.join(dataDir,mus,mus+'_CM.pkl')
    muspar,dataQRout, dataSRout, dataACTout = pickle.load(open(parFile, 'rb'))
    
    # Compute helper variables (or whatever we'll call it)
    muspar['lmtcOpt'] = (muspar['eseerelmax']+1)*muspar['lsee0'] + muspar['lce_opt'] # [m] MTC length yielding maximal isometric SEE force
    muspar['vmax'] = muspar['b']/muspar['a']*muspar['fmax'] # [m/s] maximal CE shortening velocity
    muspar['vcenorm'] = muspar['a']*((1+muspar['fmax']/muspar['a'])**0.5-1)/muspar['fmax'] # [ ] normalised CE velocity at which instantaneous CE power is maximal 
    muspar['vceopt'] = muspar['vcenorm']*muspar['vmax']
    fcenorm = muspar['vcenorm'] # [ ] normalised CE force at which instantaneous CE power is maximal 
    muspar['pmax'] = muspar['vcenorm']*fcenorm*muspar['vmax']*muspar['fmax'] # [W] value of maximal instantaneous CE power
    _,_,gamma05 = ActState(np.nan,1,muspar) # [ ] [ ] value of gamma at which q=0.5
    muspar['tHRise'] = -muspar['tact']*np.log(1-gamma05) # [s] "half-rise time"
    
    for iVar,(var,description,symbol,unit,vartype) in enumerate(zip(variables,descriptions,symbols,units,types)):
        vartype = types[iVar]
        
        # Append data for each variable and its corresponding type
        if iMus == 0:  # Only append var and type once per variable
            data['type'].append(vartype)
            data['description'].append(description)
            data['symbol'].append(symbol)
            data['unit'].append(unit)
        
        # Fill the GMe columns
        if iVar in [1,5,6,7,8,9,10,11,12,13,14]:
            value = muspar[var]*1e3
        else:
            value = muspar[var]
        data[mus].append(str(roundValue(value,3)))  # Adjust the logic as needed
        
# Create the DataFrame
df = pd.DataFrame(data)
df.map(str)
fileName = 'tbl_musprop'
df.to_csv(dataDir+'/'+fileName+'.csv', index=False, header=['type','description',
                    'symbol', 'unit', 'GMe1','GMe2','GMe3e'])
