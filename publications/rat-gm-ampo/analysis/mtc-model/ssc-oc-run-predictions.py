"""
This script predicts the maximally attainable AMPO when one SSC parameter is
imposed and MTC length, CE length and stimulation over time are otherwise free
to vary. The optimisation is formulated as a direct-collocation optimal control
problem using the Hill-type MTC model.

Specifically, the following steps were taken:

-   Define the optimal control problem for a periodic SSC.
-   Impose either cycle frequency, FTS, or MTC length excursion.
-   Optimise CE velocity and stimulation over time to maximise AMPO.
-   Save repeated optimisation results for later convergence checks.

Custom functions used:

-   `run_ssc_ocp(muspar, cf=None, fts=None, mle=None, N=200, d=3, do_plot=False, do_print=True)`
    :   Solve the optimal control problem for one specimen and one imposed SSC
        parameter.
-   `ca_func.create_funcs(muspar)`
    :   Create CasADi functions for the Hill-type MTC dynamics, mechanical
        power objective and helper variables.
-   `ca_func.get_sim_guess(f_dyn, cf0, fts0, mle0, N)`
    :   Generate an initial guess from a constant-velocity SSC simulation.
-   `ca_func.ode(x, u, muspar, b=1e3)`
    :   Evaluate the Hill-type MTC model states and outputs.
"""

#%% Load CasADi & set directories
import os, sys, pickle
import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from pathlib import Path

# Set directories
cwd = Path.cwd()
baseDir = cwd.parent.parent
dataDir = baseDir / 'data'
funcDir = baseDir / 'analysis' / 'functions'
sys.path.append(str(funcDir))

import ca_func

plt.close('all')

#%% Create optimisation function
def run_ssc_ocp(muspar, cf=None, fts=None, mle=None, N=200, d=3, do_plot=False, do_print=True):
    """
    Solve one optimal-control SSC problem.

    One of the SSC parameters can be imposed by passing `cf`, `fts` or `mle`.
    Parameters that are passed as `None` become optimisation variables. CE
    velocity and stimulation are optimised over a periodic cycle, subject to
    excitation dynamics, force-length and force-velocity properties, and basic
    path constraints. Successful solutions are returned as a table containing
    time, MTC length, stimulation, SEE force, active state and relative CE
    length.
    """
    f_dyn, f_cost, f_var = ca_func.create_funcs(muspar)
    tau = ca.collocation_points(d, 'legendre')
    [C, D, B] = ca.collocation_coeff(tau)
    
    N2 = N//2
    N1 = N-N2
    
    # Set up optimization problem
    opti = ca.Opti()
    
    # Set cf, fts, mle (and their initial guess)
    if fts is None:
        fts = opti.variable()
        opti.subject_to(opti.bounded(0.01,fts,0.99))
        fts0 = np.random.uniform(low=0.5, high=0.8)
        opti.set_initial(fts, fts0)
    else:
        fts0 = fts
        
    if cf is None:
        cf = opti.variable()
        opti.subject_to(opti.bounded(0.2,cf,12))
        cf0 = np.random.uniform(low=1, high=fts0/0.05) # minimal 50ms Tshort
        opti.set_initial(cf, cf0)
    else:
        cf0 = cf 
    
    if mle is None:
        mle0 = np.random.uniform(low=0.05, high=0.2)
    else:
        mle0 = mle
    
    # Decision variables
    x_k         = opti.variable(2,N+1)
    u_k         = opti.variable(2,N)
    p_k         = opti.variable(1)
    
    # Simulate the system first (to obtain initial guess)
    x0,u0,t0 = ca_func.get_sim_guess(f_dyn,cf0,fts0,mle0/muspar['lce_opt'],N)
    
    # Initial guesses 
    opti.set_initial(x_k, x0)
    opti.set_initial(u_k, u0)
    p_k = 1e3
        
    # Dynamics constraints
    J = 0
    t_k = [0] # [s] time-axis
    lmtc_k = [] 
    for k in range(N):
        if k<N1:
            dt = fts/cf/N1
        else:
            dt = (1-fts)/cf/N2
                
        # Collect states at k and k+1
        Xk = x_k[:, k]
        Xk_next = x_k[:, k + 1]
    
        # Collect controls at k
        Uk = u_k[:, k]
    
        # Collect helper variables at each collocation point
        Xc = opti.variable(2,d)
        # opti.set_initial(Xc, np.repeat(np.array([[1.0, 0.5]]),d,axis=0).T)
        opti.set_initial(Xc, np.tile(x0[:,k], (d, 1)).T)
        
        # Compute cost function and mechanical work
        Jp = -f_cost(Xc, Uk, p_k)
        J = J + ca.mtimes(Jp,B) * dt # cost
    
        # Get interpolating points of collocation polynomial
        Z = ca.horzcat(Xk, Xc)
        # Get slope of interpolating polynomial (normalized)
        Pidot = ca.mtimes(Z, C) / dt
        # State at end of collocation interval
        Xk_end = ca.mtimes(Z, D)
        opti.subject_to(Xk_end == Xk_next)
    
        # Explicit dynamics
        ode = f_dyn(Xc, Uk, p_k)
        opti.subject_to(Pidot == ode)
        
        # Constrain on CE force
        fce_c = f_var(Xc,Uk, p_k)[0]
        opti.subject_to(fce_c >= 0)
        
        # Time-axis
        t_k.append(t_k[-1]+dt)
        
        # MTC length
        lmtc_k.append(f_var(Xk,Uk, p_k)[1])
        
    t_k =  ca.vertcat(*t_k) 
    lmtc_k =  ca.vertcat(*lmtc_k)
    
    # Periodic constraints
    opti.subject_to(x_k[:,0]    == x_k[:,-1])
    
    # Path constraints 
    gamma_k     = x_k[0,:]
    lcerel_k    = x_k[1,:];     lce_k = lcerel_k*muspar['lce_opt']
    
    stim_k      = u_k[0,:]
    vcerel_k    = u_k[1,:];     vcerel_k1 = vcerel_k[:,:N1+1];  vcerel_k2 = vcerel_k[:,N1:]
    
    if mle is not None:
        opti.subject_to(opti.bounded(0.8*mle,lce_k[0]-lce_k[N1],mle*1.2))
        opti.subject_to(lmtc_k[0]-lmtc_k[N1] == mle)
        opti.subject_to(vcerel_k1[0] >= 2*vcerel_k1[1:])
        # opti.subject_to(lce_k[0]-lce_k[N1] == mle)
    opti.subject_to(opti.bounded(muspar['gamma_0'],gamma_k,1))
    opti.subject_to(opti.bounded(1-muspar['w'],lcerel_k,1+muspar['w']))
    opti.subject_to(opti.bounded(0,stim_k,1))
    opti.subject_to(vcerel_k1 <= 0)
    opti.subject_to(vcerel_k2 >= 0)
    
    # Optimize    
    opti.minimize(J*cf*1e1) # Objective
    
    # Set solver and solve the optimization problem
    try:
        opti.solver('ipopt',{'ipopt.max_iter': 500, 
                              'ipopt.mu_strategy': 'adaptive', # monotone = standard, other option: adaptive
                              'ipopt.hessian_approximation': 'limited-memory', # exact vs. limited-memory
                              'ipopt.tol': 5e-3
                              })
        sol = opti.solve()
    except:
        sol = opti.debug
    if mle is None:
        hessian_approx = 'limited-memory' # apparantly works  better for imposed MLE..
    else:
        hessian_approx = 'exact'
    try:        
        opti.solver('ipopt', {
            'ipopt.max_iter': 1000,
            'ipopt.mu_strategy': 'monotone',       # Good for warm start
            'ipopt.hessian_approximation': hessian_approx,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.warm_start_bound_push': 1e-4,
            'ipopt.warm_start_mult_bound_push': 1e-4,   # Also warm-start dual feasibility
            'ipopt.bound_push': 1e-4,
            'ipopt.bound_frac': 1e-4,
            'ipopt.accept_every_trial_step': 'no'  # Make Ipopt more conservative accepting steps
        })
        opti.set_initial(opti.x, sol.value(opti.x)) # Set primal warm-start
        opti.set_initial(opti.lam_g, sol.value(opti.lam_g)) #  Set constraint duals
        sol = opti.solve();
    except:
        sol = opti.debug

    # Extract and display the solution
    if sol.stats()['success'] is True: # only if succesfully optimised!
        if not isinstance(cf, float): cf = sol.value(cf)
        if not isinstance(fts, float): fts = sol.value(fts)
        
        t   = sol.value(t_k)
        x   = sol.value(x_k)
        u   = sol.value(u_k)
        u = np.hstack((u,u[:,0:1]))
        _,fce,vce,lmtc,q,fsee = ca_func.ode(x,u,muspar)[0:6]
        fce = np.squeeze(fce)
        lmtc = np.squeeze(lmtc)
        fsee = np.squeeze(fsee)
        
        gamma, lcerel = x
        stim, vcerel = u
        
        data = np.vstack((t,lmtc,stim,fsee,gamma,lcerel)).T
        df = pd.DataFrame(data)
        
        if do_plot == True:
            plt.figure()
            plt.subplot(311)
            plt.plot(t, x.T)
            
            plt.subplot(312)
            plt.plot(t, u.T)
        
        if do_print == True:
            # Calculate AMPO
            Wmech = -integrate.cumulative_trapezoid(fce,lcerel*muspar['lce_opt'])
            AMPO = Wmech[-1]*cf # [mW]
            print("AMPO = %1.2f mW" % (AMPO*1e3))
            
            # Check SSC parameters
            cf = 1/t[-1]
            iMin = np.argmin(lcerel)
            fts = t[iMin]*cf
            mle = lmtc[0]-lmtc[N1]
            print("CF = %1.2f Hz" % cf)
            print("FTS = %1.2f" % fts)
            print("MLE = %1.2f mm" % (mle*1e3))
    else:
        df = None
    return df

#%% Load muscle parameters
mus = 'GMe3'
parFile = os.path.join(dataDir, mus, mus + '_IM.pkl')
muspar = pickle.load(open(parFile, 'rb'))[0]
# Now we change lsee0 to 3mm
muspar_lsee0_new = 3e-3
muspar['ksee'] = muspar['ksee']*(muspar['lsee0']/muspar_lsee0_new)**2
muspar['lsee0'] = muspar_lsee0_new
dataDirSim = os.path.join(dataDir,mus,'simsOC','it','')

#%% Perform OC optimisation
for cf in np.arange(0.5,8.1,0.5):
    for i in range(1,6):
        df = run_ssc_ocp(muspar,cf,None,None)
        fileName = mus+f'_cf{cf:0.1f}Hz_ftsOpt_mleOpt_it{i:02d}'
        if df is not None:
            df.to_csv(dataDirSim+fileName+'.csv',index=False,header=['Time [s]','Lmtc [m]','STIM [ ]', 'Fsee [N]', 'Gamma [ ]','Lcerel [ ]'])

for fts in np.arange(0.05,0.96,0.05):
    for i in range(1,6):
        df = run_ssc_ocp(muspar,None,fts,None)
        fileName = mus+f'_cfOpt_fts{fts:0.2f}_mleOpt_it{i:02d}'
        if df is not None:
            df.to_csv(dataDirSim+fileName+'.csv',index=False,header=['Time [s]','Lmtc [m]','STIM [ ]', 'Fsee [N]', 'Gamma [ ]','Lcerel [ ]'])

for mle in np.arange(1,11.1,1):
    for i in range(1,6):
        df = run_ssc_ocp(muspar,None,None,mle/1e3)
        fileName = mus+f'_cfOpt_ftsOpt_mle{mle:04.1f}mm_it{i:02d}'
        if df is not None:
            df.to_csv(dataDirSim+fileName+'.csv',index=False,header=['Time [s]','Lmtc [m]','STIM [ ]', 'Fsee [N]', 'Gamma [ ]','Lcerel [ ]'])

