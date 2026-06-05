"""
This module provides functions to optimise stimulation timing and SSC
parameters for periodic MTC length trajectories using a Hill-type MTC model.

Functions
---------
opt_ssc_par(stim, cf, fts, mle, lmtc_avg, muspar, initial_guess={})
    Optimise free SSC parameters using bounded numerical minimisation.
    
opt_stim(x, stim, cf, fts, mle, lmtc_avg, muspar, initial_guess={})
    Optimise stimulation timing to maximise AMPO.
    
sim_periodic(t_stim, cf, fts, mle, lmtc_avg, muspar)
    Simulate a single periodic cycle of MTC dynamics.
"""

import concurrent.futures
import numpy as np
from scipy import optimize

import hillmodel, trajectories

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def opt_ssc_par(stim, cf, fts, mle, lmtc_avg, muspar, initial_guess={}):
    """
    Optimise SSC parameters using bounded numerical minimisation.

    This function optimises whichever SSC parameters are passed as `None` or
    as bounds. Fixed parameters are held constant, while free parameters are
    optimised together with stimulation timing to maximise AMPO.

    Parameters
    ----------
    stim : int
        Type of stimulation passed to `opt_stim` (1 = single pulse,
        2 = stimulation block with onset and offset).
    cf : float or tuple or None
        Cycle frequency [Hz]. If tuple, interpreted as (lower, upper) bounds.
        If float, fixed value. If None, default bounds are used.
    fts : float or tuple or None
        Fraction of cycle time spent shortening. Same rules as `cf`.
    mle : float or tuple or None
        MTC length excursion [m]. Same rules as `cf`.
    lmtc_avg : float
        Average MTC length [m].
    muspar : dict
        Muscle parameter set passed to the Hill-type MTC model.
    initial_guess : dict, optional
        Dictionary of initial guesses. Keys may include:
        - 'cfGuess'
        - 'ftsGuess'
        - 'mleGuess'
        - 'stimGuess'

    Returns
    -------
    p_mech : float
        Optimised AMPO [W].
    y : tuple
        Simulation output returned by `sim_periodic`.
    x_opt : numpy.ndarray
        Optimised SSC parameter vector.

    Notes
    -----
    Uses Nelder-Mead optimisation with bounds passed to SciPy.

    """

    bounds = []
    x0 = []

    # CF
    if isinstance(cf, tuple):
        cf_bounds = cf
    elif cf is None:
        cf_bounds = (0.2, 12)
    else:
        cf_bounds = (cf, cf)

    if not isinstance(cf, (int, float)):
        bounds.append(cf_bounds)
        cf_guess = initial_guess.get(
            "cfGuess",
            np.random.rand() * 3 + 1  # 1–4 Hz default guess
        )
        x0.append(cf_guess)

    # FTS
    if isinstance(fts, tuple):
        fts_bounds = fts
    elif fts is None:
        fts_bounds = (0.02, 0.98)
    else:
        fts_bounds = (fts, fts)

    if not isinstance(fts, (int, float)):
        bounds.append(fts_bounds)
        fts_guess = initial_guess.get(
            "ftsGuess",
            np.random.uniform(*fts_bounds)
        )
        x0.append(fts_guess)

    # MLE
    if isinstance(mle, tuple):
        mle_bounds = mle
    elif mle is None:
        mle_bounds = (0.5e-3, 16e-3)
    else:
        mle_bounds = (mle, mle)

    if not isinstance(mle, (int, float)):
        bounds.append(mle_bounds)
        mle_guess = initial_guess.get(
            "mleGuess",
            np.random.uniform(*mle_bounds)
        )
        x0.append(mle_guess)
    
    # Objective function (maximize stimOpt -> minimize negative)
    def objective(x):
        return -opt_stim(x, stim, cf, fts, mle, lmtc_avg, muspar, initial_guess)[0]

    result = optimize.minimize(
        objective,
        x0,
        method="Nelder-Mead",
        bounds=bounds,
        options={"xatol": 1e-6, "fatol": 1e-6}
    )

    x_opt = result.x

    p_mech, y = opt_stim(
        x_opt, stim, cf, fts, mle, lmtc_avg, muspar, initial_guess
    )

    return p_mech, y, x_opt

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def opt_stim(x, stim, cf, fts, mle, lmtc_avg, muspar, initial_guess={}):
    """
    Optimise stimulation onset and offset to maximise AMPO.

    Parameters
    ----------
    x : array-like
        Current values of the free SSC parameters being optimised.
    stim : int
        Type of stimulation (1 = single pulse, 2 = constant). Determines bounds.
    cf : float
        Cycle frequency [Hz].
    fts : float
        Fraction of cycle time spent shortening [-].
    mle : float
        MTC length excursion [m].
    lmtc_avg : float
        Average MTC length [m].
    muspar : dict
        Muscle parameters dictionary including gamma parameters.
    
    Returns
    -------
    p_mech : float
        AMPO [W].
    sim_result : tuple
        Full simulation results from `sim_periodic`.
    """
    
    # Print which variables we have
    print('Initial guess SSCpar = ..')
    print(x)
    print('Initial guess Stim = ..')
    print(initial_guess['stimGuess'])
    
    # Unpack SSC parameters from 'x'
    if type(cf) == tuple or cf == None:
        cf = x[0]
        x = x[1:]
        # print('Optimising CF')
    if type(fts) == tuple or fts == None: 
        fts = x[0]
        x = x[1:]
        # print('Optimising FTS')
    if type(mle) == tuple or mle == None: # we are imposing fts
        mle = x[0]
        x = x[1:]
        # print('Optimising MLE')
    
    # Make intial guess for stim
    stim_guess = initial_guess.get('stimGuess', np.nan)
    
    # Define bounds and initial guess
    if stim == 2:
        bounds = ((-np.inf, np.inf), (0, np.inf))
        if np.isnan(stim_guess).any():
            x0 = [0, fts / cf * 0.2]
        else:
            x0 = stim_guess
    elif stim == 1:
        bounds = ((0, fts / cf),)
        if np.isnan(stim_guess).any():
            x0 = [fts / cf * 0.2]
        else:
            x0 = stim_guess
            
    # Objective: negative mechanical power
    objective = lambda x: -sim_periodic(x, cf, fts, mle, lmtc_avg, muspar)[0]

    result = optimize.minimize(
        objective,
        x0,
        method='Nelder-Mead',
        bounds=bounds,
        options={'xatol': 1e-5, 'fatol': 1e-3}
    )

    t_stim_opt = result.x
    p_mech, sim_result = sim_periodic(t_stim_opt, cf, fts, mle, lmtc_avg, muspar)
    initial_guess['stimGuess'] = t_stim_opt # update initialGuess based stimOpt of previous round -> faster convergence at the end!
    
    return p_mech, sim_result

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def sim_periodic(t_stim, cf, fts, mle, lmtc_avg, muspar):
    """
    Simulate a single periodic cycle of muscle-tendon complex (MTC) dynamics.

    Parameters
    ----------
    t_stim : array-like or float
        Stimulation onset and offset times. If single value, onset = 0.
    cf : float
        Cycle frequency [Hz].
    fts : float
        Fraction of cycle time spent shortening [-].
    mle : float
        MTC length excursion [m].
    lmtc_avg : float
        Average MTC length [m].
    muspar : dict
        Muscle parameters dictionary including gamma parameters.
    
    Returns
    -------
    p_mech : float
        AMPO [W].
    sim_result : tuple
        Detailed simulation results.
    """
    
    # Determine stimulation times
    if len(t_stim) > 1:
        t_stim_on, t_stim_off = t_stim
    else:
        t_stim_on, t_stim_off = 0, t_stim[0]
    
    # Time discretization
    n_points = 2000
    time = np.unique(np.hstack((
        np.arange(0, fts / cf, 1 / n_points),
        np.arange(fts / cf, 1 / cf, 1 / n_points),
        [1 / cf]
    )))
    
    # MTC length over time
    lmtc = trajectories.cv(time, cf, fts, mle, lmtc_avg)[0]
    
    # Initial states (gamma and lcerel) at t=0
    gamma0 = hillmodel.anly_gamma(0, cf, t_stim_on, t_stim_off, 1, muspar)[0]
    lcerel0 = min(1.4, hillmodel.force_eq(lmtc[0], gamma0, muspar)[1] - 1e-2)
    lcerel_f = [lcerel0]
    
    # Setup solution dictionary
    t_on = [t_stim_on, t_stim_on+1/cf, t_stim_on+2/cf]
    t_off = [t_stim_off, t_stim_off+1/cf, t_stim_off+2/cf]
    
    inputs = {
        'time': time,
        'lmtc': lmtc,
        't_stim': np.array([t_on,t_off]).T,
        'cf': cf
    }

    # Convergence parameters
    dFsee, dLcerel = 1000, 1
    iRound, iFail = 0, 0
    timeout = 10  # seconds

    # Simulate until difference in SEE force is <10 mN
    dFsee, dLcerel = 1000, 1
    iRound, iFail = 0, 0
    timeout = 10 # [s]

    # ODE solver wrapper
    def solve_ode(gamma0, lcerel0, ode_opts, u):
        W_mech, sim_result = hillmodel.solve_simu_mtc(gamma0, lcerel0, muspar, u, ode_opts)
        
        # Extract key variables
        time, _, _, _, lcerel, _, _, _, _, fsee, *_ = sim_result
        
        # Check if solution is complete and within bounds
        if time[-1] != ode_opts['t_eval'][-1] or lcerel[-1] > 2:
            raise RuntimeError("Incomplete simulation or lcerel blew up")
        
        return W_mech, sim_result
    
    ode_opts = {}   
    ode_opts['method'] = 'Radau'
    ode_opts['rtol'] = 1e-9
    ode_opts['atol'] = 1e-6
    ode_opts['t_eval'] = time
    lcerel_f = []
    
    # DEBUGGING REMOVE
    hillmodel.solve_simu_mtc(gamma0, lcerel0, muspar, inputs, ode_opts)
    
    # Sometimes a simulation does get stuck, so if it takes longer than 10s we abort it and try again with a almost identical initial state.
    while dFsee > muspar['fmax']*0.1/100 or abs(dLcerel) > 1e-3:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Using ThreadPoolExecutor to run solve_ivp with timeout
            future = executor.submit(solve_ode,gamma0,lcerel0,ode_opts,inputs) 
            try:
                # Wait for the result with a timeout
                Wmech,y = future.result(timeout=timeout)
                time, lmtc, stim, gamma, lcerel, q, lsee, lpee, fisomrel, fsee, fpee, fce, fcerel, vcerel = y
                dFsee = np.abs(fsee[0]-fsee[-1])
                dLcerel = lcerel[-1]-lcerel[0]
                lcerel_f.append(lcerel[-1])
                # lcerel0 = np.mean(lcerel_f[-3:])  # smooth over last 3 values
                
                # sometimes we have large outliers, select the one within 2 std.
                lcerel_sel = np.array(lcerel_f[-4:]) # select last 4 values
                lcerel_sel = lcerel_sel[np.abs(lcerel_sel - np.mean(lcerel_sel)) <= 2*np.std(lcerel_sel)]
                lcerel0 = np.mean(lcerel_sel)  # avg.
            except:
                print(f"Timeout at iRound {iRound}, trying again")
                lcerel0 -= 0.1
                iFail += 1
            finally:
                # Use shutdown(wait=False) to avoid blocking while cleaning up
                executor.shutdown(wait=False)
  
            iRound += 1
            if iRound > 20 or iFail > 3:
                dFsee, dLcerel, Wmech, y = 0,0,np.nan,None
    
    Pmech = Wmech*cf
    print(f'AMPO = {Pmech*1e3:0.3f} mW')
    # import matplotlib.pyplot as plt
    # plt.figure(); plt.plot(time,lmtc)
    # plt.figure(); plt.plot(time,stim)
    # breakpoint()
    return Pmech, y
