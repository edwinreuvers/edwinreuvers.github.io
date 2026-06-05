import numpy as np
import casadi as ca

import hillmodel, trajectories

def create_funcs(muspar):
    gamma   = ca.MX.sym('gamma', 1)
    lcerel  = ca.MX.sym('lcerel', 1)
    x = ca.vertcat(gamma,lcerel)
    stim    = ca.MX.sym('stim', 1)
    vcerel  = ca.MX.sym('vcerel', 1)
    u = ca.vertcat(stim,vcerel)
    p       = ca.MX.sym('p',1)

    xdot, fce, vce, lmtc = ode(x, u, muspar, p)[0:4]
    Pmech = -vce * fce

    f_dyn = ca.Function('f_dyn', [x, u, p], [xdot])
    f_cost = ca.Function('f_cost', [x, u, p], [Pmech])
    f_var = ca.Function('f_var', [x, u, p], [fce, lmtc])
    return f_dyn, f_cost, f_var

def ode(x, u, muspar, b=1e3):
    gamma = x[0]
    lcerel = x[1]
    stim = u[0]
    vcerel = u[1]
    
    #%% Excitation dynamics
    stimOn = 0.5*np.tanh(b*(stim-gamma))+0.5
    stimOff = -0.5*np.tanh(b*(stim-gamma))+0.5
    
    gamma_0 = muspar['gamma_0'] # [ ]
    if isinstance(gamma, ca.SX) or isinstance(gamma, ca.MX):
        gamma = ca.fmax(gamma_0,gamma)
    gammad = (stimOn)*((stim*(1-gamma_0)-gamma + gamma_0)/muspar['tact']) + (stimOff)*((stim*(1-gamma_0)-gamma + gamma_0)/muspar['tdeact']) # [1/s]
    q = hillmodel.act_state(gamma,lcerel,muspar)[0] # [ ] 
    
    #%% Contraction dynamics
    vce = vcerel*muspar['lce_opt']
    fce,fcerel,f = vce2fce_dc(vce,q,lcerel,muspar)[0:3] # [1/s]   
    
    lce = lcerel*muspar['lce_opt']
    lpee = lce
    fpee = hillmodel.lee2force(0,lpee,muspar)[1] # [N]
    
    fsee = fpee+fce
    lsee = hillmodel.force2lee(fsee,fpee,muspar)[0]
    lmtc = lce+lsee
    
    return ca.vertcat(gammad,vcerel), fce, vce, lmtc, q, fsee

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def get_sim_guess(f_dyn, cf0, fts0, mle0, N):
    # Time grid
    t0 = np.linspace(0, 1/cf0, N+1)
    
    # Generate input signals
    lcerel0, vcerel0, *_ = trajectories.cv(t0[:N], cf0, fts0, mle0, 1)
    stim0 = (t0[:N] < 0.5*fts0/cf0).astype(int)
    u0 = np.vstack((stim0, vcerel0)) # shape (N, 2)
    
    #%% Simulate system
    x_init = np.array([1e-3, lcerel0[0]]) # Initial state  
    x0 = sim_casADi(f_dyn,t0,x_init,u0,N) # Do sim
    
    return x0, u0, t0

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def sim_casADi(f_dyn,t,x_init,u,N):
    x_current = ca.DM(x_init)
    x_list = [x_init]  # store NumPy arrays
    
    for i in range(N):
        dt = t[i + 1] - t[i]
        
        u_i = ca.DM(u[:,i])  # convert input to CasADi DM if needed
    
        dx = f_dyn(x_current, u_i, 1e3)  # evaluate dynamics
        x_current = x_current + dt * dx  # Euler step
    
        # Convert CasADi DM to flat NumPy array and store
        x_list.append(np.array(x_current.full()).flatten())
    
    x0 = np.array(x_list).T  # shape: (2,N+1)
    return x0

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def vce2fce_dc(vce,q,lcerel,muspar):
    """
    Compute CE force from force–velocity relationship.
    
    Parameters
    ----------
    vce : float or numpy.ndarray
        CE velocity [m/s]
    q : float or numpy.ndarray
        Active state [-]
    lcerel : float or numpy-array
        Relative CE length [-]
    muspar : dict
        Muscle parameters
    
    Returns
    -------
    fce : float or numpy.ndarray
        Contractile element force [N]
    fcerel : float or numpy.ndarray
        Relative CE force (CE force / Fcemax) [ ]
        Note: computed only if optimum CE length is known, else value is None
    """
            
    # Unravel parameter values
    a_c, b_c            = muspar['a'], muspar['b']
    fasymp, fmax        = muspar['fasymp'], muspar['fmax']
    slopfac, vfactmin   = muspar['slopfac'], muspar['vfactmin']
    q0                  = muspar['q0']  
    sloplin = vfactmin*b_c/(slopfac*0.005*0.0975*(fmax+a_c))
    
    # Scale arel and brel if necessary
    fisomrel = force_length_dc(lcerel,muspar)[0]
    
    # Smooth version of KvS brel(q)
    q0_b = (np.log(1/vfactmin-1)+q0*22)/22
    b = b_c/(1+np.exp(-22*(q-q0_b)))
    
    # Scale a
    a = a_c
    a = (lcerel>1)*a*fisomrel + (lcerel<=1)*a
    
    # Variables for various part of vce-fce relation
    dvdf_isom_con = b/(q*(fisomrel*fmax+a)) # slope in the isometric point at wrt concentric part
    dvdf_isom_ecc = dvdf_isom_con/slopfac # slope in the isometric point at wrt eccentric part
    dFdvcon0      = 1/dvdf_isom_con
    s_as          = 1/sloplin
    p1 = -(fisomrel*q*(fasymp*fmax - fmax))/(s_as - dFdvcon0*slopfac) 
    p2 =  (fisomrel**2*q**2*(fasymp*fmax - fmax)**2)/(s_as - dFdvcon0*slopfac)
    p3 =  -fasymp*fisomrel*q*fmax;
    p4 =  -s_as
    
    # Compute different regions
    r_c1 = ((vce<=0) * (dvdf_isom_con<=sloplin)) # Concentric, dvdf_isom_con<=sloplin (normal)	
    r_c2 = ((vce<=0) * (dvdf_isom_con>sloplin)) # Concentric dvdf_isom_con>sloplin (defective case)
    r_e1 = ((vce>0) * (dvdf_isom_ecc<=(sloplin/slopfac))) # Eccentric, dvdf_isom_ecc<=sloplin (normal) 
    r_e2 = ((vce>0) * (dvdf_isom_ecc>(sloplin/slopfac))) # Eccentric, dvdf_isom_ecc>sloplin (defective case)
    
    # Compute CE force
    fce_c1 = (q*(b*fisomrel*fmax+a*vce)) / (b-vce) # Concentric, dvdf_isom_con<=sloplin (normal)
    fce_c2 = q*fisomrel*fmax+vce/sloplin # Concentric dvdf_isom_con>sloplin (defective case)
    fce_e1 = (p2-(p3+p4*vce)*(p1+vce))/(p1+vce) # Eccentric, dvdf_isom_con<=sloplin (normal)
    fce_e2 = q*fisomrel*fmax+((vce*slopfac)/sloplin) # Eccentric, dvdf_isom_con>sloplin (defective case)

    # Output
    fce = r_c1*fce_c1 + r_c2*fce_c2 + r_e1*fce_e1 + r_e2*fce_e2
    if 'fmax' in muspar:
        fcerel = fce/muspar['fmax']
    else:
        fcerel = None
    
    return fce, fcerel, [fce_c1, fce_c2, fce_e1, fce_e2], [r_c1, r_c2, r_e1, r_e2]

def force_length_dc(lcerel, muspar):
    """
    Computes the relative isometric CE force based on the relative CE length.
    
    Parameters
    ----------
    lcerel : float or numpy.ndarray
        Relative CE length (Lce / Lceopt) [-].
    muspar : dict
        Muscle parameters.
    
    Returns
    -------
    fisomrel : float or numpy.ndarray
        Relative isometric CE force (CE isometric force / Fcemax) [-].
    kce : float or numpy.ndarray
        Derivative fisomrel with respect to lcerel [-].
    """
    
    # Unravel parameter values
    n = muspar['n'] # [ ]
    C = -1/muspar['w']**n # [ ]
    
    # Compute tails of parabola (exponential tails)
    Fp = 0.1 # exp function kicks in a 10% fisomrel
    xp = ((Fp-1)/C)**(1/2) # intercept of function with y=Fp
    xp = np.array([-xp+1, xp+1]) # two solutions because of root
    dFdx = 2*C*(xp-1) # first derivative of F at xp
    # function has the form: y=a*exp(b*x), so:
    b = dFdx/Fp
    a = Fp/(np.exp(b*xp))
    
    # Compute fisomrel
    expr = ca.if_else(
        lcerel < xp[0],
        a[0] * ca.exp(b[0] * lcerel),

        ca.if_else(
            lcerel <= xp[1],
            C * (lcerel - 1)**n + 1,
            a[1] * ca.exp(b[1] * lcerel)
        )
    )
    fisomrel = ca.fmax(1e-9, expr)

    # Compute derivative (i.e., fisomrel/dlcerel)
    kce = ca.if_else(
        lcerel < xp[0],
        a[0] * b[0] * ca.exp(b[0] * lcerel),

        ca.if_else(
            lcerel <= xp[1],
            2 * C * (lcerel - 1),

            a[1] * b[1] * ca.exp(b[1] * lcerel)
        )
    )
    
    return fisomrel,kce