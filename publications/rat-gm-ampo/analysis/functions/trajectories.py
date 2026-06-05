"""
This module provides functions to generate prescribed MTC length trajectories.

Functions
---------
cv(time, cf, fts, mle, lmtc_avg)
    Generate a cyclic trajectory with constant shortening and lengthening
    velocities.

scv(time, cf, fts, amp, lmtc_avg, acc)
    Generate a cyclic trajectory with smoothed acceleration around the turning
    points and constant velocity between them.
"""

import numpy as np

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def cv(time, cf, fts, mle, lmtc_avg):
    """
    Generate muscle-tendon complex length and velocity over time.

    This function computes the length (`lmtc`) and velocity (`vmtc`) of a
    muscle-tendon complex assuming a cyclic motion with constant shortening
    and lengthening velocities within each cycle.

    Parameters
    ----------
    time : array_like
        Time vector [s].
    cf : float
        Cycle frequency [Hz].
    fts : float
        Fraction of the cycle spent in shortening phase [-].
    mle : float
        Total muscle-tendon length excursion [m].
    lmtc_avg : float
        Average muscle-tendon complex length [m].

    Returns
    -------
    lmtc : ndarray
        Muscle-tendon complex length over time [m].
    vmtc : ndarray
        Muscle-tendon complex velocity over time [m/s].

    Raises
    ------
    ZeroDivisionError
        If `cf` is zero.
    ValueError
        If `fts` is not between 0 and 1.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 100)
    >>> lmtc, vmtc = cv(t, 1.0, 0.4, 0.1, 1.0)
    """
    if not 0 <= fts <= 1:
        raise ValueError("fraction_time_shortening must be between 0 and 1.")

    t_mod = np.mod(time, 1 / cf)

    # Calculate shortening and lengthening times
    t_short = fts / cf  # [s]
    t_length = (1 - fts) / cf  # [s]

    # Calculate constant shortening and lengthening velocities
    v_short = -mle / t_short
    v_length = mle / t_length

    # Compute length
    lmtc = (
        (t_mod < t_short) *
        (lmtc_avg + mle / 2 + v_short * t_mod)
        + (t_mod >= t_short) *
        (lmtc_avg - mle / 2 + v_length * (t_mod - t_short))
    )

    # Compute velocity
    vmtc = (
        (t_mod < t_short) * v_short
        + (t_mod >= t_short) * v_length
    )

    return lmtc, vmtc

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def scv(time, cf, fts, amp, lmtc_avg, acc):
    """
    Compute muscle–tendon complex (MTC) kinematics for a prescribed trajectory.

    The trajectory consists of shortening and lengthening phases with
    sinusoidal acceleration profiles near the turning points and constant
    velocity in between.

    Parameters
    ----------
    time : float or ndarray
        Time point(s) at which to evaluate the trajectory [s].
    cf : float
        Cycle frequency [Hz].
    fts : float
        Fraction of the cycle spent shortening [-].
    amp : float
        MTC amplitude [m].
    lmtc_avg : float
        Mean MTC length around which the motion is centered [m].
    acc : float
        Peak acceleration magnitude near turning points [m/s^2].

    Returns
    -------
    lmtc : float or ndarray
        MTC length at `time` [m].
    vmtc : float or ndarray
        MTC velocity at `time` [m/s].
    amtc : float or ndarray
        MTC acceleration at `time` [m/s^2].

    Notes
    -----
    The motion is periodic with period 1 / cf and consists of six phases:
    acceleration–constant–deceleration for both shortening and lengthening.
    """

    # Calculate shortening and lengthening times
    t_short = fts / cf  # [s] shortening time
    t_length = (1 - fts) / cf  # [s] lengthening time

    # Check if inputs are feasible
    # if acc * (acc * np.pi**2 * tLeng**2 + 64 * amp - 16 * amp * np.pi**2) < 0 or acc * np.pi**2 * tShort**2 + 64 * amp - 16 * amp * np.pi**2 < 0:
    #     acc = amp*(16 * np.pi**2 - 64)/(np.pi**2 * tLeng**2)
    #     return None, None, None  # Exit early if the condition is not feasible
        
    # Calculate constant shortening and lengthening velocity
    v_short = (np.pi * np.sqrt(acc * (acc * np.pi**2 * t_short**2 + 64 * amp - 16 * amp * np.pi**2)) - acc * t_short * np.pi**2) / (4 * (np.pi**2 - 4))
    v_length = -(np.pi * np.sqrt(acc * (acc * np.pi**2 * t_length**2 + 64 * amp - 16 * amp * np.pi**2)) - acc * t_length * np.pi**2) / (4 * (np.pi**2 - 4))

    # Calculate acceleration times
    t_acc1 = -v_short / (acc / 2)
    t_acc2 = v_length / (acc / 2)

    # Calculate constant velocity times
    t_con1 = t_short - 2 * t_acc1
    t_con2 = t_length - 2 * t_acc2
        
    # Check if inputs are feasible
    # if tCon1 < 0 or tCon2 < 0:
    #     return None, None, None  # Exit early if the condition is not feasible
    #     #  tCon2 <0: acc = (np.pi**2*(64*amp-16*amp*np.pi**2))/((16*tLeng**2-np.pi**4*tLeng**2))
    #     # maybe give it 0.1% more because of numerical precision
    
    # Calculate points in time
    t_points = np.cumsum([t_acc1, t_con1, t_acc1, t_acc2, t_con2, t_acc2])
    t1, t2, t3, t4, t5, t6 = t_points

    # Current time cycle position
    tc = np.mod(time, t6)

    # Calculate acceleration
    amtc = np.where(tc <= t1, -acc / 2 * (np.sin(np.pi / 2 + (np.pi * tc) / t1) + 1),
                    np.where(tc <= t2, 0,
                             np.where(tc <= t3, (np.sin((tc - t2) * np.pi / (t3 - t2) - 0.5 * np.pi) + 1) * acc / 2,
                                      np.where(tc <= t4, (np.sin((tc - t3) * np.pi / (t4 - t3) + 0.5 * np.pi) + 1) * acc / 2,
                                               np.where(tc <= t5, 0, 
                                                        (np.sin((tc - t5) * np.pi / (t6 - t5) - 0.5 * np.pi) + 1) * -acc / 2)))))

    # Calculate velocity
    vmtc = np.where(tc <= t1, -acc / 2 * tc + (-acc / 2 * t1 * np.sin(np.pi * tc / t1)) / np.pi,
                    np.where(tc <= t2, v_short,
                             np.where(tc <= t3, acc / 2 * (tc - t3) - (acc / 2 * np.sin(np.pi * (t2 - tc) / (t2 - t3) - np.pi) * (t2 - t3)) / np.pi,
                                      np.where(tc <= t4, acc / 2 * (tc - t3) + (acc / 2 * np.sin(np.pi * (t3 - tc) / (t3 - t4) - np.pi) * (t3 - t4)) / np.pi,
                                               np.where(tc <= t5, v_length, 
                                                        -acc / 2 * (tc - t6) + (-acc / 2 * np.sin(np.pi * (t5 - tc) / (t5 - t6)) * (t5 - t6)) / np.pi)))))

    # Initial position conditions
    lmtcP0 = lmtc_avg + amp - ((-acc / 2 * 0**2) / 2 - (-acc / 2 * t1**2 * np.cos((np.pi * 0) / t1)) / np.pi**2)
    lmtcP1 = lmtcP0 + (-acc / 2 * t1**2) / 2 - (-acc / 2 * t1**2 * np.cos((np.pi * t1) / t1)) / np.pi**2
    lmtcP2 = lmtcP1 + (t2 - t1) * v_short - ((acc / 2 * t2**2) / 2 - acc / 2 * t3 * t2 + (acc / 2 * np.cos((np.pi * (t2 - t2)) / (t2 - t3)) * (t2 - t3)**2) / np.pi**2)
    lmtcP3 = lmtcP2 + ((acc / 2 * t3**2) / 2 - acc / 2 * t3 * t3 + (acc / 2 * np.cos((np.pi * (t2 - t3)) / (t2 - t3)) * (t2 - t3)**2) / np.pi**2) - ((acc / 2 * (t3 - t3)**2) / 2 - (acc / 2 * np.cos((np.pi * (t3 - t3)) / (t3 - t4)) * (t3 - t4)**2) / np.pi**2)
    lmtcP4 = lmtcP3 + ((acc / 2 * (t3 - t4)**2) / 2 - (acc / 2 * np.cos((np.pi * (t3 - t4)) / (t3 - t4)) * (t3 - t4)**2) / np.pi**2)
    lmtcP5 = lmtcP4 + (t5 - t4) * v_length - (acc / 2 * t6 * t5 - (acc / 2 * t5**2) / 2 - (acc / 2 * np.cos((np.pi * (t5 - t5)) / (t5 - t6)) * (t5 - t6)**2) / np.pi**2)

    # Calculate position
    lmtc = np.where(tc <= t1, lmtcP0 + (-acc / 2 * tc**2) / 2 - (-acc / 2 * t1**2 * np.cos((np.pi * tc) / t1)) / np.pi**2,
                    np.where(tc <= t2, lmtcP1 + (tc - t1) * v_short,
                             np.where(tc <= t3, lmtcP2 + ((acc / 2 * tc**2) / 2 - acc / 2 * t3 * tc + (acc / 2 * np.cos((np.pi * (t2 - tc)) / (t2 - t3)) * (t2 - t3)**2) / np.pi**2),
                                      np.where(tc <= t4, lmtcP3 + ((acc / 2 * (t3 - tc)**2) / 2 - (acc / 2 * np.cos((np.pi * (t3 - tc)) / (t3 - t4)) * (t3 - t4)**2) / np.pi**2),
                                               np.where(tc <= t5, lmtcP4 + (tc - t4) * v_length,
                                                        lmtcP5 + (acc / 2 * t6 * tc - (acc / 2 * tc**2) / 2 - (acc / 2 * np.cos((np.pi * (t5 - tc)) / (t5 - t6)) * (t5 - t6)**2) / np.pi**2))))))


    return lmtc, vmtc, amtc
