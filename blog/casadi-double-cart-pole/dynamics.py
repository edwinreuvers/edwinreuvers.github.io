"""Dynamics models used by the optimal-control examples."""

import casadi as ca


def double_cartpole(state, force, parameters):
    """Return the first-order dynamics of the double cart-pole."""

    mc = parameters["mc"]
    g = parameters["g"]
    m1 = parameters["m1"]
    m2 = parameters["m2"]
    l1 = parameters["l1"]
    l2 = parameters["l2"]

    x_c, theta1, theta2, x_cdot, theta1dot, theta2dot = ca.vertsplit(state)

    sin = ca.sin
    cos = ca.cos

    M = ca.MX.zeros(3, 3)
    M[0, 0] = mc + m1 + m2
    M[0, 1] = -l1 * (m1 + m2) * sin(theta1)
    M[0, 2] = -l2 * m2 * sin(theta2)

    M[1, 0] = M[0, 1]
    M[1, 1] = l1**2 * (m1 + m2)
    M[1, 2] = l1 * l2 * m2 * cos(theta1 - theta2)

    M[2, 0] = M[0, 2]
    M[2, 1] = M[1, 2]
    M[2, 2] = l2**2 * m2

    h = ca.vertcat(
        -l1 * (m1 + m2) * theta1dot**2 * cos(theta1)
        - l2 * m2 * theta2dot**2 * cos(theta2),
        l1
        * (
            g * (m1 + m2) * cos(theta1)
            + l2 * m2 * theta2dot**2 * sin(theta1 - theta2)
        ),
        l2
        * m2
        * (g * cos(theta2) - l1 * theta1dot**2 * sin(theta1 - theta2)),
    )

    qdd = ca.solve(M, ca.vertcat(force, 0, 0) - h)
    return ca.vertcat(x_cdot, theta1dot, theta2dot, qdd)
