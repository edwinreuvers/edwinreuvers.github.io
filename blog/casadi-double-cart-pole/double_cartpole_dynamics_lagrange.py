"""Lagrange derivation for a double cart-pole.

The derivation follows the Lagrange formulation in the "Mathematical
derivation" callout of ``index.qmd``. The numbered code cells follow the same
order as the relevant sections and subsections in that callout.

The result has the standard mechanical form

    M(q) qdd + h(q, qd) = Q_external.
"""

import sympy as sp


# %% 1. Coordinates, parameters, and conventions
# See "Shared coordinates and kinematics".
x_c, theta1, theta2 = sp.symbols("x_c theta_1 theta_2", real=True)
x_cd, theta1d, theta2d = sp.symbols("x_cdot theta_1dot theta_2dot", real=True)
x_cdd, theta1dd, theta2dd = sp.symbols(
    "x_cddot theta_1ddot theta_2ddot", real=True
)

q = sp.Matrix([x_c, theta1, theta2])
qd = sp.Matrix([x_cd, theta1d, theta2d])
qdd = sp.Matrix([x_cdd, theta1dd, theta2dd])

u = sp.symbols("u", real=True)
mc, m1, m2 = sp.symbols("m_c m_1 m_2", positive=True)
l1, l2 = sp.symbols("l_1 l_2", positive=True)
g = sp.symbols("g", positive=True)


# %% 2. Kinematics
# See "Shared coordinates and kinematics".
# Both pendulum angles are absolute angles measured from the positive x-axis.
p_cart = sp.Matrix([x_c, 0])
p1 = sp.Matrix([
    x_c + l1 * sp.cos(theta1),
    l1 * sp.sin(theta1),
])
p2 = sp.Matrix([
    x_c + l1 * sp.cos(theta1) + l2 * sp.cos(theta2),
    l1 * sp.sin(theta1) + l2 * sp.sin(theta2),
])

# For p_i(q), the chain rule gives v_i = J_i(q) qd.
J_cart = p_cart.jacobian(q)
J1 = p1.jacobian(q)
J2 = p2.jacobian(q)

v_cart = sp.simplify(J_cart * qd)
v1 = sp.simplify(J1 * qd)
v2 = sp.simplify(J2 * qd)


# %% 3. Kinetic and potential energy
# See "Lagrange formulation > Kinetic and potential energy".
# The links are massless, so the kinetic energy is purely translational.
T = sp.simplify(
    sp.Rational(1, 2) * mc * v_cart.dot(v_cart)
    + sp.Rational(1, 2) * m1 * v1.dot(v1)
    + sp.Rational(1, 2) * m2 * v2.dot(v2)
)

# With y pointing upward, gravitational potential energy is m_i*g*y_i.
V = sp.simplify(m1 * g * p1[1] + m2 * g * p2[1])


# %% 4. Lagrangian and Euler--Lagrange equation
# See "Lagrange formulation > Lagrangian and Euler--Lagrange equation".
L = sp.simplify(T - V)

# Generalized momentum and the configuration derivative of the Lagrangian.
dLdqd = L.diff(qd)
dLdq = L.diff(q)


# %% 5. Extracting M(q) and h(q, qd)
# See "Lagrange formulation > Extracting M(q) and h(q, qdot)".
#
# Expanding d/dt(dL/dqd) with the chain rule gives
#
#   d/dt(dL/dqd)
#       = jacobian(dLdqd, q) * qd + jacobian(dLdqd, qd) * qdd.
#
# The coefficient of qdd is M. Everything that remains on the left-hand side
# of the Euler--Lagrange equation is h.
M = sp.simplify(dLdqd.jacobian(qd))
h = sp.simplify(dLdqd.jacobian(q) * qd - dLdq)


# %% 6. Complete equation of motion
# See "Lagrange formulation > Complete equation of motion".
Q_external = sp.Matrix([u, 0, 0])
equations = sp.simplify(M * qdd + h - Q_external)

# Reconstruct the Euler--Lagrange left-hand side independently and verify that
# it equals M*qdd + h.
euler_lagrange = sp.simplify(
    dLdqd.jacobian(q) * qd
    + dLdqd.jacobian(qd) * qdd
    - dLdq
)
assert sp.simplify(euler_lagrange - (M * qdd + h)) == sp.zeros(3, 1)

M_lagrange = M
h_lagrange = h

if __name__ == "__main__":
    print("\n=== M(q) ===")
    sp.pprint(M_lagrange)
    print("\nLaTeX: ", sp.latex(M_lagrange))

    print("\n=== h(q, qdot) ===")
    sp.pprint(h_lagrange)
    print("\nLaTeX: ", sp.latex(h_lagrange))

    print("\n=== Q_external ===")
    sp.pprint(Q_external)
    print("\nSymbolic Euler--Lagrange reconstruction: OK")
