"""Newton--Euler derivation for a double cart-pole.

The derivation follows the Newton--Euler formulation in the "Mathematical
derivation" callout of ``index.qmd``. The numbered code cells follow the same
order as the relevant sections and subsections in that callout.

Model assumptions
-----------------
* The cart has mass m_c; the links are massless.
* m_1 and m_2 are point masses at the ends of the links.
* theta_1 and theta_2 are absolute angles measured from the positive x-axis.
* The only input is the horizontal force u acting on the cart.

The result has the standard mechanical form

    M(q) qdd + h(q, qd) = Q_external.
"""

import sympy as sp


# %% 1. Coordinates, parameters, and conventions
# See "Shared coordinates and kinematics".
#
# q, qd, and qdd are treated as independent symbolic vectors. This makes it
# straightforward to collect all coefficients multiplying qdd.
x_c, theta1, theta2 = sp.symbols("x_c theta_1 theta_2", real=True)
x_cd, theta1d, theta2d = sp.symbols("x_cdot theta_1dot theta_2dot", real=True)
x_cdd, theta1dd, theta2dd = sp.symbols("x_cddot theta_1ddot theta_2ddot", real=True)

q = sp.Matrix([x_c, theta1, theta2])
qd = sp.Matrix([x_cd, theta1d, theta2d])
qdd = sp.Matrix([x_cdd, theta1dd, theta2dd])

u = sp.symbols("u", real=True)
mc, m1, m2 = sp.symbols("m_c m_1 m_2", positive=True)
l1, l2 = sp.symbols("l_1 l_2", positive=True)
g = sp.symbols("g", positive=True)


# %% 2. Kinematics
# See "Shared coordinates and kinematics".
#
# Position kinematics
#
# All positions are expressed in the fixed world frame. Because the angles
# are absolute, the position of m_2 contains cos(theta2), not
# cos(theta1 + theta2).
p_cart = sp.Matrix([x_c, 0])
p1 = sp.Matrix([
    x_c + l1 * sp.cos(theta1),
    l1 * sp.sin(theta1),
])
p2 = sp.Matrix([
    x_c + l1 * sp.cos(theta1) + l2 * sp.cos(theta2),
    l1 * sp.sin(theta1) + l2 * sp.sin(theta2),
])


# Jacobians and velocities
#
# For every position p_i(q), v_i = J_i(q) qd, where J_i = d p_i / d q.
# The velocities are not used later in the derivation, but are included
# explicitly to make the chain rule visible.
J_cart = p_cart.jacobian(q)
J1 = p1.jacobian(q)
J2 = p2.jacobian(q)

v_cart = sp.simplify(J_cart * qd)
v1 = sp.simplify(J1 * qd)
v2 = sp.simplify(J2 * qd)


# Time derivative of a Jacobian
def time_derivative_of_jacobian(J):
    """Compute Jdot(q, qd) using the multivariable chain rule.

    Since J depends only on q, elementwise
        Jdot = sum_k (dJ/dq_k) * qd_k.
    """

    Jdot = sp.zeros(J.rows, J.cols)
    for row in range(J.rows):
        for col in range(J.cols):
            Jdot[row, col] = sum(
                sp.diff(J[row, col], q[k]) * qd[k]
                for k in range(len(q))
            )
    return sp.simplify(Jdot)


# Cartesian accelerations
#
# Applying the product rule to v_i = J_i qd:
#     a_i = J_i qdd + Jdot_i qd.
# The first part is linear in qdd; the second contains the nonlinear velocity
# contributions (centrifugal/Coriolis).
Jdot_cart = time_derivative_of_jacobian(J_cart)
Jdot1 = time_derivative_of_jacobian(J1)
Jdot2 = time_derivative_of_jacobian(J2)

a_cart = sp.simplify(J_cart * qdd + Jdot_cart * qd)
a1 = sp.simplify(J1 * qdd + Jdot1 * qd)
a2 = sp.simplify(J2 * qdd + Jdot2 * qd)


# %% 3. Newton residuals and gravity
# See "Newton--Euler formulation > Newton residuals and gravity".
#
# Write Newton's law as m_i*a_i - F_g,i = 0 (without external force).
# Gravity on the cart is omitted: its vertical motion is constrained and the
# vertical normal force balances its weight. These forces perform no virtual
# work in the permitted horizontal direction.
F_gravity1 = sp.Matrix([0, -m1 * g])
F_gravity2 = sp.Matrix([0, -m2 * g])

R_cart = mc * a_cart
R1 = m1 * a1 - F_gravity1
R2 = m2 * a2 - F_gravity2


# %% 4. Projection using the transposed Jacobian
# See "Newton--Euler formulation > Projection using the transposed Jacobian".
#
# Virtual work gives Q_i = J_i.T * F_i. The same projection converts the
# Cartesian Newton residuals into three generalized equations.
Q_internal = sp.simplify(
    J_cart.T * R_cart
    + J1.T * R1
    + J2.T * R2
)


# %% 5. Complete equation of motion
# See "Newton--Euler formulation > Complete equation of motion".
#
# Only x is actuated. Therefore equations = 0 is equivalent to
#     Q_internal = Q_external.
Q_external = sp.Matrix([u, 0, 0])
equations = sp.simplify(Q_internal - Q_external)


# %% 6. Extracting M(q) and h(q, qd)
# See "Newton--Euler formulation > Extracting M(q) and h(q, qdot)".
#
# Q_internal = M*qdd + h. Obtain M by differentiating with respect to qdd.
# Setting qdd = 0 in Q_internal then leaves h directly. Keeping the external
# force out of this extraction makes it explicit that h cannot depend on u.
M = sp.simplify(Q_internal.jacobian(qdd))
h = sp.simplify(Q_internal.subs(dict.fromkeys(qdd, 0)))

# Symbolically verify the exact reconstruction of the residual.
reconstruction_error = sp.simplify(equations - (M * qdd + h - Q_external))
assert reconstruction_error == sp.zeros(3, 1)


# Result and output
# Keep M_newton and h_newton as descriptive aliases for use from another
# script or notebook.
M_newton = M
h_newton = h

if __name__ == "__main__":
    print("\n=== M(q) ===")
    sp.pprint(M_newton)
    print("\nLaTeX: ", sp.latex(M_newton))

    print("\n=== h(q, qdot) ===")
    sp.pprint(h_newton)
    print("\nLaTeX: ", sp.latex(h_newton))

    print("\n=== Q_external ===")
    sp.pprint(Q_external)
    print("\nSymbolic reconstruction M*qdd + h - Q_external: OK")
