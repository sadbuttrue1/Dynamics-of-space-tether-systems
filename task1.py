import numpy as np
from sympy import *
from scipy.constants import G
from astropy.constants import M_earth, R_earth
from astropy.units.si import sday
from sympy.utilities.lambdify import lambdify
import time
from scipy.integrate import ode
from matplotlib import pyplot as plt


def plot_to_file(x, y, variable_name, function_name):
    plt.figure()
    plt.plot(x, y, label='{}({})'.format(function_name, variable_name))
    plt.grid()
    plt.legend()
    plt.ylabel('{}({})'.format(function_name, variable_name))
    plt.xlabel(variable_name)
    plt.savefig('images/{}.png'.format(function_name))


start = time.time()

omega_earth = 2 * np.pi / sday.si.scale
M_e = M_earth.value
µ = G * M_e

m_1 = 6000
m_2 = 50
c = 0
l_0 = 2000
H = 250 * 1.e3
r, theta, t, l, phi = symbols('r theta t l phi')
r_1_x = r(t) * cos(theta(t))
v_1_x = diff(r_1_x, t)
r_1_y = r(t) * sin(theta(t))
v_1_y = diff(r_1_y, t)
v_1_2 = (v_1_x ** 2 + v_1_y ** 2)
alpha = phi(t) - pi / 2 + theta(t)
r_2_x = r_1_x + l(t) * sin(alpha)
v_2_x = diff(r_2_x, t)
r_2_y = r_1_y - l(t) * cos(alpha)
v_2_y = diff(r_2_y, t)
v_2_2 = (v_2_x ** 2 + v_2_y ** 2)
T = (m_1 * v_1_2 / 2 + m_2 * v_2_2 / 2)
Pe = (
    -m_1 * µ / sqrt(r_1_x ** 2 + r_1_y ** 2) - m_2 * µ / sqrt(r_2_x ** 2 + r_2_y ** 2) + c / 2.0 * (l(t) - l_0) ** 2)
L = (T - Pe)
left_phi = simplify((diff((diff(L, diff(phi(t), t))), t) - (diff(L, phi(t))))
                    .subs(diff(r(t), t), 0).subs(diff(diff(theta(t), t), t), 0)
                    .subs(diff(theta(t), t), sqrt(µ / r(t) ** 3)).subs(r(t), R_earth.value + H)
                    .subs(Derivative(0, t), 0))
left_l = simplify((diff((diff(L, diff(l(t), t))), t) - (diff(L, l(t))))
                  .subs(diff(r(t), t), 0).subs(diff(diff(theta(t), t), t), 0)
                  .subs(diff(theta(t), t), sqrt(µ / r(t) ** 3)).subs(r(t), R_earth.value + H)
                  .subs(Derivative(0, t), 0))

second_derivatives = solve([left_phi, left_l], diff(phi(t), t, t), diff(l(t), t, t))
dphi, dl = symbols('dphi dl')
dphi_tt = lambdify((phi(t), dphi(t), l(t), dl(t)), second_derivatives[diff(phi(t), t, t)]
                   .subs(diff(phi(t), t), dphi(t)).subs(diff(l(t), t), dl(t)))
dl_tt = lambdify((phi(t), dphi(t), l(t), dl(t)), second_derivatives[diff(l(t), t, t)]
                 .subs(diff(phi(t), t), dphi(t)).subs(diff(l(t), t), dl(t)))

k = 2.
l_max = 30 * 1.e3


def dl_l(l):
    if l < l_max:
        return 5. + k / 2.
    else:
        return 0


def dq(t, q):
    phi = q[0]
    dphi = q[1]
    l = q[2]
    dl = dl_l(l)
    ddphi = dphi_tt(phi, dphi, l, dl)
    ddl = dl_tt(phi, dphi, l, dl)
    result = np.append(dphi, ddphi)
    result = np.append(result, dl)
    result = np.append(result, ddl)
    return result


q_0 = [0, 0, l_0, dl_l(l_0)]
t_0 = 0.
solver = ode(dq).set_integrator('dopri5', nsteps=1)
solver.set_initial_value(q_0, t_0)

sol_t = []
sol_q = []
tk = 5000
while solver.t < tk:
    solver.integrate(tk, step=True)
    sol_t.append(solver.t)
    sol_q.append(solver.y)

sol_t = np.array(sol_t).reshape((len(sol_t), 1))
sol_q = np.array(sol_q)
sol_phi = sol_q[:, 0]
sol_dphi = sol_q[:, 1]
sol_l = sol_q[:, 2]
sol_dl = sol_q[:, 3]

plot_to_file(sol_t, sol_phi, 't', 'phi')

plot_to_file(sol_t, sol_dphi, 't', 'dphi')

plot_to_file(sol_t, sol_l, 't', 'l')

plot_to_file(sol_t, sol_dl, 't', 'dl')

print("Elapsed {} seconds".format(time.time() - start))
