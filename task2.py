import numpy as np
from sympy import *
from scipy.constants import G
from astropy.constants import M_earth, R_earth
from astropy.units.si import sday
from sympy.utilities.lambdify import lambdify
import time
from scipy.integrate import ode
from matplotlib import pyplot as plt
import os


def plot_to_file(x, y, variable_name, function_name):
    plt.figure()
    plt.plot(x, y, label='{}({})'.format(function_name, variable_name))
    plt.grid()
    plt.legend()
    plt.ylabel('{}({})'.format(function_name, variable_name))
    plt.xlabel(variable_name)
    dir = 'images/task2'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    plt.savefig('images/task2/{}.png'.format(function_name))


start = time.time()

omega_earth = 2 * np.pi / sday.si.scale
R_e = R_earth.value
M_e = M_earth.value
mu = G * M_e

V_e = 100
m_e = 100
m_c = 5000 * 1000
h = 114 * 1000 * 1000
R = R_e + h

phi, dphi, t, v_e = symbols('phi dphi t v_e')
r_c = Matrix([h / 2 * cos(phi(t)) + R_e * cos(omega_earth * t), h / 2 * sin(phi(t)) + R_e * sin(omega_earth * t)])
r_c_n = sqrt(r_c[0] ** 2 + r_c[1] ** 2)
v_c_n = sqrt((omega_earth * R_e + h / 2 * (diff(phi(t), t) + omega_earth) * cos(phi(t))) ** 2 +
             (h / 2 * (diff(phi(t), t) + omega_earth) * cos(phi(t))) ** 2)
I_c = m_c * h ** 2 / 12
r_e = Matrix([(v_e * t + R_earth.value) * sin(phi(t)), (v_e * t + R_earth.value) * cos(phi(t))])
r_e_n = sqrt(r_e[0] ** 2 + r_e[1] ** 2)

T = m_c * v_c_n ** 2 / 2 + I_c * omega_earth ** 2 / 2 + m_e * v_e ** 2 / 2
Pe = -mu * m_c / r_c_n - m_e * mu / r_e_n
L = simplify(T - Pe)

left_phi = simplify(diff((diff(L, diff(phi(t), t))), t) - (diff(L, phi(t))))
second_derivatives = solve([left_phi], diff(phi(t), t, t))

dphi_tt = lambdify((phi(t), dphi(t), t, v_e), second_derivatives[diff(phi(t), t, t)].subs(diff(phi(t), t), dphi(t)))


def dq(t, q, v):
    phi = q[0]
    dphi = q[1]
    ddphi = dphi_tt(phi, dphi, t, v)
    result = np.append(dphi, ddphi)
    return result


q_0 = [0, 0]
t_0 = 0.
solver = ode(dq).set_integrator('dopri5', nsteps=1)
solver.set_initial_value(q_0, t_0).set_f_params(V_e)

sol_t = []
sol_q = []
tk = 534000
while solver.t < tk:
    solver.integrate(tk, step=True)
    sol_t.append(solver.t)
    sol_q.append(solver.y)

sol_t = np.array(sol_t)
sol_q = np.array(sol_q)
sol_phi = sol_q[:, 0]
sol_dphi = sol_q[:, 1]

plot_to_file(sol_t, sol_phi, 't', 'phi')

plot_to_file(sol_t, sol_dphi, 't', 'dphi')

print(r_c_n)
print(r_c_n.subs(phi(t), sol_dphi[-1]).subs(t, tk))
print(r_e_n.subs(phi(t), sol_dphi[-1]).subs(t, tk).subs(v_e, V_e))

print("Elapsed {} seconds".format(time.time() - start))
