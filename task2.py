import numpy as np
from sympy import *
from scipy.constants import G
from astropy.constants import M_earth, R_earth
from astropy.units.si import sday
from sympy.utilities.lambdify import lambdify
import time
from scipy.integrate import ode
from matplotlib import pyplot as plt

start = time.time()

omega_earth = 2 * np.pi / sday.si.scale
R_e = R_earth.value
M_e = M_earth.value
µ = G * M_e

V_e = 100
m_e = 100
m_c = 5000 * 1000
h = 114 * 1000 * 1000
R = R_e + h

phi, t = symbols('phi, t')
r_c = Matrix([R * sin(phi(t)), R * cos(phi(t))])
r_c = Matrix([R / 2 * cos(phi(t)) + R_e * cos(omega_earth * t), R / 2 * sin(phi(t)) + R_e * sin(omega_earth * t)])
r_c_n = sqrt(r_c[0] ** 2 + r_c[1] ** 2)
v_c = diff(r_c, t)
v_c_n = sqrt((omega_earth * R_e + h / 2 * (diff(phi(t), t) + omega_earth) * cos(phi(t))) ** 2 +
             (h / 2 * (diff(phi(t), t) + omega_earth) * cos(phi(t))) ** 2)
# [omega_Z * Rz + h / 2 * (dphi + omega_Z) * cos(phi)] ^ 2 + [h / 2 * (dphi + omega_Z) * sin(phi)] ^ 2
I_c = m_c * h ** 2 / 12
r_e = Matrix([(V_e * t + R_earth.value) * sin(phi(t)), (V_e * t + R_earth.value) * cos(phi(t))])
r_e_n = sqrt(r_e[0] ** 2 + r_e[1] ** 2)

T = m_c * v_c_n ** 2 / 2 + I_c * omega_earth ** 2 / 2 + m_e * V_e ** 2 / 2
Pe = -µ * m_c / r_c_n - m_e * µ / r_e_n
L = simplify(T - Pe)
print(L)

left_phi = simplify(diff((diff(L, diff(phi(t), t))), t) - (diff(L, phi(t))))
print(left_phi)
second_derivatives = solve([left_phi], diff(phi(t), t, t))

print("Elapsed {} seconds".format(time.time() - start))
