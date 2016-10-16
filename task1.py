import numpy as np
from sympy import *
from scipy.constants import G
from astropy.constants import M_earth, R_earth
from astropy.units.si import sday

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
v_1_2 = simplify(v_1_x ** 2 + v_1_y ** 2)
alpha = phi(t) - pi / 2 + theta(t)
r_2_x = r_1_x + l(t) * sin(alpha)
v_2_x = diff(r_2_x, t)
r_2_y = r_1_y - l * cos(alpha)
v_2_y = diff(r_2_y, t)
v_2_2 = simplify(v_2_x ** 2 + v_2_y ** 2)
T = simplify(m_1 * v_1_2 / 2 + m_2 * v_2_2 / 2)
Pe = simplify(
    -m_1 * µ / sqrt(r_1_x ** 2 + r_1_y ** 2) - m_2 * µ / sqrt(r_2_x ** 2 + r_2_y ** 2) + c / 2.0 * (l(t) - l_0) ** 2)
L = simplify(T - Pe)
print((diff(simplify(diff(L, diff(phi(t), t))), t) - (diff(L, phi(t)))).subs(diff(r(t), t), 0).subs(
    diff(diff(theta(t), t), t), 0).subs(diff(theta(t), t), sqrt(µ / r(t) ** 3)).subs(r(t), R_earth.value + H))
print((diff(simplify(diff(L, diff(l(t), t))), t) - (diff(L, l(t)))).subs(diff(r(t), t), 0).subs(
    diff(diff(theta(t), t), t), 0).subs(diff(theta(t), t), sqrt(µ / r(t) ** 3)).subs(r(t), R_earth.value + H))
