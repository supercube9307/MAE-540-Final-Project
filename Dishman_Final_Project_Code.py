import math as m
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy as np
from rocketpy import Environment, Flight, Function, MonteCarlo, Rocket, SolidMotor # type: ignore
from rocketpy.utilities import apogee_by_mass, liftoff_speed_by_mass, fin_flutter_analysis # type: ignore


# Suppress all FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

# Given
pi = m.pi
c_star = 5210 #ft/s
g = 32.2 #ft/s^2
gamma = 1.25
a_0 = 0.03 #(in/s)[psi]^(-n)
n = 0.35
sigma_p = 0.001 #1/F
T_b0 = 70 # F 
T_bi = 70 # F
rho_p = 0.065 
N = 4
r_1 = 1 # in
r_0 = 2.375 # in
L_0 = 8 # in
A_t0 = 1 # in^2
A_p0 = (r_1**2)*pi
d_t = 2*m.sqrt(A_t0/pi)
E_0 = 4
A_e0 = E_0/A_t0
D_rocket = 6.19
A_rocket = (pi*(D_rocket/2)**2)/12**2
web_step = 0.01
print('Rocket Cross-sectional Area: {0:.3f} ft^2'.format(A_rocket))

def CF(gamma, AeAt, P1oP3):
    StopCriteria = 0.000001
    EA = StopCriteria * 1.1 
    AM2 = 1.5
    IterNo = 0
    P3oP1 = 1 / P1oP3
    while EA > StopCriteria and IterNo < 100:
        IterNo = IterNo + 1
        AFUN = (2 + (gamma - 1) * AM2 ** 2) / (gamma + 1)
        BFUN = (gamma + 1) / (2 * (gamma - 1))
        CFUN = 1 / AFUN
        DFUN = 1 / (AM2 ** 2)
        DERFUN = ((AFUN) ** BFUN) * (CFUN - DFUN)
        FUNFUN = ((1 / AM2) * AFUN ** BFUN) - AeAt
        AMOLD = AM2 
        AM2 = AM2 - FUNFUN / DERFUN
        EA = abs((AM2 - AMOLD) / AM2) * 100
    P2oP1 = (1 + 0.5 * (gamma - 1) * AM2 ** 2) ** (-gamma / (gamma - 1))
    TERM1 = 2 * gamma * gamma / (gamma - 1)
    TERM2 = 2 / (gamma + 1)
    TERM3 = (gamma + 1) / (gamma - 1)
    TERM4 = (gamma - 1) / gamma
    CF = (TERM1 * (TERM2 ** TERM3) * (1 - (P2oP1 ** TERM4))) ** 0.5 + (P2oP1 - P3oP1) * AeAt
    return CF

def CF_Opt(AK, AreaRatio):
    StopCriteria = 0.000001
    EA = StopCriteria * 1.1
    AM2 = 1.5
    IterNo = 0
    while EA > StopCriteria and IterNo < 100:
        IterNo = IterNo + 1
        AFUN = (2 + (AK - 1) * AM2**2) / (AK + 1)
        BFUN = (AK + 1) / (2 * (AK - 1))
        CFUN = 1 / AFUN
        DFUN = 1 / AM2 ** 2
        DERFUN = ((AFUN) ** BFUN) * (CFUN - DFUN)
        FUNFUN = ((1 / AM2) * AFUN ** BFUN) - AreaRatio
        AMOLD = AM2 
        AM2 = AM2 - FUNFUN / DERFUN  
        EA = abs((AM2 - AMOLD) / AM2) * 100
    P2oP1 = (1 + 0.5 * (AK - 1) * AM2 ** 2) ** (-AK / (AK - 1))
    TERM1 = 2 * AK * AK / (AK - 1)
    TERM2 = 2 / (AK + 1)
    TERM3 = (AK + 1) / (AK - 1)
    TERM4 = (AK - 1) / AK
    CF_Opt = (TERM1 * (TERM2 ** TERM3) * (1 - (P2oP1 ** TERM4))) ** 0.5
    return CF_Opt

def calc_atm(alt):
    if alt < 83000:
        P_a = -4.272981E-14*alt**3 + 0.000000008060081*alt**2-0.0005482655*alt + 14.69241
    else:
        P_a = 0.00001
    if alt<32809:
        T_a = -0.0036*alt+518.399
    else:
        T_a = 0
    if alt<82000:
        rho_a = (0.00000000001255)*alt**2-(0.0000019453)*alt+0.07579
    else:
        rho_a = 0
    return P_a, T_a, rho_a

def calc_m_p(w):
    m_p = N*rho_p*(pi*(r_0*r_0-(r_1+w)*(r_1+w))*(L_0-2*w))
    return m_p

def calc_A_b(w):
    A_b = N*2*pi*( (r_1+w)*(L_0-2*w) + (r_0**2 - (r_1+w)**2))
    return A_b

def calc_P_c(T_b0, T_bi, A_b, A_t):
    P_c = (a_0 * m.exp(sigma_p*(T_bi-T_b0))*(rho_p*c_star/g)*(A_b/A_t))**(1/(1-n))
    return P_c

def calc_r_b(T_b0, T_bi, P_c):
    r_b = a_0 * m.exp(sigma_p*(T_bi-T_b0)) * P_c**n
    return r_b

i = 0
w = 0
I_i = 0
I_sum = 0
alt = 0.00001
t_i = 0
m_pi = 1
m_ps = pd.DataFrame(); A_bs = pd.DataFrame(); AeAts = pd.DataFrame(); P_cs = pd.DataFrame(); F_is = pd.DataFrame()
c_fis = pd.DataFrame(); t_is = pd.DataFrame(); A_ts = pd.DataFrame()

print("{0:10s} {1:10s} {2:10s} {3:10s} {4:10s} {5:10s} {6:10s} {7:10s} {8:10s} {9:10s} {10:10s} {11:10s}".format(
    '   w', '   m_pi', '   t_i', '   A_bi', '   A_ti', '   P_ci', '   P_ai', '   r_bi', '   E_i', '   c_fi', '   F_i', 
    '   I_i',))
print("---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------"
        "---------- ---------- ---------- ----------")
while m_pi > 0 :
    m_pi = calc_m_p(w)
    A_bi = calc_A_b(w)
    A_ti = pi*(d_t/2)**2
    P_ci = calc_P_c(T_b0, T_bi, A_bi, A_ti)
    P_ai, T_ai, rho_ai = calc_atm(alt)
    r_bi = calc_r_b(T_b0, T_bi, P_ci)
    E_i = A_e0/A_ti
    c_fi = CF(gamma, E_i, P_ci/P_ai)
    F_i = c_fi * P_ci * A_ti
    
    if ((r_0-r_1)-w)<web_step:
        w_nxt = w + ((r_0-r_1)-w)
    else:
        w_nxt = w + web_step
    t_nxt = t_i + (w_nxt-w)/r_bi
    m_p_nxt = calc_m_p(w_nxt)
    A_b_nxt = calc_A_b(w_nxt)
    d_t_nxt = d_t + 0.000087 * (t_nxt - t_i) * P_ci
    A_t_nxt = pi*(d_t_nxt/2)**2
    P_c_nxt = calc_P_c(T_b0, T_bi, A_b_nxt, A_t_nxt)
    P_a_nxt, T_a_nxt, rho_a_nxt = calc_atm(alt)
    r_b = calc_r_b(T_b0, T_bi, P_c_nxt)
    E_nxt = A_e0/A_t_nxt
    c_f_nxt = CF(gamma, E_nxt, P_c_nxt/P_a_nxt)
    F_nxt = c_f_nxt * P_c_nxt * A_t_nxt
    I_nxt = (F_i + F_nxt)/2 * (t_nxt - t_i)
    print("{0:10.3f} {1:10.3f} {2:10.3f} {3:10.3f} {4:10.3f} {5:10.3f} {6:10.3f} {7:10.3f} {8:10.3f} {9:10.3f} {10:10.3f} {11:10.3f}".format(w, m_pi, t_i, A_bi, A_ti, P_ci, P_ai, r_bi, E_i, c_fi, F_i, I_i))
    
    m_ps = pd.concat([m_ps, pd.Series(m_pi)])
    A_bs = pd.concat([A_bs, pd.Series(A_bi)])
    A_ts = pd.concat([A_ts, pd.Series(A_ti)])
    AeAts = pd.concat([AeAts, pd.Series(E_i)])
    P_cs = pd.concat([P_cs, pd.Series(P_ci)])
    F_is = pd.concat([F_is, pd.Series(F_i)])
    c_fis = pd.concat([c_fis, pd.Series(c_fi)])
    t_is = pd.concat([t_is, pd.Series(t_i)])

    I_i = I_nxt
    w = w_nxt
    t_i = t_nxt
    d_t = d_t_nxt
    i = i + 1
    I_sum = I_sum + I_i

F_is.index = t_is[0]

print('\n')
print('Mass of the Propellant: {0:.3f} lbm'.format(float(m_ps.max())))
print('Area of the Throat at Burnout: {0:.3f} in^s'.format(float(A_ts.max())))
print('Burnout Time: {0:.3f} s'.format(float(t_is.max())))
print('Average Specific Impulse: {0:.3f} s'.format(I_sum/(float(m_ps.max()))))
print('Max Chamber Pressure: {0:.3f} psia'.format(float(P_cs.max())))
print('Max Thrust: {0:.3f} lbf'.format(float(F_is.max())))
print('A_p0/A_t0: {0:.3f} (-)'.format(A_p0/A_t0))


plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.labelsize'] = 14
plt.plot(t_is, m_ps)
plt.xlabel("Time [s]")
plt.ylabel("Mass of Propellant [lbm]")
plt.xlim(0, float(t_is.max()))
# plt.ylim( , )
plt.show()

plt.plot(t_is, P_cs)
plt.xlabel("Time [s]")
plt.ylabel("Chamber Pressure [psia]")
plt.xlim(0, float(t_is.max()))
plt.show()

plt.plot(t_is, F_is)
plt.xlabel("Time [s]")
plt.ylabel("Thrust [lbf]")
plt.xlim(0, float(t_is.max()))
plt.show()

plt.plot(AeAts, c_fis)
plt.xlabel("Area Ratio []]")
plt.ylabel("Thrust Coefficient []")
plt.show()

Motor = SolidMotor(thrust_source="C:/Users/elena/OneDrive/Documents/MAE 440/Project/thrust_curve.csv",
    dry_mass = 10/2.205,
    dry_inertia = (0, 0, 0),
    nozzle_radius = m.sqrt(A_e0/pi)/39.37,
    grain_number = N,
    grain_density = rho_p,
    grain_outer_radius = r_0/39.37,
    grain_initial_inner_radius = r_1/39.37,
    grain_initial_height = L_0/39.37,
    grain_separation = 0.125/39.37,
    grains_center_of_mass_position = N*L_0/2/39.37,
    center_of_dry_mass_position = N*L_0/2/39.37,
    nozzle_position = -L_0/39.37/2+0.03,
    burn_time = float(t_is.max()),
    throat_radius = m.sqrt(A_t0/pi)/39.37,
    coordinate_system_orientation = "nozzle_to_combustion_chamber",)
Motor.draw()