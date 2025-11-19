import math as m
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy as np
from rocketpy import Environment, Flight, Function, MonteCarlo, Rocket, SolidMotor # type: ignore
from rocketpy.utilities import apogee_by_mass, liftoff_speed_by_mass, fin_flutter_analysis # type: ignore

# Suppress all FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

# Constants
pi = m.pi
c_star = 5210       # ft/s
g = 32.2            # ft/s^2
gamma_prop = 1.25
gamma_air = 1.4
Ru = 1545.3         # (ft·lbf)/(lbmol·°R)
mw_air = 28.97      # kg/(mol)
R = Ru/mw_air * g   # (ft·lbf)/(lb·°R)
a_0 = 0.03          # (in/s)[psi]^(-n)
n = 0.35
sigma_p = 0.001     # 1/F
T_b0 = 70           # F 
rho_p = 0.065       # lbm/ft^3
grain_spacing = 0.125   # in
D_rocket = 6.19         # in
A_rocket = (pi*(D_rocket/2)**2)/12**2   # ft^2

print('Rocket Cross-sectional Area: {0:.3f} ft^2'.format(A_rocket))
print('------------------------------------------------------\n')

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
        P_a = -4.272981E-14*alt**3 + 0.000000008060081*alt**2 - 0.0005482655*alt + 14.69241
    else:
        P_a = 0.00001
    if alt<32809:
        T_a = -0.0036*alt+518
    else:
        T_a = 399
    if alt<82000:
        rho_a = (0.00000000001255)*alt**2-(0.0000019453)*alt+0.07579
    else:
        rho_a = 0
    return P_a, T_a, rho_a

def calc_m_p(N, r_1, r_0, L_0, w):
    m_p = N*rho_p*(pi*(r_0*r_0-(r_1+w)*(r_1+w))*(L_0-2*w))
    return m_p

def calc_A_b(N, r_1, r_0, L_0, w):
    A_b = N*2*pi*( (r_1+w)*(L_0-2*w) + (r_0**2 - (r_1+w)**2))
    return A_b

def calc_P_c(T_b0, T_bi, A_b, A_t):
    P_c = (a_0 * m.exp(sigma_p*(T_bi-T_b0))*(rho_p*c_star/g)*(A_b/A_t))**(1/(1-n))
    return P_c

def calc_r_b(T_b0, T_bi, P_c):
    r_b = a_0 * m.exp(sigma_p*(T_bi-T_b0)) * P_c**n
    return r_b

def FindMach(gamma, AeAt):
    StopCriteria = 0.000001 
    EA = StopCriteria * 1.1 
    AM2 = 1.5 
    IterNo = 0 
    while EA > StopCriteria and IterNo < 100:
        IterNo = IterNo + 1  
        AFUN = (2 + (gamma - 1) * AM2**2) / (gamma + 1)
        BFUN = (gamma + 1) / (2 * (gamma - 1))
        CFUN = 1 / AFUN
        DFUN = 1 / AM2**2
        DERFUN = ((AFUN)**BFUN) * (CFUN - DFUN)
        FUNFUN = ((1 / AM2) * AFUN**BFUN) - AeAt
        AMOLD = AM2 
        AM2 = AM2 - FUNFUN / DERFUN 
        EA = abs((AM2 - AMOLD) / AM2) * 100
    M = AM2
    return M

def motor_draw(A_e0, N, r_0, r_1, L_0, A_t0, offset):
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
        nozzle_position = -L_0/39.37+offset,
        burn_time = 2,
        throat_radius = m.sqrt(A_t0/pi)/39.37,
        coordinate_system_orientation = "nozzle_to_combustion_chamber",)
    Motor.draw()
    return

varT_hs = pd.DataFrame(); varT_vs = pd.DataFrame(); varT_as = pd.DataFrame(); varT_gs = pd.DataFrame(); 

def PR08D(N, r_1, r_0, L_0, A_t0, E_0, m_ballast, T_bi):
    
    m_case = N*(L_0 + grain_spacing)*0.25 #lbm -> 0.25lbm/in = density of case material
    A_p0 = (r_1**2)*pi          # in^2
    d_t = 2*m.sqrt(A_t0/pi)     # in
    A_e0 = E_0*A_t0             # -
    web_step = 0.01             # in
    m_struct = 40               # lbm
    
    print('Total Motor Casing Length (<34in): {0:.3f} in'.format(N*L_0 + N*grain_spacing))
    print('Initial Grain Port Area/Intial Throat Area (>2): {0:.3f}'.format(A_p0/A_t0))
    print('Ballast Mass (<1): {0:.3f} lbm'.format(m_ballast))
    i = 0; w = 0; I_i = 0; I_sum = 0; t_i = 0; m_pi = 1; h_i = 0; v_i = 0
    m_p0 = calc_m_p(N, r_1, r_0, L_0, w)
    m_0 = m_p0 + m_case + m_ballast + m_struct
    print('Intitial Takeoff Mass: {0:.3f} lbm'.format(m_0))
    m_ps = pd.DataFrame(); A_bs = pd.DataFrame(); AeAts = pd.DataFrame(); P_cs = pd.DataFrame(); 
    F_is = pd.DataFrame(); c_fis = pd.DataFrame(); t_is = pd.DataFrame(); A_ts = pd.DataFrame(); 
    D_is = pd.DataFrame(); h_is = pd.DataFrame(); v_is = pd.DataFrame(); a_is = pd.DataFrame(); 
    # print("{0:10s} {1:10s} {2:10s} {3:10s} {4:10s} {5:10s} {6:10s} {7:10s} {8:10s} {9:10s} {10:10s} {11:10s} {12:10s} {13:10s} {14:10s} {15:10s} {16:10s} {17:10s} {18:10s} {19:10s} {20:10s} {21:10s} {22:10s} {23:10s}".format(
    #     '   w', '   m_pi', '   t_i', '   A_bi', '   A_ti', '   P_ci', '   P_ai', '   r_bi', '   E_i', '   c_fi', '   F_i', 
    #     '   I_i', 'm_i', 'v_i', 'h_i', 'P_ai', 'T_ai', 'sos_i', 'M_i', 'CD_i', 'rho_ai', 'F/m', 'D/m', 'a_i', 'a_i (gs)'))
    # print("---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------"
    #         "---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- "
    #         "---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------")

    while m_pi > 0 :
        m_pi = calc_m_p(N, r_1, r_0, L_0, w)
        m_i = (m_pi + m_case + m_ballast + m_struct)
        A_bi = calc_A_b(N, r_1, r_0, L_0, w)
        A_ti = pi*(d_t/2)**2
        P_ci = calc_P_c(T_b0, T_bi, A_bi, A_ti)
        P_ai, T_ai, rho_ai = calc_atm(h_i)
        r_bi = calc_r_b(T_b0, T_bi, P_ci)
        E_i = A_e0/A_ti
        c_fi = CF(gamma_prop, E_i, P_ci/P_ai)
        F_i = c_fi * P_ci * A_ti
        sos_i = m.sqrt(gamma_air*R*T_ai)
        M_i = v_i/sos_i
        if M_i <= 0.6:
            CD_i = 0.15
        elif M_i <= 1.2:
            CD_i = -0.12+0.45*M_i
        elif M_i <= 1.8:
            CD_i = 0.76-0.283*M_i
        elif M_i<=4:
            CD_i = 0.311-0.034*M_i
        else:
            CD_i = 0.175
        D_i = 0.5 * rho_ai * v_i * abs(v_i)* CD_i * A_rocket
        a_i = (F_i/m_i)*g - D_i/m_i - g
        if ((r_0-r_1)-w)<web_step:
            w_nxt = w + ((r_0-r_1)-w)
        else:
            w_nxt = w + web_step
        t_nxt = t_i + (w_nxt-w)/r_bi
        m_p_nxt = calc_m_p(N, r_1, r_0, L_0, w_nxt)
        A_b_nxt = calc_A_b(N, r_1, r_0, L_0, w_nxt)
        d_t_nxt = d_t + 0.000087 * (t_nxt - t_i) * P_ci
        A_t_nxt = pi*(d_t_nxt/2)**2
        P_c_nxt = calc_P_c(T_b0, T_bi, A_b_nxt, A_t_nxt)
        v_nxt = v_i + a_i*(t_nxt-t_i)
        h_nxt = h_i + (v_nxt+v_i)/2 * (t_nxt-t_i)
        P_a_nxt, T_a_nxt, rho_a_nxt = calc_atm(h_nxt)
        r_b = calc_r_b(T_b0, T_bi, P_c_nxt)
        E_nxt = A_e0/A_t_nxt
        c_f_nxt = CF(gamma_prop, E_nxt, P_c_nxt/P_a_nxt)
        F_nxt = c_f_nxt * P_c_nxt * A_t_nxt
        I_nxt = (F_i + F_nxt)/2 * (t_nxt - t_i)
        # print("{0:10.3f} {1:10.3f} {2:10.3f} {3:10.3f} {4:10.3f} {5:10.3f} {6:10.3f} {7:10.3f} {8:10.3f} {9:10.3f} {10:10.3f} {11:10.3f} {12:10.3f} {13:10.3f} {14:10.3f} {15:10.3f} {16:10.3f} {17:10.3f} {18:10.3f} {19:10.3f} {20:10.3f} {21:10.4f} {22:10.4f} {23:10.4f}".format(w, m_pi, t_i, A_bi, A_ti, P_ci, P_ai, r_bi, E_i, c_fi, F_i, I_i, t_i, m_i, v_i, h_i, P_ai, T_ai, sos_i, M_i, CD_i, rho_ai, F_i/m_i*g, D_i/m_i, a_i, a_i/g))
        m_ps = pd.concat([m_ps, pd.Series(m_pi)])
        A_bs = pd.concat([A_bs, pd.Series(A_bi)])
        A_ts = pd.concat([A_ts, pd.Series(A_ti)])
        AeAts = pd.concat([AeAts, pd.Series(E_i)])
        P_cs = pd.concat([P_cs, pd.Series(P_ci)])
        F_is = pd.concat([F_is, pd.Series(F_i)])
        c_fis = pd.concat([c_fis, pd.Series(c_fi)])
        t_is = pd.concat([t_is, pd.Series(t_i)])
        h_is = pd.concat([h_is, pd.Series(h_i)])
        v_is = pd.concat([v_is, pd.Series(v_i)])
        a_is = pd.concat([a_is, pd.Series(a_i)])
        I_i = I_nxt
        w = w_nxt
        t_i = t_nxt
        d_t = d_t_nxt
        i = i + 1
        I_sum = I_sum + I_i
        v_i = v_nxt
        h_i = h_nxt

    print('Mass of the Propellant: {0:.3f} lbm'.format(float(m_ps.max())))
    print('Area of the Throat at Burnout: {0:.3f} in^s'.format(float(A_ts.max())))
    print('Burnout Time: {0:.3f} s'.format(float(t_is.max())))
    print('Average Specific Impulse: {0:.3f} s'.format(I_sum/(float(m_ps.max()))))
    print('Max Chamber Pressure (<1000psi): {0:.3f} psia'.format(float(P_cs.max())))
    print('Max Thrust: {0:.3f} lbf'.format(float(F_is.max())))
    print('A_p0/A_t0: {0:.3f} (-)'.format(A_p0/A_t0))
    print('Mass of the Case: {0:.4f} lbm'.format(m_case))
    print('Initial Mass of Vehicle: {0:.4f} lbm'.format(m_0))
    print('Burnout Height: {0:.4f} ft'.format(float(h_is.iloc[-1, 0])))
    print('Burnout Velocity: {0:.4f} ft/s'.format(float(v_is.iloc[-1, 0])))
    print('Burnout Acceleration: {0:.4f} ft^2/s'.format(float(a_is.iloc[-1, 0])))

    # print("{0:10s} {1:10s} {2:10s} {3:10s} {4:10s} {5:10s} {6:10s} {7:10s} {8:10s} {9:10s} {10:10s} {11:10s} {12:10s} {13:10s} {14:10s} {15:10s} {16:10s} {17:10s} {18:10s} {19:10s} {20:10s} {21:10s} {22:10s} {23:10s}".format(
    #     '   w', '   m_pi', '   t_i', '   A_bi', '   A_ti', '   P_ci', '   P_ai', '   r_bi', '   E_i', '   c_fi', '   F_i', 
    #     '   I_i', 'm_i', 'v_i', 'h_i', 'P_ai', 'T_ai', 'sos_i', 'M_i', 'CD_i', 'rho_ai', 'F/m', 'D/m', 'a_i', 'a_i (gs)'))
    # print("---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------"
    #         "---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- "
    #         "---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------")

    while h_i>=0: 
        P_ai, T_ai, rho_ai = calc_atm(h_i)
        sos_i = m.sqrt(gamma_air*R*T_ai)
        M_i = v_i/sos_i
        if M_i <= 0.6:
            CD_i = 0.15
        elif M_i <= 1.2:
            CD_i = -0.12+0.45*M_i
        elif M_i <= 1.8:
            CD_i = 0.76-0.283*M_i
        elif M_i<=4:
            CD_i = 0.311-0.034*M_i
        else:
            CD_i = 0.175
        
        D_i = 0.5 * rho_ai * v_i * abs(v_i)* CD_i * A_rocket
        a_i = (F_i/m_i)*g - D_i/m_i - g
        t_nxt = t_i + 0.1
        v_nxt = v_i + a_i*(t_nxt-t_i)
        h_nxt = h_i + (v_nxt+v_i)/2 * (t_nxt-t_i)
        P_a_nxt, T_a_nxt, rho_a_nxt = calc_atm(h_nxt)
        F_nxt = 0
        # print("{0:10.3f} {1:10.3f} {2:10.3f} {3:10.3f} {4:10.3f} {5:10.3f} {6:10.3f} {7:10.3f} {8:10.3f} {9:10.3f} {10:10.3f} {11:10.3f} {12:10.3f} {13:10.3f} {14:10.3f} {15:10.3f} {16:10.3f} {17:10.3f} {18:10.3f} {19:10.3f} {20:10.3f} {21:10.4f} {22:10.4f} {23:10.4f}".format(w, m_pi, t_i, A_bi, A_ti, P_ci, P_ai, r_bi, E_i, c_fi, F_i, I_i, t_i, m_i, v_i, h_i, P_ai, T_ai, sos_i, M_i, CD_i, rho_ai, F_i/m_i*g, D_i/m_i, a_i, a_i/g))
        m_ps = pd.concat([m_ps, pd.Series(m_pi)])
        A_bs = pd.concat([A_bs, pd.Series(A_bi)])
        A_ts = pd.concat([A_ts, pd.Series(A_ti)])
        AeAts = pd.concat([AeAts, pd.Series(E_i)])
        P_cs = pd.concat([P_cs, pd.Series(P_ci)])
        F_is = pd.concat([F_is, pd.Series(F_i)])
        c_fis = pd.concat([c_fis, pd.Series(c_fi)])
        t_is = pd.concat([t_is, pd.Series(t_i)])
        h_is = pd.concat([h_is, pd.Series(h_i)])
        v_is = pd.concat([v_is, pd.Series(v_i)])
        a_is = pd.concat([a_is, pd.Series(a_i)])
        t_i = t_nxt
        v_i = v_nxt
        h_i = h_nxt
        F_i = 0
    F_is.index = t_is[0]

    h_max = float(h_is.max())
    v_max = float(v_is.max())
    a_max = float(a_is.max())
    g_max = float(a_is.max())/g
    print('Max Height: {0:.4f} ft'.format(h_max))
    print('Max Velocity: {0:.4f} ft/s'.format(v_max))
    print('Max Acceleration: {0:.4f} ft^2/s'.format(a_max))
    print('Max Acceleration in gees (<15): {0:.4f}'.format(g_max))
    
    # plt.rcParams.update({'font.size': 14})
    # plt.rcParams['axes.labelsize'] = 14
    # plt.plot(t_is, m_ps)
    # plt.xlabel("Time [s]")
    # plt.ylabel("Mass of Propellant [lbm]")
    # plt.xlim(0, float(t_is.max()))
    # plt.show()

    # plt.plot(t_is, P_cs)
    # plt.xlabel("Time [s]")
    # plt.ylabel("Chamber Pressure [psia]")
    # plt.xlim(0, float(t_is.max()))
    # plt.show()

    # plt.plot(t_is, F_is)
    # plt.xlabel("Time [s]")
    # plt.ylabel("Thrust [lbf]")
    # plt.xlim(0, float(t_is.max()))
    # plt.show()

    # plt.plot(AeAts, c_fis)
    # plt.xlabel("Area Ratio []]")
    # plt.ylabel("Thrust Coefficient []")
    # plt.show()

    # plt.plot(t_is, h_is)
    # plt.xlabel("Time [s]")
    # plt.ylabel("Height [ft]")
    # plt.xlim(0, float(t_is.max()))
    # plt.show()

    # plt.plot(t_is, v_is)
    # plt.xlabel("Time [s]")
    # plt.ylabel("Velocity [ft/s]")
    # plt.xlim(0, float(t_is.max()))
    # plt.show()

    # plt.plot(t_is, a_is)
    # plt.xlabel("Time [s]")
    # plt.ylabel("Acceleration [ft^2/s]")
    # plt.xlim(0, float(t_is.max()))
    # plt.show()

    # motor_draw(A_e0, N, r_0, r_1, L_0, A_t0, -0.025)
    return h_max, v_max, a_max


print('Baseline:')
PR08D(4, 1, 2.375, 8, 1, 4, 1, 70) 
print('------------------------------------------------------\n')
print('5k:')
PR08D(2, 1, 2.375, 2, 0.302, 3, 0.015, 70) 
print('------------------------------------------------------\n')
print('15k:')
PR08D(5, 1, 2.375, 2, 1.5, 3.5, 0.807, 70) 
print('------------------------------------------------------\n')

varT_hs = pd.DataFrame(); varT_vs = pd.DataFrame(); varT_as = pd.DataFrame(); varT_gs = pd.DataFrame(); 
print('10k:')
print('---------------------(30F)---------------------')
h_max_30F, v_max_30F, a_max_30F = PR08D(3, 1, 2.375, 2, 0.445, 4, 0.301, 30) 
print('\n')
print('---------------------(70F)---------------------')
h_max_70F, v_max_70F, a_max_70F = PR08D(3, 1, 2.375, 2, 0.445, 4, 0.301, 70)
print('\n')
print('---------------------(120F)---------------------')
h_max_120F, v_max_120F, a_max_120F = PR08D(3, 1, 2.375, 2, 0.445, 4, 0.301, 120) 


# varTs = pd.DataFrame([30, 70, 120])
# varT_hs = pd.DataFrame([h_max_30F, h_max_70F, h_max_120F])
# plt.plot(varTs, varT_hs, 'bo-')
# plt.xlabel("Initial Propellant Temperature [F]")
# plt.ylabel("Max Altiude [ft]")
# plt.xlim(float(varTs.min()), float(varTs.max()))
# plt.show()

# varT_vs = pd.DataFrame([v_max_30F, v_max_70F, v_max_120F])
# plt.plot(varTs, varT_vs, 'bo-')
# plt.xlabel("Initial Propellant Temperature [F]")
# plt.ylabel("Max Velocity [ft/s]")
# plt.xlim(float(varTs.min()), float(varTs.max()))
# plt.show()

# varT_as = pd.DataFrame([a_max_30F/g, a_max_70F/g, a_max_120F/g])
# plt.plot(varTs, varT_as, 'bo-')
# plt.xlabel("Initial Propellant Temperature [F]")
# plt.ylabel("Max Acceleration (a_max/g_0)")
# plt.xlim(float(varTs.min()), float(varTs.max()))
# plt.show()
