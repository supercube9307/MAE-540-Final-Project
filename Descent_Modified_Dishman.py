import math as m
import random
import Gradient_Descent_Test as GD
import scipy.optimize

def apogee_function(number_grains, inner_radius, outer_radius, grain_length, case_length, mass_ballast, throat_area, expansion_ratio):

	def CF(gamma, AeAt, P1oP3):
	    StopCriteria = 0.001
	    EA = StopCriteria * 1.1
	    AM2 = 3
	    IterNo = 0
	    P3oP1 = 1 / P1oP3
	    #while EA > StopCriteria and IterNo < 100:
	    #    IterNo = IterNo + 1
	    #    AFUN = (2 + (gamma - 1) * AM2 ** 2) / (gamma + 1)
	    #    BFUN = (gamma + 1) / (2 * (gamma - 1))
	    #    CFUN = 1 / AFUN
	    #    DFUN = 1 / (AM2 ** 2)
	    #    FUN = ((1 / AM2) * AFUN ** BFUN) - AeAt
	    #    DERFUN = ((AFUN) ** BFUN) * (CFUN - DFUN)
	    #    AMOLD = AM2
	    #    AM2 = AM2 - FUN / DERFUN
	    #    EA = abs((AM2 - AMOLD) / AM2) * 100
	    AM2 = FindMach(AeAt,gamma)
	    P2oP1 = (1 + 0.5 * (gamma - 1) * AM2 ** 2) ** (-gamma / (gamma - 1))
	    TERM1 = 2 * gamma * gamma / (gamma - 1)
	    TERM2 = 2 / (gamma + 1)
	    TERM3 = (gamma + 1) / (gamma - 1)
	    TERM4 = (gamma - 1) / gamma
	    CF = (TERM1 * (TERM2 ** TERM3) * (1 - (P2oP1 ** TERM4))) ** 0.5 + (P2oP1 - P3oP1) * AeAt

	    return CF

	def calc_atm(alt):
	    if alt < 83000:
	        P_a = -4.272981*10**-14*alt**3 + 0.000000008060081*alt**2 - 0.0005482655*alt + 14.69241
	    else:
	        P_a = 0.00001
	    if alt<32809:
	        T_a = -0.0036*alt+518
	    else:
	        T_a = 399
	    if alt<82000:
	        rho_a = 0.00000000001255*alt**2-0.0000019453*alt+0.07579
	    else:
	        rho_a = 0

	    return [P_a, T_a, rho_a]

	def calc_m_p(w):
	    m_p = N*rho_p*(pi*(r_0*r_0-(r_1+w)*(r_1+w))*(L_0-2*w))
	    return m_p

	def calc_A_b(w):
	    A_b = N*2*pi*((r_1+w)*(L_0-2*w) + (r_0**2 - (r_1+w)**2))
	    return A_b

	def calc_P_c(T_b0, T_bi, A_b, A_t):
	    P_c = (a_0 * m.exp(sigma_p*(T_bi-T_b0))*(rho_p*c_star/g)*(A_b/A_t))**(1/(1-n))
	    return P_c

	def calc_r_b(T_b0, T_bi, P_c):
	    r_b = a_0 * m.exp(sigma_p*(T_bi-T_b0)) * P_c**n
	    return r_b

	def FindMach(AeAt,gamma):
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

	#define constant variables

	pi = m.pi
	c_star = 5210 #ft/s
	g = 32.2 #ft/s^2
	gamma_prop = 1.25
	gamma_air = 1.4
	Ru = 1545.3 # (ft-lbf)/(lbm*R)
	mw_air = 28.97
	R = Ru/mw_air * g
	a_0 = 0.03 #(in/s)[psi]^(-n)
	n = 0.35
	sigma_p = 0.001 #1/F
	T_b0 = 70 # F
	T_bi = 70 # F
	rho_p = 0.065
	D_rocket = 6.19
	A_rocket = (pi*(D_rocket/2)**2)/12**2
	web_step = 0.01

	# define function variables

	N = number_grains
	r_1 = inner_radius # in
	r_0 = outer_radius # in
	L_0 = grain_length # in
	A_t0 = throat_area # in^2
	A_p0 = (r_1**2)*pi
	d_t = 2*m.sqrt(A_t0/pi)
	E_0 = expansion_ratio
	A_e0 = expansion_ratio*throat_area
	m_case = case_length*0.25 #lbm -> 0.25lbm/in = denisty of case material
	m_ballast = mass_ballast # lbm
	m_struct = 40 # lbm

	i = 0; w = 0; I_i = 0; I_sum = 0; t_i = 0; m_pi = 1; h_i = 0; v_i = 0; P_cmax = 0
	m_p0 = calc_m_p(w); m_0 = m_p0 + m_case + m_ballast + m_struct
	h_max = 0
	a_max = 0

	web_max = min(r_0-r_1,L_0/2)

	# begin burn loop

	while m_pi > 0.0000000001 :

	    m_pi = calc_m_p(w)
	    m_i = (m_pi + m_case + m_ballast + m_struct)
	    A_bi = calc_A_b(w)
	    A_ti = pi*(d_t/2)**2
	    P_ci = calc_P_c(T_b0, T_bi, A_bi, A_ti)
	    [P_ai, T_ai, rho_ai] = calc_atm(h_i)
	    sos_i = m.sqrt(gamma_air*R*T_ai)
	    r_bi = calc_r_b(T_b0, T_bi, P_ci)
	    E_i = A_e0/A_ti
	    c_fi = CF(gamma_prop, E_i, P_ci/P_ai)
	    F_i = c_fi * P_ci * A_ti
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
	    a_max = max(a_i,a_max)
	    if (web_max-w)<web_step:
	        w_nxt = w + (web_max-w)
	    else:
	        w_nxt = w + web_step
	    t_nxt = t_i + (w_nxt-w)/r_bi
	    m_p_nxt = calc_m_p(w_nxt)
	    A_b_nxt = calc_A_b(w_nxt)
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
	    P_cmax = max(P_ci,P_cmax)
	    I_i = I_nxt
	    w = w_nxt
	    t_i = t_nxt
	    d_t = d_t_nxt
	    i = i + 1
	    I_sum = I_sum + I_i
	    v_i = v_nxt
	    h_i = h_nxt



	#begin trajectory loop


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
	    h_max = max(h_max,h_i)
	    a_max = max(a_max,a_i)
	    t_i = t_nxt
	    v_i = v_nxt
	    h_i = h_nxt
	    F_i = 0

	# return apogee, max chamber pressure, max acceleration in gees

	return(h_max,P_cmax,a_max/g)

def altitude_function(inner_radius, outer_radius, grain_length, ballast_5k, ballast_10k, ballast_15k, throat_area, expansion_ratio):

	# define constants

	grain_spacing = 0.125
	alt_err_tol = 100

	# define function variables

	case_length = N_15k*(grain_length+grain_spacing)

	# find performance characteristics

	[h_max_low, pc_max_low, a_max_low] = apogee_function(N_5k, inner_radius, outer_radius, grain_length, case_length, ballast_5k, throat_area, expansion_ratio)
	[h_max_mid, pc_max_mid, a_max_mid] = apogee_function(N_10k, inner_radius, outer_radius, grain_length, case_length, ballast_10k, throat_area, expansion_ratio)
	[h_max_high, pc_max_high, a_max_high] = apogee_function(N_15k, inner_radius, outer_radius, grain_length, case_length, ballast_15k, throat_area, expansion_ratio)

	# find altitude differences

	low_diff = h_max_low-5000
	mid_diff = h_max_mid-10000
	high_diff = h_max_high-15000

	return(low_diff, mid_diff, high_diff)


def error_function(inner_radius, outer_radius, grain_length, ballast_5k, ballast_10k, ballast_15k, throat_area, expansion_ratio):

	# define constants

	grain_spacing = 0.125
	alt_err_tol = 100

	# define function variables

	case_length = N_15k*(grain_length+grain_spacing)

	# check for valid configuration

	error_multiplier = 0

	if case_length > 34:
		error_multiplier += 1 + case_length - 34

	if inner_radius  > outer_radius:
		error_multiplier += 1 + inner_radius - outer_radius

	if inner_radius*m.pi**2/throat_area < 2:
		error_multiplier += 1 + (2 - inner_radius*m.pi**2/throat_area)

	for ballast in [ballast_5k, ballast_10k, ballast_15k]:
		if ballast > 1:
			error_multiplier += ballast

		if ballast < 0:
			error_multiplier -= ballast - 1

	# find performance characteristics

	[h_max_low, pc_max_low, a_max_low] = apogee_function(N_5k, inner_radius, outer_radius, grain_length, case_length, ballast_5k, throat_area, expansion_ratio)
	[h_max_mid, pc_max_mid, a_max_mid] = apogee_function(N_10k, inner_radius, outer_radius, grain_length, case_length, ballast_10k, throat_area, expansion_ratio)
	[h_max_high, pc_max_high, a_max_high] = apogee_function(N_15k, inner_radius, outer_radius, grain_length, case_length, ballast_15k, throat_area, expansion_ratio)

	# check for valid performance
	for pc_max in [pc_max_low, pc_max_mid, pc_max_high]:
		if pc_max > 1000:
			error_multiplier += pc_max/1000

	for a_max in [a_max_low, a_max_mid, a_max_high]:
		if a_max > 15:
			error_multiplier += a_max/15

	# find altitude differences

	low_diff = h_max_low-4900
	mid_diff = h_max_mid-10100
	high_diff = h_max_high-14900

	# find error

	error = 0

	for diff in [low_diff, mid_diff, high_diff]:

#		if abs(diff) < alt_err_tol:
			# ignore differences within tolerance
#			diff *= 0.2

		error += diff ** 2

	error = error**0.5
	error *= 1000**error_multiplier

	return(error)

#print(apogee_function(4, 1, 2.375, 8, 32.5, 1, 1, 4))
#print(apogee_function(2, 1, 2.375, 2, 4.25, 0.015, 0.302, 3))
#print(apogee_function(3, 1, 2.375, 2, 6.375, 0.301, 0.445, 4))
#print(apogee_function(5, 1, 2.375, 2, 10.625, 0.807, 1.5, 3.5))
#print()

N_5k = 3
N_10k = 4
N_15k = 5


#initial_guess = [1, 2.375, 2, 0.015, 0.301, 0.807, 1.5, 3.5]

#this guy hangs for some reason, investigate later ig
#initial_guess = [0.68925, 2.22273, 2.94149, 0.73477, 0.47957, 0.26708, 1.93996, 3.03405]
#it was floating point imprecision preventing the mass of propellant from reaching 0

initial_guess = [0.4897046, 1.8590839, 2.9147516, 2.7e-06, 0.9999973, 2.7e-06, 1.1063312, 2.0927006]
guess_limits = [[0,1],[1,2.375],[1,3],[0,1],[0,1],[0,1],[1,2],[2,4]]

#initial_guess = []
#for limit_pair in guess_limits:
#	initial_guess.append(random.uniform(*limit_pair))

error = 0
step_size = 0.0000001


updated_guess = [x for x in initial_guess]

print(f"Current_performance: {[round(x,5) for x in altitude_function(*initial_guess)]}\nPosition {[round(x,4) for x in initial_guess]}\n")
#print(error_function(*initial_guess))

run = True
if run:
	try:
		[updated_guess, value] = GD.gradient_descent(error,error_function,updated_guess,step_size,debug=True,debug_precision=7,max_iterations=1000000000)
	except KeyboardInterrupt:
		print("Bye Bye!")
	finally:
		print(f"Current Performance: {[round(x,5) for x in altitude_function(*updated_guess)]}\nPosition: {[round(x,7) for x in updated_guess]}\n")
#(inner_radius, outer_radius, grain_length, ballast_5k, ballast_10k, ballast_15k, throat_area, expansion_ratio)


#numbers for 2 3 4 configuration
#Current Performance: -188.22 364.71 -248.85
#Position [0.32264, 2.36696, 1.85811, 0.00108, 0.99862, 0.44157, 0.63607, 2.20978]))


#numbers for 3 4 5 configuraiton
#Current Performance: [-187.3, 279.15, -293.36]
#Position [0.3411, 2.1637, 1.8533, 0.0275, 0.9293, 0.6508, 1.2185, 3.6753])

#Current Performance: [-165.83, 243.62, -135.28]
#Position: [0.4946, 1.8645, 2.9228, 0.4114, 0.8932, 0.5209, 1.1245, 2.1047])

#Current Performance: [-165.83, 243.62, -135.28]
#Position: [0.4946, 1.8645, 2.9228, 0.4114, 0.8932, 0.5209, 1.1245, 2.1047])

#Current Performance: [-91.78019323366243, 190.29051297345723, -22.504723257241494]
#Position: [0.4946, 1.8645, 2.920, 0, 1, 0, 1.1245, 2.1047]

#iterate above through 200,000 iterations VVV

#Current Performance: [-93.58243, 139.61448, -99.95299]
#Position: [0.4897046, 1.8590839, 2.9147516, 2.7e-06, 0.9999973, 2.7e-06, 1.1063312, 2.0927006]


#solution involving fractional grains

#N5k = 3.15161, N10k = 3.81157, N15k = 4.66668,
#print(altitude_function(*[0.45263, 2.18333, 2.32621, 0.35305, 0.23819, 0.93539, 1.67634, 3.38209]))
#(-77.3685560410022, -97.53787542411555, -67.24469208793198)


