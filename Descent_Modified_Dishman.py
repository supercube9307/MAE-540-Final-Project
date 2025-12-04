import math as m
import random
import Gradient_Descent_Test as GD

def CF(gamma, AeAt, P1oP3, StopCriteria = 0.001, AM2 = 3, IterNo = 0):
    EA = StopCriteria * 1.1
    P3oP1 = 1 / P1oP3

    AM2 = FindMach(AeAt,gamma)

    P2oP1 = (1 + 0.5 * (gamma - 1) * AM2 ** 2) ** (-gamma / (gamma - 1))

    TERM1 = 2 * gamma * gamma / (gamma - 1)
    TERM2 = 2 / (gamma + 1)
    TERM3 = (gamma + 1) / (gamma - 1)
    TERM4 = (gamma - 1) / gamma

    CF = (TERM1 * (TERM2 ** TERM3) * (1 - (P2oP1 ** TERM4))) ** 0.5 + (P2oP1 - P3oP1) * AeAt

    return CF

def FindMach(AeAt, gamma, StopCriteria = 0.000001):

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

    return AM2

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

def calc_A_b(w, N, r_1, r_0, L_0):
    A_b = N*2*m.pi*((r_1+w)*(L_0-2*w) + (r_0**2 - (r_1+w)**2))
    return A_b

def calc_P_c(T_b0, T_bi, A_b, A_t, a_0, sigma_p, rho_p, c_star, g, n):
    P_c = (a_0 * m.exp(sigma_p*(T_bi-T_b0))*(rho_p*c_star/g)*(A_b/A_t))**(1/(1-n))
    return P_c

def calc_r_b(T_b0, T_bi, P_c, a_0, sigma_p, n):
    r_b = a_0 * m.exp(sigma_p*(T_bi-T_b0)) * P_c**n
    return r_b

def calc_m_p(w, N, rho_p, r_0, r_1, L_0):
    m_p = N*rho_p*(m.pi*(r_0*r_0-(r_1+w)*(r_1+w))*(L_0-2*w))
    return m_p

def apogee_function(case_length, number_grains, inner_radius, outer_radius, grain_length, mass_ballast, throat_area, expansion_ratio):

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
	m_ballast = mass_ballast # lbm
	m_struct = 40 # lbm
	grain_spacing = 0.125
	m_case = case_length*0.25 #lbm -> 0.25lbm/in = denisty of case material

	i = 0; w = 0; I_i = 0; I_sum = 0; t_i = 0; m_pi = 1; h_i = 0; v_i = 0; P_cmax = 0
	m_p0 = calc_m_p(w, N, rho_p, r_0, r_1, L_0); m_0 = m_p0 + m_case + m_ballast + m_struct
	h_max = 0
	a_max = 0

	web_max = min(r_0-r_1,L_0/2)

	# begin burn loop

	while m_pi > 0.0000000001 :

	    m_pi = calc_m_p(w, N, rho_p, r_0, r_1, L_0)
	    m_i = (m_pi + m_case + m_ballast + m_struct)
	    A_bi = calc_A_b(w, N, r_1, r_0, L_0)
	    A_ti = pi*(d_t/2)**2
	    P_ci = calc_P_c(T_b0, T_bi, A_bi, A_ti, a_0, sigma_p, rho_p, c_star, g, n)

	    [P_ai, T_ai, rho_ai] = calc_atm(h_i)
	    sos_i = m.sqrt(gamma_air*R*T_ai)
	    r_bi = calc_r_b(T_b0, T_bi, P_ci, a_0, sigma_p, n)

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
	    m_p_nxt = calc_m_p(w_nxt, N, rho_p, r_0, r_1, L_0)
	    A_b_nxt = calc_A_b(w_nxt, N, r_1, r_0, L_0)
	    d_t_nxt = d_t + 0.000087 * (t_nxt - t_i) * P_ci
	    A_t_nxt = pi*(d_t_nxt/2)**2
	    P_c_nxt = calc_P_c(T_b0, T_bi, A_b_nxt, A_t_nxt, a_0, sigma_p, rho_p, c_star, g, n)
	    v_nxt = v_i + a_i*(t_nxt-t_i)
	    h_nxt = h_i + (v_nxt+v_i)/2 * (t_nxt-t_i)
	    P_a_nxt, T_a_nxt, rho_a_nxt = calc_atm(h_nxt)
	    r_b = calc_r_b(T_b0, T_bi, P_c_nxt, a_0, sigma_p, n)
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

def check_configuration(N_5k, N_10k, N_15k, inner_radius, outer_radius, grain_length, ballast, throat_area, expansion_ratio):

	# define constants

	if fractional_grains == False:
		N_5k = round(N_5k,0)
		N_10k = round(N_10k,0)
		N_15k = round(N_15k,0)

	error_output = ""
	error_messages = ["5k", "10k", "15k"]
	grain_spacing = 0.125
	max_grains = max(N_5k, N_10k, N_15k)
	case_length = max_grains*(grain_length+grain_spacing)

	# check for valid configuration

	constraint_error_multiplier = 0
	grain_error_multiplier = 0

	if case_length > 34:
		constraint_error_multiplier += 1 + case_length - 34
		error_output += f"Case Length (<34): {round(case_length,2)}\n"

	if inner_radius > outer_radius:
		constraint_error_multiplier += 1 + inner_radius - outer_radius
		error_output += f"Inner Radius > Outer Radius: {round(inner_radius,3)}, {round(outer_radius,3)}\n"

	port_area = (inner_radius**2)*m.pi
	if port_area / throat_area < 2:
		constraint_error_multiplier += 1 + (2 - port_area / throat_area)
		error_output += f"Port / Throat Area Ratio (>2): {round(port_area/throat_area,3)}\n"

	if ballast > 1:
		constraint_error_multiplier += ballast
		error_output += f"Ballast (<1): {round(balast,3)}\n"

	if ballast < 0:
		constraint_error_multiplier += (-ballast) + 1
		error_output += f"Ballast (>0): {round(case_length,2)}\n"

	if fractional_grains == True:
		grain_list = [N_5k, N_10k, N_15k]
		for index in range(len(grain_list)):
			grain = grain_list[index]
			grain_error_multiplier += abs(grain - round(grain,0))
			if grain != round(grain):
				error_output += "Noninteger Grain in " + error_messages[index] + f" Case : {round(grain,10)}\n"

	# find performance characteristics

	[h_max_low, pc_max_low, a_max_low] = apogee_function(max_grains, N_5k, inner_radius, outer_radius, grain_length, ballast, throat_area, expansion_ratio)
	[h_max_mid, pc_max_mid, a_max_mid] = apogee_function(max_grains, N_10k, inner_radius, outer_radius, grain_length, ballast, throat_area, expansion_ratio)
	[h_max_high, pc_max_high, a_max_high] = apogee_function(max_grains, N_15k, inner_radius, outer_radius, grain_length, ballast, throat_area, expansion_ratio)

	# check for valid performance

	pc_list = [pc_max_low, pc_max_mid, pc_max_high]
	a_list = [a_max_low, a_max_mid, a_max_high]

	for index in range(len(pc_list)):
		pc_max = pc_list[index]
		a_max = a_list[index]

		if pc_max > 1000:
			constraint_error_multiplier += pc_max/1000
			error_output += "Chamber Pressure in " + error_messages[index] + f" Case (<1000): {round(Chamber_Pressure,1)}\n"
		if a_max > 15:
			constraint_error_multiplier += a_max/15
			error_output += "Acceleration in " + error_messages[index] + f" Case (<15): {round(a_max,2)}\n"

	# find altitude differences

	low_diff = h_max_low-5000
	mid_diff = h_max_mid-10000
	high_diff = h_max_high-15000

	if error_output != "":
		error_output = "Problems:\n" + error_output

	return(low_diff, mid_diff, high_diff, constraint_error_multiplier, grain_error_multiplier, error_output)



def error_function(N_5k, N_10k, N_15k, inner_radius, outer_radius, grain_length, ballast, throat_area, expansion_ratio):

	# find performance of configuration

	configuration = [N_5k, N_10k, N_15k, inner_radius, outer_radius, grain_length, ballast, throat_area, expansion_ratio]
	[low_diff, mid_diff, high_diff, constraint_error_exp, grain_error_exp, error_output] = check_configuration(*configuration)

	# find error

	error = 0

	for alt_diff in [low_diff, mid_diff, high_diff]:
		if ignore_solved_cases and alt_diff < 100:
			alt_diff = 0
		error += abs(alt_diff) ** alt_diff_exponent

	error = error ** (1/alt_diff_exponent)

	error *= constraint_error_base ** constraint_error_exp
	error *= grain_error_base ** grain_error_exp

	return(error)



def altitude_function(N_5k, N_10k, N_15k, inner_radius, outer_radius, grain_length, ballast, throat_area, expansion_ratio):

	# find performance of configuration

	configuration = [N_5k, N_10k, N_15k, inner_radius, outer_radius, grain_length, ballast, throat_area, expansion_ratio]
	[low_diff, mid_diff, high_diff, constraint_error_exp, grain_error_exp, error_output] = check_configuration(*configuration)

	return(low_diff, mid_diff, high_diff)



def pretty_output(position):
	print(f"Current Performance: {[round(x,4) for x in altitude_function(*position)]}\n" \
		f"Configuration: {[round(x,10) for x in position]}")

# define operating parameters

fractional_grains = True
random_seed = False
ignore_solved_cases = False
gradient_debug_output = False
configuration_debug_output = True
run = True

step_size = 0.000001
alt_diff_exponent = 2
constraint_error_base = 1000
grain_error_base = 100

error_tolerance = 0
update_interval = 100
max_iterations = 10000

# setup initial guess

initial_guess = [7.0, 8.0, 9.0, 1.7897980649, 2.4946672557, 2.4239819896, 0.6621373368, 4.8787486621, 4.3774970358]

if random_seed:
	guess_limits = [[0,3],[3,6],[6,9],[0,1],[1,2.375],[1,3],[0,1],[1,2],[2,4]]
	initial_guess = []
	for limit_pair in guess_limits:
		initial_guess.append(random.uniform(*limit_pair))

updated_guess = [x for x in initial_guess]

#begin main execution

pretty_output(initial_guess)
print(check_configuration(*initial_guess)[-1])

if run:
	try:
		iterations = 1
		while iterations < max_iterations:

			[updated_guess, value] = GD.gradient_descent(error_tolerance,error_function,updated_guess,step_size,debug=gradient_debug_output,debug_precision=7,max_iterations=1)

			if iterations % update_interval == 0:
				print(f"Iterations: {iterations}")
				pretty_output(updated_guess)

				if configuration_debug_output == True:
					print(check_configuration(*updated_guess)[-1])

			iterations += 1


	except KeyboardInterrupt:
		print("\nBye Bye!\n")

	finally:
		print(f"Iterations: {iterations}\n")
		print("Started With:")
		pretty_output(initial_guess)
		print("")
		print("Ended At:")
		pretty_output(updated_guess)
		print("")

#(inner_radius, outer_radius, grain_length, ballast, throat_area, expansion_ratio)


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

#Current Performance: [-93.58243, 139.61448, -99.95299]
#Position: [0.4897046, 1.8590839, 2.9147516, 2.7e-06, 0.9999973, 2.7e-06, 1.1063312, 2.0927006]

#Current Performance: [-115.670397184, 115.731965351, -115.6721469]
#Position: [0.483760984, 1.857996605, 2.914397813, 5.2e-08, 0.999999986, 3.8e-08, 1.106791396, 2.090445684]



#numbers for 4 5 6 configuration

#Current Performance: [-61.367741191, 60.876200493, -62.85698039]
#Position: [0.355078451, 1.747524512, 2.999791451, 0.077366224, 0.974321392, 0.0063114, 1.716156602, 2.294976554]



#change to using single ballast

#numbers for 5 6 7 configuration

#Current Performance: [756.00705, 376.7791, -693.39905]
#Position: [0.4169956, 1.6579927, 2.8709978, 0.5089858, 1.9441472, 2.3801612]

#Current Performance: [398.63123, 345.42524, -396.57627]
#Position: [0.470887, 1.7348351, 2.844881, 0.5192668, 2.2096228, 2.5249239]

#Current Performance: [-142.18951, 180.71958, -164.62971]
#Position: [0.5907065, 1.8019178, 2.9062376, 0.4720588, 2.6396978, 2.6191151]

#Current Performance: [-161.27843, 166.44122, -162.33403]
#Position: [0.5491636, 1.7794092, 2.9417269, 0.4712267, 2.6063295, 2.5957065]



# numbers for 7 8 9 configuration

#Current Performance: [0.00039, 0.00069, 0.00055]
#Position: [1.2757137, 2.3533341, 2.5009745, 0.6656629, 4.6653368, 4.2024949]

#fix port area check

#Current_performance: [-3836.8373, -7738.5062, -11407.9878]
#Position: [1.9281671, 2.1562932, 2.6430464, 0.662075, 4.5719097, 4.1721649]

#Current_performance: [366.1829, 251.5188, -711.3972]
#Position: [1.7671103, 2.4409167, 2.5236486, 0.6610638, 4.7815007, 4.3139776]

#Current Performance: [38.13289, 254.90743, -341.52976]
#Position: [1.7612084, 2.514782, 2.4291327, 0.6620642, 4.8598378, 4.3651373]

#Current Performance: [29.27174, 265.30938, -308.8693]
#Position: [1.7598759, 2.5192301, 2.4242438, 0.6621349, 4.863731, 4.3676703]

#Current_performance: [45.708, 286.2022, -291.4788]
#Position: [1.7601831, 2.5189767, 2.4267905, 0.6621081, 4.8654999, 4.3689204]

#Current_performance: [42.0549, 286.1776, -288.5611]
#Position: [1.760887, 2.5195429, 2.4279883, 0.6621054, 4.8694012, 4.3715825]

#Current Performance: [51.2529, 290.7686, -294.4654]
#Position: [1.7621503, 2.5175244, 2.432753, 0.6619122, 4.8714055, 4.3730739]

#Current Performance: [10.5236, 262.7507, -295.5196]
#Position: [1.7624208681, 2.5223884116, 2.427260894, 0.6620070677, 4.8789685133, 4.3780786046]



#solution involving fractional grains

#N5k = 3.15161, N10k = 3.81157, N15k = 4.66668,
#print(altitude_function(*[0.45263, 2.18333, 2.32621, 0.35305, 0.23819, 0.93539, 1.67634, 3.38209]))
#(-77.3685560410022, -97.53787542411555, -67.24469208793198)



#this guy hangs for some reason, investigate later ig
#initial_guess = [0.68925, 2.22273, 2.94149, 0.73477, 0.47957, 0.26708, 1.93996, 3.03405]
#it was floating point imprecision preventing the mass of propellant from reaching 0

