import math
import matplotlib
import scipy.optimize

#define useful quantities
web_step = 0.01
pi = math.pi
g = 32.2
atmospheric_pressure = 14.7
str_format = lambda number, length: str(number).strip("[]")[0:length].ljust(length,"0") 

#define SRM geometry
number_of_grains = 4
inner_radius = 1
outer_radius = 2.375
initial_grain_length = 8
initial_throat_area = 1
initial_area_ratio = 4
exit_area = initial_throat_area*initial_area_ratio

initial_throat_diameter = math.sqrt(4/pi*initial_throat_area)

if outer_radius-inner_radius <= initial_grain_length/2:
	web_max = outer_radius-inner_radius
else:
	web_max = initial_grain_length/2

#define propellant characteristics
c_star = 5210
gamma = 1.25
a0 = 0.030
n = 0.35
sigma_p = 0.001
initial_bulk_temperature = 70
bulk_temperature = 70
propellant_density = 0.065


#define mach number relations
area_ratio_mach_number = lambda mach_number: 1/mach_number*((2+(gamma-1)*mach_number**2)/(gamma+1))**((gamma+1)/(2*gamma-2))
pressure_ratio_mach_number = lambda mach_number: (1+(gamma-1)/2*mach_number**2)**(-gamma/(gamma-1))
thrust_coefficient = lambda chamber_pressure_ratio, atmospheric_pressure_ratio, area_ratio: (((2*gamma**2)/(gamma-1))*(2/(gamma+1))**((gamma+1)/(gamma-1))*(1-chamber_pressure_ratio**((gamma-1)/(gamma))))**0.5+(chamber_pressure_ratio-atmospheric_pressure_ratio)*area_ratio


#create funtions involving burn
a = a0*math.exp(sigma_p*(bulk_temperature-initial_bulk_temperature))
area_web = lambda web: number_of_grains*(2*pi*(initial_grain_length-2*web)*(inner_radius+web)+2*pi*(outer_radius**2-(inner_radius+web)**2))
chamber_pressure = lambda burn_area, throat_area: ((a*propellant_density*burn_area*c_star*12)/(g*12*throat_area))**(1/(1-n))
burn_rate = lambda chamber_pressure: a*chamber_pressure**n
volume_web = lambda web: number_of_grains*(pi*(initial_grain_length-2*web)*(outer_radius**2-(inner_radius+web)**2))


#set up thrust calculation loop
web = 0
burn_time = 0
old_force = 0
impulse = 0
pc_max = 0
thrust_max = 0
throat_diameter = initial_throat_diameter
break_next = False

debug = True

#begin thrust calculation loop
while web <= web_max:

	#area ratio math
	throat_area = pi/4*throat_diameter**2
	area_ratio = exit_area/throat_area

	area_ratio_solver = lambda mach_number: area_ratio_mach_number(mach_number)-area_ratio
	exit_mach = scipy.optimize.fsolve(area_ratio_solver, 3)

	#presssure math
	chamber_pressure_ratio = pressure_ratio_mach_number(exit_mach)
	pc = chamber_pressure(area_web(web), throat_area)
	atmospheric_pressure_ratio = atmospheric_pressure/pc
	pc_max = max(pc,pc_max)

	#time math
	delta_time = web_step/burn_rate(pc)
	burn_time += delta_time
	throat_diameter += 0.000087*delta_time*pc

	#thrust math
	cf = thrust_coefficient(chamber_pressure_ratio,atmospheric_pressure_ratio,area_ratio)
	new_force = cf*initial_throat_area*pc
	thrust_max = max(new_force, thrust_max)

	#impulse math
	delta_impulse = 0.5*(new_force+old_force)*delta_time
	impulse += delta_impulse

	#update variables and move to next loop

	output = f"Web distance [in]: {str_format(web,5)}, Time [s]: {str_format(burn_time,6)}, Force [lbf]: {str_format(new_force,6)}"
	if debug == True:
		output += f"\nThroat Area [in^2]: {str_format(throat_area,5)}, Exit Mach: {str_format(exit_mach,3)}, Chamber Pressure [psi]: {str_format(pc,6)}\n"
	print(output)

	old_force = new_force

	if break_next:
		break

	web += web_step
	if web > web_max:
		web = web_max
		break_next = True


mass_propellant = volume_web(0)*propellant_density
Isp = impulse/(mass_propellant*g/32.2)

#report results
print()
print(f"Impulse [lbfs]: {impulse}")
print(f"Isp [s]:        {Isp}")
print(f"Burn Time [s]:  {burn_time}")

if debug:
	print()
	print(f"Mass of Propellant:        {mass_propellant}")
	print(f"Initial Web Area:          {area_web(0)}")
	print(f"FInal Web Area:            {area_web(web_max)}")
	print(f"Initial Chamber Pressure:  {chamber_pressure(area_web(0),initial_throat_area)}")
	print(f"Final Chamber Pressure:    {chamber_pressure(area_web(web_max),throat_area)}")
	print(f"Max Chamber Pressure:      {pc_max}")
	print(f"Max Thrust: 		   {thrust_max}")
