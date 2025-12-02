import math
import statistics as stat

def centered_difference(function, position, step_size):
	dfunction = (function(position + step_size)-function(position-step_size))/(2*step_size)
	return dfunction


def d_test_d_argument(function,position,argument_index,step_size):

	def test_partial_argument(function,argument_index,position):
	#return new function with inputs as position except for argument at argument index
		def new_function(argument):
			new_position = [x for x in position]
			new_position[argument_index] = argument
			return(function(*new_position))
		return new_function

	return centered_difference(test_partial_argument(function,argument_index,position),position[argument_index],step_size)

def gradient(function, position, step_size):
	if function.__code__.co_argcount != len(position):
		print("ERROR: argument disagreement between function and position")
	else:
		#initialize gradient output vector with same number of inputs as function
		gradient_vector = [x for x in position]
		for argument_index in range(len(position)):
			gradient_vector[argument_index] =  d_test_d_argument(function,position,argument_index,step_size)
		return gradient_vector


def gradient_normalized(function, position, step_size):

	gradient_magnitude = 0
	gradient_vector = gradient(function, position, step_size)

	for index in range(len(gradient_vector)):
		gradient_magnitude += gradient_vector[index]**2

	gradient_magnitude = gradient_magnitude**0.5

	if gradient_magnitude == 0:
		gradient_magnitude = 1

	for index in range(len(gradient_vector)):
		gradient_vector[index] = gradient_vector[index]/gradient_magnitude

	return gradient_vector

def gradient_descent(error_tolerance, function, position_input, step_size, max_iterations=1000, debug=False, debug_precision=5):

	position = [x for x in position_input]
	dimension = range(len(position))
	iterations = 0
	previous_values = [0]*20

	# gradent step size should be radius of insphere of n-dim octahedron sampled by center difference method
	# weird math thing to make sure gradient step doesn't exceed region sampled by center difference

	gradient_step_size = step_size
	CD_step_size = 10 * step_size * max(dimension)**0.5

	while True:
		value = function(*position)
		previous_values[iterations % len(previous_values)] = value
		deviation = stat.stdev(previous_values)
		gradient_vector = gradient_normalized(function,position,CD_step_size)

		if debug and iterations % 100 == 0:
			debug_output =  f"Standard Deviation: {round(deviation,debug_precision)}\n" \
					f"Iterations: {iterations}\n" \
					f"Value: {round(value,debug_precision)}\n" \
					f"Position: {[round(x,debug_precision) for x in position]}\n" \
					f"Direction: {[round(-x,debug_precision) for x in gradient_vector]}\n"
			print(debug_output)

		# do the actual descent part
		for index in dimension:
			position[index] = position[index] - gradient_vector[index] * gradient_step_size

		if deviation < error_tolerance:
			break

		if iterations > max_iterations:
			print("Max Iterations Reached")
			break

		iterations += 1

	return(position, value)

if __name__ == "__main__":

	def test_function (x,y,z):
		f = (x+1)**2 + y**2 + (z-2)**4 + 1
		return f

	location = [1,2,5]
	error_tolerance = 0.001

	gradient_step_size = 0.1
	output = gradient_descent(error_tolerance,test_function,location,gradient_step_size,debug=True)
	print(output)

	print("\nChanging step Size to 0.01\n")

	gradient_step_size = 0.01
	output = gradient_descent(error_tolerance,test_function,location,gradient_step_size)
	print(output)




