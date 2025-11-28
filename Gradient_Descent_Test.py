import math

def centered_difference(function, position, step_size):
	dfunction = (function(position + step_size)-function(position-step_size))/(2*step_size)
	return dfunction

def test_partial_argument(function,argument_index,position):
	#return new function with inputs as position except for argument at argument index
	def new_function(argument):
		new_position = [x for x in position]
		new_position[argument_index] = argument
		return(function(*new_position))
	return new_function

def d_test_d_argument(function,position,argument_index,step_size):
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

	gradient_vector = gradient(function, position, step_size)
	gradient_magnitude = 0
	dimension = range(len(gradient_vector))

	for index in dimension:
		gradient_magnitude += gradient_vector[index]**2

	gradient_magnitude = gradient_magnitude**0.5

	for index in dimension:
		gradient_vector[index] = gradient_vector[index]/gradient_magnitude
	return gradient_vector


def gradient_descent(error_tolerance, function, position_input, gradient_step_size, CD_step_size):
	if CD_step_size < gradient_step_size:
		print("Warning, center difference approximation step size smaller than gradient step size, may lead to erratic behavior due to overshooting")
	position = [x for x in position_input]
	error = error_tolerance + 1
	dimension = range(len(position))
	old_value = 999
	iterations = 0
	while error > error_tolerance:
		if iterations > 1000:
			return(position, value)
			break
		value = function(*position)
		error = abs(value-old_value)
		gradient_vector = gradient_normalized(function,position,CD_step_size)

		print(f"Iterations: {iterations}")
		print(f"Value: {round(value)}")
		#print(f"Error: {round(error)}")
		print(f"Position: {[round(x,5) for x in position]}")
		print(f"Gradient: {[round(x,5) for x in gradient_vector]}")
		print("")
		for index in dimension:
			position[index] = position[index] - gradient_vector[index] * gradient_step_size
		old_value = value
		iterations += 1
	return(position, value)

if __name__ == "__main__":

	def test_function (x,y):
		f = (x+1)**2 + y**2 + 1
		return f

	location = [1,2]
	CD_step_size = 0.0000001
	gradient_step_size = 0.1
	error_tolerance = 0.001

	output = gradient_descent(error_tolerance,test_function,location,gradient_step_size,CD_step_size)
	print(output)
	print("\nChanging step Size\n")

	gradient_step_size = 0.001
	error_tolerance = 0.0000001

	print(gradient_descent(error_tolerance,test_function,location,gradient_step_size,CD_step_size))



