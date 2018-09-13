import math
import numpy as np

start_point = [1, 1]
learning_rate = 0.1
error_threshold = 10 ** -14
iterations_threshold = 15

def error_surface(uv_list=None):
    u = uv_list[0]
    v = uv_list[1]
    return (u * math.exp(v) - 2 * v * math.exp(-u)) ** 2


def error_surface_gradient(uv_list=None, u=None, v=None):
    u = uv_list[0]
    v = uv_list[1]
    gradient_u = 2 * (math.exp(v) + 2 * v * math.exp(-u)) * (u * math.exp(v) - 2 * v * math.exp(-u))
    gradient_v = 2 * (u * math.exp(v) - 2 * v * math.exp(-u)) * (u * math.exp(v) - 2 * math.exp(-u))
    return [gradient_u, gradient_v]


def list_to_string(list):
    u = list[0]
    v = list[1]
    return ("(" + str(u) + ", " + str(v) + ")")


def update_weight(weight):
    u = weight[0]
    v = weight[1]
    gradient = error_surface_gradient(weight)
    gradient_u = gradient[0]
    gradient_v = gradient[1]
    new_u = u - gradient_u * learning_rate
    new_v = v - gradient_v * learning_rate
    return [new_u, new_v]


def two_step_update_weight(weight):
    u = weight[0]
    v = weight[1]

    first_gradient = error_surface_gradient(weight)
    gradient_u = first_gradient[0]
    new_u = u - gradient_u * learning_rate
    weight_after_first_step = [new_u, v]

    second_gradient = error_surface_gradient(weight_after_first_step)
    gradient_v = second_gradient[1]
    new_v = v - gradient_v * learning_rate
    weight_after_second_step = [new_u, new_v]

    # TESTING
    # print("U STEP")
    # print("Gradient: " + str(first_gradient))
    # print("New Weight: " + str(weight_after_first_step))
    # print("V STEP")
    # print("Gradient: " + str(second_gradient))
    # print("New Weight: " + str(weight_after_second_step))

    return weight_after_second_step


# MAIN CODE

error = error_surface(start_point)
current_weight = start_point
number_of_iterations = 0
while number_of_iterations < iterations_threshold:
    number_of_iterations += 1
    current_weight = two_step_update_weight(current_weight)
    error = error_surface(current_weight)

    print("iteration: " + str(number_of_iterations))
    print("weights: " + list_to_string(current_weight))
    print("error: " + str(error))

# TESTING

# ERROR AND GRADIENT FUNCTIONS
# start_point = [np.random.uniform(0, 1), np.random.uniform(0, 1)]
# start_point_error = error_surface(start_point)
# start_point_error_gradient = error_surface_gradient(start_point)
# print("Start Point: " + list_to_string(start_point))
# print("Start Point Error: " + str(start_point_error))
# print("Start Point Error Gradient: " + list_to_string(start_point_error_gradient))

# UPDATING WEIGHTS
# start_point_error = error_surface(start_point)
# start_point_error_gradient = error_surface_gradient(start_point)
# new_weight = update_weight(start_point)
# print("Start Point: " + list_to_string(start_point))
# print("Start Point Error: " + str(start_point_error))
# print("Start Point Error Gradient: " + list_to_string(start_point_error_gradient))
# print("New Point: " + list_to_string(new_weight))

# UPDATING TWO STEP WEIGHTS
# start_point_error = error_surface(start_point)
# start_point_error_gradient = error_surface_gradient(start_point)
# new_weight = two_step_update_weight(start_point)
# print("Start Point: " + list_to_string(start_point))
# print("Start Point Error: " + str(start_point_error))
# print("Start Point Error Gradient: " + list_to_string(start_point_error_gradient))
# print("New Point: " + list_to_string(new_weight))
