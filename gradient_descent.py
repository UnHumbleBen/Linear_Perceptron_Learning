import math
import numpy as np

start_point = [1, 1]
learning_rate = 0.1


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


def list_to_string(point):
    u = point[0]
    v = point[1]
    return ("(" + str(u) + ", " + str(v) + ")")


# TESTING

# ERROR AND GRADIENT FUNCTIONS
# start_point = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
# start_point_error = error_surface(start_point)
# start_point_error_gradient = error_surface_gradient(start_point)
# print("Start Point: " + list_to_string(start_point))
# print("Start Point Error: " + str(start_point_error))
# print("Start Point Error Gradient: " + list_to_string(start_point_error_gradient))
