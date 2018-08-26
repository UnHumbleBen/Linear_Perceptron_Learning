import random

testing = True
testing_with_fixed_points = False
number_of_training_points = 10


def sign(n):
    if n == 0:
        return 0
    elif n > 0:
        return 1
    else:
        return -1


first_rand_point_x1 = random.uniform(-1, 1)
first_rand_point_x2 = random.uniform(-1, 1)
second_rand_point_x1 = random.uniform(-1, 1)
second_rand_point_x2 = random.uniform(-1, 1)

# fixed points for testing
if testing and testing_with_fixed_points:
    first_rand_point_x1 = -1
    first_rand_point_x2 = -1
    second_rand_point_x1 = 1
    second_rand_point_x2 = 1


def slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


def b_constant(m, x1, x2):
    return x2 - m * x1


def dot_product(W, X):
    total = 0
    for index in range(0, len(W)):
        total += W[index] * X[index]
    return total


def target_function(inputs):
    return sign(dot_product(target_function_weights, inputs))


def g_function(inputs):
    return sign(dot_product(g_function_weights, inputs))


slope = slope(first_rand_point_x1, first_rand_point_x2, second_rand_point_x1, second_rand_point_x2)
b_constant = b_constant(slope, first_rand_point_x1, first_rand_point_x2)
target_function_weights = [-b_constant, -slope, 1]

data_set_x = []

for _ in range(number_of_training_points):
    x0 = 1
    x1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    data_set_x.append([x0, x1, x2])

# testing purpose (printing data set x)
# data_set_x_str = ""
# for input in data_set_x:
#     x1 = input[1]
#     x2 = input[2]
#     data_set_x_str += "(" + str(x1) + ", " + str(x2) + ") "
# print(data_set_x_str)

data_set_y = []
for input in data_set_x:
    y = target_function(input)
    data_set_y.append(y)

number_of_iterations = 0
g_function_weights = [0, 0, 0]
misclassified_points = range(number_of_training_points)
while len(misclassified_points) > 0:
    number_of_iterations += 1
    random_misclassified_point_index = misclassified_points[random.randrange(len(misclassified_points))]
    random_misclassified_point_y = data_set_y[random_misclassified_point_index]
    random_misclassified_point_x = data_set_x[random_misclassified_point_index]

    # testing purpose
    # prior_g_function_weights = g_function_weights.copy()

    for d in range(0, len(g_function_weights)):
        g_function_weights[d] = g_function_weights[d] + random_misclassified_point_y * random_misclassified_point_x[d]

    # testing purpose (printing g function)
    # g_function_w0 = g_function_weights[0]
    # g_function_w1 = g_function_weights[1]
    # g_function_w2 = g_function_weights[2]
    # g_function_str = "f(x,y) = " + str(g_function_w0) + " + " + str(g_function_w1) + "x + " + str(g_function_w2) + "y"

    misclassified_points = []
    for n in range(len(data_set_x)):
        input = data_set_x[n]
        actual_y = target_function(input)
        learned_y = g_function(input)
        if (actual_y != learned_y):
            misclassified_points.append(n)



    # printing iterations
    # print("iteration: " + str(number_of_iterations))
    # print("random index: " + str(random_misclassified_point_index))
    # print("y: " + str(random_misclassified_point_y))
    # print("x: " + str(random_misclassified_point_x))
    # print("weights before: " + str(prior_g_function_weights))
    # print("weights after:  " + str(g_function_weights))
    # print("g_function: " + g_function_str)
    # print("Data set Y (target)    : " + str(data_set_y))
    # data_set_y_g = []
    # for point in data_set_x:
    #     data_set_y_g.append(g_function(point))
    # print("Data set Y (based on g): " + str(data_set_y_g))
    # print("new list of misclassified points: " + str(misclassified_points))

    # if number_of_iterations >= 2:
    #     break

# testing data set generation
# print("first point is (" + str(first_rand_point_x1) + ", " + str(first_rand_point_x2) + ")")
# print("second point is (" + str(second_rand_point_x1) + ", " + str(second_rand_point_x2) + ")")
# print("slope: " + str(slope))
# print("b constant: " + str(b_constant))
# print("Equation of line: " + str(slope) + "x + " + str(b_constant))
# print("target function weights: " + str(target_function_weights))

# print("Data set X: " + str(data_set_x))
# # for point in data_set_x:
# #     print("(" + str(point[1]) + ", " + str(point[2]) + ")")
#
# print("Data set Y: " + str(data_set_y))
# data_set_y_g = []
# for point in data_set_x:
#     data_set_y_g.append(g_function(point))
# print("Data set G: " + str(data_set_y_g))
# test_inputs = [[1, 0.5, 0.5], [1, -0.5, 0.5], [1, -0.5, -0.5], [1, 0.5, -0.5]]

# testing line generation
# if testing:
#     print("first point is (" + str(first_rand_point_x1) + ", " + str(first_rand_point_x2) + ")")
#     print("second point is (" + str(second_rand_point_x1) + ", " + str(second_rand_point_x2) + ")")
#     print("slope: " + str(slope))
#     print("b constant: " + str(b_constant))
#     print("target function weights: " + str(target_function_weights))
#
#     for test_point in test_inputs:
#         print("test inputs: " + str(test_point))
#         print("dot product: " + str(dot_product(test_point, target_function_weights)))
#         print("test_target_function: " + str(target_function(test_point)))


# testing dot product
# A = [0, 4, -2]
# B = [2, -1, 7]
# print(dot_product(A, B))

# testing slope-intercept
# first_x1 = -8
# first_x2 = 8
# second_x1 = 1
# second_x2 = -10
# slope = slope(first_x1, first_x2, second_x1, second_x2)
# print("slope: " + str(slope))
# print("y-intercept: " + str(b_constant(slope, first_x1, first_x2)))

# testing slope function
# print(str(slope(-10, 1, 0, -4)))

# testing random
# print("first point is (" + str(first_rand_point_x1) + ", " + str(first_rand_point_x2) + ")")
# print("second point is (" + str(second_rand_point_x1) + ", " + str(second_rand_point_x2) + ")")

# learning how to use random
# for _ in range(10):
#     value = random.uniform(-1, 1)
#     print(value)


# testing sign function
# test = [-1, 2, 3, 0, -20, 500, 9.4, -7.6]
# for n in test:
#     print("number: " + str(n))
#     print("sign: " + str(sign(n)))
