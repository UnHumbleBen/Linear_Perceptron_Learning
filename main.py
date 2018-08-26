import random

testing = True
testing_with_fixed_points = False


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


slope = slope(first_rand_point_x1, first_rand_point_x2, second_rand_point_x1, second_rand_point_x2)
b_constant = b_constant(slope, first_rand_point_x1, first_rand_point_x2)
target_function_weights = [-b_constant, -slope, 1]
test_inputs = [[1, 0.5, 0.5], [1, -0.5, 0.5], [1, -0.5, -0.5], [1, 0.5, -0.5]]

# testing line generation
if testing:
    print("first point is (" + str(first_rand_point_x1) + ", " + str(first_rand_point_x2) + ")")
    print("second point is (" + str(second_rand_point_x1) + ", " + str(second_rand_point_x2) + ")")
    print("slope: " + str(slope))
    print("b constant: " + str(b_constant))
    print("target function weights: " + str(target_function_weights))

    for test_point in test_inputs:
        print("test inputs: " + str(test_point))
        print("dot product: " + str(dot_product(test_point, target_function_weights)))
        print("test_target_function: " + str(target_function(test_point)))

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
