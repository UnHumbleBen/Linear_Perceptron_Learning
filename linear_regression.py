from random import uniform
from random import randrange
from numpy import reshape
from numpy import linalg
from numpy import matmul


def experiment(number_of_training_points, number_of_out_of_sample_points):
    def sign(n):
        if n == 0:
            return 0
        elif n > 0:
            return 1
        else:
            return -1

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
        return sign(dot_product(g_function_weight, inputs))

    def g_misclassified_points():
        misclassified_points = []
        for n in range(len(data_set_x)):
            input = data_set_x[n]
            actual_y = target_function(input)
            learned_y = g_function(input)
            if (actual_y != learned_y):
                misclassified_points.append(n)
        return misclassified_points

    # choosing two random points in [-1, 1] x [-1, 1]
    first_rand_point_x1 = uniform(-1, 1)
    first_rand_point_x2 = uniform(-1, 1)
    second_rand_point_x1 = uniform(-1, 1)
    second_rand_point_x2 = uniform(-1, 1)

    # finding line between the two random points and setting target function weights
    slope = slope(first_rand_point_x1, first_rand_point_x2, second_rand_point_x1, second_rand_point_x2)
    b_constant = b_constant(slope, first_rand_point_x1, first_rand_point_x2)
    target_function_weights = [-b_constant, -slope, 1]

    # choosing data set inputs randomly
    data_set_x = []
    for _ in range(number_of_training_points):
        x0 = 1
        x1 = uniform(-1, 1)
        x2 = uniform(-1, 1)
        data_set_x.append([x0, x1, x2])

    # evaluating corresponding output
    data_set_y = []
    for input in data_set_x:
        y = target_function(input)
        data_set_y.append(y)

    # construct x_matrix
    x_matrix = reshape(data_set_x, (number_of_training_points, 3))

    # construct y_matrix
    y_matrix = reshape(data_set_y, (number_of_training_points, 1))

    # solve for weight
    psuedo_inverse_x_matrix = linalg.pinv(x_matrix)
    w_matrix = matmul(psuedo_inverse_x_matrix, y_matrix)
    g_function_weight = []
    for weight in w_matrix:
        g_function_weight.append(weight[0])

    # calculating fraction of in-sample points
    # create list of misclassified points for PLA
    misclassified_points = g_misclassified_points()
    fraction_in_sample = len(misclassified_points) / number_of_training_points

    # calculating fraction of out-of-sample points
    number_of_incorrect_out_of_sample = 0
    for _ in range(number_of_out_of_sample_points):
        random_point_x1 = uniform(-1, 1)
        random_point_x2 = uniform(-1, 1)
        random_input = [1, random_point_x1, random_point_x2]
        f_value = target_function(random_input)
        g_value = g_function(random_input)
        if f_value != g_value:
            number_of_incorrect_out_of_sample += 1
    fraction_out_of_sample = number_of_incorrect_out_of_sample / number_of_out_of_sample_points

    # iterating PLA
    number_of_iterations = 0
    while len(misclassified_points) > 0:
        number_of_iterations += 1

        # choosing a random misclassified point
        random_misclassified_point_index = misclassified_points[randrange(len(misclassified_points))]
        random_misclassified_point_y = data_set_y[random_misclassified_point_index]
        random_misclassified_point_x = data_set_x[random_misclassified_point_index]

        # update weights to properly classify point
        for d in range(0, len(g_function_weight)):
            g_function_weight[d] = g_function_weight[d] + random_misclassified_point_y * random_misclassified_point_x[d]

        # creating new set of misclassified points
        misclassified_points = g_misclassified_points()

    return [fraction_in_sample, fraction_out_of_sample, number_of_iterations]


number_of_training_points = 10
number_of_experiments = 1000
number_of_out_of_sample_points = 1000
results = [0, 0, 0]
for _ in range(number_of_experiments):
    experiment_results = experiment(number_of_training_points, number_of_out_of_sample_points)
    results[0] += experiment_results[0]
    results[1] += experiment_results[1]
    results[2] += experiment_results[2]

avg_fraction_in = results[0] / number_of_experiments
avg_fraction_out = results[1] / number_of_experiments
avg_iterations = results[2] / number_of_experiments

print("in sample errors:             " + str(avg_fraction_in))
print("out of sample errors:         " + str(avg_fraction_out))
print("average number of iterations: " + str(avg_iterations))
