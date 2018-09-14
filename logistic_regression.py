import numpy as np


def slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


def b_constant(m, x1, x2):
    return x2 - m * x1


def dot_product(W, X):
    total = 0
    for index in range(0, len(W)):
        total += W[index] * X[index]
    return total


def random_target_weights():
    # choosing two random points in [-1, 1] x [-1, 1]
    first_rand_point_x1 = np.random.uniform(-1, 1)
    first_rand_point_x2 = np.random.uniform(-1, 1)
    second_rand_point_x1 = np.random.uniform(-1, 1)
    second_rand_point_x2 = np.random.uniform(-1, 1)

    # finding line between the two random points and setting target function weights
    rand_slope = slope(first_rand_point_x1, first_rand_point_x2, second_rand_point_x1, second_rand_point_x2)
    rand_b_constant = b_constant(rand_slope, first_rand_point_x1, first_rand_point_x2)
    rand_weight = [-rand_b_constant, -rand_slope, 1]

    return rand_weight


def target_function(target_function_weights, inputs):
    value = dot_product(target_function_weights, inputs)
    if value > 0:
        return 1
    else:
        return -1


def training_data_set_x():
    set = []
    for _ in range(number_of_training_points):
        x0 = 1
        x1 = np.random.uniform(-1, 1)
        x2 = np.random.uniform(-1, 1)
        set.append([x0, x1, x2])
    return set


def training_data_set_y(weight, set_x):
    set_y = []
    for input in set_x:
        value = target_function(weight, input)
        set_y.append(value)
    return set_y


def weight_to_equation(weight):
    b = -weight[0]
    m = -weight[1]
    return ("y = " + str(m) + "x + " + str(b))


def print_data_sets(X, Y):
    print("PRINTING DATA SET")
    for index in range(len(X)):
        x = X[index]
        x1 = x[1]
        x2 = x[2]
        y = Y[index]
        print("Data " + str(index) + ": f(" + str(x1) + ", " + str(x2) + ") = " + str(y))


def in_sample_error(W, X, Y):
    sum_error = 0
    for n in range(number_of_training_points):
        x = X[n]
        y = Y[n]
        w_dot_x = dot_product(W, x)
        this_error = np.log(1 + np.exp(-y * w_dot_x))
        sum_error += this_error

        # TESTING
        # print("Index: " + str(n))
        # print("X: " + str(x))
        # print("Y: " + str(y))
        # print("W dot X: " + str(dot_product(W, x)))
        # print("this error: " + str(this_error))
    return sum_error / number_of_training_points


def error_gradient(W, X, Y):
    gradient = []
    for index in range(3):
        this_partial = 0
        for n in range(number_of_training_points):
            x = X[n]
            y = Y[n]
            w_dot_x = dot_product(W, x)
            x_index = x[index]
            this_error = y * x_index / (1 + np.exp(y * w_dot_x))
            this_partial += this_error
        this_partial /= -number_of_training_points
        gradient.append(this_partial)
    return gradient


def stochastic_error_gradient(W, X, Y, n):
    gradient = []
    for index in range(3):
        x = X[n]
        y = Y[n]
        w_dot_x = dot_product(W, x)
        x_index = x[index]
        this_partial = -y * x_index / (1 + np.exp(y * w_dot_x))
        gradient.append(this_partial)
    return gradient


def update_weight(weight, gradient):
    new_weight = []
    for index in range(len(weight)):
        this_current_weight = weight[index]
        this_gradient = gradient[index]
        this_new_weight = this_current_weight - learning_rate * this_gradient
        new_weight.append(this_new_weight)
    return new_weight


def stochastic_update_weight(weight, X, Y):
    new_weight = weight.copy()
    shuffled_indices = (list(range(number_of_training_points)))
    np.random.shuffle(shuffled_indices)
    for n in shuffled_indices:
        this_gradient = stochastic_error_gradient(new_weight, X, Y, n)
        for index in range(len(weight)):
            new_weight[index] = new_weight[index] - learning_rate * this_gradient[index]
    return new_weight


def difference_norm(weight, new_weight):
    difference = weight.copy()
    for index in range(len(weight)):
        difference[index] -= new_weight[index]
    # print("difference: " + str(difference))
    return np.sqrt(dot_product(difference, difference))


def experiment():
    target_function_weight = random_target_weights()
    data_set_x = training_data_set_x()
    data_set_y = training_data_set_y(target_function_weight, data_set_x)

    # TESTING
    # print("Target Boundary: " + weight_to_equation(target_function_weight))
    # print_data_sets(data_set_x, data_set_y)

    hypothesis_weight = [0, 0, 0]
    hypothesis_error_in = 0
    epoch = 0
    change = change_threshold + 1
    while change >= change_threshold:
        epoch += 1
        hypothesis_error_in = in_sample_error(hypothesis_weight, data_set_x, data_set_y)
        # hypothesis_gradient = error_gradient(hypothesis_weight, data_set_x, data_set_y)
        new_hypothesis_weight = stochastic_update_weight(hypothesis_weight, data_set_x, data_set_y)
        change = difference_norm(hypothesis_weight, new_hypothesis_weight)

        # print("Epoch: " + str(epoch))
        # print("Hypothesis weight: " + str(hypothesis_weight))
        # print("Hypothesis error: " + str(hypothesis_error_in))
        # print("Hypothesis gradient: " + str(hypothesis_gradient))
        # print("New hypothesis: " + str(new_hypothesis_weight))
        # print("Change: " + str(change))

        hypothesis_weight = new_hypothesis_weight

    # FINDING E OUT
    out_of_sample_x = training_data_set_x()
    out_of_sample_y = training_data_set_y(hypothesis_weight, out_of_sample_x)
    hypothesis_error_out = in_sample_error(hypothesis_weight, out_of_sample_x, out_of_sample_y)

    return [hypothesis_error_out, epoch]

number_of_training_points = 100
learning_rate = 0.01
change_threshold = 0.01

number_of_experiments = 20
sum_result = experiment()
print("Experiments conducted: 1")
for i in range(1, number_of_experiments):
    new_result = experiment()
    print("Experiments conducted: " + str(i + 1))
    for index in range(len(sum_result)):
        sum_result[index] += new_result[index]

average_result = [totals / number_of_experiments for totals in sum_result]
e_out = average_result[0]
epoch = average_result[1]

print("\nRESULTS\n")
print("Error (out-of-sample): " + str(e_out))
print("Epoch: " + str(epoch))

