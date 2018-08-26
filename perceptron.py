import random


def PLA(number_of_training_points):
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

    # choosing two random points in [-1, 1] x [-1, 1]
    first_rand_point_x1 = random.uniform(-1, 1)
    first_rand_point_x2 = random.uniform(-1, 1)
    second_rand_point_x1 = random.uniform(-1, 1)
    second_rand_point_x2 = random.uniform(-1, 1)

    # finding line between the two random points and setting target function weights
    slope = slope(first_rand_point_x1, first_rand_point_x2, second_rand_point_x1, second_rand_point_x2)
    b_constant = b_constant(slope, first_rand_point_x1, first_rand_point_x2)
    target_function_weights = [-b_constant, -slope, 1]

    # choosing data set inputs randomly
    data_set_x = []

    for _ in range(number_of_training_points):
        x0 = 1
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        data_set_x.append([x0, x1, x2])

    # evaluating corresponding output
    data_set_y = []
    for input in data_set_x:
        y = target_function(input)
        data_set_y.append(y)

    # initializing PLA process
    number_of_iterations = 0
    g_function_weight = [0, 0, 0]
    misclassified_points = range(number_of_training_points)

    # iterating PLA
    while len(misclassified_points) > 0:
        number_of_iterations += 1

        # choosing a random misclassified point
        random_misclassified_point_index = misclassified_points[random.randrange(len(misclassified_points))]
        random_misclassified_point_y = data_set_y[random_misclassified_point_index]
        random_misclassified_point_x = data_set_x[random_misclassified_point_index]

        # update weights to properly classify point
        for d in range(0, len(g_function_weight)):
            g_function_weight[d] = g_function_weight[d] + random_misclassified_point_y * random_misclassified_point_x[d]

        # creating new set of misclassified points
        misclassified_points = []
        for n in range(len(data_set_x)):
            input = data_set_x[n]
            actual_y = target_function(input)
            learned_y = g_function(input)
            if (actual_y != learned_y):
                misclassified_points.append(n)

    # finding probability that f and g disagree
    number_of_random_points = 1000
    number_of_disagreements = 0
    for _ in range(number_of_random_points):
        random_point_x1 = random.uniform(-1, 1)
        random_point_x2 = random.uniform(-1, 1)
        random_input = [1, random_point_x1, random_point_x2]
        f_value = target_function(random_input)
        g_value = g_function(random_input)

        # testing purpose
        # print(f_value - g_value)

        if f_value != g_value:
            number_of_disagreements += 1
    probability_that_f_and_g_disagree = number_of_disagreements / number_of_random_points

    # return number of iterations and probability in a list
    return [number_of_iterations, probability_that_f_and_g_disagree]

number_of_runs = 1000
number_of_training_points = 100

total_number_of_iterations = 0
total_probability_that_f_and_g_disagree = 0
for _ in range(number_of_runs):
    results = PLA(number_of_training_points)
    total_number_of_iterations += results[0]
    total_probability_that_f_and_g_disagree += results[1]

average_number_of_iterations = total_number_of_iterations / number_of_runs
average_probability_that_f_and_g_disagree = total_probability_that_f_and_g_disagree / number_of_runs

print("For N = " + str(number_of_training_points))
print("Average number of iterations: " + str(average_number_of_iterations))
print("Average P[f(x) != g(x)]: " + str(average_probability_that_f_and_g_disagree))