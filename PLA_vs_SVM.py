import numpy as np


def point_to_str(x1, x2):
    return "(" + str(x1) + ", " + str(x2) + ")"


def slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


def b_constant(m, x1, x2):
    return x2 - m * x1


def create_target_function():
    print("Taking number of dimensions  d = " + str(number_of_dimensions))
    print("Choosing a random line as target function  f")
    print("Taking two randomly distributed points on input space  X")
    print(input_space_str)
    first_rand_point_x1 = np.random.uniform(domain_min, domain_max)
    first_rand_point_x2 = np.random.uniform(domain_min, domain_max)
    second_rand_point_x1 = np.random.uniform(domain_min, domain_max)
    second_rand_point_x2 = np.random.uniform(domain_min, domain_max)
    first_rand_point_str = point_to_str(first_rand_point_x1, first_rand_point_x2)
    second_rand_point_str = point_to_str(second_rand_point_x1, second_rand_point_x2)
    print("First point: " + str(first_rand_point_str))
    print("Second point: " + str(second_rand_point_str))
    rand_slope = slope(first_rand_point_x1, first_rand_point_x2, second_rand_point_x1, second_rand_point_x2)
    print("Slope between the points: " + str(rand_slope))
    rand_b_constant = b_constant(rand_slope, first_rand_point_x1, first_rand_point_x2)
    print("Y intercept of the line: " + str(rand_b_constant))
    print("Equation of the line connecting the two points: y = " + str(rand_slope) + "x + " + str(rand_b_constant))
    print("Converting equation into weights for target function")
    rand_weight = [-rand_b_constant, -rand_slope, 1]
    print("Target function (weights): " + str(rand_weight))
    return rand_weight


def choose_inputs(number_of_points=None):
    if number_of_points is None:
        number_of_points = number_of_training_points

    set = []
    print("Generating " + str(number_of_points) + " random points")
    for n in range(number_of_points):
        x0 = 1
        x1 = np.random.uniform(domain_min, domain_max)
        x2 = np.random.uniform(domain_min, domain_max)
        x_str = point_to_str(x1, x2)
        print("X_" + str(n) + " = " + x_str)
        print("Converting point into input matrix")
        x_list = [x0, x1, x2]
        print("X_" + str(n) + " = " + str(x_list))
        set.append(x_list)
    print("Inputs: " + str(set))
    return set


def dot_product(W, X):
    total = 0
    for index in range(0, len(W)):
        total += W[index] * X[index]
    return total


def classification(W, X):
    value = dot_product(W, X)
    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0


def choose_outputs(weight, set_x):
    set_y = []
    print("Generating output")
    for input in set_x:
        print("For x = " + str(input))
        value = classification(weight, input)
        print("y = " + str(value))
        set_y.append(value)
    print("Outputs: " + str(set_y))
    return set_y


def checkEqual(lst):
    return lst[1:] == lst[:-1]


def print_data(X, Y=None):
    if Y is None:
        for n in range(len(X)):
            print("Input: " + str(X[n]))
    else:
        for n in range(len(X)):
            print("Input: " + str(X[n]) + " Output: " + str(Y[n]))


def choose_inputs_and_outputs(weight):
    inputs = choose_inputs()
    outputs = choose_outputs(weight, inputs)
    is_same_side = checkEqual(outputs)
    print("Are all the points on the same side?: " + str(is_same_side))
    while is_same_side:
        print("All the points are on the same side, creating new data set")
        inputs = choose_inputs()
        outputs = choose_outputs(weight, inputs)
        is_same_side = checkEqual(outputs)
        print("Are all the points on the same side?: " + str(is_same_side))
    print("There are points on both sides of the line, finalizing data set")
    print_data(inputs, outputs)
    return (inputs, outputs)


def list_of_misclassified_points(W, X, Y):
    misclassified_points = []
    print("Creating list of misclassified points")
    for n in range(len(Y)):
        str_n = str(n)
        input = X[n]
        target_y = Y[n]
        learned_y = classification(W, input)
        print("x_" + str_n + " = " + str(input))
        print("y_" + str_n + " = " + str(target_y))
        print("h_" + str_n + " = " + str(learned_y))
        is_correct = target_y == learned_y
        print("Hypothesis matches data: " + str(is_correct))
        if not is_correct:
            print("Adding data index " + str(n) + " to list of misclassified points")
            misclassified_points.append(n)
    print("Final list of misclassified points: " + str(misclassified_points))
    return misclassified_points


def PLA(X, Y):
    pla_weight = []
    for _ in range(number_of_dimensions + 1):
        pla_weight.append(0)
    print("Initial weight = " + str(pla_weight))
    misclassified_points = list_of_misclassified_points(pla_weight, X, Y)
    while len(misclassified_points) > 0:
        rand_misclassified_point_index = misclassified_points[np.random.randint(0, len(misclassified_points))]
        rand_misclassified_point_x = X[rand_misclassified_point_index]
        rand_misclassified_point_y = Y[rand_misclassified_point_index]
        print("Randomly selecting a misclassified point: n = " + str(rand_misclassified_point_index))
        print("X_" + str(rand_misclassified_point_index) + " = " + str(rand_misclassified_point_x))
        print("Y_" + str(rand_misclassified_point_index) + " = " + str(rand_misclassified_point_y))

        print("Updating weight to classify the misclassified point")
        for i in range(number_of_dimensions + 1):
            print("(Before) W_" + str(i) + " = " + str(pla_weight[i]))
            pla_weight[i] += rand_misclassified_point_y * rand_misclassified_point_x[i]
            print("(After)  W_" + str(i) + " = " + str(pla_weight[i]))
        print("Updated weights = " + str(pla_weight))

        misclassified_points = list_of_misclassified_points(pla_weight, X, Y)

    print("Final weights after PLA = " + str(pla_weight))
    return pla_weight


def differences(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i != j for i, j in zip(a, b))


def probability_of_disagreement(f, g):
    print("Generating a random out of sample set of " + str(number_of_out_of_sample_points) + " points")
    out_x = choose_inputs(number_of_out_of_sample_points)
    print("Calculating target outputs")
    f_x = choose_outputs(f, out_x)
    print("Calculating PLA outputs")
    g_x = choose_outputs(g, out_x)
    print("Target ouputs on the left, PLA outputs on the right")
    print_data(f_x, g_x)
    number_of_disagreements = differences(f_x, g_x)
    print("Number of disagreements: " + str(number_of_disagreements))
    measure_of_disagreement = number_of_disagreements / number_of_out_of_sample_points
    print("Estimate disagreement probability = " + str(measure_of_disagreement))
    return measure_of_disagreement


number_of_dimensions = 2
domain_min = -1
domain_max = 1
domain_str = "[" + str(domain_min) + ", " + str(domain_max) + "]"
input_space_str = "X  =  "
for _ in range(number_of_dimensions - 1):
    input_space_str += domain_str + " x "
input_space_str += domain_str
number_of_training_points = 10
number_of_out_of_sample_points = 1000

print("\nCreating target function  f  and data set  D\n")
target_function_weight = create_target_function()

print("\nChoosing inputs  X_n  as random points in  " + input_space_str + "\n")
(training_set_x, training_set_y) = choose_inputs_and_outputs(target_function_weight)

print("\nRunning PLA to find final hypothesis  g_pla\n")
g_pla = PLA(training_set_x, training_set_y)

print("\nMeasuring disagreement between f and g_pla\n")
P_pla = probability_of_disagreement(target_function_weight, g_pla)