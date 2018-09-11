import numpy as np

number_of_training_examples = 2
number_of_experiments = 10000
number_of_out_of_sample_points = 1000


def target_function(x):
    return np.sin(np.pi * x)


def squared_error_ax(hypothesis, next_hypothesis = None):
    x = np.random.uniform(-1, 1)
    if next_hypothesis is None:
        f = target_function(x)
    else:
        f = next_hypothesis * x
    g = hypothesis * x
    return ((g - f) ** 2)


def bias(hypothesis):
    total_squared_error = 0
    for _ in range(number_of_out_of_sample_points):
        total_squared_error += squared_error_ax(hypothesis)
    return total_squared_error / number_of_out_of_sample_points


def var(average_g, set_g):
    total_squared_error_of_set_g = 0
    g_index = 1
    for g in set_g:
        total_squared_error = 0
        for _ in range(number_of_out_of_sample_points):
            total_squared_error += squared_error_ax(average_g, g)
        expected_squared_error = total_squared_error / number_of_out_of_sample_points
        total_squared_error_of_set_g += expected_squared_error
        # print("Squared Error with respect to X: " + str(expected_squared_error))
        print("Number of hypotheses checked: " + str(g_index))
        g_index += 1
    return total_squared_error_of_set_g / len(set_g)

def experiment():
    def hypothesis_ax(set_of_order_pairs):
        sum_of_products = 0
        sum_of_x_squares = 0
        for order_pair in set_of_order_pairs:
            x = order_pair[0]
            y = order_pair[1]
            sum_of_products += x * y
            sum_of_x_squares += x * x
        return sum_of_products / sum_of_x_squares

    training_set = []
    for _ in range(number_of_training_examples):
        x = np.random.uniform(-1, 1)
        y = target_function(x)
        training_set.append([x, y])
        # print("(" + str(x) + ", " + str(y) + ")")
    # print("Training Set: " + str(training_set))

    a = hypothesis_ax(training_set)
    print("Final hypothesis: g(x) = " + str(a) + "x")

    result = []
    result.append(a)
    return result


sum_result = experiment()
all_g = []
for _ in range(number_of_experiments - 1):
    new_result = experiment()
    sum_result = [sum_result[index] + new_result[index] for index in range(len(sum_result))]
    all_g.append(new_result[0])

avg_result = [total / number_of_experiments for total in sum_result]
a_hat = avg_result[0]
bias = bias(a_hat)
var = var(a_hat, all_g)
expected_value_of_out_of_sample_error = bias + var
print("\nEXPERIMENT COMPLETE\n")

print("Average hypothesis: g(x) = " + str(a_hat) + "x")

print("Bias: " + str(bias))

print("Variance: " + str(var))

print("Expected out-of-sample error: " + str(expected_value_of_out_of_sample_error))