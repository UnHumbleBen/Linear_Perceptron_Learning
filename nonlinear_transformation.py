import random
import numpy


def experiment(number_of_training_points, noise_probability, number_of_out_of_sample_points):
    def sign(n):
        if n == 0:
            return 0
        elif n > 0:
            return 1
        else:
            return -1

    def dot_product(W, X):
        total = 0
        for index in range(0, len(W)):
            total += W[index] * X[index]
        return total

    def target_function(inputs):
        x1 = inputs[1]
        x2 = inputs[2]
        return sign(x1 ** 2 + x2 ** 2 - 0.6)

    def linear_regression(input, output):
        number_of_features = len(input[0])
        x_matrix = numpy.reshape(input, (number_of_training_points, number_of_features))
        y_matrix = numpy.reshape(output, (number_of_training_points, 1))
        w_matrix = numpy.matmul(numpy.linalg.pinv(x_matrix), y_matrix)
        w_list = []
        for w_row in w_matrix:
            w_list.append(w_row[0])
        return w_list

    def g_function(inputs):
        return sign(dot_product(g_function_weights, inputs))

    def g_misclassified_points(input_set=None, output_set=None):
        if input_set is None:
            input_set = data_set_x
        if output_set is None:
            output_set = data_set_y
        misclassified_points = []
        for n in range(len(input_set)):
            input = input_set[n]
            actual_y = output_set[n]
            learned_y = g_function(input)
            if (actual_y != learned_y):
                misclassified_points.append(n)
        return misclassified_points

    def in_sample_error():
        return len(g_misclassified_points()) / number_of_training_points

    def out_of_sample_error(number_of_out_of_sample_points):
        out_of_sample_x = []
        for _ in range(number_of_out_of_sample_points):
            x0 = 1
            x1 = random.uniform(-1, 1)
            x2 = random.uniform(-1, 1)
            out_of_sample_x.append([x0, x1, x2])
        out_of_sample_z = []
        for input in out_of_sample_x:
            out_of_sample_z.append(nonlinear_transform(input))

        out_of_sample_y = []
        for input in out_of_sample_x:
            y = target_function(input)
            out_of_sample_y.append(y)

        number_of_noise_outputs = int(number_of_out_of_sample_points * noise_probability)
        noise_indices = random.sample(range(number_of_out_of_sample_points), number_of_noise_outputs)
        for n in noise_indices:
            out_of_sample_y[n] = -out_of_sample_y[n]

        return len(g_misclassified_points(out_of_sample_z, out_of_sample_y)) / number_of_training_points



    def nonlinear_transform(inputs):
        x1 = inputs[1]
        x2 = inputs[2]
        return [1, x1, x2, x1 * x2, x1 ** 2, x2 ** 2]

    # initialize list for result
    result = []

    # generating a training set
    data_set_x = []
    for _ in range(number_of_training_points):
        x0 = 1
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        data_set_x.append([x0, x1, x2])

    # evaluating output of training set
    data_set_y = []
    for input in data_set_x:
        y = target_function(input)
        data_set_y.append(y)

    # TESTING SETUP
    # for testing noise generation
    # no_noise_data_set_y = data_set_y.copy()

    # generating noise
    number_of_noise_outputs = int(number_of_training_points * noise_probability)
    noise_indices = random.sample(range(number_of_training_points), number_of_noise_outputs)
    for n in noise_indices:
        data_set_y[n] = -data_set_y[n]

    # linear regression
    g_function_weights = linear_regression(data_set_x, data_set_y)

    # in-sample error
    result.append(in_sample_error())

    # transform data set
    data_set_z = []
    for input in data_set_x:
        data_set_z.append(nonlinear_transform(input))

    # linear regression with transformed data set
    g_function_weights = linear_regression(data_set_z, data_set_y)
    for weight in g_function_weights:
        result.append(weight)

    result.append(out_of_sample_error(number_of_out_of_sample_points))
    # TESTING
    # testing tilde w
    # print(g_function_weights)

    # testing nonlinear transformation
    # for n in range(len(data_set_x)):
    #     x1 = data_set_x[n][1]
    #     x2 = data_set_x[n][2]
    #     print("(" + str(x1) + ", " + str(x2) + ")")
    #     print(data_set_x[n])
    #     print(data_set_z[n])

    # testing weight list

    # testing noise generation
    # number_of_noise_outputs = 0
    # for n in range(number_of_training_points):
    #     if no_noise_data_set_y[n] != data_set_y[n]:
    #         number_of_noise_outputs += 1
    # print(number_of_noise_outputs)

    # testing target function
    # for n in range(number_of_training_points):
    #     input_x = data_set_x[n]
    #     x1 = input_x[1]
    #     x2 = input_x[2]
    #     output_y = data_set_y[n]
    #     print("f(" + str(x1) + ", " + str(x2) + ")")
    #     print(output_y)

    return result


N = 1000
p_noise = 0.1
num_out_sample = 1000

number_of_experiments = 1000
sum_result = experiment(N, p_noise, num_out_sample)
for _ in range(1, number_of_experiments):
    new_result = experiment(N, p_noise, num_out_sample)
    for index in range(len(sum_result)):
        sum_result[index] += new_result[index]
average_result = [totals / number_of_experiments for totals in sum_result]
E_in = average_result[0]
x0 = average_result[1]
x1 = average_result[2]
x2 = average_result[3]
x3 = average_result[4]
x4 = average_result[5]
x5 = average_result[6]
E_out = average_result[7]
print(average_result)
print("average in-sample error (before transformation): " + str(E_in))
print("g(x_1, x_2) = sign(" + str(x0) + " + " + str(x1) + "x_1 + " + str(x2) + "x_2 + " + str(x3) + "x_1*x_2 + " + str(x4) + "x_1^2 + " + str(x5) + "x_2^2)")
print("average out-of-sample error (after transformation): " + str(E_out))
# sample output 1
# f(-0.8694395529642278, -0.1664175742706442)
# 1
# f(0.7184201665265715, 0.5037062830110837)
# 1
# f(0.8998257214991578, 0.19089764186785563)
# 1
# f(0.9281877927974556, -0.4721470937893555)
# 1
# f(-0.5508965554212537, -0.7803933106986078)
# 1
# f(0.44439436438448165, -0.48781132220635715)
# -1
# f(-0.8626624858191068, -0.5022334690002903)
# 1
# f(-0.38993190739636185, -0.479471186343283)
# -1
# f(-0.6975183897071731, -0.9616393689938105)
# 1
# f(-0.7824862296941602, -0.1480020596679894)
# 1

# sample output 2
# f(-0.7759695051609068, -0.8488745841873364)
# 1
# f(0.9064112929630144, 0.006264398747369393)
# 1
# f(-0.13867919721165434, -0.2920153864596131)
# -1
# f(0.5927076976364465, -0.7927348556360576)
# 1
# f(0.10494636002483637, 0.43709721397210544)
# -1
# f(-0.433573630208141, 0.5391360595600982)
# -1
# f(0.7990405901805464, -0.7103951483703881)
# 1
# f(0.4782995698696755, 0.06010779683371914)
# -1
# f(-0.9408550412969336, 0.6538986866096861)
# 1
# f(-0.5882530357927014, 0.5950136653075475)
# 1
