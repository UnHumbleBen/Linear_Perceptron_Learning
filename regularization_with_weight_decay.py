import numpy as np


def file_to_data(file_name):
    set_x = []
    set_y = []
    with open(file_name, 'r') as data_file:
        for line in data_file:
            # print(line, end='')
            numbers = line.strip().split("  ")
            # print(numbers)
            # print(len(numbers))
            X = [1]
            for n in range(len(numbers) - 1):
                X.append(float(numbers[n]))
            set_x.append(X)
            set_y.append(int(float(numbers[-1])))

            # print("Input: " + str(X) + "  Output: " + str(Y))

        # data_file_content = data_file.readline()
        #
        # while len(data_file_content) > 0:
        #     # print(data_file_content, end='')
        #     data_file_numbers = map(float, data_file_content)
        #     print(data_file_numbers)
        #
        #     data_file_content = data_file.readline()
    return (set_x, set_y)


def print_data(X, Y=None):
    if Y is None:
        for n in range(len(X)):
            print("Input: " + str(X[n]))
    else:
        for n in range(len(X)):
            print("Input: " + str(X[n]) + " Output: " + str(Y[n]))


def classification_transformation(x):
    if (type(x[0]) == int):
        x1 = x[1]
        x2 = x[2]
        return [1, x1, x2, x1 ** 2, x2 ** 2, x1 * x2, abs(x1 - x2), abs(x1 + x2)]
    else:
        set_z = []
        for n in range(len(x)):
            set_z.append(classification_transformation(x[n]))
        return set_z


def linear_regression(set_x, set_y, weight_decay=None):
    number_of_features = len(set_x[0])
    print("Number of features: " + str(number_of_features))
    number_of_training_points = len(set_x)
    print("Number of training points: " + str(number_of_training_points))
    x_matrix = np.reshape(set_x, (number_of_training_points, number_of_features))
    print("Constructing X matrix: ")
    print(np.matrix(x_matrix))
    y_matrix = np.reshape(set_y, (number_of_training_points, 1))
    print("Constructing Y matrix: ")
    print(np.matrix(y_matrix))

    w_matrix = None
    if weight_decay is None:
        print("Solving W matrix: ")
        w_matrix = np.matmul(np.linalg.pinv(x_matrix), y_matrix)
    else:
        print("Transposing X matrix ")
        transposed_x_matrix = np.matrix.transpose(x_matrix)
        print(np.matrix(transposed_x_matrix))
        print("Multplying X matrix with transposed X matrix")
        x_matrix_squared = np.matmul(transposed_x_matrix, x_matrix)
        print(np.matrix(x_matrix_squared))
        print("Constructing identity matrix")
        identity_matrix = np.identity(number_of_features)
        print(np.matrix(identity_matrix))
        inverted_matrix = np.linalg.inv(x_matrix_squared + weight_decay * identity_matrix)
        print("Calculating (Z(t)Z + lambda * I)^-1")
        w_matrix = np.matmul(inverted_matrix, transposed_x_matrix)
        print(np.matrix(w_matrix))
        print("Solving W matrix")
        w_matrix = np.matmul(w_matrix, y_matrix)

    print(np.matrix(w_matrix))
    w_list = []
    for w_row in w_matrix:
        w_list.append(w_row[0])
    print("Converting W matrix into w list")
    return w_list


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


def classification(weight, input):
    return sign(dot_product(weight, input))


def classification_error(W, X, Y):
    number_of_individual_errors = 0
    data_set_size = len(Y)
    print("Data set size: " + str(data_set_size))
    for n in range(len(X)):
        print("Data input " + str(n))
        x = X[n]
        print("Inputs " + str(x))
        y = Y[n]
        print("Target output " + str(y))
        hypothesis_y = classification(W, x)
        print("Hypothesis output " + str(hypothesis_y))
        if y != hypothesis_y:
            print("Hypothesis does not match target")
            number_of_individual_errors += 1
        print("Number of errors: " + str(number_of_individual_errors))

    print("Total number of errors: " + str(number_of_individual_errors))
    # print(number_of_individual_errors)
    return number_of_individual_errors / data_set_size

k = 0
weight_decay = None

in_data_file_name = 'in_data.txt'
out_data_file_name = 'out_data.txt'
(data_set_x, data_set_y) = file_to_data(in_data_file_name)
(out_data_set_x, out_data_set_y) = file_to_data(out_data_file_name)
print("\nConverting " + in_data_file_name + " to data set\n")
print_data(data_set_x, data_set_y)
print("\nConverting " + out_data_file_name + " to data set\n")
print_data(out_data_set_x, out_data_set_y)

data_set_z = classification_transformation(data_set_x)
out_data_set_z = classification_transformation(out_data_set_x)
print("\nTransforming in_data set using nonlinear transformation\n")
print_data(data_set_x, data_set_z)
print("\nTransforming out_data set using nonlinear transformation\n")
print_data(out_data_set_x, out_data_set_z)

print("\nExecuting linear regression\n")
hypothesis_weight = linear_regression(data_set_z, data_set_y, weight_decay)
print("Hypothesis weight: " + str(hypothesis_weight) + "\n")

hypothesis_in_sample_error = classification_error(hypothesis_weight, data_set_z, data_set_y)
print("In-sample classification error: " + str(hypothesis_in_sample_error) + "\n")
hypothesis_out_sample_error = classification_error(hypothesis_weight, out_data_set_z, out_data_set_y)
print("Out-of-sample classification error: " + str(hypothesis_out_sample_error))
euclidean_distance_of_error = np.sqrt(hypothesis_in_sample_error ** 2 + hypothesis_out_sample_error ** 2)
print("Euclidean distance of in-sample and out-of-sample classification errors: " + str(euclidean_distance_of_error))

print("\nResults: \n")
print("Weight decay factor: " + str(weight_decay))
print("In-sample classification error: " + str(hypothesis_in_sample_error))
print("Out-of-sample classification error: " + str(hypothesis_out_sample_error))
