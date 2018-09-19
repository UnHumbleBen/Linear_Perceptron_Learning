import numpy as np


def file_to_data(file_name):
    set_x = []
    set_y = []
    with open(file_name, 'r') as data_file:
        for line in data_file:
            numbers = line.strip().split("  ")
            X = [1]
            for n in range(len(numbers) - 1):
                X.append(float(numbers[n]))
            set_x.append(X)
            set_y.append(int(float(numbers[-1])))
    return (set_x, set_y)


def print_data(X, Y=None):
    if Y is None:
        for n in range(len(X)):
            print("Input: " + str(X[n]))
    else:
        for n in range(len(X)):
            print("Input: " + str(X[n]) + " Output: " + str(Y[n]))


def classification_transformation(x, k):
    if (type(x[0]) == int):
        x1 = x[1]
        x2 = x[2]
        transformation = [1, x1, x2, x1 ** 2, x2 ** 2, x1 * x2, abs(x1 - x2), abs(x1 + x2)]
        return transformation[:k + 1]
    else:
        set_z = []
        for n in range(len(x)):
            set_z.append(classification_transformation(x[n], k))
        return set_z


def linear_regression(set_x, set_y):
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
    print("Solving W matrix: ")
    w_matrix = np.matmul(np.linalg.pinv(x_matrix), y_matrix)
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
    return number_of_individual_errors / data_set_size


def training(X, Y, k):
    Z = classification_transformation(X, k)
    return linear_regression(Z, Y)


def validation(W, X, Y, k):
    Z = classification_transformation(X, k)
    return classification_error(W, Z, Y)

in_data_file_name = 'in_data.txt'
out_data_file_name = 'out_data.txt'
number_of_training_points = 25
models = [3, 4, 5, 6, 7]

# reading in data files
print("\nConverting " + in_data_file_name + " to data set\n")
(data_set_x, data_set_y) = file_to_data(in_data_file_name)
print_data(data_set_x, data_set_y)
print("\nConverting " + out_data_file_name + " to data set\n")
(out_data_set_x, out_data_set_y) = file_to_data(out_data_file_name)
print_data(out_data_set_x, out_data_set_y)

# splitting data to training and validation sets
print("\nSplitting " + in_data_file_name + " into training and validation\n")
training_set_x = data_set_x[number_of_training_points:]
training_set_y = data_set_y[number_of_training_points:]
print("Training set: ")
print_data(training_set_x, training_set_y)
print("Training size: " + str(len(training_set_y)))
validation_set_x = data_set_x[:number_of_training_points]
validation_set_y = data_set_y[:number_of_training_points]
print("Validation set: ")
print_data(validation_set_x, validation_set_y)
print("Validation size: " + str(len(validation_set_y)))

# training the models
print("\nTraining for k = " + str(models))
model_hypotheses = []
for model_index in models:
    print("Calculating weights for k = " + str(model_index))
    this_model_weight = training(training_set_x, training_set_y, model_index)
    model_hypotheses.append(this_model_weight)
    print("Weights: " + str(this_model_weight))
print("Finished calculating weights for all models")
print("Final weights: ")
print_data(models, model_hypotheses)

# calculate classification errors in model using validation set
print("\nCalculating errors using validation set\n")
validation_model_errors = []
for index in range(len(models)):
    this_k = models[index]
    print("Calculating error for k = " + str(this_k))
    this_hypothesis = model_hypotheses[index]
    print("Hypothesis weight = " + str(this_hypothesis))
    this_error = validation(this_hypothesis, validation_set_x, validation_set_y, this_k)
    print("Classification error: " + str(this_error))
    validation_model_errors.append(this_error)
print("Finished calculating errors for all models")
print("Final errors: ")
print_data(models, validation_model_errors)

# calculating classification errors in models using out data
print("\nCalculating errors using " + str(out_data_file_name) + "\n")
out_model_errors = []
for index in range(len(models)):
    this_k = models[index]
    print("Calculating error for k = " + str(this_k))
    this_hypothesis = model_hypotheses[index]
    print("Hypothesis weight = " + str(this_hypothesis))
    this_error = validation(this_hypothesis, out_data_set_x, out_data_set_y, this_k)
    print("Classification error: " + str(this_error))
    out_model_errors.append(this_error)
print("Finished calculating errors for all models")
print("Final errors: ")
print_data(models, out_model_errors)

# compare validation and out of sample error
print("\nComparing validation and out-of-sample error\n")
print("Validation error on left, out-of-sample error on right")
print_data(validation_model_errors, out_model_errors)