import random

def experiment(number_of_training_points):
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

    # testing
    # testing target function
    for n in range(number_of_training_points):
        input_x = data_set_x[n]
        x1 = input_x[1]
        x2 = input_x[2]
        output_y = data_set_y[n]
        print("f(" + str(x1) + ", " + str(x2) + ")")
        print(output_y)


experiment(10)

# sample output
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
