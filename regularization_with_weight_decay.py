import string

def file_to_data(file_name):
    set_x = []
    set_y = []
    with open(file_name, 'r') as data_file:
        for line in data_file:
            # print(line, end='')
            numbers = line.strip().split("  ")
            # print(numbers)
            # print(len(numbers))
            X = []
            for n in range(len(numbers) - 1):
                X.append(float(numbers[n]))
            set_x.append(X)
            set_y.append(float(numbers[-1]))

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


def print_data(X, Y):
    for n in range(len(X)):
        print("Input: " + str(X[n]) + " Output: " + str(Y[n]))


in_data_file_name = 'in_data.txt'
(data_set_x, data_set_y) = file_to_data(in_data_file_name)
print_data(data_set_x, data_set_y)


