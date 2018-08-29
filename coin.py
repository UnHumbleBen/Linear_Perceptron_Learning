import numpy as np
from random import randrange


def flip(number_of_flips):
    return np.count_nonzero(np.random.randint(2, size=number_of_flips)) / number_of_flips


def experiment(number_of_coins, number_of_flips):
    v_1 = flip(number_of_flips)
    random_index = randrange(number_of_coins)
    v_rand = v_1
    v_min = v_1

    for i in range(1, random_index):
        v_new = flip(number_of_flips)
        if v_new < v_min:
            v_min = v_new

    if random_index != 0:
        v_rand = flip(number_of_flips)

    for i in range(random_index + 1, number_of_coins):
        v_new = flip(number_of_flips)
        if v_new < v_min:
            v_min = v_new

    return [v_1, v_rand, v_min]


number_of_coins = 1000
number_of_flips = 10
number_of_experiments = 100000
v_1_sum = 0
v_rand_sum = 0
v_min_sum = 0

for _ in range(number_of_experiments):
    result = experiment(number_of_coins, number_of_flips)
    v_1_sum += result[0]
    v_rand_sum += result[1]
    v_min_sum += result[2]

print("v1:    " + str(v_1_sum / number_of_experiments))
print("v_rand " + str(v_rand_sum / number_of_experiments))
print("v_min  " + str(v_min_sum / number_of_experiments))
