import random


class Coin:
    def __init__(self, number_of_flips=None):
        self.coin_log = []
        if number_of_flips != None:
            self.flip(number_of_flips)

    def flip(self, number_of_flips=None):
        if number_of_flips is None:
            number_of_flips = 1

        for _ in range(number_of_flips):
            flip = random.randint(0, 1)
            self.coin_log.append(flip)

    def frequency_of_head(self):
        number_of_heads = 0
        for flip in self.coin_log:
            if flip == 1:
                number_of_heads += 1
        return number_of_heads / len(self.coin_log)


def experiment(number_of_coins, number_of_flips):
    c_1 = Coin(number_of_flips)
    c_min = c_1
    coin_list = [c_1]

    for _ in range(number_of_coins - 1):
        new_coin = Coin(number_of_flips)
        coin_list.append(new_coin)
        if new_coin.frequency_of_head() < c_min.frequency_of_head():
            c_min = new_coin

    c_rand = coin_list[random.randrange(number_of_coins)]

    return [c_1.frequency_of_head(), c_rand.frequency_of_head(), c_min.frequency_of_head()]


number_of_coins = 1000
number_of_flips = 10
number_of_experiments = 100
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
print("v_min  %.5f" % (v_min_sum / number_of_experiments))
