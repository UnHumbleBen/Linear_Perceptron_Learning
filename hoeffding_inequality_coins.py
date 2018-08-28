import random


class Coin:
    def __init__(self):
        self.coin_log = []

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


number_of_coins = 1000
number_of_flips = 10
coin_list = []
for _ in range(number_of_coins):
    new_coin = Coin()
    new_coin.flip(number_of_flips)
    coin_list.append(new_coin)
    