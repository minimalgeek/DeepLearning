class Action:
    BUY = 'buy'
    SELL = 'sell'
    CLOSE = 'close'
    HOLD = 'hold'

    acts = [BUY, SELL, CLOSE, HOLD]

    def __init__(self, ticker, act):
        self.ticker = ticker
        self.act = act

    def __repr__(self):
        return '{} - {}'.format(self.ticker, self.act)

    def __str__(self):
        return self.__repr__()
