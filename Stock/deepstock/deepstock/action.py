class Action:

    BUY = 'buy'
    SELL = 'sell'
    SKIP = 'skip'

    acts = [BUY, SELL]

    def __init__(self, ticker, act, days, percentage):
        self.ticker = ticker
        self.act = act
        self.days = days
        self.percentage = percentage

    def __str__(self):
        return '{} - {} - {}'.format(self.ticker, self.act, self.days)

    def __repr__(self):
        return str(self)