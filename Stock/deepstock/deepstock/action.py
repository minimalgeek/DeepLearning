class Action:

    BUY = 'buy'
    SELL = 'sell'
    SKIP = 'skip'

    acts = [BUY, SELL, SKIP]

    def __init__(self, ticker, act, days, percentage):
        self.ticker = ticker
        self.act = act
        self.days = days
        self.percentage = percentage
