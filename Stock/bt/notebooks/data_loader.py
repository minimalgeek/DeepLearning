import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pandas_datareader.data as web

DATA = '/work/data/'

def common_part(*args, fn):
    file_name = '{}_{}_{}.pkl'.format(args[0], str(args[1]), str(args[2]))
    full_path = DATA + file_name
    if os.path.isfile(full_path):
        df = pd.read_pickle(full_path)
    else:
        df = fn(*args)
        df.to_pickle(full_path)
    return df

def get_minute_data_and_cache(ticker, period, days):
    """
    ticker = symbol, like 'AAPL'
    period = the HFT data time interval in seconds\n\t(if you want to download 1 minute data use 60 for PERIOD)
    days = number of days HFT data that you want
    """
    def download_data(ticker, period, days):
        url = 'http://www.google.com/finance/getprices?q={}&i={}&p={}d&f=d,o,h,l,c,v'.format(ticker, period, days)
        raw = pd.read_csv(url,skiprows=7,header=None)
        x=np.array(raw)
        date=[]
        for i in range(0,len(x)):
            if x[i][0][0]=='a':
                t = datetime.fromtimestamp(int(x[i][0].replace('a','')))
                date.append(t)
            else:
                date.append(t+timedelta(minutes =int(x[i][0])))

        data=pd.DataFrame(x,index=date)
        data.columns=['a','Open','High','Low','Close','Volume']
        del data['a']
        return data
    return common_part(ticker, period, days, fn=download_data)

def get_daily_data_and_cache(ticker, from_date, to_date):
    def download_data(ticker, from_date, to_date):
        df = web.DataReader(ticker, 'yahoo', from_date, to_date)
        return df.dropna()
    return common_part(ticker, from_date, to_date, fn=download_data)
