import numpy as np
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


class Collector:
    def __init__(self, 
                 start_date, 
                 end_date,
                 ticker):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = ticker

    def collect(self):
        pass
