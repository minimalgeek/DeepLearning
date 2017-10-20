import numpy as np
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report