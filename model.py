################################   ################################   ################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split, KFold

import re
import gc
import warnings
warnings.filterwarnings('ignore')
warnings.warn("this will not show")


pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
pd.options.display.max_colwidth = 1000
import imblearn

from imblearn.under_sampling import RandomUnderSampler



import pandas as pd
import numpy as np
import multiprocessing
import warnings
import gc
from time import time
import datetime
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score

# for modeling 
import sklearn
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report, recall_score, f1_score
from sklearn.metrics import plot_precision_recall_curve, precision_recall_curve, plot_roc_curve, roc_curve, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import datasets, metrics
from sklearn.decomposition import PCA

# to avoid warnings
import warnings
warnings.filterwarnings('ignore')
warnings.warn("this will not show")

import pickle

warnings.simplefilter('ignore')
sns.set()

################################  1. Data Importing ################################


## 
train_id = pd.read_csv(r"train_identity.csv")
train_tr = pd.read_csv(r"train_transaction.csv")
train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
del train_id, train_tr
gc.collect()


################################  2. Data Preprosessing ################################

def null_values(df, rate=0):
    """a function to show null values with percentage"""
    nv=pd.concat([df.isnull().sum(), 100 * df.isnull().sum()/df.shape[0]],axis=1).rename(columns={0:'Missing_Records', 1:'Percentage (%)'})
    return nv[nv['Percentage (%)']>rate].sort_values('Percentage (%)', ascending=False)

null_val = null_values(train)
col_with_null_val = null_val[null_val["Percentage (%)"] >= 50.0].index
train.drop(col_with_null_val, axis=1, inplace=True)

## 
for f in train.columns:
    if (train[f].dtype!='object') and (train[f].dtype!='int64') and (f!='isFraud'): 
        if (train[f].isna().sum()>0):
            mean = np.mean(train[~(train[f].isna())][f].values)  
            if mean:
                # print(f'{f: >10} mean: {mean: >10.3f}, n_train_missing = {train[f].isna().sum(): >10,}')
                train[f] = np.where(train[f].isna(), mean, train[f])


## 

null_val = null_values(train)

for t in null_val.index:
    train[t].fillna("inco", inplace=True)

import datetime
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')
us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())

for k, df in enumerate([train]):
  df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
  df['DT_M'] = ((df['DT'].dt.year-2017-k)*12 + df['DT'].dt.month).astype(np.int8).apply(lambda x: x%12 if x>12 else x)
  df['DT_W'] = ((df['DT'].dt.year-2017-k)*52 + df['DT'].dt.weekofyear).astype(np.int8).apply(lambda x: x%52 if x>52 else x)
  df['DT_D'] = ((df['DT'].dt.year-2017-k)*365 + df['DT'].dt.dayofyear).astype(np.int16).apply(lambda x: x%365 if x>365 else x)
  
  df['DT_hour'] = (df['DT'].dt.hour).astype(np.int8)
  df['DT_day_week'] = (df['DT'].dt.dayofweek).astype(np.int8)
  df['DT_day_month'] = (df['DT'].dt.day).astype(np.int8)

  # Holidays
  df['DT_holiday'] = (df['DT'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)

train = train.drop('DT',axis=1)

train['TransactionAmt'] = train['TransactionAmt'].clip(0,5000)

################################ 3. Model Building ################################

df = train
df = pd.get_dummies(df)
train , test = train_test_split(df, test_size=0.10, random_state =42)
X = train.drop(['isFraud'], axis=1)
y = train['isFraud']

xgb = XGBClassifier()
xgb.fit(X, y)

# Saving model to disk
pickle.dump(xgb, open('xgb.pkl','wb'))