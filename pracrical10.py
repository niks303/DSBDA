# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 00:23:27 2022

@author: DELL
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
%matplotlib inline
from sklearn import datasets
data = datasets.load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df.columns = ['col1','col2', 'col3','col4']

df
df['col5'] = data.target

df
df.info()
np.unique(df['col5'])
fig, axes = plt.subplots(2,2, figsize=(16,8))
axes[0,0].hist(df.col1)
axes[0,0].set_title("col1")

axes[0,1].hist(df.col2)
axes[0,1].set_title("col1")

axes[1,0].hist(df.col3)
axes[1,0].set_title("col1")

axes[1,1].hist(df.col4)
axes[1,1].set_title("col1")
# sns.set_style('whitegrid')
# fig, ax = plt.subplots()
col = ['col1','col2', 'col3','col4']
df.boxplot(col)
df.describe()
