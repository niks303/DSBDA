# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 11:27:56 2022

@author: DELL
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
df= sns.load_dataset('titanic')
df.head()
cols = df.columns
cols
df.info()
df.describe()
df.isnull().sum()
sns.boxplot(df['sex'] ,df['age'])
sns.boxplot(df['sex'] ,df['age'],df['survived'])

