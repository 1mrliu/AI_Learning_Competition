# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df1 = pd.read_csv("../input/keras-baseline-feature-hashing-cnn/baseline_submission.csv").rename(columns={'project_is_approved': 'project_is_approved1'})
df2 = pd.read_csv("../input/keras-baseline-feature-hashing-price-tfidf/baseline_submission.csv").rename(columns={'project_is_approved': 'project_is_approved2'})
df3 = pd.read_csv("../input/extensive-data-analysis-modelling-donors-choose/XGBMarch32018.csv").rename(columns={'project_is_approved': 'project_is_approved3'})
df4 = pd.read_csv("../input/the-choice-is-yours/blend_submission.csv").rename(columns={'project_is_approved': 'project_is_approved4'})
df5 = pd.read_csv("../input/the-choice-is-yours/xgb_submission.csv").rename(columns={'project_is_approved': 'project_is_approved5'})
df6 = pd.read_csv("../input/tf-idf-and-features-logistic-regression/logistic_sub.csv").rename(columns={'project_is_approved': 'project_is_approved6'})
df7 = pd.read_csv("../input/opanichevlightgbmandtfidfstarter/submission.csv").rename(columns={'project_is_approved': 'project_is_approved7'})


df = pd.merge(df1, df2, on='id')
df = pd.merge(df, df3, on='id')
df = pd.merge(df, df4, on='id')
df = pd.merge(df, df5, on='id')
df = pd.merge(df, df6, on='id')
df = pd.merge(df, df7, on='id')


df['project_is_approved'] = (df['project_is_approved1']**3*df['project_is_approved2']**3*df['project_is_approved3']*df['project_is_approved4']*df['project_is_approved5']*df['project_is_approved6']**3*df['project_is_approved7']**6)**(1/18)

df[['id', 'project_is_approved']].to_csv("simple_average.csv", index=False)

# Any results you write to the current directory are saved as output.