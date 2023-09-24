#!/bin/env python
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
"""
Script for computing the logloss value using the predicted probabilty for 
each test example and its actual label.
The distribution of this logloss can be then plotted as a histogram or 
the examples that have the highest logloss values can be examined for better
understanding of the learning
"""
filename = sys.argv[1]
threshold = float(sys.argv[2])
df = pd.read_csv(filename)

df['logloss'] = -((df['Ground_Truth']* np.log(df['Prediction'])) \
        + (( 1 - df['Ground_Truth']) * np.log( 1 - df['Prediction'])))

# print(df.sort_values('logloss' ,ascending=False)[df.Ground_Truth==1])

fig, axes = plt.subplots(1, 2, figsize=(12,4))

# correct predictions = TP + TN
filt1 = df['logloss'][((df.Ground_Truth==1) & (df.Prediction > threshold)) | \
        ((df.Ground_Truth==0) & (df.Prediction < threshold))]
# incorrect predictions = FP + FN
filt2 = df['logloss'][((df.Ground_Truth==1) & (df.Prediction < threshold)) | \
        ((df.Ground_Truth==0) & (df.Prediction > threshold))]
axes[0].hist(filt1, bins = 30, range=(0,.1))
axes[1].hist(filt2, bins = 30, color='orange' )
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(12,4))
tp = df['logloss'][(df.Ground_Truth==1) & (df.Prediction > threshold)] 
fn = df['logloss'][(df.Ground_Truth==1) & (df.Prediction < threshold)]
tn = df['logloss'][(df.Ground_Truth==0) & (df.Prediction < threshold)]
fp = df['logloss'][(df.Ground_Truth==0) & (df.Prediction > threshold)]

axes[0].hist(tn, bins=20, density=True, label='True Negatives')
axes[1].hist(fn, bins=20, color='brown', density=True, label='False Negatives')
axes[1].hist(tp, bins=20, label='True Positives')
axes[1].hist(fp, bins=20, color='orange', label='False Positives')
axes[0].set_xlabel("Logloss")
axes[1].set_xlabel("Logloss")
axes[0].legend()
axes[1].legend()
fig.suptitle('Distribution of Logloss in our Test Set', fontsize=13)
plt.savefig("hist_logloss_test.png", bbox_inches="tight")
plt.show()

print(df.loc[fn.index])
