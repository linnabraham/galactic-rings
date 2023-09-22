import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve,
                              PrecisionRecallDisplay)

import matplotlib
#matplotlib.use("Agg")
#Filename,Prediction,Label

if __name__=="__main__":

    filename = sys.argv[1]
    gt_colname = sys.argv[2]
    prob_colname = sys.argv[3]
    df = pd.read_csv(filename)
    y_test = df[gt_colname]
    predictions = df[prob_colname]
    probabilities = predictions.apply(lambda x: float(x.strip('[]')))
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.savefig('plot.png')
    plt.show()
