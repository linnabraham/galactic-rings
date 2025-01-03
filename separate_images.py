#!/bin/env python
"""
Script to read filepaths from prediction file and separate into different
directories based on the confusion matrix
"""
import sys
import csv
import re
import os
import shutil

def identify_label(filename):
    """
    assert identify_label("data/E6/NonRings/J110800.98+072632.21.jpeg") == "NonRings"
    assert identify_label("data/E6/Rings/94.jpeg") == "Rings"
    """
    if re.search(r'/NonRings/', filename):
        return "NonRings"
    elif re.search(r'/Rings/', filename):
        return "Rings"
    else:
        return "Unknown"

def main(predictions):

    dest = "separated"

    tp_list = []
    fp_list = []
    tn_list = []
    fn_list = []

    with open(predictions, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            prediction = float(row['Prediction'])
            label = row['Label']
            filename = row['Filename']

            # Determine TP, FP, TN, FN based on conditions
            if identify_label(filename) == "Rings"  and label == 'Rings':
                tp_list.append(row['Filename'])
            elif identify_label(filename) == "NonRings" and label == 'Rings':
                fp_list.append(row['Filename'])
            elif identify_label(filename) == "NonRings" and label == 'NonRings':
                tn_list.append(row['Filename'])
            elif identify_label(filename) == "Rings" and label == 'NonRings':
                fn_list.append(row['Filename'])

    # Print the results
    #print("True Positives:", tp_list)
    #print("False Positives:", fp_list)
    #print("True Negatives:", tn_list)
    #print("False Negatives:", fn_list)

    print(len(tp_list))
    print(len(fp_list))
    print(len(tn_list))
    print(len(fn_list))

    if not os.path.exists(dest):
        os.mkdir(dest)
        os.makedirs(os.path.join(dest,"TP"))
        os.makedirs(os.path.join(dest,"FP"))
        os.makedirs(os.path.join(dest,"TN"))
        os.makedirs(os.path.join(dest,"FN"))
    else:
        raise ValueError(f"{dest} already exists..exiting")
        sys.exit(0)

    for imglist,folder in zip([tp_list, fp_list, tn_list, fn_list], ["TP","FP","TN","FN"]):
        for imgpath in imglist:
            shutil.copy2(imgpath, os.path.join(dest,folder))

if __name__=="__main__":
    predictions_file_path=sys.argv[1]
    main(predictions_file_path)
