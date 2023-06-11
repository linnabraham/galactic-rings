from astropy.io import ascii
import pandas as pd
import numpy as np

def read_Buta(datafile):
    # Read catalogue file using astropy
    col_starts=(0,2,4,11,13,15,17,23,54,63,70,77,89,100,114,120,124,130)
    col_ends=  (1,3,8,12,14,16,20,53,60,68,74,87,97,111,118,121,127,159)
    names = ["RAh",
    "RAM",
    "RAs",
    "DE-",
    "DEd",
    "DEm",
    "DEs",
    "Names",
    "PGCName",
    "FILEN",
    "z",
    "OV",
    "Fam",
    "IV",
    "ST",
    "T",
    "F",
    "CVRHS"
    ]
    tab = ascii.read(datafile,format='fixed_width_no_header', col_starts=col_starts,col_ends=col_ends,names=names)

    butadata = tab.to_pandas()
    return tab

def get_bin_edges(feature, num_bins=10):
    bin_edges = np.linspace(min(feature), max(feature), num_bins + 1)
    return bin_edges

def subsample_df(df, colname, nsamples=100):
    dfs = []
    feature = df[colname]

    bin_edges = get_bin_edges(feature,num_bins=10)
    
    # Assign data points to bins using np.digitize
    indices = np.digitize(feature, bin_edges)
    
    # iterate over the actual bins
    for binplace in range(1,len(bin_edges)):
        subset = df.loc[np.where(indices==binplace)[0]]

        # compute number of elements in each bin in the subsample
        targ_count = int(len(subset)/len(feature)*nsamples)
        dfs.append(subset.sample(targ_count))
    return pd.concat(dfs)

