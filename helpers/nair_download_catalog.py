#!/bin/env python

import pandas as pd
from astropy.io import ascii

def read_catalog(filePath):

    data = ascii.read(filePath, include_names=['JID','RAdeg','DEdeg','z','TType','Bar','Ring','FRing','RC3'])
    df = data.to_pandas()
    return df

def get_clean_nonrings(df):
    subset = df.loc[df['Ring']==0].loc[df['FRing']==0]
    return subset

if __name__=="__main__":

    filePath = "data/catalogs/apjs316193t2_mrt.txt"
    outputFile = "data/catalogs/apjs316193t2_mrt_Ring_0_FRing_0.csv"
    df = read_catalog(filePath)
    subset = get_clean_nonrings(df)
    subset.to_csv(outputFile,index=False)
