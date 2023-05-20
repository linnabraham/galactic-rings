#!/bin/env python

from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord
import pandas as pd

def read_catalog(filePath):

    # Read catalogue file using astropy
    col_starts=(0,2,4,11,13,15,17,23,53,63,70,77,89,100,114,120,124,130)
    col_ends=  (1,3,8,12,14,16,20,52,60,68,74,87,97,111,118,121,127,159)

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

    table = ascii.read(filePath,format='fixed_width_no_header', col_starts=col_starts,col_ends=col_ends,names=names)
    table = table.to_pandas()
    return table

def convCordRA(x):
    y=SkyCoord(x,unit=(u.hourangle,u.deg))
    return y.ra.degree

def convCordDEC(x):
    y=SkyCoord(x,unit=(u.hourangle,u.deg))
    return y.dec.degree

def get_ra_dec_deg(df):
    df['RA'] = df['RAh'].astype('str') + " " + df['RAM'].astype('str') + " " + df['RAs'].astype('str')+" "+df['DE-'].astype(str)+df['DEd'].astype(str)+" " + df['DEm'].astype(str)+" "+df['DEs'].astype(str)

    df['coord'] = df['RA'].astype('str')

    coord = pd.Series(df['RA'],dtype='string')


    df['RAdeg'] = df['coord'].apply(convCordRA)
    df['DECdeg'] = df['coord'].apply(convCordDEC)
    return df

if __name__=="__main__":

    filePath="data/catalogs/buta_2017_table2.dat"
    catalog = read_catalog(filePath)
    df = get_ra_dec_deg(catalog)
    df.to_csv("data/catalogs/buta_2017_with_ra_dec.csv", index=True, index_label='Index')
