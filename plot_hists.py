#!/bin/env python
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#plt.style.use('ggplot')


def make_subplots(df1, df2):
    """
    df1: buta's data
    df2: nair abraham data
    """
    ndf = df2
    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)
    plt.hist(df['z'], bins=50, histtype='step', edgecolor='black', density=True)
    # plt.xlabel('Redshift')
    # plt.ylabel('Counts')
    # plt.ylim(0,900)
    # plt.xlim(0,0.21)

    # print(np.max(df['z'])) = 0.204
    #plt.savefig('buta_zdist_python.png')
    # plt.close()

    plt.subplot(1, 2, 2)  # Right subplot
    plt.hist(ndf['zs'], bins=50, histtype='step', edgecolor='black', density=True)
    # plt.xlabel('Redshift')
    # plt.ylabel('Counts')
    plt.subplot(1, 2, 2).get_shared_x_axes().join(plt.subplot(1, 2, 1), plt.subplot(1, 2, 2))
    plt.figtext(0.5, 0.01, 'Redshift', ha='center', va='center')
    #plt.savefig('na_zdist_python.png')
    plt.figtext(0.05, 0.5, 'Normalized Counts', ha='center', va='center', rotation='vertical')
    # plt.show()
    plt.savefig('zdist_comb.png',bbox_inches='tight')
    # plt.close()
    # plt.ylim(0,900)
    # plt.xlim(0,0.21)
    # Add a common y-axis label to the left of the subplots

    # plt.close()

def make_subplots_mag(df1, df2):
    """
    df1: buta data with mag
    df2: nair abraham data
    """
    # plt.hist(df['z'], bins=50, stacked=True, edgecolor='black', color=None, histtype='step', linestyle='dashed')
    # plt.hist(ndf['zs'], bins=50, stacked=True, edgecolor='black' , color=None, histtype='step')
    # plt.ylim(0,900)
    # plt.xlim(0,0.21)
    # plt.show()
    # plt.close()
    # def subpl_1r2c(
    # fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
    g_mag = buta_mag['gmag']-buta_mag['e_gmag']

    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)
    plt.hist(ndf['g_mag'], range=(12,16), bins=30, histtype='step', edgecolor='black', density=True)# the extinction corrected g_mag column
    # plt.xlabel('Extinction corrected g-mag')
    # plt.ylabel('Counts')
    #plt.savefig('na_gmag_dist_python.png')
    # plt.show()
    #plt.close()

    plt.subplot(1, 2, 2)
    plt.hist(g_mag, range=(12,18), bins=30, histtype='step', edgecolor='black', density=True)# the extinction corrected g_mag column
    # plt.hist(g_mag, bins=30)# the extinction corrected g_mag column
    # plt.xlabel('Extinction corrected g-mag')
    # plt.ylabel('Counts')
    #plt.savefig('buta_gmag_dist_python.png')
    plt.subplot(1, 2, 2).get_shared_x_axes().join(plt.subplot(1, 2, 1), plt.subplot(1, 2, 2))
    # import sys
    # sys.exit(0)
    plt.figtext(0.5, 0.01, 'Extinction corrected g-mag', ha='center', va='center')
    plt.figtext(0.05, 0.5, 'Normalized Counts', ha='center', va='center', rotation='vertical')
    # plt.show()
    plt.savefig('gmag_dist_comb.png',bbox_inches='tight')
    # plt.close()

hdul = fits.open("data/catalogs/buta_vizier.fit")
df = Table(hdul[1].data).to_pandas()
# Topcat reads the fits file as missing data instead of 0.0 may be we should do the same in python
df['z'][df['z']==0.0]=np.nan

hdul = fits.open("data/catalogs/nair_abraham_2010_vizier.fit")
ndf = Table(hdul[1].data).to_pandas()

make_subplots(df, ndf)

buta_mag = pd.read_csv("data/catalogs/buta_with_mag.csv")

make_subplots_mag(buta_mag, ndf)

