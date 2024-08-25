"""
This script is used to combine the prediction output with more details about each galaxy
like ra, dec, redshift, model fit parameters etc. where available
This is done by joining on the unique id like the sdss objid
"""
import pandas as pd
import os,sys

prediction_sample = "/home/linn/may/1_galaxy/galaxy-git/data/catalogs/prediction_catalog_nair_selection_DR18.csv"
sample_df = pd.read_csv(prediction_sample)
sample_df = sample_df.drop_duplicates(subset=["objid"], keep='first')
#print(sample_df.shape)
predicted  = "/home/linn/may/1_galaxy/galaxy-git/pred_output_20230925_pt2.csv"
predicted_df = pd.read_csv(predicted)
predicted_df["UID"] = predicted_df["Filename"].apply(lambda x: int(os.path.splitext(os.path.basename(x))[0]))
#print(predicted_df.shape)
merged = pd.merge(sample_df, predicted_df, left_on="objid", right_on="UID", how="inner", validate="one_to_one")
sel_cols = ['ra', 'dec', 'objid','gmag', 'deVRad_r', 'deVAB_g', 'redshift', 'Prediction', 'Label']
#print(merged[sel_cols])
#merged[sel_cols].to_csv("ring_catalog.csv", index=False)
predicted = "/home/linn/may/1_galaxy/galaxy-git/pred_output_20230925_pt2_rings+barred.csv"
predicted_df = pd.read_csv(predicted)
predicted_df["UID"] = predicted_df["Filename"].apply(lambda x: int(os.path.splitext(os.path.basename(x))[0]))
bar_catalog_path = "/home/linn/sept/Catalogue_Predicted_CNN.csv"
bar_catalog = pd.read_csv(bar_catalog_path)
#print(bar_catalog)
merged = pd.merge(bar_catalog, predicted_df, left_on="SDSS_Objid", right_on="UID", how="inner", validate="one_to_one")
sel_cols = [ 'ra', 'dec', 'SDSS_Objid', 'Prediction', 'Label' ]
#print(merged.columns)
print(merged[sel_cols])
#merged[sel_cols].to_csv("barred_rings_catalog.csv", index=False)

