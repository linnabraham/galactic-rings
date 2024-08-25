import pandas as pd
import os

selection_file = "prediction_catalog_nair_selection_DR18.csv"
pred_output_file = "pred_output_20230925_pt2_thresh90.csv"
catalogue_dest_path = "prediction_catalogue_full.csv"
selection = pd.read_csv(selection_file)
pred_output = pd.read_csv(pred_output_file)

# remove duplicate rows with objids keeping first
selection = selection.drop_duplicates(subset=['objid'], keep='first')

print("No. of rows in selection table", len(selection))
print("No. of rows in prediction output", len(pred_output))

pred_output["objid"] = pred_output.Filename.apply(lambda x: int(os.path.basename(x)[:-5]))
pred_output = pred_output[~pred_output.Filename.str.contains("control")]
df_merged = pd.merge(pred_output, selection, on="objid", validate="one_to_one", how="left")

print("No. of Rings in prediction catalogue", (df_merged.Label=="Rings").sum())
print("Total no. of galaxies in prediction catalogue", len(df_merged))
print("Saving catalogue to file", catalogue_dest_path)

df_merged.to_csv(catalogue_dest_path, index=False)
df_merged_subset = df_merged[["ra","dec","objid","gmag","deVRad_r","deVAB_g","redshift","Prediction","Label"]]
df_merged_subset.to_csv("prediction_catalogue.csv", index=False)

# save latex code for preview of catalogue subset for inclusion in paper
catalogue_preview_table = df_merged_subset[:20].to_latex(index=False)
with open('prediction_catalogue_preview.tex', 'w') as file:
        file.write(catalogue_preview_table)
