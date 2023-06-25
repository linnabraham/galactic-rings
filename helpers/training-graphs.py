import json
import matplotlib.pyplot as plt
import sys

jsonfile=sys.argv[1]
# Load the history data from the JSON file
with open(jsonfile, 'r') as file:
    history = json.load(file)

# Extract the training and validation metrics from the history data
train_loss = history['loss']
val_loss = history['val_loss']
train_auc = history['auc_pr']
val_auc = history['val_auc_pr']

# Plot the training and validation loss
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Plot the training and validation accuracy
plt.plot(train_auc, label='Training AUC_PR')
plt.plot(val_auc, label='Validation AUC_PR')
plt.xlabel('Epoch')
plt.ylabel('AUC_PR')
plt.legend()
plt.title('Training and Validation AUC_PR')
plt.show()

