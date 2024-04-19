# ./wandb/latest-run/files/model.pikcle
#
#
#
#

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load the model
model = pickle.load(open("./wandb/latest-run/files/model.pickle", "rb"))

# Get the feature importances
importances = model.feature_importances_

# Get the feature names
features = model.feature_name_

# Create a dataframe
df = pd.DataFrame({"feature": features, "importance": importances})

# Sort the dataframe
df = df.sort_values("importance", ascending=False)

df.to_csv("./feature_importance.csv", index=False)
