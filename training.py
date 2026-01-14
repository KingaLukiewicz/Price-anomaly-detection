import pandas as pd
import numpy as np
from models import BaseModel, AdvancedModel, compute_ground_truth, evaluate

train_df = pd.read_csv("data/data.csv")
val_df = pd.read_csv("data/val_data.csv")
test_df = pd.read_csv("data/test_data.csv")

feature_cols = [col for col in train_df.columns if col not in ["price",
                                                               "log_price"]]

X_train = train_df[feature_cols]
X_val = val_df[feature_cols]
X_test = test_df[feature_cols]

X_train_array = X_train.to_numpy()
X_val_array = X_val.to_numpy()
X_test_array = X_test.to_numpy()

y_train = compute_ground_truth(train_df)
y_val = compute_ground_truth(val_df)
y_test = compute_ground_truth(test_df)

print("Tuning and training BaseModel")
base_model = BaseModel()
base_model.tune(X_train_array, y_train, X_val_array, y_val,
                cluster_values=[3, 5, 7], lof_neighbors=[10, 20, 30])

y_pred_base = base_model.predict(X_test_array)
metrics_base = evaluate(y_test, y_pred_base)
print("BaseModel metrics:", metrics_base)

print("Tuning and training AdvancedModel")
adv_model = AdvancedModel()
adv_model.tune(X_train_array, y_train, X_val_array, y_val,
               cluster_sizes=[5, 10, 15], contamination_values=[0.03, 0.05, 0.07])

y_pred_adv = adv_model.predict(X_test_array)
metrics_adv = evaluate(y_test, y_pred_adv)
print("AdvancedModel metrics:", metrics_adv)

base_model.save("models/base_model.pkl")
adv_model.save("models/advanced_model.pkl")
print("Modele zapisane: base_model.pkl i advanced_model.pkl")
