import pandas as pd
import hdbscan
from models import BaseModel, AdvancedModel, compute_ground_truth, evaluate

train_df = pd.read_csv("data/data.csv")
val_df = pd.read_csv("data/val_data.csv")
test_df = pd.read_csv("data/test_data.csv")

feature_cols = [col for col in train_df.columns if col not in ["price",
                                                               "log_price"]]

X_train = train_df[feature_cols]
X_val = val_df[feature_cols]
X_test = test_df[feature_cols]

y_train_price = train_df["price"]
y_val_price = val_df["price"]
y_test_price = test_df["price"]

y_test_log = test_df["log_price"]

# BaseModel
print("Tuning and training BaseModel...")
base_model = BaseModel()
base_model.tune(
    X_train, y_train_price,
    X_val, y_val_price,
    cluster_values=[3, 5, 7],
    threshold_values=[0.2, 0.25, 0.3, 0.35]
)

cluster_labels_test = base_model.kmeans.predict(X_test)
df_test_base = pd.DataFrame({
    "log_price": y_test_log,
    "cluster_labels": cluster_labels_test
})
y_test_base = compute_ground_truth(df_test_base, cluster_col="cluster_labels")
y_pred_base = base_model.predict(X_test, y_test_price)
metrics_base = evaluate(y_test_base, y_pred_base)
print("BaseModel metrics:", metrics_base)

# AdvancedModel
print("Tuning and training AdvancedModel...")
adv_model = AdvancedModel()
adv_model.tune(
    X_train, y_train_price,
    X_val, y_val_price,
    cluster_sizes=[5, 10, 15],
    threshold_values=[0.2, 0.25, 0.3, 0.35]
)

cluster_labels_test, strengths = hdbscan.approximate_predict(adv_model.clusterer, X_test)
df_test_adv = pd.DataFrame({
    "log_price": y_test_log,
    "cluster_labels": cluster_labels_test
})
y_test_adv = compute_ground_truth(df_test_adv, cluster_col="cluster_labels")
y_pred_adv = adv_model.predict(X_test, y_test_price)
metrics_adv = evaluate(y_test_adv, y_pred_adv)
print("AdvancedModel metrics:", metrics_adv)

base_model.save("models/base_model.pkl")
adv_model.save("models/advanced_model.pkl")
print("Modele zapisane: base_model.pkl i advanced_model.pkl")
