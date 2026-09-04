import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("Dataset/cleaned_supply_chain.csv")

# Create target variable
df["Shipping_Delay_Days"] = (
    df["Days for shipping (real)"] -
    df["Days for shipment (scheduled)"]
)


# ==============================
# 1. ACTUAL VS PREDICTED
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

features = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Sales",
    "Order Item Discount Rate"
]

target = "Shipping_Delay_Days"

model_data = df[features + [target]].dropna()

X = model_data[features]
y = model_data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)


plt.figure(figsize=(8, 6))

plt.scatter(y_test, rf_pred, alpha=0.3)

plt.xlabel("Actual Shipping Delay (Days)")
plt.ylabel("Predicted Shipping Delay (Days)")
plt.title("Actual vs Predicted Shipping Delay")

plt.tight_layout()

plt.savefig(
    "Visualization/Week-4/01_actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ==============================
# 2. FEATURE IMPORTANCE
# ==============================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)


plt.figure(figsize=(8, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "Visualization/Week-4/02_feature_importance.png",
    dpi=300
)

plt.show()


# ==============================
# 3. MODEL COMPARISON
# ==============================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)


linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
linear_r2 = r2_score(y_test, linear_pred)


rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)


models = ["Linear Regression", "Random Forest"]

mae_values = [linear_mae, rf_mae]
rmse_values = [linear_rmse, rf_rmse]
r2_values = [linear_r2, rf_r2]


plt.figure(figsize=(8, 6))

x = np.arange(len(models))
width = 0.25

plt.bar(x - width, mae_values, width, label="MAE")
plt.bar(x, rmse_values, width, label="RMSE")
plt.bar(x + width, r2_values, width, label="R²")

plt.xticks(x, models)
plt.ylabel("Score")
plt.title("Predictive Model Performance Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    "Visualization/Week-4/03_model_comparison.png",
    dpi=300
)

plt.show()


print("WEEK 4 VISUALIZATIONS COMPLETED!")