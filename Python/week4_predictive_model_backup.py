import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD CLEANED DATASET
# ============================================================

df = pd.read_csv("Dataset/cleaned_supply_chain.csv")

print("Dataset Shape:", df.shape)


# ============================================================
# 2. CREATE TARGET VARIABLE
# ============================================================

# Delivery delay = actual shipping days - scheduled shipping days

df["Shipping_Delay_Days"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)


# ============================================================
# 3. SELECT FEATURES
# ============================================================

features = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Sales",
    "Order Item Discount Rate"
]

target = "Shipping_Delay_Days"


# ============================================================
# 4. PREPARE MODEL DATA
# ============================================================

# Shipping Mode and Market are kept for later
# operational risk analysis.
required_columns = features + [
    target,
    "Shipping Mode",
    "Market"
]

model_data = df[required_columns].dropna().copy()

print("Records available for modeling:", len(model_data))


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

train_data, test_data = train_test_split(
    model_data,
    test_size=0.20,
    random_state=42
)

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 6. LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_pred = linear_model.predict(X_test)


# ============================================================
# 7. BASELINE RANDOM FOREST
# ============================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(X_test)


# ============================================================
# 8. EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    print("\n", name)
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R² Score:", round(r2, 4))

    return mae, rmse, r2


# ============================================================
# 9. EVALUATE BASELINE MODELS
# ============================================================

linear_results = evaluate_model(
    "Linear Regression",
    y_test,
    linear_pred
)

rf_results = evaluate_model(
    "Baseline Random Forest",
    y_test,
    rf_pred
)


# ============================================================
# 10. BASELINE MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Baseline Random Forest"
    ],
    "MAE": [
        linear_results[0],
        rf_results[0]
    ],
    "RMSE": [
        linear_results[1],
        rf_results[1]
    ],
    "R2_Score": [
        linear_results[2],
        rf_results[2]
    ]
})

print("\n========== BASELINE MODEL COMPARISON ==========")
print(comparison)


# ============================================================
# 11. FEATURE IMPORTANCE - BASELINE RANDOM FOREST
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance["Importance_Percentage"] = (
    importance["Importance"] * 100
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance)


# ============================================================
# 12. CROSS-VALIDATION - BASELINE RANDOM FOREST
# ============================================================

cv_scores = cross_val_score(
    rf_model,
    model_data[features],
    model_data[target],
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("\n========== 5-FOLD CROSS-VALIDATION ==========")

for i, score in enumerate(cv_scores, start=1):
    print(
        f"Fold {i} R²:",
        round(score, 4)
    )

print(
    "Mean Cross-Validation R²:",
    round(cv_scores.mean(), 4)
)

print(
    "Standard Deviation:",
    round(cv_scores.std(), 4)
)

cv_results = pd.DataFrame({
    "Fold": [1, 2, 3, 4, 5],
    "R2_Score": cv_scores
})

cv_results.to_csv(
    "Reports/Week-4/cross_validation_results.csv",
    index=False
)


# ============================================================
# 13. HYPERPARAMETER TUNING
# ============================================================

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    ),
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

print("\n========== HYPERPARAMETER TUNING ==========")

print(
    "Best Parameters:",
    grid_search.best_params_
)

print(
    "Best Cross-Validation R²:",
    round(grid_search.best_score_, 4)
)


# ============================================================
# 14. FINAL TUNED RANDOM FOREST
# ============================================================

best_model = grid_search.best_estimator_

y_pred_tuned = best_model.predict(
    X_test
)

tuned_mae = mean_absolute_error(
    y_test,
    y_pred_tuned
)

tuned_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_tuned
    )
)

tuned_r2 = r2_score(
    y_test,
    y_pred_tuned
)

print("\n========== TUNED RANDOM FOREST TEST RESULTS ==========")

print(
    "MAE:",
    round(tuned_mae, 4)
)

print(
    "RMSE:",
    round(tuned_rmse, 4)
)

print(
    "R² Score:",
    round(tuned_r2, 4)
)


# ============================================================
# 15. BASELINE VS TUNED IMPROVEMENT
# ============================================================

mae_improvement = (
    (rf_results[0] - tuned_mae)
    / rf_results[0]
) * 100

rmse_improvement = (
    (rf_results[1] - tuned_rmse)
    / rf_results[1]
) * 100

r2_improvement = (
    (tuned_r2 - rf_results[2])
    / rf_results[2]
) * 100

improvement = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R2"
    ],
    "Baseline": [
        rf_results[0],
        rf_results[1],
        rf_results[2]
    ],
    "Tuned": [
        tuned_mae,
        tuned_rmse,
        tuned_r2
    ],
    "Improvement_Percentage": [
        mae_improvement,
        rmse_improvement,
        r2_improvement
    ]
})

print("\n========== MODEL IMPROVEMENT ==========")
print(improvement)

improvement.to_csv(
    "Reports/Week-4/model_improvement.csv",
    index=False
)


# ============================================================
# 16. FINAL MODEL FEATURE IMPORTANCE
# ============================================================

tuned_importance = pd.DataFrame({
    "Feature": features,
    "Importance": best_model.feature_importances_
})

tuned_importance = tuned_importance.sort_values(
    by="Importance",
    ascending=False
)

tuned_importance["Importance_Percentage"] = (
    tuned_importance["Importance"] * 100
)

print("\n========== TUNED MODEL FEATURE IMPORTANCE ==========")
print(tuned_importance)

tuned_importance.to_csv(
    "Reports/Week-4/tuned_feature_importance.csv",
    index=False
)


# ============================================================
# 17. ACTUAL VS PREDICTED DELAY ANALYSIS
# ============================================================

results = test_data.copy()

results["Actual_Delay"] = y_test.values

results["Predicted_Delay"] = y_pred_tuned


# High-delay threshold
# More than 1 day predicted delay

optimization_threshold = 1

results["Actual_High_Delay"] = (
    results["Actual_Delay"]
    > optimization_threshold
)

results["Predicted_High_Delay"] = (
    results["Predicted_Delay"]
    > optimization_threshold
)


# ============================================================
# 18. OVERALL OPTIMIZATION ANALYSIS
# ============================================================

total_shipments = len(results)

actual_high_delay = (
    results["Actual_High_Delay"].sum()
)

predicted_high_delay = (
    results["Predicted_High_Delay"].sum()
)

actual_high_delay_percentage = (
    actual_high_delay
    / total_shipments
) * 100

predicted_high_delay_percentage = (
    predicted_high_delay
    / total_shipments
) * 100


print("\n========== OVERALL OPTIMIZATION ANALYSIS ==========")

print(
    "Total Test Shipments:",
    total_shipments
)

print(
    "Actual High-Delay Shipments:",
    actual_high_delay
)

print(
    "Predicted High-Delay Shipments:",
    predicted_high_delay
)

print(
    "Actual High-Delay Percentage:",
    round(
        actual_high_delay_percentage,
        2
    ),
    "%"
)

print(
    "Predicted High-Delay Percentage:",
    round(
        predicted_high_delay_percentage,
        2
    ),
    "%"
)

print(
    "Delay Threshold:",
    optimization_threshold,
    "day(s)"
)


optimization_summary = pd.DataFrame({
    "Metric": [
        "Total Test Shipments",
        "Actual High-Delay Shipments",
        "Predicted High-Delay Shipments",
        "Actual High-Delay Percentage",
        "Predicted High-Delay Percentage",
        "Delay Threshold (Days)"
    ],
    "Value": [
        total_shipments,
        actual_high_delay,
        predicted_high_delay,
        round(
            actual_high_delay_percentage,
            2
        ),
        round(
            predicted_high_delay_percentage,
            2
        ),
        optimization_threshold
    ]
})

optimization_summary.to_csv(
    "Reports/Week-4/optimization_analysis_tuned.csv",
    index=False
)


# ============================================================
# 19. SHIPPING MODE OPERATIONAL RISK ANALYSIS
# ============================================================

shipping_mode_analysis = (
    results
    .groupby("Shipping Mode")
    .agg(
        Shipments=("Predicted_High_Delay", "size"),
        Actual_High_Delay_Shipments=(
            "Actual_High_Delay",
            "sum"
        ),
        Predicted_High_Delay_Shipments=(
            "Predicted_High_Delay",
            "sum"
        ),
        Average_Actual_Delay=(
            "Actual_Delay",
            "mean"
        ),
        Average_Predicted_Delay=(
            "Predicted_Delay",
            "mean"
        )
    )
    .reset_index()
)

shipping_mode_analysis[
    "Actual_High_Delay_Percentage"
] = (
    shipping_mode_analysis[
        "Actual_High_Delay_Shipments"
    ]
    / shipping_mode_analysis["Shipments"]
) * 100

shipping_mode_analysis[
    "Predicted_High_Delay_Percentage"
] = (
    shipping_mode_analysis[
        "Predicted_High_Delay_Shipments"
    ]
    / shipping_mode_analysis["Shipments"]
) * 100

shipping_mode_analysis = shipping_mode_analysis.sort_values(
    by="Average_Predicted_Delay",
    ascending=False
)

print("\n========== SHIPPING MODE OPERATIONAL RISK ==========")
print(shipping_mode_analysis)

shipping_mode_analysis.to_csv(
    "Reports/Week-4/shipping_mode_risk_analysis.csv",
    index=False
)


# ============================================================
# 20. MARKET OPERATIONAL RISK ANALYSIS
# ============================================================

market_analysis = (
    results
    .groupby("Market")
    .agg(
        Shipments=("Predicted_High_Delay", "size"),
        Actual_High_Delay_Shipments=(
            "Actual_High_Delay",
            "sum"
        ),
        Predicted_High_Delay_Shipments=(
            "Predicted_High_Delay",
            "sum"
        ),
        Average_Actual_Delay=(
            "Actual_Delay",
            "mean"
        ),
        Average_Predicted_Delay=(
            "Predicted_Delay",
            "mean"
        )
    )
    .reset_index()
)

market_analysis[
    "Actual_High_Delay_Percentage"
] = (
    market_analysis[
        "Actual_High_Delay_Shipments"
    ]
    / market_analysis["Shipments"]
) * 100

market_analysis[
    "Predicted_High_Delay_Percentage"
] = (
    market_analysis[
        "Predicted_High_Delay_Shipments"
    ]
    / market_analysis["Shipments"]
) * 100

market_analysis = market_analysis.sort_values(
    by="Average_Predicted_Delay",
    ascending=False
)

print("\n========== MARKET OPERATIONAL RISK ==========")
print(market_analysis)

market_analysis.to_csv(
    "Reports/Week-4/market_risk_analysis.csv",
    index=False
)
# ============================================================
# 20A. SHIPPING MODE OPTIMIZATION ANALYSIS
# ============================================================

# Analyze operational priority using actual and predicted delay.

shipping_mode_optimization = (
    results
    .groupby("Shipping Mode")
    .agg(
        Shipments=("Predicted_Delay", "size"),
        Average_Actual_Delay=("Actual_Delay", "mean"),
        Average_Predicted_Delay=("Predicted_Delay", "mean"),
        Actual_High_Delay_Shipments=(
            "Actual_High_Delay",
            "sum"
        ),
        Predicted_High_Delay_Shipments=(
            "Predicted_High_Delay",
            "sum"
        )
    )
    .reset_index()
)

shipping_mode_optimization[
    "Actual_High_Delay_Percentage"
] = (
    shipping_mode_optimization[
        "Actual_High_Delay_Shipments"
    ]
    / shipping_mode_optimization["Shipments"]
) * 100

shipping_mode_optimization[
    "Predicted_High_Delay_Percentage"
] = (
    shipping_mode_optimization[
        "Predicted_High_Delay_Shipments"
    ]
    / shipping_mode_optimization["Shipments"]
) * 100


# Operational priority based on predicted delay.
shipping_mode_optimization["Priority"] = np.where(
    shipping_mode_optimization[
        "Average_Predicted_Delay"
    ] > 1,
    "High Priority",
    np.where(
        shipping_mode_optimization[
            "Average_Predicted_Delay"
        ] > 0,
        "Monitor",
        "Low Priority"
    )
)

shipping_mode_optimization = (
    shipping_mode_optimization
    .sort_values(
        by="Average_Predicted_Delay",
        ascending=False
    )
)

print("\n========== SHIPPING MODE OPTIMIZATION ==========")
print(shipping_mode_optimization)

shipping_mode_optimization.to_csv(
    "Reports/Week-4/shipping_mode_optimization.csv",
    index=False
)


# ============================================================
# 20B. MARKET OPTIMIZATION PRIORITY
# ============================================================

market_optimization = (
    results
    .groupby("Market")
    .agg(
        Shipments=("Predicted_Delay", "size"),
        Average_Actual_Delay=("Actual_Delay", "mean"),
        Average_Predicted_Delay=("Predicted_Delay", "mean"),
        Actual_High_Delay_Shipments=(
            "Actual_High_Delay",
            "sum"
        ),
        Predicted_High_Delay_Shipments=(
            "Predicted_High_Delay",
            "sum"
        )
    )
    .reset_index()
)

market_optimization[
    "Actual_High_Delay_Percentage"
] = (
    market_optimization[
        "Actual_High_Delay_Shipments"
    ]
    / market_optimization["Shipments"]
) * 100

market_optimization[
    "Predicted_High_Delay_Percentage"
] = (
    market_optimization[
        "Predicted_High_Delay_Shipments"
    ]
    / market_optimization["Shipments"]
) * 100


market_optimization["Priority"] = np.where(
    market_optimization[
        "Predicted_High_Delay_Percentage"
    ] >= 20,
    "High Priority",
    np.where(
        market_optimization[
            "Predicted_High_Delay_Percentage"
        ] >= 19,
        "Monitor",
        "Normal"
    )
)

market_optimization = (
    market_optimization
    .sort_values(
        by="Predicted_High_Delay_Percentage",
        ascending=False
    )
)

print("\n========== MARKET OPTIMIZATION PRIORITY ==========")
print(market_optimization)

market_optimization.to_csv(
    "Reports/Week-4/market_optimization.csv",
    index=False
)


# ============================================================
# 20C. HIGH-RISK SHIPMENT PRIORITY SUMMARY
# ============================================================

high_risk_shipments = results[
    results["Predicted_Delay"] > optimization_threshold
].copy()

high_risk_summary = pd.DataFrame({
    "Metric": [
        "Total Test Shipments",
        "Predicted High-Risk Shipments",
        "Predicted High-Risk Percentage",
        "Average Predicted Delay of High-Risk Shipments"
    ],
    "Value": [
        len(results),
        len(high_risk_shipments),
        round(
            len(high_risk_shipments)
            / len(results) * 100,
            2
        ),
        round(
            high_risk_shipments[
                "Predicted_Delay"
            ].mean(),
            4
        )
    ]
})

print("\n========== HIGH-RISK SHIPMENT PRIORITY ==========")
print(high_risk_summary)

high_risk_summary.to_csv(
    "Reports/Week-4/high_risk_priority_summary.csv",
    index=False
)
# ============================================================
# 21. SAVE DETAILED PREDICTION RESULTS
# ============================================================

results.to_csv(
    "Reports/Week-4/prediction_results.csv",
    index=False
)


# ============================================================
# 22. FINAL MODEL SUMMARY
# ============================================================

final_summary = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Baseline Random Forest",
        "Tuned Random Forest"
    ],
    "MAE": [
        linear_results[0],
        rf_results[0],
        tuned_mae
    ],
    "RMSE": [
        linear_results[1],
        rf_results[1],
        tuned_rmse
    ],
    "R2_Score": [
        linear_results[2],
        rf_results[2],
        tuned_r2
    ]
})

print("\n========== FINAL MODEL SUMMARY ==========")
print(final_summary)

final_summary.to_csv(
    "Reports/Week-4/final_model_comparison.csv",
    index=False
)


print("\n============================================")
print("WEEK 4 PREDICTIVE MODELING COMPLETED!")
print("All analysis reports saved successfully.")
print("============================================")