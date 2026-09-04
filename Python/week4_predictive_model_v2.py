import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("Dataset/cleaned_supply_chain.csv")

print("Dataset Shape:", df.shape)


# =========================================================
# 2. CREATE TARGET VARIABLE
# =========================================================

df["Shipping_Delay_Days"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)

print("\nTarget variable created: Shipping_Delay_Days")


# =========================================================
# 3. SELECT NUMERICAL + CATEGORICAL FEATURES
# =========================================================

numerical_features = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Sales",
    "Order Item Discount Rate"
]

categorical_features = [
    "Shipping Mode",
    "Market"
]

target = "Shipping_Delay_Days"


# =========================================================
# 4. REMOVE MISSING VALUES
# =========================================================

model_data = df[
    numerical_features + categorical_features + [target]
].dropna().copy()

print("Records used for modeling:", len(model_data))


# =========================================================
# 5. ENCODE CATEGORICAL FEATURES
# =========================================================

model_data_encoded = pd.get_dummies(
    model_data,
    columns=categorical_features,
    drop_first=False
)

X = model_data_encoded.drop(columns=[target])
y = model_data_encoded[target]

print("\nEncoded feature count:", X.shape[1])


# =========================================================
# 6. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# =========================================================
# 7. LINEAR REGRESSION
# =========================================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)


# =========================================================
# 8. BASELINE RANDOM FOREST
# =========================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)


# =========================================================
# 9. MODEL EVALUATION FUNCTION
# =========================================================

def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    print("\n", name)
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

    return mae, rmse, r2


# =========================================================
# 10. EVALUATE MODELS
# =========================================================

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


# =========================================================
# 11. BASELINE MODEL COMPARISON
# =========================================================

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


# =========================================================
# 12. CROSS VALIDATION
# =========================================================

cv_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

print("\n========== 5-FOLD CROSS-VALIDATION ==========")

for i, score in enumerate(cv_scores, 1):
    print(
        "Fold",
        i,
        "R2:",
        round(score, 4)
    )

print(
    "Mean CV R2:",
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
    "Reports/Week-4/v2_cross_validation_results.csv",
    index=False
)


# =========================================================
# 13. HYPERPARAMETER TUNING
# =========================================================

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


grid_search.fit(X_train, y_train)


print("\n========== HYPERPARAMETER TUNING ==========")

print(
    "Best Parameters:",
    grid_search.best_params_
)

print(
    "Best CV R2:",
    round(grid_search.best_score_, 4)
)


# =========================================================
# 14. TUNED MODEL
# =========================================================

best_model = grid_search.best_estimator_

tuned_pred = best_model.predict(X_test)


tuned_mae = mean_absolute_error(
    y_test,
    tuned_pred
)

tuned_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        tuned_pred
    )
)

tuned_r2 = r2_score(
    y_test,
    tuned_pred
)


print("\n========== TUNED RANDOM FOREST ==========")

print(
    "MAE:",
    round(tuned_mae, 4)
)

print(
    "RMSE:",
    round(tuned_rmse, 4)
)

print(
    "R2 Score:",
    round(tuned_r2, 4)
)


# =========================================================
# 15. FINAL MODEL COMPARISON
# =========================================================

final_comparison = pd.DataFrame({

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


print("\n========== FINAL MODEL COMPARISON ==========")

print(final_comparison)


final_comparison.to_csv(
    "Reports/Week-4/v2_final_model_comparison.csv",
    index=False
)


# =========================================================
# 16. FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": best_model.feature_importances_
})


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


importance["Importance_Percentage"] = (
    importance["Importance"] * 100
)


print("\n========== TUNED MODEL FEATURE IMPORTANCE ==========")

print(
    importance.head(15)
)


importance.to_csv(
    "Reports/Week-4/v2_feature_importance.csv",
    index=False
)


# =========================================================
# 17. MODEL IMPROVEMENT
# =========================================================

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
    / abs(rf_results[2])
) * 100


print("\n========== MODEL IMPROVEMENT ==========")

print(
    "MAE Improvement:",
    round(mae_improvement, 2),
    "%"
)

print(
    "RMSE Improvement:",
    round(rmse_improvement, 2),
    "%"
)

print(
    "R2 Improvement:",
    round(r2_improvement, 2),
    "%"
)


# =========================================================
# 18. OPTIMIZATION / HIGH DELAY RISK
# =========================================================

optimization_results = X_test.copy()

optimization_results[
    "Actual_Delay"
] = y_test.values

optimization_results[
    "Predicted_Delay"
] = tuned_pred


optimization_threshold = 1


optimization_results[
    "Predicted_High_Delay"
] = (
    optimization_results["Predicted_Delay"]
    > optimization_threshold
)


total_shipments = len(
    optimization_results
)


high_delay_shipments = (
    optimization_results[
        "Predicted_High_Delay"
    ].sum()
)


high_delay_percentage = (
    high_delay_shipments
    / total_shipments
) * 100


actual_high_delay_shipments = (
    optimization_results[
        "Actual_Delay"
    ] > optimization_threshold
).sum()


actual_high_delay_percentage = (
    actual_high_delay_shipments
    / total_shipments
) * 100


print("\n========== OPTIMIZATION ANALYSIS ==========")

print(
    "Total Test Shipments:",
    total_shipments
)

print(
    "Actual High-Delay Shipments:",
    actual_high_delay_shipments
)

print(
    "Predicted High-Delay Shipments:",
    high_delay_shipments
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
        high_delay_percentage,
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

        "Delay Threshold"
    ],

    "Value": [

        total_shipments,

        actual_high_delay_shipments,

        high_delay_shipments,

        round(
            actual_high_delay_percentage,
            2
        ),

        round(
            high_delay_percentage,
            2
        ),

        optimization_threshold
    ]
})


optimization_summary.to_csv(

    "Reports/Week-4/v2_optimization_summary.csv",

    index=False
)


# =========================================================
# 19. SAVE PREDICTION RESULTS
# =========================================================

prediction_output = pd.DataFrame({

    "Actual_Delay": y_test.values,

    "Predicted_Delay": tuned_pred

})


prediction_output.to_csv(

    "Reports/Week-4/v2_predictions.csv",

    index=False
)


# =========================================================
# 20. FINAL SUMMARY
# =========================================================

print("\n==============================================")
print("WEEK 4 V2 PREDICTIVE MODELING COMPLETED")
print("==============================================")

print(
    "Final Model: Tuned Random Forest"
)

print(
    "Final MAE:",
    round(tuned_mae, 4)
)

print(
    "Final RMSE:",
    round(tuned_rmse, 4)
)

print(
    "Final R2:",
    round(tuned_r2, 4)
)

print(
    "Mean 5-Fold CV R2:",
    round(cv_scores.mean(), 4)
)

print(
    "Predicted High-Delay:",
    high_delay_shipments,
    "/",
    total_shipments
)

print(
    "Predicted High-Delay %:",
    round(
        high_delay_percentage,
        2
    ),
    "%"
)

print("\nAll V2 reports saved successfully!")