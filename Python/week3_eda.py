import pandas as pd
import os

# -----------------------------
# 1. Load cleaned dataset
# -----------------------------

file_path = "Dataset/cleaned_supply_chain.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# -----------------------------
# 2. Basic Dataset Information
# -----------------------------

print("\n--- Dataset Information ---")
print(df.info())


# -----------------------------
# 3. Descriptive Statistics
# -----------------------------

print("\n--- Descriptive Statistics ---")
print(df.describe())


# -----------------------------
# 4. Delivery Time Analysis
# -----------------------------

actual_shipping = df["Days for shipping (real)"]
scheduled_shipping = df["Days for shipment (scheduled)"]

print("\n--- Delivery Time Analysis ---")

print("Average actual shipping days:",
      actual_shipping.mean())

print("Average scheduled shipping days:",
      scheduled_shipping.mean())

print("Minimum actual shipping days:",
      actual_shipping.min())

print("Maximum actual shipping days:",
      actual_shipping.max())


# Difference between actual and scheduled delivery time
df["Shipping_Delay_Days"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)

print("Average shipping delay:",
      df["Shipping_Delay_Days"].mean())

print("Orders with positive delay:",
      (df["Shipping_Delay_Days"] > 0).sum())


# -----------------------------
# 5. Sales and Profit Analysis
# -----------------------------

print("\n--- Sales and Profit Analysis ---")

print("Total Sales:",
      df["Sales"].sum())

print("Average Sales per order:",
      df["Sales"].mean())

print("Total Profit:",
      df["Order Profit Per Order"].sum())

print("Average Profit per order:",
      df["Order Profit Per Order"].mean())


# -----------------------------
# 6. Late Delivery Analysis
# -----------------------------

print("\n--- Late Delivery Analysis ---")

late_orders = (df["Late_delivery_risk"] == 1).sum()
total_orders = len(df)

late_percentage = (late_orders / total_orders) * 100

print("Total orders:", total_orders)
print("Late-risk orders:", late_orders)
print("Late delivery risk percentage:",
      late_percentage)


# -----------------------------
# 7. Shipping Mode Analysis
# -----------------------------

print("\n--- Shipping Mode Analysis ---")

shipping_analysis = df.groupby("Shipping Mode").agg(
    Orders=("Order Id", "count"),
    Average_Actual_Days=("Days for shipping (real)", "mean"),
    Average_Scheduled_Days=("Days for shipment (scheduled)", "mean"),
    Average_Sales=("Sales", "mean"),
    Average_Profit=("Order Profit Per Order", "mean")
).sort_values("Orders", ascending=False)

print(shipping_analysis)


# -----------------------------
# 8. Market Analysis
# -----------------------------

print("\n--- Market Analysis ---")

market_analysis = df.groupby("Market").agg(
    Orders=("Order Id", "count"),
    Total_Sales=("Sales", "sum"),
    Average_Sales=("Sales", "mean"),
    Average_Profit=("Order Profit Per Order", "mean"),
    Late_Risk=("Late_delivery_risk", "mean")
).sort_values("Total_Sales", ascending=False)

market_analysis["Late_Risk_Percentage"] = (
    market_analysis["Late_Risk"] * 100
)

print(market_analysis)


# -----------------------------
# 9. Customer Segment Analysis
# -----------------------------

print("\n--- Customer Segment Analysis ---")

segment_analysis = df.groupby("Customer Segment").agg(
    Orders=("Order Id", "count"),
    Total_Sales=("Sales", "sum"),
    Average_Sales=("Sales", "mean"),
    Average_Profit=("Order Profit Per Order", "mean"),
    Late_Risk=("Late_delivery_risk", "mean")
).sort_values("Total_Sales", ascending=False)

segment_analysis["Late_Risk_Percentage"] = (
    segment_analysis["Late_Risk"] * 100
)

print(segment_analysis)


# -----------------------------
# 10. Category Analysis
# -----------------------------

print("\n--- Top Product Categories ---")

category_analysis = df.groupby("Category Name").agg(
    Orders=("Order Id", "count"),
    Total_Sales=("Sales", "sum"),
    Average_Profit=("Order Profit Per Order", "mean")
).sort_values("Total_Sales", ascending=False)

print(category_analysis.head(10))


# -----------------------------
# 11. Correlation Analysis
# -----------------------------

print("\n--- Correlation Analysis ---")

numeric_columns = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Sales",
    "Order Item Quantity",
    "Order Item Discount",
    "Order Item Profit Ratio",
    "Order Profit Per Order",
    "Late_delivery_risk"
]

correlation = df[numeric_columns].corr()

print(correlation)


# -----------------------------
# 12. Save Analysis Results
# -----------------------------

os.makedirs("Reports/Week-3", exist_ok=True)

shipping_analysis.to_csv(
    "Reports/Week-3/shipping_mode_analysis.csv"
)

market_analysis.to_csv(
    "Reports/Week-3/market_analysis.csv"
)

segment_analysis.to_csv(
    "Reports/Week-3/customer_segment_analysis.csv"
)

category_analysis.to_csv(
    "Reports/Week-3/category_analysis.csv"
)

correlation.to_csv(
    "Reports/Week-3/correlation_matrix.csv"
)

print("\n================================")
print("WEEK 3 ANALYSIS COMPLETED!")
print("Results saved in Reports/Week-3")
print("================================")

# =========================================================
# 13. Detailed Insights for Week 3 Report
# =========================================================

print("\n" + "=" * 50)
print("DETAILED WEEK 3 INSIGHTS")
print("=" * 50)

# 1. Delivery gap
avg_actual = df["Days for shipping (real)"].mean()
avg_scheduled = df["Days for shipment (scheduled)"].mean()
delivery_gap = avg_actual - avg_scheduled

print("\n1. DELIVERY PERFORMANCE")
print("Average actual shipping days:", round(avg_actual, 2))
print("Average scheduled shipping days:", round(avg_scheduled, 2))
print("Average delay compared with schedule:", round(delivery_gap, 2), "days")

# 2. Positive delay percentage
positive_delay = (df["Shipping_Delay_Days"] > 0).sum()
positive_delay_percentage = positive_delay / len(df) * 100

print("\n2. ORDERS WITH SHIPPING DELAY")
print("Orders delayed beyond schedule:", positive_delay)
print("Percentage of delayed orders:", round(positive_delay_percentage, 2), "%")

# 3. Shipping mode comparison
mode_delay = df.groupby("Shipping Mode").agg(
    Average_Actual=("Days for shipping (real)", "mean"),
    Average_Scheduled=("Days for shipment (scheduled)", "mean")
)

mode_delay["Delay"] = (
    mode_delay["Average_Actual"]
    - mode_delay["Average_Scheduled"]
)

print("\n3. SHIPPING MODE DELAY")
print(mode_delay.sort_values("Delay", ascending=False))

# 4. Market with highest late-delivery risk
market_risk = (
    df.groupby("Market")["Late_delivery_risk"]
    .mean()
    .sort_values(ascending=False)
)

print("\n4. MARKET LATE DELIVERY RISK")
print((market_risk * 100).round(2))

# 5. Market with highest sales
market_sales = (
    df.groupby("Market")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n5. MARKET SALES")
print(market_sales.round(2))

# 6. Top category
category_sales = (
    df.groupby("Category Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\n6. TOP 5 PRODUCT CATEGORIES BY SALES")
print(category_sales.head(5).round(2))

# 7. Profit comparison
print("\n7. PROFIT")
print("Total profit:",
      round(df["Order Profit Per Order"].sum(), 2))

print("Average profit per order:",
      round(df["Order Profit Per Order"].mean(), 2))

# 8. Correlation with late delivery risk
correlations = df[
    [
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Sales",
        "Order Item Quantity",
        "Order Item Discount",
        "Order Item Profit Ratio",
        "Order Profit Per Order",
        "Late_delivery_risk"
    ]
].corr()["Late_delivery_risk"].sort_values(ascending=False)

print("\n8. CORRELATION WITH LATE DELIVERY RISK")
print(correlations.round(3))

print("\n" + "=" * 50)
print("DETAILED INSIGHTS COMPLETED")
print("=" * 50)