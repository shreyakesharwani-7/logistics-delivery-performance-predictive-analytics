import pandas as pd
import os

# ---------------------------------------
# WEEK 1: STRATEGIC PLANNING & DATA EXPLORATION
# Logistics Delivery Performance & Predictive Analytics
# ---------------------------------------

# Dataset path
file_path = "Dataset/DataCoSupplyChainDataset.csv"

# Load dataset
df = pd.read_csv(file_path, encoding="latin1")

print("\n========== DATASET OVERVIEW ==========")

# Basic information
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# ---------------------------------------
# KEY LOGISTICS KPIs
# ---------------------------------------

print("\n========== KEY LOGISTICS KPIs ==========")

# Average shipping time
avg_shipping_days = df["Days for shipping (real)"].mean()

# Average scheduled shipping days
avg_scheduled_days = df["Days for shipment (scheduled)"].mean()

# Late delivery percentage
late_delivery_rate = df["Late_delivery_risk"].mean() * 100

# Total sales
total_sales = df["Sales"].sum()

# Total profit
total_profit = df["Order Profit Per Order"].sum()

# Average order value
avg_order_value = df["Sales"].mean()

print(f"Average Actual Shipping Days: {avg_shipping_days:.2f}")
print(f"Average Scheduled Shipping Days: {avg_scheduled_days:.2f}")
print(f"Late Delivery Rate: {late_delivery_rate:.2f}%")
print(f"Total Sales: ${total_sales:,.2f}")
print(f"Total Profit: ${total_profit:,.2f}")
print(f"Average Order Value: ${avg_order_value:,.2f}")

# ---------------------------------------
# DELIVERY STATUS ANALYSIS
# ---------------------------------------

print("\n========== DELIVERY STATUS ==========")
print(df["Delivery Status"].value_counts())

# ---------------------------------------
# SHIPPING MODE ANALYSIS
# ---------------------------------------

print("\n========== SHIPPING MODE ==========")
print(df["Shipping Mode"].value_counts())

# ---------------------------------------
# MARKET ANALYSIS
# ---------------------------------------

print("\n========== MARKET PERFORMANCE ==========")

market_sales = df.groupby("Market")["Sales"].sum().sort_values(
    ascending=False
)

print(market_sales)

# ---------------------------------------
# PRODUCT CATEGORY ANALYSIS
# ---------------------------------------

print("\n========== TOP PRODUCT CATEGORIES ==========")

category_sales = df.groupby("Category Name")["Sales"].sum().sort_values(
    ascending=False
)

print(category_sales.head(10))

# ---------------------------------------
# CUSTOMER SEGMENT ANALYSIS
# ---------------------------------------

print("\n========== CUSTOMER SEGMENT ==========")

segment_sales = df.groupby("Customer Segment")["Sales"].sum().sort_values(
    ascending=False
)

print(segment_sales)

# ---------------------------------------
# SAVE BASIC SUMMARY
# ---------------------------------------

summary = pd.DataFrame({
    "KPI": [
        "Average Actual Shipping Days",
        "Average Scheduled Shipping Days",
        "Late Delivery Rate (%)",
        "Total Sales",
        "Total Profit",
        "Average Order Value"
    ],
    "Value": [
        round(avg_shipping_days, 2),
        round(avg_scheduled_days, 2),
        round(late_delivery_rate, 2),
        round(total_sales, 2),
        round(total_profit, 2),
        round(avg_order_value, 2)
    ]
})

os.makedirs("Reports", exist_ok=True)

summary.to_csv("Reports/week1_kpi_summary.csv", index=False)

print("\n========== COMPLETE ==========")
print("Week 1 analysis completed successfully!")
print("KPI summary saved in Reports/week1_kpi_summary.csv")