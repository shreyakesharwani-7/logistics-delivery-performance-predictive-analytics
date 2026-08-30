import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("Dataset/DataCoSupplyChainDataset.csv", encoding="latin1")

# Create Visualization folder if it doesn't exist
os.makedirs("Visualization", exist_ok=True)

# ---------------------------------------
# 1. DELIVERY STATUS
# ---------------------------------------

delivery_status = df["Delivery Status"].value_counts()

plt.figure(figsize=(8, 5))
delivery_status.plot(kind="bar")
plt.title("Delivery Status Distribution")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Orders")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("Visualization/delivery_status.png")
plt.show()


# ---------------------------------------
# 2. SHIPPING MODE
# ---------------------------------------

shipping_mode = df["Shipping Mode"].value_counts()

plt.figure(figsize=(8, 5))
shipping_mode.plot(kind="bar")
plt.title("Orders by Shipping Mode")
plt.xlabel("Shipping Mode")
plt.ylabel("Number of Orders")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("Visualization/shipping_mode.png")
plt.show()


# ---------------------------------------
# 3. MARKET-WISE SALES
# ---------------------------------------

market_sales = df.groupby("Market")["Sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
market_sales.plot(kind="bar")
plt.title("Market-wise Sales Performance")
plt.xlabel("Market")
plt.ylabel("Total Sales")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("Visualization/market_sales.png")
plt.show()


# ---------------------------------------
# 4. TOP 10 PRODUCT CATEGORIES
# ---------------------------------------

category_sales = (
    df.groupby("Category Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
category_sales.plot(kind="bar")
plt.title("Top 10 Product Categories by Sales")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.savefig("Visualization/top_10_categories.png")
plt.show()


# ---------------------------------------
# 5. LATE DELIVERY RISK
# ---------------------------------------

late_delivery = df["Late_delivery_risk"].value_counts()

plt.figure(figsize=(7, 5))
late_delivery.plot(kind="bar")
plt.title("Late Delivery Risk Distribution")
plt.xlabel("Late Delivery Risk (0 = No, 1 = Yes)")
plt.ylabel("Number of Orders")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("Visualization/late_delivery_risk.png")
plt.show()


print("All Week 1 visualizations created successfully!")