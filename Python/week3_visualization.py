import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("Dataset/cleaned_supply_chain.csv")

# Create output folder
output_folder = "Visualization/Week-3"
os.makedirs(output_folder, exist_ok=True)


# =========================================================
# 1. Delivery Time: Actual vs Scheduled
# =========================================================

avg_actual = df["Days for shipping (real)"].mean()
avg_scheduled = df["Days for shipment (scheduled)"].mean()

plt.figure(figsize=(7, 5))

plt.bar(
    ["Scheduled", "Actual"],
    [avg_scheduled, avg_actual]
)

plt.title("Average Scheduled vs Actual Shipping Days")
plt.ylabel("Days")

plt.tight_layout()
plt.savefig(
    f"{output_folder}/01_actual_vs_scheduled_shipping.png"
)
plt.close()


# =========================================================
# 2. Shipping Delay Distribution
# =========================================================

df["Shipping_Delay_Days"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)

plt.figure(figsize=(8, 5))

plt.hist(
    df["Shipping_Delay_Days"],
    bins=15
)

plt.title("Distribution of Shipping Delay")
plt.xlabel("Delay (Days)")
plt.ylabel("Number of Orders")

plt.tight_layout()
plt.savefig(
    f"{output_folder}/02_shipping_delay_distribution.png"
)
plt.close()


# =========================================================
# 3. Shipping Mode Performance
# =========================================================

shipping_mode = df.groupby("Shipping Mode")[
    "Days for shipping (real)"
].mean().sort_values()

plt.figure(figsize=(8, 5))

plt.bar(
    shipping_mode.index,
    shipping_mode.values
)

plt.title("Average Actual Shipping Days by Shipping Mode")
plt.xlabel("Shipping Mode")
plt.ylabel("Average Shipping Days")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig(
    f"{output_folder}/03_shipping_mode_performance.png"
)
plt.close()


# =========================================================
# 4. Market-wise Sales
# =========================================================

market_sales = (
    df.groupby("Market")["Sales"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

plt.bar(
    market_sales.index,
    market_sales.values
)

plt.title("Total Sales by Market")
plt.xlabel("Market")
plt.ylabel("Total Sales")

plt.xticks(rotation=25)

plt.tight_layout()
plt.savefig(
    f"{output_folder}/04_market_sales.png"
)
plt.close()


# =========================================================
# 5. Top 10 Product Categories
# =========================================================

category_sales = (
    df.groupby("Category Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))

plt.barh(
    category_sales.index[::-1],
    category_sales.values[::-1]
)

plt.title("Top 10 Product Categories by Sales")
plt.xlabel("Total Sales")
plt.ylabel("Product Category")

plt.tight_layout()
plt.savefig(
    f"{output_folder}/05_top_categories_sales.png"
)
plt.close()


# =========================================================
# 6. Sales vs Profit Relationship
# =========================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Sales"],
    df["Order Profit Per Order"],
    alpha=0.3
)

plt.title("Relationship Between Sales and Profit")
plt.xlabel("Sales")
plt.ylabel("Profit per Order")

plt.tight_layout()
plt.savefig(
    f"{output_folder}/06_sales_vs_profit.png"
)
plt.close()


print("====================================")
print("WEEK 3 VISUALIZATIONS COMPLETED!")
print("====================================")
print("Graphs saved in:")
print("Visualization/Week-3")
# =========================================================
# 7. Correlation Heatmap
# =========================================================

correlation_columns = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Sales",
    "Order Item Quantity",
    "Order Item Discount",
    "Order Item Profit Ratio",
    "Order Profit Per Order",
    "Late_delivery_risk"
]

correlation = df[correlation_columns].corr()

plt.figure(figsize=(10, 7))

plt.imshow(correlation, aspect="auto")

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation_columns)),
    correlation_columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_columns)),
    correlation_columns
)

plt.title("Correlation Heatmap of Key Logistics Variables")

plt.tight_layout()

plt.savefig(
    f"{output_folder}/07_correlation_heatmap.png"
)

plt.close()

print("Correlation heatmap created successfully!")