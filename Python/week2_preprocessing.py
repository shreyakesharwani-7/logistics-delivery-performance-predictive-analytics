import pandas as pd

# Load the original dataset
df = pd.read_csv(
    "Dataset/DataCoSupplyChainDataset.csv",
    encoding="latin1"
)

print("========== WEEK 2: DATA PREPROCESSING ==========")

# Dataset size
print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Duplicate rows:", df.duplicated().sum())

# Data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== INITIAL DATA QUALITY CHECK COMPLETED ==========")


# ==========================================
# STEP 2: HANDLE MISSING VALUES
# ==========================================

# Make a copy of the original dataset
cleaned_df = df.copy()

# Remove columns with extremely high missing values
cleaned_df = cleaned_df.drop(
    columns=["Order Zipcode", "Product Description"]
)

# Fill missing customer last names
cleaned_df["Customer Lname"] = cleaned_df["Customer Lname"].fillna("Unknown")

# Fill missing customer zipcodes with the median
cleaned_df["Customer Zipcode"] = cleaned_df["Customer Zipcode"].fillna(
    cleaned_df["Customer Zipcode"].median()
)

# Check missing values after cleaning
print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(cleaned_df.isnull().sum())

print("\nMissing values handled successfully!")

# ==========================================
# STEP 3: HANDLE DUPLICATE RECORDS
# ==========================================

# Count duplicate rows before removing
duplicate_count = cleaned_df.duplicated().sum()

print("\n========== DUPLICATE RECORDS ==========")
print("Duplicate rows before cleaning:", duplicate_count)

# Remove duplicate rows
cleaned_df = cleaned_df.drop_duplicates()

# Check dataset size after removing duplicates
print("\n========== AFTER DUPLICATE REMOVAL ==========")
print("Rows:", cleaned_df.shape[0])
print("Columns:", cleaned_df.shape[1])

print("Duplicate records handled successfully!")


# ==========================================
# STEP 4: OUTLIER DETECTION
# ==========================================

# Function to detect outliers using IQR
def detect_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_limit) |
        (data[column] > upper_limit)
    ]

    return outliers, lower_limit, upper_limit


# Check Sales
sales_outliers, sales_lower, sales_upper = detect_outliers(
    cleaned_df, "Sales"
)

print("\n========== SALES OUTLIERS ==========")
print("Lower limit:", sales_lower)
print("Upper limit:", sales_upper)
print("Number of Sales outliers:", len(sales_outliers))


# Check Order Item Quantity
quantity_outliers, quantity_lower, quantity_upper = detect_outliers(
    cleaned_df, "Order Item Quantity"
)

print("\n========== QUANTITY OUTLIERS ==========")
print("Lower limit:", quantity_lower)
print("Upper limit:", quantity_upper)
print("Number of Quantity outliers:", len(quantity_outliers))

# ==========================================
# STEP 5: DATA NORMALIZATION
# ==========================================

from sklearn.preprocessing import MinMaxScaler

# Create scaler
scaler = MinMaxScaler()

# Normalize Sales column
cleaned_df["Sales_Normalized"] = scaler.fit_transform(
    cleaned_df[["Sales"]]
)

print("\n========== NORMALIZATION ==========")
print(cleaned_df[["Sales", "Sales_Normalized"]].head())

print("\nNormalization completed successfully!")

# ==========================================
# STEP 6: SAVE CLEANED DATASET
# ==========================================

cleaned_df.to_csv(
    "Dataset/cleaned_supply_chain.csv",
    index=False
)

print("\n========== CLEANED DATASET SAVED ==========")
print("File: Dataset/cleaned_supply_chain.csv")
print("Rows:", cleaned_df.shape[0])
print("Columns:", cleaned_df.shape[1])

# ==========================================
# STEP 7: FINAL DATA QUALITY CHECK
# ==========================================

print("\n========== FINAL DATA QUALITY CHECK ==========")

print("Missing values:", cleaned_df.isnull().sum().sum())
print("Duplicate rows:", cleaned_df.duplicated().sum())

print("\nFinal dataset shape:")
print("Rows:", cleaned_df.shape[0])
print("Columns:", cleaned_df.shape[1])

print("\n========== WEEK 2 PREPROCESSING COMPLETED ==========")