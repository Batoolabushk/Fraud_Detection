import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

n_transactions = 12000
n_customers = 2000
fraud_rate = 0.08  # 8% fraudulent transactions (for pattern generation, but label will be dropped)

merchant_categories = [
    'Grocery', 'Gas Station', 'Restaurant', 'Retail', 'Online Shopping',
    'ATM Withdrawal', 'Transfer', 'Bill Payment', 'Entertainment',
    'Healthcare', 'Travel', 'Hotel', 'Pharmacy', 'Department Store',
    'Coffee Shop', 'Fast Food', 'Electronics', 'Clothing', 'Fuel'
]

countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'Australia', 'Brazil', 'Mexico', 'India']
country_weights = [0.7, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02]

# Transaction types
transaction_types = ['POS', 'Online', 'ATM', 'Transfer']
transaction_type_weights = [0.4, 0.35, 0.15, 0.1]

def generate_customer_profiles():
    """Generate customer profiles with spending patterns"""
    customers = {}
    for customer_id in range(1, n_customers + 1):
        customers[customer_id] = {
            'avg_transaction_amount': np.random.lognormal(mean=3.5, sigma=1.2),
            'transaction_frequency': np.random.gamma(shape=2, scale=5),
            'preferred_categories': np.random.choice(merchant_categories, size=3, replace=False),
            'home_country': np.random.choice(countries, p=country_weights),
            'risk_profile': np.random.choice(['low', 'medium', 'high'], p=[0.7, 0.25, 0.05])
        }
    return customers

def generate_transaction_datetime(start_date, days_span):
    """Generate realistic transaction datetime"""
    # Weight toward business hours and weekdays
    random_days = np.random.randint(0, days_span)
    base_date = start_date + timedelta(days=random_days)
    
    # Weight hours toward business hours (8 AM - 10 PM)
    hour_weights = np.array([0.5, 0.5, 0.3, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 
                            4.5, 4.0, 3.5, 3.5, 4.0, 4.5, 5.0, 4.5, 4.0, 3.5, 
                            3.0, 2.5, 2.0, 1.0])
    hour_weights = hour_weights / hour_weights.sum()
    hour = np.random.choice(24, p=hour_weights)
    
    minute = np.random.randint(0, 60)
    second = np.random.randint(0, 60)
    
    return base_date.replace(hour=hour, minute=minute, second=second)

def generate_normal_transaction(customer_id, customer_profile, transaction_id):
    """Generate a normal transaction for a customer"""
    base_amount = customer_profile['avg_transaction_amount']
    amount = np.random.lognormal(mean=np.log(base_amount), sigma=0.5)
    amount = round(max(amount, 1.0), 2)  # Minimum $1.00
    
    if np.random.random() < 0.7:
        category = np.random.choice(customer_profile['preferred_categories'])
    else:
        category = np.random.choice(merchant_categories)
    
    # Adjust amount based on category
    category_multipliers = {
        'ATM Withdrawal': 1.5, 'Transfer': 2.0, 'Bill Payment': 1.8,
        'Travel': 3.0, 'Hotel': 4.0, 'Electronics': 2.5,
        'Coffee Shop': 0.2, 'Fast Food': 0.3, 'Gas Station': 1.2
    }
    
    if category in category_multipliers:
        amount *= category_multipliers[category]
        amount = round(amount, 2)
    
    if np.random.random() < 0.95:
        country = customer_profile['home_country']
    else:
        country = np.random.choice([c for c in countries if c != customer_profile['home_country']])
    
    transaction_type = np.random.choice(transaction_types, p=transaction_type_weights)
    
    start_date = datetime(2023, 1, 1)
    transaction_datetime = generate_transaction_datetime(start_date, 365)
    
    return {
        'Transaction_ID': f'TXN_{transaction_id:08d}',
        'Customer_ID': f'CUST_{customer_id:06d}',
        'Transaction_DateTime': transaction_datetime,
        'Transaction_Amount': amount,
        'Merchant_Category': category,
        'Location_Country': country,
        'Transaction_Type': transaction_type,
        'transaction_pattern': 'normal'  # Internal tracking, will be removed
    }

def generate_suspicious_transaction(customer_id, customer_profile, transaction_id):
    """Generate a transaction with suspicious patterns (but no fraud label)"""
    fraud_patterns = ['large_amount', 'unusual_location', 'unusual_category', 'unusual_time', 'multiple_small']
    pattern = np.random.choice(fraud_patterns)
    
    # Base transaction
    transaction = generate_normal_transaction(customer_id, customer_profile, transaction_id)
    transaction['transaction_pattern'] = 'suspicious'  # Internal tracking, will be removed
    
    if pattern == 'large_amount':
        # Unusually large transaction (5-50x normal)
        multiplier = np.random.uniform(5, 50)
        transaction['Transaction_Amount'] = round(customer_profile['avg_transaction_amount'] * multiplier, 2)
        transaction['Merchant_Category'] = np.random.choice(['Online Shopping', 'Electronics', 'Transfer'])
    
    elif pattern == 'unusual_location':
        unusual_countries = ['Nigeria', 'Russia', 'China', 'Romania', 'Ukraine']
        transaction['Location_Country'] = np.random.choice(unusual_countries)
        transaction['Transaction_Type'] = 'Online'
    
    elif pattern == 'unusual_category':
        unusual_categories = ['Electronics', 'Travel', 'Hotel', 'Online Shopping']
        transaction['Merchant_Category'] = np.random.choice(unusual_categories)
        transaction['Transaction_Amount'] = round(customer_profile['avg_transaction_amount'] * np.random.uniform(3, 15), 2)
    
    elif pattern == 'unusual_time':
        # Transaction at unusual time (very early morning)
        base_time = transaction['Transaction_DateTime']
        unusual_hour = np.random.choice([1, 2, 3, 4, 5])
        transaction['Transaction_DateTime'] = base_time.replace(hour=unusual_hour)
        transaction['Transaction_Type'] = 'Online'
    
    elif pattern == 'multiple_small':
        # Small amount but suspicious pattern
        transaction['Transaction_Amount'] = round(np.random.uniform(1, 50), 2)
        transaction['Merchant_Category'] = 'Online Shopping'
        transaction['Transaction_Type'] = 'Online'
    
    return transaction

# Generate customer profiles
print("Generating customer profiles...")
customer_profiles = generate_customer_profiles()

# Generate transactions
print("Generating transactions...")
transactions = []
suspicious_count = int(n_transactions * fraud_rate)
normal_count = n_transactions - suspicious_count

# Generate normal transactions
for i in range(normal_count):
    customer_id = np.random.randint(1, n_customers + 1)
    transaction = generate_normal_transaction(customer_id, customer_profiles[customer_id], i + 1)
    transactions.append(transaction)

# Generate suspicious transactions (but without fraud labels)
for i in range(suspicious_count):
    customer_id = np.random.randint(1, n_customers + 1)
    transaction = generate_suspicious_transaction(customer_id, customer_profiles[customer_id], normal_count + i + 1)
    transactions.append(transaction)

# Create DataFrame
print("Creating DataFrame...")
df = pd.DataFrame(transactions)

# Sort by Customer_ID first, then by datetime for grouping
df = df.sort_values(['Customer_ID', 'Transaction_DateTime']).reset_index(drop=True)

# Add some additional features
df['Transaction_Hour'] = df['Transaction_DateTime'].dt.hour
df['Transaction_Day_of_Week'] = df['Transaction_DateTime'].dt.dayofweek
df['Transaction_Month'] = df['Transaction_DateTime'].dt.month

# Calculate customer statistics for feature engineering
print("Calculating customer statistics...")
customer_stats = df.groupby('Customer_ID').agg({
    'Transaction_Amount': ['mean', 'std', 'count'],
    'Location_Country': lambda x: x.nunique(),
    'Merchant_Category': lambda x: x.nunique()
}).round(2)

customer_stats.columns = ['Avg_Amount', 'Std_Amount', 'Transaction_Count', 'Unique_Countries', 'Unique_Categories']
customer_stats = customer_stats.reset_index()

# Merge customer statistics back to main dataframe
df = df.merge(customer_stats, on='Customer_ID', how='left')

# Add amount deviation from customer average
df['Amount_Deviation'] = abs(df['Transaction_Amount'] - df['Avg_Amount']) / (df['Std_Amount'] + 0.01)

# Remove internal tracking column and keep only required columns
columns_to_keep = [
    'Transaction_ID', 'Customer_ID', 'Transaction_DateTime', 'Transaction_Amount',
    'Merchant_Category', 'Location_Country', 'Transaction_Type', 'Transaction_Hour',
    'Transaction_Day_of_Week', 'Transaction_Month', 'Avg_Amount', 'Std_Amount',
    'Transaction_Count', 'Unique_Countries', 'Unique_Categories', 'Amount_Deviation'
]

df = df[columns_to_keep]

# Display dataset information
print(f"\nDataset generated successfully!")
print(f"Total transactions: {len(df):,}")
print(f"Unique customers: {df['Customer_ID'].nunique():,}")
print(f"Date range: {df['Transaction_DateTime'].min()} to {df['Transaction_DateTime'].max()}")
print(f"Transactions per customer (avg): {len(df)/df['Customer_ID'].nunique():.1f}")

# Show dataset grouped by Customer ID (first few customers)
print("\n=== DATASET OVERVIEW (Grouped by Customer ID) ===")
print("First 15 transactions (showing grouping by Customer ID):")
print(df.head(15)[['Transaction_ID', 'Customer_ID', 'Transaction_DateTime', 'Transaction_Amount', 'Merchant_Category', 'Location_Country']])

print("\n=== CUSTOMER TRANSACTION SUMMARY ===")
customer_summary = df.groupby('Customer_ID').agg({
    'Transaction_ID': 'count',
    'Transaction_Amount': ['mean', 'sum', 'std'],
    'Location_Country': 'nunique',
    'Merchant_Category': 'nunique'
}).round(2)

customer_summary.columns = ['Num_Transactions', 'Avg_Amount', 'Total_Amount', 'Std_Amount', 'Countries_Used', 'Categories_Used']
print("Sample customer summary:")
print(customer_summary.head(10))

print("\n=== MERCHANT CATEGORY DISTRIBUTION ===")
category_dist = df['Merchant_Category'].value_counts()
print(category_dist)

print("\n=== COUNTRY DISTRIBUTION ===")
country_dist = df['Location_Country'].value_counts()
print(country_dist)

print("\n=== TRANSACTION TYPE DISTRIBUTION ===")
type_dist = df['Transaction_Type'].value_counts()
print(type_dist)

# Save to CSV
filename = 'bank_transactions_dataset_grouped.csv'
df.to_csv(filename, index=False)
print(f"\nDataset saved as '{filename}'")

# Show sample of transactions with high amount deviation (potentially suspicious patterns)
print("\n=== TRANSACTIONS WITH HIGH AMOUNT DEVIATION ===")
high_deviation = df.nlargest(10, 'Amount_Deviation')[['Transaction_ID', 'Customer_ID', 'Transaction_Amount', 'Avg_Amount', 'Amount_Deviation', 'Merchant_Category', 'Location_Country']]
print(high_deviation)

# Additional analysis: Customers with most diverse spending patterns
print("\n=== CUSTOMERS WITH MOST DIVERSE SPENDING PATTERNS ===")
diverse_customers = df.groupby('Customer_ID').agg({
    'Merchant_Category': 'nunique',
    'Location_Country': 'nunique',
    'Transaction_Type': 'nunique',
    'Transaction_Amount': ['count', 'std']
}).round(2)
diverse_customers.columns = ['Categories', 'Countries', 'Types', 'Num_Transactions', 'Amount_Std']
diverse_customers = diverse_customers.sort_values(['Categories', 'Countries'], ascending=False)
print("Top 10 most diverse customers:")
print(diverse_customers.head(10))
