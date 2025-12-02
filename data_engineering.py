import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

class BankDatasetGenerator:
    """
    Comprehensive Bank Transaction Dataset Generator with Data Cleaning and Feature Engineering
    """
    
    def __init__(self, n_transactions=12000, n_customers=2000, fraud_rate=0.08, random_seed=42):
        """Initialize the dataset generator with parameters"""
        self.n_transactions = n_transactions
        self.n_customers = n_customers
        self.fraud_rate = fraud_rate
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Define constants
        self.merchant_categories = [
            'Grocery', 'Gas Station', 'Restaurant', 'Retail', 'Online Shopping',
            'ATM Withdrawal', 'Transfer', 'Bill Payment', 'Entertainment',
            'Healthcare', 'Travel', 'Hotel', 'Pharmacy', 'Department Store',
            'Coffee Shop', 'Fast Food', 'Electronics', 'Clothing', 'Fuel'
        ]
        
        self.countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'Australia', 'Brazil', 'Mexico', 'India']
        self.country_weights = [0.7, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02]
        
        self.high_risk_countries = ['Nigeria', 'Russia', 'Ukraine', 'Romania', 'China']
        
        self.transaction_types = ['POS', 'Online', 'ATM', 'Transfer']
        self.transaction_type_weights = [0.4, 0.35, 0.15, 0.1]
        
        print(f"Initialized Bank Dataset Generator:")
        print(f"  - Transactions: {self.n_transactions:,}")
        print(f"  - Customers: {self.n_customers:,}")
        print(f"  - Fraud Rate: {self.fraud_rate*100:.1f}%")
    
    def generate_customer_profiles(self):
        """Generate customer profiles with spending patterns"""
        print("Generating customer profiles...")
        customers = {}
        for customer_id in range(1, self.n_customers + 1):
            customers[customer_id] = {
                'avg_transaction_amount': np.random.lognormal(mean=3.5, sigma=1.2),
                'transaction_frequency': np.random.gamma(shape=2, scale=5),
                'preferred_categories': np.random.choice(self.merchant_categories, size=3, replace=False),
                'home_country': np.random.choice(self.countries, p=self.country_weights),
                'risk_profile': np.random.choice(['low', 'medium', 'high'], p=[0.7, 0.25, 0.05])
            }
        return customers
    
    def generate_transaction_datetime(self, start_date, days_span):
        """Generate realistic transaction datetime with business hour weighting"""
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
    
    def generate_normal_transaction(self, customer_id, customer_profile, transaction_id):
        """Generate a normal transaction for a customer"""
        base_amount = customer_profile['avg_transaction_amount']
        amount = np.random.lognormal(mean=np.log(base_amount), sigma=0.5)
        amount = round(max(amount, 1.0), 2)  # Minimum $1.00
        
        # Category selection (70% from preferred categories)
        if np.random.random() < 0.7:
            category = np.random.choice(customer_profile['preferred_categories'])
        else:
            category = np.random.choice(self.merchant_categories)
        
        # Category-based amount adjustments
        category_multipliers = {
            'ATM Withdrawal': 1.5, 'Transfer': 2.0, 'Bill Payment': 1.8,
            'Travel': 3.0, 'Hotel': 4.0, 'Electronics': 2.5,
            'Coffee Shop': 0.2, 'Fast Food': 0.3, 'Gas Station': 1.2
        }
        
        if category in category_multipliers:
            amount *= category_multipliers[category]
            amount = round(amount, 2)
        
        # Country selection (95% home country)
        if np.random.random() < 0.95:
            country = customer_profile['home_country']
        else:
            country = np.random.choice([c for c in self.countries if c != customer_profile['home_country']])
        
        transaction_type = np.random.choice(self.transaction_types, p=self.transaction_type_weights)
        
        start_date = datetime(2023, 1, 1)
        transaction_datetime = self.generate_transaction_datetime(start_date, 365)
        
        return {
            'Transaction_ID': f'TXN_{transaction_id:08d}',
            'Customer_ID': f'CUST_{customer_id:06d}',
            'Transaction_DateTime': transaction_datetime,
            'Transaction_Amount': amount,
            'Merchant_Category': category,
            'Location_Country': country,
            'Transaction_Type': transaction_type,
            'transaction_pattern': 'normal'
        }
    
    def generate_suspicious_transaction(self, customer_id, customer_profile, transaction_id):
        """Generate a transaction with suspicious patterns"""
        fraud_patterns = ['large_amount', 'unusual_location', 'unusual_category', 'unusual_time', 'multiple_small']
        pattern = np.random.choice(fraud_patterns)
        
        transaction = self.generate_normal_transaction(customer_id, customer_profile, transaction_id)
        transaction['transaction_pattern'] = 'suspicious'
        
        if pattern == 'large_amount':
            # Unusually large transaction (5-50x normal)
            multiplier = np.random.uniform(5, 50)
            transaction['Transaction_Amount'] = round(customer_profile['avg_transaction_amount'] * multiplier, 2)
            transaction['Merchant_Category'] = np.random.choice(['Online Shopping', 'Electronics', 'Transfer'])
        
        elif pattern == 'unusual_location':
            transaction['Location_Country'] = np.random.choice(self.high_risk_countries)
            transaction['Transaction_Type'] = 'Online'
        
        elif pattern == 'unusual_category':
            unusual_categories = ['Electronics', 'Travel', 'Hotel', 'Online Shopping']
            transaction['Merchant_Category'] = np.random.choice(unusual_categories)
            transaction['Transaction_Amount'] = round(customer_profile['avg_transaction_amount'] * np.random.uniform(3, 15), 2)
        
        elif pattern == 'unusual_time':
            base_time = transaction['Transaction_DateTime']
            unusual_hour = np.random.choice([1, 2, 3, 4, 5])
            transaction['Transaction_DateTime'] = base_time.replace(hour=unusual_hour)
            transaction['Transaction_Type'] = 'Online'
        
        elif pattern == 'multiple_small':
            transaction['Transaction_Amount'] = round(np.random.uniform(1, 50), 2)
            transaction['Merchant_Category'] = 'Online Shopping'
            transaction['Transaction_Type'] = 'Online'
        
        return transaction
    
    def generate_raw_dataset(self):
        """Generate the raw transaction dataset"""
        print("Generating transactions...")
        
        # Generate customer profiles
        customer_profiles = self.generate_customer_profiles()
        
        # Generate transactions
        transactions = []
        suspicious_count = int(self.n_transactions * self.fraud_rate)
        normal_count = self.n_transactions - suspicious_count
        
        # Generate normal transactions
        for i in range(normal_count):
            customer_id = np.random.randint(1, self.n_customers + 1)
            transaction = self.generate_normal_transaction(customer_id, customer_profiles[customer_id], i + 1)
            transactions.append(transaction)
        
        # Generate suspicious transactions
        for i in range(suspicious_count):
            customer_id = np.random.randint(1, self.n_customers + 1)
            transaction = self.generate_suspicious_transaction(customer_id, customer_profiles[customer_id], normal_count + i + 1)
            transactions.append(transaction)
        
        # Create DataFrame
        df = pd.DataFrame(transactions)
        df = df.sort_values(['Customer_ID', 'Transaction_DateTime']).reset_index(drop=True)
        
        print(f"Raw dataset generated: {len(df):,} transactions")
        return df
    
    def clean_data(self, df):
        """Comprehensive data cleaning pipeline"""
        print("\n=== DATA CLEANING PIPELINE ===")
        df_cleaned = df.copy()
        
        # 1. Fix data types
        print("1. Fixing data types...")
        df_cleaned['Transaction_DateTime'] = pd.to_datetime(df_cleaned['Transaction_DateTime'])
        df_cleaned['Transaction_Amount'] = pd.to_numeric(df_cleaned['Transaction_Amount'], errors='coerce')
        
        # Clean string columns
        string_columns = ['Transaction_ID', 'Customer_ID', 'Merchant_Category', 'Location_Country', 'Transaction_Type']
        for col in string_columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
        
        # 2. Handle missing values
        print("2. Handling missing values...")
        missing_before = df_cleaned.isnull().sum().sum()
        
        # Fill missing transaction amounts with customer median
        df_cleaned['Transaction_Amount'] = df_cleaned.groupby('Customer_ID')['Transaction_Amount'].transform(
            lambda x: x.fillna(x.median())
        )
        df_cleaned['Transaction_Amount'].fillna(df_cleaned['Transaction_Amount'].median(), inplace=True)
        
        # Fill categorical missing values with mode
        categorical_cols = ['Merchant_Category', 'Location_Country', 'Transaction_Type']
        for col in categorical_cols:
            mode_value = df_cleaned[col].mode().iloc[0] if not df_cleaned[col].mode().empty else 'Unknown'
            df_cleaned[col].fillna(mode_value, inplace=True)
        
        missing_after = df_cleaned.isnull().sum().sum()
        print(f"   Missing values: {missing_before} → {missing_after}")
        
        # 3. Remove duplicates
        print("3. Removing duplicates...")
        duplicates_before = df_cleaned.duplicated().sum()
        df_cleaned = df_cleaned.drop_duplicates()
        duplicates_after = df_cleaned.duplicated().sum()
        print(f"   Duplicates: {duplicates_before} → {duplicates_after}")
        
        # 4. Validate data consistency
        print("4. Validating data consistency...")
        
        # Fix negative amounts
        negative_amounts = (df_cleaned['Transaction_Amount'] <= 0).sum()
        if negative_amounts > 0:
            df_cleaned['Transaction_Amount'] = df_cleaned['Transaction_Amount'].clip(lower=0.01)
            print(f"   Fixed {negative_amounts} non-positive amounts")
        
        # 5. Handle outliers using IQR method
        print("5. Handling outliers...")
        Q1 = df_cleaned['Transaction_Amount'].quantile(0.25)
        Q3 = df_cleaned['Transaction_Amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_cleaned[(df_cleaned['Transaction_Amount'] < lower_bound) | 
                             (df_cleaned['Transaction_Amount'] > upper_bound)]
        print(f"   Outliers found: {len(outliers)} ({len(outliers)/len(df_cleaned)*100:.2f}%)")
        
        # Cap outliers instead of removing
        df_cleaned['Transaction_Amount'] = df_cleaned['Transaction_Amount'].clip(lower=lower_bound, upper=upper_bound)
        
        print("✓ Data cleaning completed")
        return df_cleaned
    
    def engineer_features(self, df):
        """Comprehensive feature engineering pipeline"""
        print("\n=== FEATURE ENGINEERING PIPELINE ===")
        df_features = df.copy()
        
        # 1. Basic time-based features
        print("1. Creating time-based features...")
        df_features['Transaction_Hour'] = df_features['Transaction_DateTime'].dt.hour
        df_features['Transaction_Day_of_Week'] = df_features['Transaction_DateTime'].dt.dayofweek
        df_features['Transaction_Month'] = df_features['Transaction_DateTime'].dt.month
        df_features['Is_Weekend'] = (df_features['Transaction_Day_of_Week'] >= 5).astype(int)
        df_features['Is_Night'] = ((df_features['Transaction_Hour'] <= 6) | 
                                  (df_features['Transaction_Hour'] >= 23)).astype(int)
        
        # 2. Customer aggregation features
        print("2. Creating customer aggregation features...")
        customer_stats = df_features.groupby('Customer_ID').agg({
            'Transaction_Amount': ['mean', 'std', 'count', 'min', 'max'],
            'Location_Country': lambda x: x.nunique(),
            'Merchant_Category': lambda x: x.nunique(),
            'Transaction_Type': lambda x: x.nunique()
        }).round(2)
        
        customer_stats.columns = ['Avg_Amount', 'Std_Amount', 'Transaction_Count', 
                                 'Min_Amount', 'Max_Amount', 'Unique_Countries', 
                                 'Unique_Categories', 'Unique_Transaction_Types']
        customer_stats = customer_stats.reset_index()
        
        df_features = df_features.merge(customer_stats, on='Customer_ID', how='left')
        
        # 3. Amount deviation features
        print("3. Creating amount deviation features...")
        df_features['Amount_Deviation'] = abs(df_features['Transaction_Amount'] - df_features['Avg_Amount']) / (df_features['Std_Amount'] + 0.01)
        df_features['Amount_Ratio_to_Avg'] = df_features['Transaction_Amount'] / df_features['Avg_Amount']
        df_features['Amount_Ratio_to_Max'] = df_features['Transaction_Amount'] / df_features['Max_Amount']
        
        # 4. Transaction velocity features
        print("4. Creating transaction velocity features...")
        df_features = df_features.sort_values(['Customer_ID', 'Transaction_DateTime']).reset_index(drop=True)
        
        velocity_24h = []
        velocity_1h = []
        
        for idx, row in df_features.iterrows():
            current_time = row['Transaction_DateTime']
            customer_id = row['Customer_ID']
            
            customer_txns = df_features[df_features['Customer_ID'] == customer_id].copy()
            
            # 24-hour velocity
            time_window_24h = current_time - timedelta(hours=24)
            recent_txns_24h = customer_txns[
                (customer_txns['Transaction_DateTime'] >= time_window_24h) & 
                (customer_txns['Transaction_DateTime'] < current_time)
            ]
            velocity_24h.append(len(recent_txns_24h))
            
            # 1-hour velocity
            time_window_1h = current_time - timedelta(hours=1)
            recent_txns_1h = customer_txns[
                (customer_txns['Transaction_DateTime'] >= time_window_1h) & 
                (customer_txns['Transaction_DateTime'] < current_time)
            ]
            velocity_1h.append(len(recent_txns_1h))
            
            if idx % 2000 == 0:
                print(f"   Processed {idx:,}/{len(df_features):,} transactions for velocity")
        
        df_features['Transaction_Velocity_24h'] = velocity_24h
        df_features['Transaction_Velocity_1h'] = velocity_1h
        
        # 5. Category and location risk features
        print("5. Creating risk-based features...")
        
        # High-risk country flag
        df_features['High_Risk_Country'] = df_features['Location_Country'].isin(self.high_risk_countries).astype(int)
        
        # Unusual spending category (customer-specific)
        customer_categories = df_features.groupby('Customer_ID')['Merchant_Category'].apply(
            lambda x: x.value_counts(normalize=True).to_dict()
        ).to_dict()
        
        unusual_category_flags = []
        for idx, row in df_features.iterrows():
            customer_id = row['Customer_ID']
            category = row['Merchant_Category']
            
            if customer_id in customer_categories:
                category_freq = customer_categories[customer_id].get(category, 0)
            else:
                category_freq = 0
            
            is_unusual = category_freq < 0.1  # Less than 10% of customer's transactions
            unusual_category_flags.append(1 if is_unusual else 0)
        
        df_features['Unusual_Spending_Category'] = unusual_category_flags
        
        # 6. Category spending spike
        print("6. Creating category spending spike features...")
        category_stats = df_features.groupby('Merchant_Category')['Transaction_Amount'].agg(['mean', 'std']).to_dict('index')
        
        spending_spike_flags = []
        for idx, row in df_features.iterrows():
            category = row['Merchant_Category']
            amount = row['Transaction_Amount']
            
            if category in category_stats:
                cat_mean = category_stats[category]['mean']
                cat_std = category_stats[category]['std']
                
                if cat_std > 0:
                    z_score = (amount - cat_mean) / cat_std
                    is_spike = (z_score > 3) or (amount > cat_mean * 5)
                else:
                    is_spike = amount > cat_mean * 5
            else:
                is_spike = amount > 1000
            
            spending_spike_flags.append(1 if is_spike else 0)
        
        df_features['Category_Spending_Spike'] = spending_spike_flags
        
        # 7. Micro-transaction patterns
        print("7. Creating micro-transaction pattern features...")
        micro_transaction_flags = []
        
        for idx, row in df_features.iterrows():
            current_amount = row['Transaction_Amount']
            current_time = row['Transaction_DateTime']
            customer_id = row['Customer_ID']
            
            if current_amount <= 200:
                micro_transaction_flags.append(0)
                continue
            
            customer_txns = df_features[df_features['Customer_ID'] == customer_id].copy()
            time_window_start = current_time - timedelta(hours=2)
            
            recent_micro_txns = customer_txns[
                (customer_txns['Transaction_DateTime'] >= time_window_start) & 
                (customer_txns['Transaction_DateTime'] < current_time) &
                (customer_txns['Transaction_Amount'] < 10)
            ]
            
            has_micro_pattern = len(recent_micro_txns) >= 2
            micro_transaction_flags.append(1 if has_micro_pattern else 0)
        
        df_features['Micro_Txns_Before_Large'] = micro_transaction_flags
        
        # 8. Composite risk score
        print("8. Creating composite risk score...")
        risk_features = ['Transaction_Velocity_24h', 'Unusual_Spending_Category', 
                        'Category_Spending_Spike', 'High_Risk_Country', 'Micro_Txns_Before_Large']
        
        # Normalize velocity to 0-1 scale
        max_velocity = df_features['Transaction_Velocity_24h'].max()
        velocity_normalized = df_features['Transaction_Velocity_24h'] / max_velocity if max_velocity > 0 else 0
        
        # Calculate composite risk score
        df_features['Risk_Score'] = (
            velocity_normalized * 0.3 +
            df_features['Unusual_Spending_Category'] * 0.2 +
            df_features['Category_Spending_Spike'] * 0.2 +
            df_features['High_Risk_Country'] * 0.2 +
            df_features['Micro_Txns_Before_Large'] * 0.1
        ).round(3)
        
        # Remove internal tracking column
        df_features = df_features.drop('transaction_pattern', axis=1, errors='ignore')
        
        print("✓ Feature engineering completed")
        print(f"Final dataset shape: {df_features.shape}")
        
        return df_features
    
    def generate_summary_report(self, df):
        """Generate comprehensive dataset summary report"""
        print("\n" + "="*60)
        print("DATASET SUMMARY REPORT")
        print("="*60)
        
        print(f"\n📊 BASIC STATISTICS")
        print(f"Total transactions: {len(df):,}")
        print(f"Unique customers: {df['Customer_ID'].nunique():,}")
        print(f"Date range: {df['Transaction_DateTime'].min()} to {df['Transaction_DateTime'].max()}")
        print(f"Avg transactions per customer: {len(df)/df['Customer_ID'].nunique():.1f}")
        print(f"Dataset shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print(f"\n💰 TRANSACTION AMOUNTS")
        print(f"Total transaction value: ${df['Transaction_Amount'].sum():,.2f}")
        print(f"Average transaction: ${df['Transaction_Amount'].mean():.2f}")
        print(f"Median transaction: ${df['Transaction_Amount'].median():.2f}")
        print(f"Min transaction: ${df['Transaction_Amount'].min():.2f}")
        print(f"Max transaction: ${df['Transaction_Amount'].max():.2f}")
        print(f"Standard deviation: ${df['Transaction_Amount'].std():.2f}")
        
        print(f"\n🛍️ MERCHANT CATEGORIES")
        category_dist = df['Merchant_Category'].value_counts().head(10)
        for category, count in category_dist.items():
            print(f"  {category}: {count:,} ({count/len(df)*100:.1f}%)")
        
        print(f"\n🌍 LOCATION DISTRIBUTION")
        country_dist = df['Location_Country'].value_counts().head(10)
        for country, count in country_dist.items():
            print(f"  {country}: {count:,} ({count/len(df)*100:.1f}%)")
        
        print(f"\n🔒 RISK INDICATORS")
        print(f"High-risk country transactions: {df['High_Risk_Country'].sum():,} ({df['High_Risk_Country'].mean()*100:.1f}%)")
        print(f"Unusual category spending: {df['Unusual_Spending_Category'].sum():,} ({df['Unusual_Spending_Category'].mean()*100:.1f}%)")
        print(f"Category spending spikes: {df['Category_Spending_Spike'].sum():,} ({df['Category_Spending_Spike'].mean()*100:.1f}%)")
        print(f"Micro-transaction patterns: {df['Micro_Txns_Before_Large'].sum():,} ({df['Micro_Txns_Before_Large'].mean()*100:.1f}%)")
        
        print(f"\n⚡ TRANSACTION VELOCITY")
        print(f"Average 24h velocity: {df['Transaction_Velocity_24h'].mean():.2f}")
        print(f"Max 24h velocity: {df['Transaction_Velocity_24h'].max()}")
        print(f"Customers with >5 transactions/24h: {len(df[df['Transaction_Velocity_24h'] > 5]):,}")
        
        print(f"\n📈 RISK SCORE DISTRIBUTION")
        print(f"Average risk score: {df['Risk_Score'].mean():.3f}")
        print(f"High-risk transactions (score >0.5): {len(df[df['Risk_Score'] > 0.5]):,} ({len(df[df['Risk_Score'] > 0.5])/len(df)*100:.1f}%)")
        print(f"Very high-risk transactions (score >0.8): {len(df[df['Risk_Score'] > 0.8]):,} ({len(df[df['Risk_Score'] > 0.8])/len(df)*100:.1f}%)")
        
        # Show sample of highest risk transactions
        print(f"\n⚠️ HIGHEST RISK TRANSACTIONS (Top 10)")
        high_risk = df.nlargest(10, 'Risk_Score')[['Transaction_ID', 'Customer_ID', 'Transaction_Amount', 
                                                   'Merchant_Category', 'Location_Country', 'Risk_Score']]
        print(high_risk.to_string(index=False))
        
        print("\n" + "="*60)
    
    def run_full_pipeline(self, output_filename='cleaned_transactions.csv'):
        """Execute the complete data engineering pipeline"""
        print("🚀 STARTING BANK DATASET ENGINEERING PIPELINE")
        print("="*60)
        
        start_time = datetime.now()
        
        # Step 1: Generate raw dataset
        raw_df = self.generate_raw_dataset()
        
        # Step 2: Clean the data
        cleaned_df = self.clean_data(raw_df)
        
        # Step 3: Engineer features
        final_df = self.engineer_features(cleaned_df)
        
        # Step 4: Generate summary report
        self.generate_summary_report(final_df)
        
        # Step 5: Save the final dataset
        final_df.to_csv(output_filename, index=False)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        print(f"\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Final dataset saved as: '{output_filename}'")
        print(f"Dataset ready for analysis and modeling!")
        
        return final_df


# Main execution
if __name__ == "__main__":
    # Initialize and run the data engineering pipeline
    generator = BankDatasetGenerator(
        n_transactions=12000,
        n_customers=2000,
        fraud_rate=0.08,
        random_seed=42
    )
    
    # Run the complete pipeline
    final_dataset = generator.run_full_pipeline('cleaned_transactions.csv')
    
    print(f"\n🎯 Dataset is ready for machine learning and analysis!")
    print(f"Features included: {list(final_dataset.columns)}")

