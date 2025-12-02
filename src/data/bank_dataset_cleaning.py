import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_inspect_data(filename='bank_transactions_dataset_grouped.csv'):
    """Load the dataset and perform initial inspection"""
    try:
        df = pd.read_csv(filename)
        print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"File {filename} not found. Please ensure the file exists in the current directory.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def inspect_data_quality(df):
    """Comprehensive data quality inspection"""
    print("\n=== DATA QUALITY INSPECTION ===")
    
    # Basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values
    print("\n--- Missing Values ---")
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_percentages
    }).round(2)
    print(missing_summary[missing_summary['Missing_Count'] > 0])
    
    # Data types
    print("\n--- Data Types ---")
    print(df.dtypes)
    
    # Duplicates
    print(f"\n--- Duplicates ---")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Duplicate Transaction_IDs: {df['Transaction_ID'].duplicated().sum()}")
    
    # Unique values in categorical columns
    categorical_cols = ['Customer_ID', 'Merchant_Category', 'Location_Country', 'Transaction_Type']
    print("\n--- Categorical Column Summary ---")
    for col in categorical_cols:
        if col in df.columns:
            print(f"{col}: {df[col].nunique()} unique values")
    
    return missing_summary

def clean_data_types(df):
    """Fix data type inconsistencies"""
    print("\n=== CLEANING DATA TYPES ===")
    df_cleaned = df.copy()
    
    # Convert Transaction_DateTime to datetime
    if 'Transaction_DateTime' in df_cleaned.columns:
        try:
            df_cleaned['Transaction_DateTime'] = pd.to_datetime(df_cleaned['Transaction_DateTime'])
            print("✓ Transaction_DateTime converted to datetime")
        except Exception as e:
            print(f"✗ Error converting Transaction_DateTime: {str(e)}")
    
    # Ensure numeric columns are properly typed
    numeric_columns = [
        'Transaction_Amount', 'Transaction_Hour', 'Transaction_Day_of_Week', 
        'Transaction_Month', 'Avg_Amount', 'Std_Amount', 'Transaction_Count', 
        'Unique_Countries', 'Unique_Categories', 'Amount_Deviation'
    ]
    
    for col in numeric_columns:
        if col in df_cleaned.columns:
            try:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                print(f"✓ {col} converted to numeric")
            except Exception as e:
                print(f"✗ Error converting {col}: {str(e)}")
    
    # Ensure string columns are properly typed
    string_columns = ['Transaction_ID', 'Customer_ID', 'Merchant_Category', 'Location_Country', 'Transaction_Type']
    for col in string_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            print(f"✓ {col} cleaned as string")
    
    return df_cleaned

def handle_missing_values(df):
    """Handle missing values with appropriate strategies"""
    print("\n=== HANDLING MISSING VALUES ===")
    df_cleaned = df.copy()
    
    # Check for missing values
    missing_summary = df_cleaned.isnull().sum()
    
    if missing_summary.sum() == 0:
        print("✓ No missing values found")
        return df_cleaned
    
    # Handle missing values by column type
    for col in df_cleaned.columns:
        missing_count = df_cleaned[col].isnull().sum()
        if missing_count > 0:
            print(f"\nHandling {missing_count} missing values in {col}")
            
            if col == 'Transaction_Amount':
                # For transaction amounts, use median of customer's other transactions
                df_cleaned[col] = df_cleaned.groupby('Customer_ID')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # If still missing, use overall median
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                print(f"✓ {col}: Filled with customer median, then overall median")
            
            elif col in ['Avg_Amount', 'Std_Amount']:
                # Recalculate these based on available data
                customer_stats = df_cleaned.groupby('Customer_ID')['Transaction_Amount'].agg(['mean', 'std'])
                df_cleaned['Avg_Amount'] = df_cleaned['Customer_ID'].map(customer_stats['mean'])
                df_cleaned['Std_Amount'] = df_cleaned['Customer_ID'].map(customer_stats['std']).fillna(0)
                print(f"✓ {col}: Recalculated from transaction data")
            
            elif col in ['Transaction_Count', 'Unique_Countries', 'Unique_Categories']:
                # Recalculate based on available data
                if col == 'Transaction_Count':
                    customer_counts = df_cleaned.groupby('Customer_ID').size()
                    df_cleaned[col] = df_cleaned['Customer_ID'].map(customer_counts)
                elif col == 'Unique_Countries':
                    country_counts = df_cleaned.groupby('Customer_ID')['Location_Country'].nunique()
                    df_cleaned[col] = df_cleaned['Customer_ID'].map(country_counts)
                elif col == 'Unique_Categories':
                    category_counts = df_cleaned.groupby('Customer_ID')['Merchant_Category'].nunique()
                    df_cleaned[col] = df_cleaned['Customer_ID'].map(category_counts)
                print(f"✓ {col}: Recalculated from available data")
            
            elif col == 'Amount_Deviation':
                # Recalculate amount deviation
                df_cleaned['Amount_Deviation'] = abs(
                    df_cleaned['Transaction_Amount'] - df_cleaned['Avg_Amount']
                ) / (df_cleaned['Std_Amount'] + 0.01)
                print(f"✓ {col}: Recalculated")
            
            elif col in ['Merchant_Category', 'Location_Country', 'Transaction_Type']:
                # Use mode (most frequent value)
                mode_value = df_cleaned[col].mode().iloc[0] if not df_cleaned[col].mode().empty else 'Unknown'
                df_cleaned[col].fillna(mode_value, inplace=True)
                print(f"✓ {col}: Filled with mode value: {mode_value}")
            
            else:
                # For other columns, use appropriate default values
                if df_cleaned[col].dtype in ['int64', 'float64']:
                    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                    print(f"✓ {col}: Filled with median")
                else:
                    df_cleaned[col].fillna('Unknown', inplace=True)
                    print(f"✓ {col}: Filled with 'Unknown'")
    
    return df_cleaned

def handle_duplicates(df):
    """Remove duplicate records"""
    print("\n=== HANDLING DUPLICATES ===")
    df_cleaned = df.copy()
    
    # Check for exact duplicates
    duplicate_rows = df_cleaned.duplicated().sum()
    if duplicate_rows > 0:
        df_cleaned = df_cleaned.drop_duplicates()
        print(f"✓ Removed {duplicate_rows} duplicate rows")
    
    # Check for duplicate Transaction_IDs
    duplicate_ids = df_cleaned['Transaction_ID'].duplicated().sum()
    if duplicate_ids > 0:
        print(f"⚠ Found {duplicate_ids} duplicate Transaction_IDs")
        # Keep first occurrence and remove duplicates
        df_cleaned = df_cleaned.drop_duplicates(subset=['Transaction_ID'], keep='first')
        print(f"✓ Kept first occurrence of duplicate Transaction_IDs")
    
    if duplicate_rows == 0 and duplicate_ids == 0:
        print("✓ No duplicates found")
    
    return df_cleaned

def validate_data_consistency(df):
    """Validate and fix data consistency issues"""
    print("\n=== VALIDATING DATA CONSISTENCY ===")
    df_cleaned = df.copy()
    
    # Fix Transaction_Hour range (0-23)
    if 'Transaction_Hour' in df_cleaned.columns:
        invalid_hours = df_cleaned[(df_cleaned['Transaction_Hour'] < 0) | (df_cleaned['Transaction_Hour'] > 23)]
        if len(invalid_hours) > 0:
            print(f"⚠ Found {len(invalid_hours)} invalid hours, capping to 0-23 range")
            df_cleaned['Transaction_Hour'] = df_cleaned['Transaction_Hour'].clip(0, 23)
        else:
            print("✓ Transaction hours are valid (0-23)")
    
    # Fix Transaction_Day_of_Week range (0-6)
    if 'Transaction_Day_of_Week' in df_cleaned.columns:
        invalid_days = df_cleaned[(df_cleaned['Transaction_Day_of_Week'] < 0) | (df_cleaned['Transaction_Day_of_Week'] > 6)]
        if len(invalid_days) > 0:
            print(f"⚠ Found {len(invalid_days)} invalid days, capping to 0-6 range")
            df_cleaned['Transaction_Day_of_Week'] = df_cleaned['Transaction_Day_of_Week'].clip(0, 6)
        else:
            print("✓ Transaction days are valid (0-6)")
    
    # Fix Transaction_Month range (1-12)
    if 'Transaction_Month' in df_cleaned.columns:
        invalid_months = df_cleaned[(df_cleaned['Transaction_Month'] < 1) | (df_cleaned['Transaction_Month'] > 12)]
        if len(invalid_months) > 0:
            print(f"⚠ Found {len(invalid_months)} invalid months, capping to 1-12 range")
            df_cleaned['Transaction_Month'] = df_cleaned['Transaction_Month'].clip(1, 12)
        else:
            print("✓ Transaction months are valid (1-12)")
    
    # Validate Transaction_Amount (should be positive)
    if 'Transaction_Amount' in df_cleaned.columns:
        negative_amounts = df_cleaned[df_cleaned['Transaction_Amount'] <= 0]
        if len(negative_amounts) > 0:
            print(f"⚠ Found {len(negative_amounts)} non-positive transaction amounts")
            # Set minimum amount to $0.01
            df_cleaned['Transaction_Amount'] = df_cleaned['Transaction_Amount'].clip(lower=0.01)
            print("✓ Set minimum transaction amount to $0.01")
        else:
            print("✓ All transaction amounts are positive")
    
    # Validate Customer_ID and Transaction_ID formats
    if 'Customer_ID' in df_cleaned.columns:
        invalid_cust_ids = df_cleaned[~df_cleaned['Customer_ID'].str.match(r'^CUST_\d{6}$', na=False)]
        if len(invalid_cust_ids) > 0:
            print(f"⚠ Found {len(invalid_cust_ids)} invalid Customer_ID formats")
            # Attempt to fix format
            df_cleaned['Customer_ID'] = df_cleaned['Customer_ID'].apply(
                lambda x: f"CUST_{str(x).split('_')[-1].zfill(6)}" if '_' in str(x) else f"CUST_{str(x).zfill(6)}"
            )
            print("✓ Standardized Customer_ID format")
        else:
            print("✓ Customer_ID format is consistent")
    
    if 'Transaction_ID' in df_cleaned.columns:
        invalid_txn_ids = df_cleaned[~df_cleaned['Transaction_ID'].str.match(r'^TXN_\d{8}$', na=False)]
        if len(invalid_txn_ids) > 0:
            print(f"⚠ Found {len(invalid_txn_ids)} invalid Transaction_ID formats")
            # Attempt to fix format
            df_cleaned['Transaction_ID'] = df_cleaned['Transaction_ID'].apply(
                lambda x: f"TXN_{str(x).split('_')[-1].zfill(8)}" if '_' in str(x) else f"TXN_{str(x).zfill(8)}"
            )
            print("✓ Standardized Transaction_ID format")
        else:
            print("✓ Transaction_ID format is consistent")
    
    return df_cleaned

def handle_outliers(df, method='iqr'):
    """Handle outliers in numerical columns"""
    print(f"\n=== HANDLING OUTLIERS (Method: {method.upper()}) ===")
    df_cleaned = df.copy()
    
    numerical_cols = ['Transaction_Amount', 'Amount_Deviation']
    
    for col in numerical_cols:
        if col not in df_cleaned.columns:
            continue
            
        original_count = len(df_cleaned)
        
        if method == 'iqr':
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                print(f"\n{col}:")
                print(f"  Outliers found: {outlier_count} ({outlier_count/original_count*100:.2f}%)")
                print(f"  Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
                
                # Cap outliers instead of removing them
                df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"  ✓ Outliers capped to valid range")
            else:
                print(f"✓ {col}: No outliers found")
        
        elif method == 'zscore':
            z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
            outliers = df_cleaned[z_scores > 3]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                print(f"\n{col}:")
                print(f"  Outliers found: {outlier_count} ({outlier_count/original_count*100:.2f}%)")
                # Cap at 3 standard deviations
                mean_val = df_cleaned[col].mean()
                std_val = df_cleaned[col].std()
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"  ✓ Outliers capped to ±3 standard deviations")
            else:
                print(f"✓ {col}: No outliers found")
    
    return df_cleaned

def recalculate_derived_features(df):
    """Recalculate derived features to ensure consistency"""
    print("\n=== RECALCULATING DERIVED FEATURES ===")
    df_cleaned = df.copy()
    
    # Recalculate customer statistics
    customer_stats = df_cleaned.groupby('Customer_ID').agg({
        'Transaction_Amount': ['mean', 'std', 'count'],
        'Location_Country': lambda x: x.nunique(),
        'Merchant_Category': lambda x: x.nunique()
    }).round(2)
    
    customer_stats.columns = ['Avg_Amount', 'Std_Amount', 'Transaction_Count', 'Unique_Countries', 'Unique_Categories']
    customer_stats = customer_stats.reset_index()
    
    # Merge back to main dataframe
    df_cleaned = df_cleaned.drop(['Avg_Amount', 'Std_Amount', 'Transaction_Count', 'Unique_Countries', 'Unique_Categories'], axis=1, errors='ignore')
    df_cleaned = df_cleaned.merge(customer_stats, on='Customer_ID', how='left')
    
    # Recalculate Amount_Deviation
    df_cleaned['Amount_Deviation'] = abs(df_cleaned['Transaction_Amount'] - df_cleaned['Avg_Amount']) / (df_cleaned['Std_Amount'] + 0.01)
    
    # Recalculate time-based features if Transaction_DateTime exists
    if 'Transaction_DateTime' in df_cleaned.columns:
        df_cleaned['Transaction_Hour'] = df_cleaned['Transaction_DateTime'].dt.hour
        df_cleaned['Transaction_Day_of_Week'] = df_cleaned['Transaction_DateTime'].dt.dayofweek
        df_cleaned['Transaction_Month'] = df_cleaned['Transaction_DateTime'].dt.month
    
    print("✓ All derived features recalculated")
    return df_cleaned

def final_data_validation(df):
    """Perform final validation of the cleaned dataset"""
    print("\n=== FINAL DATA VALIDATION ===")
    
    # Check for remaining issues
    issues = []
    
    # Missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        issues.append(f"Still has {missing_values} missing values")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Still has {duplicates} duplicate rows")
    
    # Data types
    if 'Transaction_DateTime' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Transaction_DateTime']):
            issues.append("Transaction_DateTime is not datetime type")
    
    # Negative amounts
    if 'Transaction_Amount' in df.columns:
        negative_amounts = (df['Transaction_Amount'] <= 0).sum()
        if negative_amounts > 0:
            issues.append(f"Has {negative_amounts} non-positive transaction amounts")
    
    # Report results
    if issues:
        print("⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All validations passed!")
    
    # Summary statistics
    print(f"\n--- Final Dataset Summary ---")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['Transaction_DateTime'].min()} to {df['Transaction_DateTime'].max()}")
    print(f"Unique customers: {df['Customer_ID'].nunique():,}")
    print(f"Unique transactions: {df['Transaction_ID'].nunique():,}")
    print(f"Amount range: ${df['Transaction_Amount'].min():.2f} - ${df['Transaction_Amount'].max():.2f}")
    print(f"Average transaction: ${df['Transaction_Amount'].mean():.2f}")
    
    return len(issues) == 0

def main():
    """Main data cleaning pipeline"""
    print("=== BANK DATASET CLEANING PIPELINE ===")
    
    # Step 1: Load and inspect data
    df = load_and_inspect_data()
    if df is None:
        return
    
    # Step 2: Initial data quality inspection
    inspect_data_quality(df)
    
    # Step 3: Clean data types
    df = clean_data_types(df)
    
    # Step 4: Handle missing values
    df = handle_missing_values(df)
    
    # Step 5: Handle duplicates
    df = handle_duplicates(df)
    
    # Step 6: Validate data consistency
    df = validate_data_consistency(df)
    
    # Step 7: Handle outliers
    df = handle_outliers(df, method='iqr')
    
    # Step 8: Recalculate derived features
    df = recalculate_derived_features(df)
    
    # Step 9: Final validation
    is_clean = final_data_validation(df)
    
    # Step 10: Save cleaned dataset
    output_filename = 'bank_transactions_dataset_cleaned.csv'
    df.to_csv(output_filename, index=False)
    print(f"\n✓ Cleaned dataset saved as '{output_filename}'")
    
    if is_clean:
        print("\n🎉 Dataset cleaning completed successfully!")
    else:
        print("\n⚠ Dataset cleaning completed with some remaining issues. Please review the output.")
    
    return df

# Execute the cleaning pipeline
if __name__ == "__main__":
    cleaned_df = main()
