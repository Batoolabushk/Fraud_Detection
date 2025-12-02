import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    def __init__(self, contamination=0.08, random_state=42):
        """
        Initialize the fraud detection system.
        
        Parameters:
        contamination (float): Expected proportion of outliers (fraud rate)
        random_state (int): Random state for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.is_fitted = False
        
    def load_and_prepare_data(self, filepath):
        """
        Load and prepare the transaction data for modeling.
        
        Parameters:
        filepath (str): Path to the CSV file
        
        Returns:
        pd.DataFrame: Processed dataframe
        """
        try:
            print("Loading transaction data...")
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df):,} transactions")
            
            # Convert datetime column
            df['Transaction_DateTime'] = pd.to_datetime(df['Transaction_DateTime'])
            
            # Create additional time-based features
            df['Hour_Sin'] = np.sin(2 * np.pi * df['Transaction_Hour'] / 24)
            df['Hour_Cos'] = np.cos(2 * np.pi * df['Transaction_Hour'] / 24)
            df['Day_Sin'] = np.sin(2 * np.pi * df['Transaction_Day_of_Week'] / 7)
            df['Day_Cos'] = np.cos(2 * np.pi * df['Transaction_Day_of_Week'] / 7)
            df['Month_Sin'] = np.sin(2 * np.pi * df['Transaction_Month'] / 12)
            df['Month_Cos'] = np.cos(2 * np.pi * df['Transaction_Month'] / 12)
            
            # Log transform amount features (add small constant to avoid log(0))
            df['Log_Transaction_Amount'] = np.log1p(df['Transaction_Amount'])
            df['Log_Avg_Amount'] = np.log1p(df['Avg_Amount'])
            
            # Create ratio features
            df['Amount_to_Avg_Ratio'] = df['Transaction_Amount'] / (df['Avg_Amount'] + 0.01)
            df['Velocity_per_Transaction'] = df['Transaction_Velocity_24h'] / (df['Transaction_Count'] + 1)
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def engineer_features(self, df):
        """
        Engineer and select features for the model.
        
        Parameters:
        df (pd.DataFrame): Input dataframe
        
        Returns:
        pd.DataFrame: Feature matrix ready for modeling
        """
        try:
            print("Engineering features...")
            
            # Select numerical features
            numerical_features = [
                'Transaction_Amount', 'Log_Transaction_Amount', 'Log_Avg_Amount',
                'Std_Amount', 'Transaction_Count', 'Unique_Countries', 'Unique_Categories',
                'Amount_Deviation', 'Amount_to_Avg_Ratio', 'Transaction_Velocity_24h',
                'Velocity_per_Transaction', 'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
                'Month_Sin', 'Month_Cos'
            ]
            
            # Select categorical features to encode
            categorical_features = ['Merchant_Category', 'Location_Country', 'Transaction_Type']
            
            # Select binary fraud indicator features
            fraud_features = [
                'Unusual_Spending_Category', 'Category_Spending_Spike',
                'High_Risk_Country', 'Micro_Txns_Before_Large'
            ]
            
            # Start with numerical and fraud features
            feature_df = df[numerical_features + fraud_features].copy()
            
            # Handle missing values
            feature_df = feature_df.fillna(feature_df.median())
            
            # Encode categorical features
            for cat_feature in categorical_features:
                if cat_feature not in self.label_encoders:
                    self.label_encoders[cat_feature] = LabelEncoder()
                    feature_df[f'{cat_feature}_encoded'] = self.label_encoders[cat_feature].fit_transform(df[cat_feature].astype(str))
                else:
                    # Handle unseen categories during prediction
                    try:
                        feature_df[f'{cat_feature}_encoded'] = self.label_encoders[cat_feature].transform(df[cat_feature].astype(str))
                    except ValueError:
                        # Assign a default value for unseen categories
                        feature_df[f'{cat_feature}_encoded'] = 0
            
            # Store feature names for later use
            self.feature_names = feature_df.columns.tolist()
            
            print(f"Engineered {len(self.feature_names)} features")
            return feature_df
            
        except Exception as e:
            print(f"Error in feature engineering: {str(e)}")
            raise
    
    def train_model(self, X):
        """
        Train the Isolation Forest model.
        
        Parameters:
        X (pd.DataFrame): Feature matrix
        """
        try:
            print("Training Isolation Forest model...")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Initialize and train Isolation Forest
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=200,
                max_samples='auto',
                max_features=1.0,
                bootstrap=False,
                n_jobs=-1
            )
            
            # Fit the model
            self.model.fit(X_scaled)
            self.is_fitted = True
            
            print("Model training completed successfully!")
            
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            raise
    
    def predict_anomalies(self, X):
        """
        Predict anomalies using the trained model.
        
        Parameters:
        X (pd.DataFrame): Feature matrix
        
        Returns:
        tuple: (predictions, anomaly_scores)
        """
        try:
            if not self.is_fitted:
                raise ValueError("Model must be trained before making predictions")
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions (-1 for anomaly, 1 for normal)
            predictions = self.model.predict(X_scaled)
            
            # Get anomaly scores (lower scores indicate more anomalous)
            scores = self.model.decision_function(X_scaled)
            
            return predictions, scores
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise
    
    def analyze_results(self, df, predictions, scores):
        """
        Analyze and display the results of fraud detection.
        
        Parameters:
        df (pd.DataFrame): Original dataframe
        predictions (np.array): Model predictions
        scores (np.array): Anomaly scores
        """
        try:
            print("\n" + "="*60)
            print("FRAUD DETECTION ANALYSIS RESULTS")
            print("="*60)
            
            # Add predictions and scores to dataframe
            df_results = df.copy()
            df_results['Is_Anomaly'] = predictions == -1
            df_results['Anomaly_Score'] = scores
            df_results['Risk_Level'] = pd.cut(scores, 
                                            bins=[-np.inf, -0.1, 0, 0.1, np.inf],
                                            labels=['Very High', 'High', 'Medium', 'Low'])
            
            # Basic statistics
            n_anomalies = sum(predictions == -1)
            anomaly_rate = n_anomalies / len(predictions) * 100
            
            print(f"Total Transactions Analyzed: {len(predictions):,}")
            print(f"Flagged as Potentially Fraudulent: {n_anomalies:,} ({anomaly_rate:.2f}%)")
            print(f"Average Anomaly Score: {scores.mean():.4f}")
            print(f"Min Anomaly Score: {scores.min():.4f}")
            print(f"Max Anomaly Score: {scores.max():.4f}")
            
            # Risk level distribution
            print(f"\nRisk Level Distribution:")
            risk_dist = df_results['Risk_Level'].value_counts()
            for level, count in risk_dist.items():
                print(f"  {level}: {count:,} ({count/len(df_results)*100:.1f}%)")
            
            # Top anomalous transactions
            print(f"\nTop 10 Most Suspicious Transactions:")
            top_anomalies = df_results.nsmallest(10, 'Anomaly_Score')[
                ['Transaction_ID', 'Customer_ID', 'Transaction_Amount', 'Merchant_Category',
                 'Location_Country', 'Transaction_Velocity_24h', 'Anomaly_Score', 'Risk_Level']
            ]
            print(top_anomalies.to_string(index=False))
            
            # Fraud flag correlation analysis
            print(f"\nCorrelation with Existing Fraud Flags:")
            fraud_flags = ['Unusual_Spending_Category', 'Category_Spending_Spike',
                          'High_Risk_Country', 'Micro_Txns_Before_Large']
            
            for flag in fraud_flags:
                if flag in df_results.columns:
                    correlation = df_results['Is_Anomaly'].astype(int).corr(df_results[flag])
                    flag_in_anomalies = df_results[df_results['Is_Anomaly']][flag].sum()
                    total_flag = df_results[flag].sum()
                    overlap_rate = flag_in_anomalies / total_flag * 100 if total_flag > 0 else 0
                    print(f"  {flag}: Correlation = {correlation:.3f}, Overlap = {overlap_rate:.1f}%")
            
            # Customer analysis
            print(f"\nCustomer-Level Analysis:")
            customer_anomalies = df_results.groupby('Customer_ID').agg({
                'Is_Anomaly': ['count', 'sum'],
                'Anomaly_Score': 'mean',
                'Transaction_Amount': 'sum'
            })
            customer_anomalies.columns = ['Total_Txns', 'Anomalous_Txns', 'Avg_Risk_Score', 'Total_Amount']
            customer_anomalies['Anomaly_Rate'] = customer_anomalies['Anomalous_Txns'] / customer_anomalies['Total_Txns']
            
            high_risk_customers = customer_anomalies[customer_anomalies['Anomaly_Rate'] > 0.5]
            print(f"Customers with >50% anomalous transactions: {len(high_risk_customers)}")
            
            if len(high_risk_customers) > 0:
                print("Top 5 highest risk customers:")
                print(high_risk_customers.nlargest(5, 'Anomaly_Rate')[
                    ['Total_Txns', 'Anomalous_Txns', 'Anomaly_Rate', 'Avg_Risk_Score']
                ].round(3).to_string())
            
            return df_results
            
        except Exception as e:
            print(f"Error in result analysis: {str(e)}")
            raise
    
    def evaluate_model_performance(self, df_results):
        """
        Evaluate model performance using available metrics.
        """
        try:
            print(f"\n" + "="*60)
            print("MODEL PERFORMANCE EVALUATION")
            print("="*60)
            
            # Since we don't have true labels, we'll use proxy evaluation methods
            
            # 1. Feature importance analysis (using correlation with anomaly scores)
            print("Feature Importance Analysis (Correlation with Anomaly Scores):")
            feature_importance = {}
            
            # Calculate correlation between features and anomaly scores
            numeric_cols = df_results.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['Is_Anomaly', 'Anomaly_Score']]
            
            for feature in numeric_cols[:15]:  # Top 15 features
                if feature in df_results.columns:
                    corr = abs(df_results[feature].corr(df_results['Anomaly_Score']))
                    feature_importance[feature] = corr
            
            # Sort by importance
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            print("Top 10 Most Important Features:")
            for i, (feature, importance) in enumerate(sorted_importance[:10], 1):
                print(f"  {i:2d}. {feature}: {importance:.4f}")
            
            # 2. Distribution analysis
            print(f"\nAnomaly Score Distribution Analysis:")
            print(f"  Mean: {df_results['Anomaly_Score'].mean():.4f}")
            print(f"  Std:  {df_results['Anomaly_Score'].std():.4f}")
            print(f"  25th percentile: {df_results['Anomaly_Score'].quantile(0.25):.4f}")
            print(f"  50th percentile: {df_results['Anomaly_Score'].median():.4f}")
            print(f"  75th percentile: {df_results['Anomaly_Score'].quantile(0.75):.4f}")
            
            # 3. Business rule validation
            print(f"\nBusiness Rule Validation:")
            
            # High amount transactions
            high_amount_threshold = df_results['Transaction_Amount'].quantile(0.95)
            high_amount_anomaly_rate = df_results[df_results['Transaction_Amount'] > high_amount_threshold]['Is_Anomaly'].mean()
            print(f"  Anomaly rate for top 5% amounts: {high_amount_anomaly_rate:.2%}")
            
            # High velocity transactions
            high_velocity_anomaly_rate = df_results[df_results['Transaction_Velocity_24h'] > 5]['Is_Anomaly'].mean()
            print(f"  Anomaly rate for high velocity (>5 txns/24h): {high_velocity_anomaly_rate:.2%}")
            
            # High-risk countries
            if 'High_Risk_Country' in df_results.columns:
                high_risk_country_anomaly_rate = df_results[df_results['High_Risk_Country'] == 1]['Is_Anomaly'].mean()
                print(f"  Anomaly rate for high-risk countries: {high_risk_country_anomaly_rate:.2%}")
            
            return sorted_importance
            
        except Exception as e:
            print(f"Error in model evaluation: {str(e)}")
            raise
    
    def create_visualizations(self, df_results):
        """
        Create visualizations for fraud detection results.
        """
        try:
            print(f"\nCreating visualizations...")
            
            plt.style.use('default')
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Fraud Detection Analysis Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Anomaly Score Distribution
            axes[0, 0].hist(df_results['Anomaly_Score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(df_results['Anomaly_Score'].mean(), color='red', linestyle='--', label='Mean')
            axes[0, 0].set_title('Anomaly Score Distribution')
            axes[0, 0].set_xlabel('Anomaly Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            
            # 2. Transaction Amount vs Anomaly Score
            sample_data = df_results.sample(n=min(5000, len(df_results)))  # Sample for performance
            scatter = axes[0, 1].scatter(sample_data['Transaction_Amount'], sample_data['Anomaly_Score'], 
                                       c=sample_data['Is_Anomaly'], cmap='RdYlBu', alpha=0.6)
            axes[0, 1].set_title('Transaction Amount vs Anomaly Score')
            axes[0, 1].set_xlabel('Transaction Amount ($)')
            axes[0, 1].set_ylabel('Anomaly Score')
            axes[0, 1].set_xscale('log')
            plt.colorbar(scatter, ax=axes[0, 1])
            
            # 3. Anomalies by Transaction Hour
            hourly_anomalies = df_results.groupby('Transaction_Hour')['Is_Anomaly'].agg(['count', 'sum', 'mean'])
            axes[0, 2].bar(hourly_anomalies.index, hourly_anomalies['mean'], alpha=0.7, color='orange')
            axes[0, 2].set_title('Anomaly Rate by Hour of Day')
            axes[0, 2].set_xlabel('Hour')
            axes[0, 2].set_ylabel('Anomaly Rate')
            
            # 4. Top Merchant Categories with Anomalies
            category_anomalies = df_results.groupby('Merchant_Category')['Is_Anomaly'].agg(['count', 'sum', 'mean'])
            category_anomalies = category_anomalies.sort_values('mean', ascending=False).head(10)
            axes[1, 0].barh(range(len(category_anomalies)), category_anomalies['mean'], alpha=0.7, color='lightcoral')
            axes[1, 0].set_yticks(range(len(category_anomalies)))
            axes[1, 0].set_yticklabels(category_anomalies.index, fontsize=8)
            axes[1, 0].set_title('Top 10 Categories by Anomaly Rate')
            axes[1, 0].set_xlabel('Anomaly Rate')
            
            # 5. Risk Level Distribution
            risk_counts = df_results['Risk_Level'].value_counts()
            axes[1, 1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Risk Level Distribution')
            
            # 6. Fraud Flags vs Anomalies Heatmap
            fraud_flags = ['Unusual_Spending_Category', 'Category_Spending_Spike',
                          'High_Risk_Country', 'Micro_Txns_Before_Large']
            
            if all(flag in df_results.columns for flag in fraud_flags):
                fraud_corr_data = df_results[fraud_flags + ['Is_Anomaly']].astype(int).corr()
                sns.heatmap(fraud_corr_data, annot=True, cmap='coolwarm', center=0, 
                           ax=axes[1, 2], cbar_kws={'shrink': 0.8})
                axes[1, 2].set_title('Fraud Flags Correlation Matrix')
            else:
                axes[1, 2].text(0.5, 0.5, 'Fraud flags not available', 
                               transform=axes[1, 2].transAxes, ha='center', va='center')
                axes[1, 2].set_title('Fraud Flags Correlation Matrix')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
    
    def save_results(self, df_results, filepath='fraud_detection_results.csv'):
        """
        Save the fraud detection results to a CSV file.
        """
        try:
            # Select relevant columns for export
            export_columns = [
                'Transaction_ID', 'Customer_ID', 'Transaction_DateTime', 'Transaction_Amount',
                'Merchant_Category', 'Location_Country', 'Transaction_Type',
                'Is_Anomaly', 'Anomaly_Score', 'Risk_Level'
            ]
            
            # Add fraud flags if available
            fraud_flags = ['Unusual_Spending_Category', 'Category_Spending_Spike',
                          'High_Risk_Country', 'Micro_Txns_Before_Large']
            for flag in fraud_flags:
                if flag in df_results.columns:
                    export_columns.append(flag)
            
            df_export = df_results[export_columns].copy()
            df_export.to_csv(filepath, index=False)
            print(f"\nResults saved to: {filepath}")
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")

def main():
    """
    Main function to run the fraud detection system.
    """
    try:
        # Initialize the fraud detection system
        print("Initializing Fraud Detection System...")
        fraud_detector = FraudDetectionSystem(contamination=0.08, random_state=42)
        
        # Load and prepare data
        df = fraud_detector.load_and_prepare_data('bank_transactions_enhanced_fraud_detection.csv')
        
        # Engineer features
        X = fraud_detector.engineer_features(df)
        
        # Train the model
        fraud_detector.train_model(X)
        
        # Make predictions
        print("Making predictions...")
        predictions, scores = fraud_detector.predict_anomalies(X)
        
        # Analyze results
        df_results = fraud_detector.analyze_results(df, predictions, scores)
        
        # Evaluate model performance
        feature_importance = fraud_detector.evaluate_model_performance(df_results)
        
        # Create visualizations
        fraud_detector.create_visualizations(df_results)
        
        # Save results
        fraud_detector.save_results(df_results)
        
        print(f"\n" + "="*60)
        print("FRAUD DETECTION SYSTEM COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Key Outputs:")
        print(f"- Fraud detection results saved to 'fraud_detection_results.csv'")
        print(f"- {sum(predictions == -1):,} transactions flagged as potentially fraudulent")
        print(f"- Model ready for deployment and real-time fraud detection")
        print(f"\n" + "="*60)
        print("FRAUD DETECTION SYSTEM COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Key Outputs:")
        print(f"- Fraud detection results saved to 'fraud_detection_results.csv'")
        print(f"- {sum(predictions == -1):,} transactions flagged as potentially fraudulent")
        print(f"- Model ready for deployment and real-time fraud detection")
        
        # ADD THESE LINES TO PAUSE:
        print(f"\nProgram completed! Check 'fraud_detection_results.csv' for results.")
        print("Press Enter to exit...")
        input()  # This pauses the program until you press Enter
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        print(f"\nAn error occurred. Press Enter to exit...")
        input()  # Pause even when there's an error
        raise
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()