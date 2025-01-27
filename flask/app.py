from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle
import scorecardpy as sc


app = Flask(__name__)

# Load the machine learning model
model = joblib.load('Logistic_Regression_model.pkl')

# Load WOE bins from a file
with open('woe_bins.pkl', 'rb') as file:
    bins_adj = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Log the form data to check what's being received
            app.logger.info(f"Form data received: {request.form}")

            # Get form data
            date_str = request.form['Date']
            date = datetime.strptime(date_str, '%Y-%m-%d')
            total_amount = float(request.form['Monetary'])
            frequency = int(request.form['Frequency'])
            stability = float(request.form['Stability'])
            average_amount = float(request.form['Average_Transaction_Value'])
        
            # Create a DataFrame with the input features
            input_features = pd.DataFrame([{
                'Average_Transaction_Value': average_amount,
                'Value': total_amount,
                'Date': date
            }])

            
            # Generate additional features (replicate training feature engineering)
            input_features['Transaction_Year'] = input_features['Date'].dt.year
            input_features['Transaction_Day'] = input_features['Date'].dt.day
            input_features['Transaction_Month'] = input_features['Date'].dt.month
            input_features['Transaction_Hour'] = 0

            input_features.drop(columns='Date', inplace=True)

            # converting df into woe values
            df_woe = sc.woebin_ply(input_features, bins_adj)

            print(df_woe.isnull().sum())
            print(df_woe.head())

            date = pd.to_datetime(input_features['Transaction_Year'].astype(str) + '-' + 
                                                input_features['Transaction_Month'].astype(str) + '-' + 
                                                input_features['Transaction_Day'].astype(str))
            reference_date = pd.to_datetime('2019-02-13')
            
            add_features = pd.DataFrame([{
                'Stability': stability,
                'Frequency': frequency,
                'Monetary': total_amount,
                'Recency' : (reference_date-date).dt.days
            }])

            input_features = pd.concat([input_features.reset_index(drop=True), add_features.reset_index(drop=True)], axis=1)
            print(f"Add Merged Fetaures: {input_features.isnull().sum()}")
            print(input_features.head())

            
            # Compute RFMS score (simple average of components)
            input_features['RFMS_Score'] = (input_features['Recency'] + input_features['Frequency'] + input_features['Monetary'] + input_features['Stability']) / 4
            
            print(input_features.isnull().sum())
            print(input_features.head())
            # Merge by index
            df_merged = pd.concat([input_features.reset_index(drop=True), df_woe.reset_index(drop=True)], axis=1)

            
            print(df_merged.isnull().sum())
            print(df_merged.head())


            required_order = [  'Transaction_Day', 'Monetary', 'Frequency', 'Average_Transaction_Value',
                                'Transaction_Year_woe', 'Transaction_Month_woe', 'Value',
                                'Transaction_Day_woe', 'Recency', 'Transaction_Hour',
                                'Transaction_Month', 'Average_Transaction_Value_woe',
                                'Transaction_Hour_woe', 'Stability', 'Value_woe', 'Transaction_Year',
                                'RFMS_Score']
            
            for col in required_order:
                if col not in df_merged.columns:
                    df_merged[col] = 0 

            #reorder
            df_merged = df_merged[required_order]

            ## Scale the data
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df_merged)

            # Make prediction
            y_pred = model.predict(df_scaled)[0]
            y_prob = model.predict_proba(df_scaled)[:, 1]
           
            def risk_to_credit_score(probability):
                min_score = 300
                max_score = 850
                return min_score + (max_score - min_score) * (1 - probability)  # Inverse of risk
            
            credit_score = risk_to_credit_score(y_prob)[0]

            # Combine the columns into a single datetime column
            df_merged['Transaction_Date'] = pd.to_datetime(df_merged['Transaction_Year'].astype(str) + '-' + 
                                                df_merged['Transaction_Month'].astype(str) + '-' + 
                                                df_merged['Transaction_Day'].astype(str))

            # Render result
            return render_template('result.html', date=df_merged['Transaction_Date'].iloc[0], total_amount=total_amount,frequency=frequency,
                                    stability=stability,average_amount=average_amount, risk_predicted=y_pred, credit_score=credit_score)

        except Exception as e:
            app.logger.error(f"Error during form processing: {str(e)}")
            return render_template('error.html', error_message=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
           