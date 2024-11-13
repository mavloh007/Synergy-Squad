import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, BatchNormalization, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def lstm_prediction(df, base_product, variation_detail, seq_length=10, epochs=100, batch_size=16):
    
    df_1 = df[(df['Base Product'] == base_product) & (df['Variation Detail'] == variation_detail)].copy()
    
    lstm = df_1.copy()
    
    # Preprocess data
    low_q = lstm['Quantity'].quantile(0.15)
    high_q = lstm['Quantity'].quantile(0.85)
    lstm = lstm[(lstm['Quantity'] >= low_q) & (lstm['Quantity'] <= high_q)]
    lstm.set_index('Date', inplace=True)
    lstm['Logged_Qty'] = np.log1p(lstm['Quantity'])
    lstm['Logged_Price'] = np.log1p(lstm['Price'])
    lstm = lstm.drop(columns=['Base Product', 'Description', 'Variation Type', 'Variation Detail', 'Material', 'Country', 'Customisation Complexity', 'Price', 'Quantity'], axis=1)
    
    scaler_qty = MinMaxScaler()
    scaler = MinMaxScaler()
    lstm['Logged_Price'] = scaler.fit_transform(lstm[['Logged_Price']])
    lstm['Logged_Qty'] = scaler_qty.fit_transform(lstm[['Logged_Qty']])
    
    target_variable = 'Logged_Qty'

    # Define the model
    num_features = lstm.shape[1] - 1  # Exclude target from features
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, num_features)),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        BatchNormalization(),
        Dropout(0.5),
        Dense(25),
        Dropout(0.5),
        Dense(1)
    ])
    optimizer = RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_absolute_error')

    # Prepare sequential data
    def create_sequential_data(dataset, seq_length, target_variable):
        num_samples = len(dataset) - seq_length
        X_data = np.zeros((num_samples, seq_length, num_features))
        y_data = np.zeros(num_samples)
        
        feature_columns = dataset.drop(columns=[target_variable]).values
        target_column = dataset[target_variable].values
        
        for i in range(num_samples):
            X_data[i] = feature_columns[i:i + seq_length]
            y_data[i] = target_column[i + seq_length]
        
        return X_data, y_data

    # Create train and test splits
    train_data, test_data = train_test_split(lstm, test_size=0.2, shuffle=False)
    X_train, y_train = create_sequential_data(train_data, seq_length, target_variable)
    X_test, y_test = create_sequential_data(test_data, seq_length, target_variable)

    train_dates, test_dates = train_data.index, test_data.index

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

    # Predict and transform predictions
    predictions = model.predict(X_test)
    predictions = scaler_qty.inverse_transform(predictions)
    predictions_unlogged = np.expm1(predictions).flatten()
    y_test = scaler_qty.inverse_transform(y_test.reshape(-1, 1))
    y_test_unlogged = np.expm1(y_test)

    predicted_data = []
    for date, actual, pred in zip(test_dates, y_test_unlogged, predictions_unlogged):
        predicted_data.append({
            'Date': date,
            'Actual Quantity': actual,
            'Predicted Quantity': pred
        })
    predictions_df = pd.DataFrame(predicted_data)
    output_df = df_1.merge(predictions_df, on=['Date'], how='left')

    return output_df

def dynamic_pricing(df1, low_inventory_threshold, high_inventory_threshold):
    df2 = df1.copy()
    
    material_price = df2.groupby('Material')['Price'].mean().sort_values()

    variation_price = df2.groupby('Variation Type')['Price'].mean().sort_values()

    complexity_price = df2.groupby('Customisation Complexity')['Price'].mean().sort_values()

    variation_price_rank = variation_price.reset_index().rename(columns={'Price': 'Average Price'})
    variation_price_rank['Rank'] = variation_price_rank['Average Price'].rank(ascending=False)

    complexity_price_rank = complexity_price.reset_index().rename(columns={'Price': 'Average Price'})
    complexity_price_rank['Rank'] = complexity_price_rank['Average Price'].rank(ascending=False)

    material_price_rank = material_price.reset_index().rename(columns={'Price': 'Average Price'})
    material_price_rank['Rank'] = material_price_rank['Average Price'].rank(ascending=False)

    variation_price_rank.rename(columns={'Variation Type': 'Type'}, inplace=True)
    complexity_price_rank.rename(columns={'Customisation Complexity': 'Type'}, inplace=True)
    material_price_rank.rename(columns={'Material': 'Type'}, inplace=True)

    # Define ranking summary
    ranking_summary = {
        'Variation Type': variation_price_rank,
        'Customisation Complexity': complexity_price_rank,
        'Material': material_price_rank
    }

    # Concatenate with multi-index keys and reset the index
    summary_df = pd.concat(ranking_summary.values(), keys=ranking_summary.keys(), names=['Category'])
    summary_df.reset_index(level=0, inplace=True)  # Make 'Category' a column

    def generate_rank_dict(summary_df, category):
        return dict(zip(summary_df[summary_df['Category'] == category]['Type'], summary_df[summary_df['Category'] == category]['Rank']))

    variation_ranks = generate_rank_dict(summary_df, 'Variation Type')
    material_ranks = generate_rank_dict(summary_df, 'Material')
    complexity_ranks = generate_rank_dict(summary_df, 'Customisation Complexity')

    variation_price_rank = variation_price.reset_index().rename(columns={'Price': 'Average Price'})

    df2 = df2.dropna(subset=['Predicted Quantity'])
    df2['Date'] = pd.to_datetime(df2['Date'])
    
    holidays = {
    (1, 1): "New Year's Day",
    (3, 29): "Good Friday",
    (4, 1): "Easter Monday",
    (5, 6): "Early May Bank Holiday",
    (5, 27): "Spring Bank Holiday",
    (8, 26): "Summer Bank Holiday",
    (12, 25): "Christmas Day",
    (12, 26): "Boxing Day"
    }

    df2['Holiday'] = df2['Date'].apply(lambda x: 1 if (x.month, x.day) in holidays else 0)
    
    df2.drop(columns=['Quantity'], inplace=True)

    df2['Actual Revenue'] = df2['Actual Quantity'] * df2['Price']
    df2['Predicted Revenue'] = df2['Predicted Quantity'] * df2['Price']

    def adjust_price(row, variation_ranks, material_ranks, complexity_ranks):
        # Initial price adjustment multiplier
        price_adjustment = 1.0
        
        # Adjustments based on ranks and factors
        variation_rank = variation_ranks.get(row['Variation Type'], 1)
        price_adjustment += (variation_rank / max(variation_ranks.values())) * 0.1
        
        material_rank = material_ranks.get(row['Material'], 1)
        price_adjustment += (material_rank / max(material_ranks.values())) * 0.1
        
        complexity_rank = complexity_ranks.get(row['Customisation Complexity'], 1)
        price_adjustment += (complexity_rank / max(complexity_ranks.values())) * 0.05
        
        # Additional holiday price increase
        if row['Holiday'] == 1:
            price_adjustment += 0.15

        return price_adjustment

    def optimize_dynamic_pricing(row, variation_ranks, material_ranks, complexity_ranks, max_iterations=30, target_revenue_increase=1.20, adjustment_step=0.05):
        initial_price = row['Price']
        best_price = initial_price
        best_revenue = row['Predicted Revenue']
        actual_revenue = row['Actual Revenue']
    
        
        if best_revenue >= actual_revenue * target_revenue_increase:
            best_price = initial_price * adjust_price(row, variation_ranks, material_ranks, complexity_ranks)
            if row['Inventory'] < low_inventory_threshold:
                best_price *= (1 + adjustment_step)  
            elif row['Inventory'] > high_inventory_threshold:
                best_price *= (1 - adjustment_step) 
            best_revenue = row['Predicted Quantity'] * best_price
            return best_price, best_revenue

        for iteration in range(max_iterations):
            # Apply custom adjustments to the price based on factors
            price_adjustment = adjust_price(row, variation_ranks, material_ranks, complexity_ranks)
            
            # Adjust price based on inventory levels
            if row['Inventory'] < low_inventory_threshold:
                price_adjustment *= (1 + adjustment_step)  # Increase price in steps for low inventory
            elif row['Inventory'] > high_inventory_threshold:
                price_adjustment *= (1 - adjustment_step)  # Decrease price in steps for high inventory
            else:
                price_adjustment *= (1 + adjustment_step)  
                
            new_price = best_price * price_adjustment
            
            # Calculate predicted revenue with the new price
            predicted_revenue = row['Predicted Quantity'] * new_price

            # If the new revenue is higher than the best recorded revenue, update best price and revenue
            if predicted_revenue > best_revenue:
                best_price = new_price
                best_revenue = predicted_revenue
            
            # Stop if the target revenue increase is achieved
            if best_revenue >= actual_revenue * target_revenue_increase:
                break

        return best_price, best_revenue

    df2[['Optimized Price', 'Optimized Predicted Revenue']] = df2.apply(
        lambda row: pd.Series(optimize_dynamic_pricing(row, variation_ranks, material_ranks, complexity_ranks)),
        axis=1
    )

    # Calculate improvement over actual revenue
    df2['Revenue Improvement'] = df2['Optimized Predicted Revenue'] - df2['Actual Revenue']

    return df2

def generate_inventory(df):
    """
    Placeholder function to generate random inventory values and set quantile-based thresholds.
    """
    # Seed for reproducibility
    np.random.seed(42)
    
    # Generate random inventory levels between 1 and 100 (or adjust as needed)
    df['Inventory'] = np.random.randint(1, 101, size=len(df))
    
    # Calculate the lower and upper quantiles (e.g., 25th and 75th percentiles)
    lower_threshold = df['Inventory'].quantile(0.25)
    upper_threshold = df['Inventory'].quantile(0.75)
    
    return df, lower_threshold, upper_threshold