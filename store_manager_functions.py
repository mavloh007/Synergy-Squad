import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, BatchNormalization, Dropout, Dense
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, date, timedelta

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

def generate_inventory(df1, base_product, variation_detail):
    df1["Date"] = pd.to_datetime(df1["Date"])
    description = df1[(df1["Base Product"] == base_product) & (df1["Variation Detail"] == variation_detail)].iloc[0]["Description"]
    predictions = df1.dropna(subset = "Predicted Quantity", ignore_index = True)
    # Function to run simulation for a given safety stock and reorder point
    def run_inventory_simulation(safety_stock, reorder_point, predictions_description_df, mean_demand):
        total_holding_cost = 0
        total_ordering_cost = 0
        total_stockout_cost = 0
        total_units_short = 0
        orders_placed = 0
        service_level = 0
        inventory_level = safety_stock  # Starting inventory level includes safety stock

        num_days = len(predictions_description_df)
        last_date = predictions_description_df.iloc[num_days-1]["Date"]
        
        for i in range(num_days):
            date = predictions_description_df.iloc[i]["Date"]
            daily_demand = max(0, int(predictions_description_df.iloc[i]["Predicted Quantity"]))

            # Check if reorder is needed
            if inventory_level <= reorder_point:
                # Generate a lead time for this order
                lead_time = max(1, int(np.random.normal(mean_lead_time, std_lead_time)))
                orders_placed += 1
                total_ordering_cost += ordering_cost_per_unit * (reorder_point-safety_stock) # Reorder Quantity

                # Add new stock after lead time
                if date + timedelta(days = lead_time) < last_date:
                    inventory_level += reorder_point - safety_stock # Reorder Quantity

            # Calculate stockout (if any) and update inventory
            if inventory_level >= daily_demand:
                inventory_level -= daily_demand
            else:
                units_short = daily_demand - inventory_level
                total_stockout_cost += units_short * stockout_cost_per_unit
                total_units_short += units_short
                inventory_level = 0

            # Calculate daily holding cost
            total_holding_cost += inventory_level * holding_cost_per_unit_per_day

        # Calculate service level
        service_level = 1 - (total_units_short / (mean_demand * num_days))

        return {
            "Total Holding Cost": total_holding_cost,
            "Total Ordering Cost": total_ordering_cost,
            "Total Stockout Cost": total_stockout_cost,
            "Total Cost": total_holding_cost + total_ordering_cost + total_stockout_cost,
            "Orders Placed": orders_placed,
            "Service Level": service_level
        }
    
    ## Simulation Hyperparameter ##
    # Cost Parameter #
    holding_cost_per_unit_per_day = df1["Price"].mean() # Cost to hold one unit in inventory for one day (Mean Price of all products in inventory)
    
    product_df = df1[df1['Base Product'] == base_product].copy()
    predictions_df = predictions[predictions["Base Product"] == base_product].copy()

    description_df = product_df[product_df['Description'] == description].drop(columns=['Description', 'Base Product'])
    predictions_description_df = predictions_df[predictions_df["Description"] == description].drop(columns=['Description', 'Base Product'])
                
    ## Simulation Hyperparams ##
    mean_demand = int(description_df["Quantity"].mean())
    # Assumptions #
    max_lead_time = 9
    mean_lead_time = 5
    std_lead_time = 2   
                
    # Cost parameters #
    ordering_cost_per_unit = description_df["Price"].mean()/120*100 # Fixed order cost per unit (Assuming retail price is 120% of cost price)
    stockout_cost_per_unit = description_df["Price"].mean() # Penalty cost for each unit not met (The retail price basically, since we do not have customer satisfaction values to include)
            
    # Recommended Inventory Control Metrics
    # Safety Stock and Reorder Point Strategy
    # using Safaety Stock = 75% Quantile of Demand During Lead Time * Max Lead Time (due to big outliers and computational resources)
    # using Reorder Point = Mean Demand During Lead time * Mean Lead Time + Safety Stock
    rec_safety_stock = description_df["Quantity"].quantile(q = 0.75).astype(int) * max_lead_time
    rec_reorder_point = mean_demand * mean_lead_time + rec_safety_stock
            
    safety_stock_levels = range(description_df["Quantity"].quantile(q = 0.75).astype(int), rec_safety_stock+100, 100)  # Different safety stock levels to test
                
    ## Collect the results of running the simulation to find the optimal levels
    results = []
    for safety_stock in safety_stock_levels:
        reorder_points = range(description_df["Quantity"].quantile(q = 0.75).astype(int), rec_reorder_point+50, 50)    # Different reorder points to test
        for reorder_point in reorder_points:
            if (reorder_point > safety_stock):
                result = run_inventory_simulation(safety_stock, reorder_point, predictions_description_df, mean_demand)
                total_cost = result["Total Cost"]
                service_level = result["Service Level"]
                results.append({
                "Base Product": base_product,
                "Description": description,
                "Safety Stock": safety_stock,
                "Reorder Point": reorder_point,
                "Total Cost": total_cost,
                "Service Level": service_level
    })
    df_results = pd.DataFrame(results)
                
    ## Find the optimal Safety Stock Level and Reorder Point for each description in every base product
    best_cost = df_results["Total Cost"].max()
    best_service = df_results["Service Level"].min()
    optimal_safety_stock = df_results.iloc[0]["Safety Stock"]
    optimal_reorder_point = df_results.iloc[0]["Reorder Point"]
                
    for i in range(len(df_results)):
        if (df_results.iloc[i]["Total Cost"] <= best_cost) & (df_results.iloc[i]["Service Level"] >= best_service):
            best_cost = df_results.iloc[i]["Total Cost"]
            best_service = df_results.iloc[i]["Service Level"]
            optimal_safety_stock = df_results.iloc[i]["Safety Stock"]
            optimal_reorder_point = df_results.iloc[i]["Reorder Point"]
                
    inventory_optimisation = {
        "Base Product": base_product,
        "Description": description,
        "Safety Stock": optimal_safety_stock,
        "Reorder Point": optimal_reorder_point
    }

    final = []
    df2 = predictions[(predictions["Base Product"] == base_product) & (predictions["Description"] == description)].copy()
    df2["Inventory"] = 0
    df2["Predicted Quantity"] = df2["Predicted Quantity"].astype(int)

    base_product = inventory_optimisation["Base Product"]
    description = inventory_optimisation["Description"]
    product_safety_stock = inventory_optimisation["Safety Stock"]
    product_reorder_point = inventory_optimisation["Reorder Point"]
        
    days = len(df2)
    max_date = df2.iloc[days-1]["Date"]
    product_inventory = product_safety_stock
        
    for i in range(days):
        # Copy the current row's values
        curr_date = df2.iloc[i]["Date"]
        product_demand = df2.iloc[i]["Predicted Quantity"]  

        # Check if reorder is needed
        if product_inventory <= product_reorder_point:
            # Generate a lead time for this order
            lead_time = max(1, int(np.random.normal(5, 2)))

            # Add new stock after lead time
            if curr_date + timedelta(days = lead_time) < max_date:
                product_inventory += product_reorder_point - product_safety_stock # Reorder Quantity
            
        # Calculate stockout (if any) and update inventory
        if product_inventory >= product_demand:
            product_inventory -= product_demand
        else:
            product_inventory = 0
        final.append({
            "Base Product": base_product,
            "Description": description,
            "Date": curr_date,
            "Inventory": product_inventory
        })
    df3 = pd.DataFrame(final) 
    
    # Calculate the lower and upper quantiles (e.g., 25th and 75th percentiles)
    lower_threshold = df3['Inventory'].quantile(0.25)
    upper_threshold = df3['Inventory'].quantile(0.75)
    
    df4 = df1[(df1["Base Product"] == base_product) & (df1["Description"] == description)].merge(df3, on = ["Base Product", "Description", "Date"], how = "left")
    
    return df4, lower_threshold, upper_threshold