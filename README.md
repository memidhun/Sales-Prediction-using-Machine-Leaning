# Sales-Prediction-using-Machine-Leaning
---
jupyter:
  colab:
    authorship_tag: ABX9TyNwab6xvTU23CQK9c86gfa9
    mount_file_id: 14cGBLbF81ZIcTKbKAMf7z0boo62siKAI
    provenance:
    - file_id: 14cGBLbF81ZIcTKbKAMf7z0boo62siKAI
      timestamp: 1715974227834
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .code execution_count="23" executionInfo="{\"elapsed\":387,\"status\":\"ok\",\"timestamp\":1715971468699,\"user\":{\"displayName\":\"MIDHUN MATHEW\",\"userId\":\"16632471415797437956\"},\"user_tz\":-330}" id="2KwMiz0E1oWf"}
``` python
#importing requires Libraries
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
```
:::

::: {.cell .code execution_count="7" executionInfo="{\"elapsed\":27925,\"status\":\"ok\",\"timestamp\":1715971015783,\"user\":{\"displayName\":\"MIDHUN MATHEW\",\"userId\":\"16632471415797437956\"},\"user_tz\":-330}" id="qiMAU26Q1qd5"}
``` python
# Listing files in the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Loading the data
path = '/content/drive/MyDrive/Coffee Shop Sales.xlsx'
df = pd.read_excel(path)
```
:::

::: {.cell .code execution_count="44" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":374,\"status\":\"ok\",\"timestamp\":1715972173071,\"user\":{\"displayName\":\"MIDHUN MATHEW\",\"userId\":\"16632471415797437956\"},\"user_tz\":-330}" id="MC_ZeUTh19Qv" outputId="93ab5cbd-ac95-4edd-e569-b3c96ce10ee1"}
``` python
# Displaying the relevant columns of the dataframe
df['total_cost'] = df['transaction_qty'] * df['unit_price']
df['Month'] = pd.to_datetime(df['transaction_date']).dt.to_period('M')
df_relevant = df[['transaction_date', 'Month', 'total_cost', 'product_type']]
print(df_relevant.head())
```

::: {.output .stream .stdout}
      transaction_date    Month  total_cost           product_type
    0       2023-01-01  2023-01         6.0  Gourmet brewed coffee
    1       2023-01-01  2023-01         6.2        Brewed Chai tea
    2       2023-01-01  2023-01         9.0          Hot chocolate
    3       2023-01-01  2023-01         2.0            Drip coffee
    4       2023-01-01  2023-01         6.2        Brewed Chai tea
:::
:::

::: {.cell .code execution_count="32" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":3,\"status\":\"ok\",\"timestamp\":1715971544024,\"user\":{\"displayName\":\"MIDHUN MATHEW\",\"userId\":\"16632471415797437956\"},\"user_tz\":-330}" id="lpvVXk7a2ZvV" outputId="01895eae-a9c0-47a9-d576-6c3041194a4c"}
``` python
# Displaying the last 5 transaction dates
print(df['transaction_date'].tail())
```

::: {.output .stream .stdout}
    149111   2023-06-30
    149112   2023-06-30
    149113   2023-06-30
    149114   2023-06-30
    149115   2023-06-30
    Name: transaction_date, dtype: datetime64[ns]
:::
:::

::: {.cell .code execution_count="33" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":382,\"status\":\"ok\",\"timestamp\":1715971546784,\"user\":{\"displayName\":\"MIDHUN MATHEW\",\"userId\":\"16632471415797437956\"},\"user_tz\":-330}" id="YpMynBJh2e1F" outputId="e99ef819-08d5-4c13-ab0e-1100614f35f5"}
``` python
# Calculating total sales for each month
df['transaction_date'] = pd.to_datetime(df['transaction_date'])
df['Month'] = df['transaction_date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['total_cost'].sum().reset_index()
monthly_sales['Month'] = monthly_sales['Month'].astype(str)
print(monthly_sales)
```

::: {.output .stream .stdout}
         Month  total_cost
    0  2023-01    81677.74
    1  2023-02    76145.19
    2  2023-03    98834.68
    3  2023-04   118941.08
    4  2023-05   156727.76
    5  2023-06   166485.88
:::
:::

::: {.cell .code execution_count="34" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":957}" executionInfo="{\"elapsed\":715,\"status\":\"ok\",\"timestamp\":1715971550481,\"user\":{\"displayName\":\"MIDHUN MATHEW\",\"userId\":\"16632471415797437956\"},\"user_tz\":-330}" id="8ouQzq4y2c3u" outputId="844ff1d9-0be0-4b28-fb21-8fed985e7a1d"}
``` python
# Data visualization
plt.figure(figsize=(10, 5))
plt.plot(monthly_sales['Month'], monthly_sales['total_cost'], marker='o', color='b')
plt.title('Monthly Sales')
plt.xlabel('Months')
plt.ylabel('Sales ($)')
plt.grid(True)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(monthly_sales['Month'], monthly_sales['total_cost'], color='g')
plt.title('Monthly Sales Amounts')
plt.xlabel('Months')
plt.ylabel('Sales Amount ($)')
plt.grid(True)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()
```

::: {.output .display_data}
![](vertopal_02264bc4a6234512ade8eaa6d7fa205a/de177821bea32d33e016c852eeb0df4825210828.png)
:::

::: {.output .display_data}
![](vertopal_02264bc4a6234512ade8eaa6d7fa205a/754d0f5c806429c5d3af4617b62d9e668f0523b6.png)
:::
:::

::: {.cell .code execution_count="35" executionInfo="{\"elapsed\":362,\"status\":\"ok\",\"timestamp\":1715971555148,\"user\":{\"displayName\":\"MIDHUN MATHEW\",\"userId\":\"16632471415797437956\"},\"user_tz\":-330}" id="-5pegoHY287n"}
``` python
# Machine Learning: Predicting future sales
# Convert Month to numerical format for ML model
monthly_sales['Month_num'] = pd.to_datetime(monthly_sales['Month']).map(pd.Timestamp.toordinal)
# Preparing data for model
X = monthly_sales[['Month_num']]
y = monthly_sales['total_cost']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
:::

::: {.cell .code execution_count="50" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":4,\"status\":\"ok\",\"timestamp\":1715972619596,\"user\":{\"displayName\":\"MIDHUN MATHEW\",\"userId\":\"16632471415797437956\"},\"user_tz\":-330}" id="VUINLQsS3Dmm" outputId="f8722e0f-b45a-41b9-ebdf-3bcdeaedc317"}
``` python
# Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
```

::: {.output .stream .stdout}
    Mean Squared Error: 418969916.05479974
    R-squared: -53.75101012273224
:::
:::

::: {.cell .code execution_count="51" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":416,\"status\":\"ok\",\"timestamp\":1715972621972,\"user\":{\"displayName\":\"MIDHUN MATHEW\",\"userId\":\"16632471415797437956\"},\"user_tz\":-330}" id="GF98dcNu4Lxu" outputId="aa920630-7dda-48d6-c36d-10ec65145a65"}
``` python
# Predicting future sales
future_months = pd.date_range(start='2023-07-01', periods=6, freq='M').to_period('M')
future_month_nums = future_months.to_timestamp().map(pd.Timestamp.toordinal).to_frame(name='Month_num')
future_sales_predictions = model.predict(future_month_nums)

future_sales = pd.DataFrame({
    'Month': future_months.astype(str),
    'Predicted_Sales': future_sales_predictions
})

print(future_sales)
```

::: {.output .stream .stdout}
         Month  Predicted_Sales
    0  2023-07    194962.303097
    1  2023-08    219319.718176
    2  2023-09    243677.133255
    3  2023-10    267248.825267
    4  2023-11    291606.240346
    5  2023-12    315177.932358
:::
:::

::: {.cell .code execution_count="52" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":524}" executionInfo="{\"elapsed\":1108,\"status\":\"ok\",\"timestamp\":1715972624507,\"user\":{\"displayName\":\"MIDHUN MATHEW\",\"userId\":\"16632471415797437956\"},\"user_tz\":-330}" id="pv7vdjuD4Os2" outputId="2cc97b46-79f4-4a98-c932-91f495a71709"}
``` python
# Visualizing future sales predictions
plt.figure(figsize=(10, 5))
plt.plot(monthly_sales['Month'], monthly_sales['total_cost'], marker='o', color='b', label='Historical Sales')
plt.plot(future_sales['Month'], future_sales['Predicted_Sales'], marker='o', color='r', linestyle='--', label='Predicted Sales')
plt.title('Monthly Sales: Historical and Predicted')
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
```

::: {.output .display_data}
![](vertopal_02264bc4a6234512ade8eaa6d7fa205a/dd6e0f8ccbae7540806d4a359110d8c7e5c9ca7c.png)
:::
:::

::: {.cell .code execution_count="54" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":706}" executionInfo="{\"elapsed\":1260,\"status\":\"ok\",\"timestamp\":1715972826892,\"user\":{\"displayName\":\"MIDHUN MATHEW\",\"userId\":\"16632471415797437956\"},\"user_tz\":-330}" id="xYUk-rDX8u9R" outputId="6c75ef79-66d0-4b49-9d04-c47a45b6bc9d"}
``` python
#NOW USING RandomForestRegressor

# Creating and training the model with hyperparameters
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Predicting future sales
future_months = pd.date_range(start='2023-07-01', periods=6, freq='M').to_period('M')
future_month_nums = future_months.to_timestamp().map(pd.Timestamp.toordinal).to_frame(name='Month_num')
future_sales_predictions = model.predict(future_month_nums)

future_sales = pd.DataFrame({
    'Month': future_months.astype(str),
    'Predicted_Sales': future_sales_predictions
})

print(future_sales)

# Visualizing future sales predictions
plt.figure(figsize=(10, 5))
plt.plot(monthly_sales['Month'], monthly_sales['total_cost'], marker='o', color='b', label='Historical Sales')
plt.plot(future_sales['Month'], future_sales['Predicted_Sales'], marker='o', color='r', linestyle='--', label='Predicted Sales')
plt.title('Monthly Sales: Historical and Predicted')
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
```

::: {.output .stream .stdout}
    Mean Squared Error: 908111371.9073001
    Root Mean Squared Error: 30134.88629325321
    R-squared: -117.67204066595
         Month  Predicted_Sales
    0  2023-07       160992.598
    1  2023-08       160992.598
    2  2023-09       160992.598
    3  2023-10       160992.598
    4  2023-11       160992.598
    5  2023-12       160992.598
:::

::: {.output .display_data}
![](vertopal_02264bc4a6234512ade8eaa6d7fa205a/34570c42ea283abe1f1bc8aa58c541c30cd0bfd3.png)
:::
:::

::: {.cell .code execution_count="56" executionInfo="{\"elapsed\":399,\"status\":\"ok\",\"timestamp\":1715973184982,\"user\":{\"displayName\":\"MIDHUN MATHEW\",\"userId\":\"16632471415797437956\"},\"user_tz\":-330}" id="TUfYZAJx9-XB"}
``` python
# I Tried chaging the hyperparameters but doesnt refelected enough chaanges , so I am going to use Gradient Boosting Machines
import xgboost as xgb
```
:::

::: {.cell .code execution_count="57" colab="{\"base_uri\":\"https://localhost:8080/\"}" executionInfo="{\"elapsed\":2282,\"status\":\"ok\",\"timestamp\":1715973238863,\"user\":{\"displayName\":\"MIDHUN MATHEW\",\"userId\":\"16632471415797437956\"},\"user_tz\":-330}" id="uvYUxH5f-PYY" outputId="c04b0b82-c553-4035-c500-19ee6463c8d4"}
``` python
# Creating and training the XGBoost model
model = xgb.XGBRegressor(objective ='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Predicting future sales
future_months = pd.date_range(start='2023-07-01', periods=6, freq='M').to_period('M')
future_month_nums = future_months.to_timestamp().map(pd.Timestamp.toordinal).to_frame(name='Month_num')
future_sales_predictions = model.predict(future_month_nums)
```

::: {.output .stream .stdout}
    Mean Squared Error: 404587694.76107836
    Root Mean Squared Error: 20114.365383006207
    R-squared: -51.87154070627676
:::
:::

::: {.cell .code execution_count="58" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":652}" executionInfo="{\"elapsed\":2205,\"status\":\"ok\",\"timestamp\":1715973256951,\"user\":{\"displayName\":\"MIDHUN MATHEW\",\"userId\":\"16632471415797437956\"},\"user_tz\":-330}" id="kMI5Kk9k-d0_" outputId="1b50fe85-639d-4fca-e54a-e5c088691e21"}
``` python
future_sales = pd.DataFrame({
    'Month': future_months.astype(str),
    'Predicted_Sales': future_sales_predictions
})

print(future_sales)

# Visualizing future sales predictions
plt.figure(figsize=(10, 5))
plt.plot(monthly_sales['Month'], monthly_sales['total_cost'], marker='o', color='b', label='Historical Sales')
plt.plot(future_sales['Month'], future_sales['Predicted_Sales'], marker='o', color='r', linestyle='--', label='Predicted Sales')
plt.title('Monthly Sales: Historical and Predicted')
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
```

::: {.output .stream .stdout}
         Month  Predicted_Sales
    0  2023-07    166485.828125
    1  2023-08    166485.828125
    2  2023-09    166485.828125
    3  2023-10    166485.828125
    4  2023-11    166485.828125
    5  2023-12    166485.828125
:::

::: {.output .display_data}
![](vertopal_02264bc4a6234512ade8eaa6d7fa205a/ea6dcafb69ced15fd653d9ff13647815eeecb830.png)
:::
:::

::: {.cell .code execution_count="74" colab="{\"base_uri\":\"https://localhost:8080/\"}" collapsed="true" executionInfo="{\"elapsed\":10356,\"status\":\"ok\",\"timestamp\":1715973955900,\"user\":{\"displayName\":\"MIDHUN MATHEW\",\"userId\":\"16632471415797437956\"},\"user_tz\":-330}" id="QfISwF-G-3Ig" outputId="9601c3a4-1fed-49af-bf52-5495795cda2d"}
``` python
#CHECKING THE BEST HYPER PARAMETERS OF BOOSTER MODEL
from sklearn.model_selection import GridSearchCV
# Define the XGBoost model
model = xgb.XGBRegressor(objective ='reg:squarederror', random_state=42)

# Define the grid of hyperparameters to search
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of boosting rounds
    'learning_rate': [0.05, 0.1, 0.2],  # Step size shrinkage
    'max_depth': [3, 5, 7],  # Maximum depth of trees
    'subsample': [0.6, 0.8, 1.0],  # Subsample ratio of the training instances
}

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use the best model for prediction
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')
```

::: {.output .stream .stdout}
    Fitting 3 folds for each of 81 candidates, totalling 243 fits
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=200, subsample=0.8; total time=   0.1s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=300, subsample=0.6; total time=   0.1s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=300, subsample=0.6; total time=   0.1s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=300, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=300, subsample=0.8; total time=   0.1s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=300, subsample=0.8; total time=   0.1s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=300, subsample=1.0; total time=   0.1s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=300, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=3, n_estimators=300, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=300, subsample=0.6; total time=   0.1s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=300, subsample=0.6; total time=   0.1s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=300, subsample=0.8; total time=   0.1s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=300, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=300, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=300, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=300, subsample=1.0; total time=   0.1s
    [CV] END learning_rate=0.05, max_depth=5, n_estimators=300, subsample=1.0; total time=   0.1s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=200, subsample=0.6; total time=   0.1s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=300, subsample=0.6; total time=   0.1s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=300, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=300, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=300, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=300, subsample=1.0; total time=   0.1s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=300, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.05, max_depth=7, n_estimators=300, subsample=1.0; total time=   0.1s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=100, subsample=1.0; total time=   0.1s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=300, subsample=0.6; total time=   0.1s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=300, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=300, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=300, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=300, subsample=1.0; total time=   0.1s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=300, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=3, n_estimators=300, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=300, subsample=0.6; total time=   0.1s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=300, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=300, subsample=0.8; total time=   0.1s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=300, subsample=0.8; total time=   0.1s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=300, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=300, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=5, n_estimators=300, subsample=1.0; total time=   0.1s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=100, subsample=0.8; total time=   0.1s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=300, subsample=0.6; total time=   0.1s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=300, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=300, subsample=0.8; total time=   0.1s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=300, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=300, subsample=1.0; total time=   0.1s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=300, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.1, max_depth=7, n_estimators=300, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=300, subsample=0.8; total time=   0.1s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=300, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=300, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=300, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=300, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=3, n_estimators=300, subsample=1.0; total time=   0.1s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=300, subsample=0.8; total time=   0.1s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=300, subsample=0.8; total time=   0.1s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=300, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=300, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=300, subsample=1.0; total time=   0.1s
    [CV] END learning_rate=0.2, max_depth=5, n_estimators=300, subsample=1.0; total time=   0.1s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=100, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=100, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=100, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=200, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=200, subsample=0.6; total time=   0.1s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=200, subsample=0.8; total time=   0.1s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=200, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=200, subsample=1.0; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=300, subsample=0.6; total time=   0.1s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=300, subsample=0.6; total time=   0.1s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=300, subsample=0.6; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=300, subsample=0.8; total time=   0.0s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=300, subsample=0.8; total time=   0.1s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=300, subsample=0.8; total time=   0.1s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=300, subsample=1.0; total time=   0.1s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=300, subsample=1.0; total time=   0.1s
    [CV] END learning_rate=0.2, max_depth=7, n_estimators=300, subsample=1.0; total time=   0.2s
    Best Hyperparameters: {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.6}
    Mean Squared Error: 510229678.00482833
    Root Mean Squared Error: 22588.264165376415
    R-squared: -65.67684049588632
:::
:::

::: {.cell .code execution_count="76" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":524}" executionInfo="{\"elapsed\":1004,\"status\":\"ok\",\"timestamp\":1715974046225,\"user\":{\"displayName\":\"MIDHUN MATHEW\",\"userId\":\"16632471415797437956\"},\"user_tz\":-330}" id="L5PPxf1z_ZCI" outputId="ae575f3a-bddc-4ac7-8875-f6449b3acfd2"}
``` python
# Convert Month to numerical format for ML model
monthly_sales['Month_num'] = pd.to_datetime(monthly_sales['Month']).map(pd.Timestamp.toordinal)

# Preparing data for model
X = monthly_sales[['Month_num']]
y = monthly_sales['total_cost']

# Define the XGBoost model with best hyperparameters
model = xgb.XGBRegressor(learning_rate=0.05, max_depth=3, n_estimators=100, subsample=0.6, objective='reg:squarederror', random_state=42)

# Train the model on the entire dataset
model.fit(X, y)

# Predicting future sales
future_months = pd.date_range(start='2023-07-01', periods=6, freq='M')
future_month_nums = future_months.to_series().apply(lambda x: x.toordinal()).values.reshape(-1, 1)
future_sales_predictions = model.predict(future_month_nums)

# Create a DataFrame for future predictions
future_sales = pd.DataFrame({
    'Month': future_months.strftime('%Y-%m'),  # Format the date to match historical data format
    'Predicted_Sales': future_sales_predictions
})

# Plotting the predictions
plt.figure(figsize=(10, 5))
plt.plot(monthly_sales['Month'], monthly_sales['total_cost'], marker='o', color='b', label='Historical Sales')
plt.plot(future_sales['Month'], future_sales['Predicted_Sales'], marker='o', color='r', linestyle='--', label='Predicted Sales')
plt.title('Monthly Sales: Historical and Predicted')
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
```

::: {.output .display_data}
![](vertopal_02264bc4a6234512ade8eaa6d7fa205a/9c61ec538713973a49ebcb6ca63f67af8991d3ae.png)
:::
:::
