import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
import numpy as np


# # Loading data
data = pd.read_csv('train_data.csv')
data.isnull().mean() * 100

# Preprocessing
def fill_mode_per_group(group):
    mode_values = group['CurrentGameMode'].mode()
    if not mode_values.empty:
        group['CurrentGameMode'] = group['CurrentGameMode'].fillna(mode_values.iloc[0])
    return group
# Apply the function to each group by 'UserID'
data = data.groupby('UserID').apply(fill_mode_per_group)

# Reset the index after grouping
data.reset_index(drop=True, inplace=True)

mode_value = data['CurrentGameMode'].mode()[0]
data['CurrentGameMode'].fillna(mode_value, inplace=True)

data.drop("QuestionType", axis = 1 , inplace = True)
data.drop("CurrentTask", axis = 1 , inplace = True)
data.drop("LastTaskCompleted", axis = 1 , inplace = True)

# data.isnull().mean() * 100

def fill_mean_per_group(group):
    group['LevelProgressionAmount'].fillna(group['LevelProgressionAmount'].mean(), inplace=True)
    return group

# Apply the function to each group by 'UserID'
data = data.groupby('UserID').apply(fill_mean_per_group)

# Reset the index after grouping
data.reset_index(drop=True, inplace=True)

data['LevelProgressionAmount'].fillna(data['LevelProgressionAmount'].mean(), inplace=True)
# data.isnull().mean() * 100

from sklearn.preprocessing import MinMaxScaler
data['QuestionTiming'] = data['QuestionTiming'].map({'System Initiated': 1, 'User Initiated': 0})
data['TimeUtc'] = pd.to_datetime(data['TimeUtc'], errors='coerce')
data['hour'] = data['TimeUtc'].dt.hour
data['dayofweek'] = data['TimeUtc'].dt.dayofweek
data['day'] = data['TimeUtc'].dt.day
data['month'] = data['TimeUtc'].dt.month
data['year'] = data['TimeUtc'].dt.year

# Drop the original TimeUtc column
data.drop('TimeUtc', axis=1, inplace=True)

data['CurrentGameMode'] = LabelEncoder().fit_transform(data['CurrentGameMode'])
data['UserID_encoded'] = LabelEncoder().fit_transform(data['UserID'])

scaler=MinMaxScaler()
data['CurrentSessionLength'] = scaler.fit_transform(data[['CurrentSessionLength']])


# data.to_csv('date-test.csv')


def create_user_feature(df):
    # Compute user-specific statistics
    user_stats = df.groupby('UserID_encoded')['ResponseValue'].agg(['mean', 'std']).reset_index()
    user_stats.columns = ['UserID_encoded', 'UserResponseMean', 'UserResponseStd']
    
    # Merge user statistics back to the original dataframe
    df = pd.merge(df, user_stats, on='UserID_encoded', how='left')
    
    return df

# Create the new feature
data = create_user_feature(data)


# data.isnull().mean() * 100


data['UserResponseStd'].fillna(0, inplace=True)


# # Initialize and train the Random Forest model

X = data.drop(columns=['ResponseValue','UserID'])
y = data['ResponseValue']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # RandomForest Implementation
# model = RandomForestRegressor(random_state=42)
# model.fit(X_train, y_train)

# # Predict on the test set
# y_pred = model.predict(X_test)

# # Calculate Mean Absolute Error (MAE)
# test_mae = mean_absolute_error(y_test, y_pred)
# print(f"Test MAE: {test_mae}")


unique_data=data[['UserID','UserResponseMean','UserResponseStd']]

unique_data = unique_data.drop_duplicates(subset='UserID')


# # Dummy Regressor
# dummy_model = DummyRegressor(strategy="mean")
# dummy_model.fit(X_train, y_train)
# pred_dummy= dummy_model.predict(X_test)
# mae = mean_absolute_error(y_test, pred_dummy)
# print(f'Mean Absolute Error: {mae}')


# # Linear Regression
# Linear_Reg_model = LinearRegression()
# Linear_Reg_model.fit(X_train, y_train)
# pred_Linear_Reg= Linear_Reg_model.predict(X_test)
# mae = mean_absolute_error(y_test, pred_Linear_Reg)
# print(f'Mean Absolute Error: {mae}')


# # Polynomial
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
# # Example with degree 2 polynomial
# model = Pipeline([
#     ('poly', PolynomialFeatures(degree=2)),
#     ('linear', LinearRegression())
# ])
# model.fit(X_train, y_train)
# pred_poly = model.predict(X_test)
# mae = mean_absolute_error(y_test, pred_poly)
# print(f'Mean Absolute Error: {mae}')


# # Ridge
# from sklearn.linear_model import Ridge
# ridge_model = Ridge(alpha=1.0)
# ridge_model.fit(X_train, y_train)
# pred_ridge = ridge_model.predict(X_test)
# mae = mean_absolute_error(y_test, pred_ridge)
# print(f'Mean Absolute Error: {mae}')


# Lasso
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
pred_lasso = lasso_model.predict(X_test)
mae = mean_absolute_error(y_test, pred_lasso)
print(f'Mean Absolute Error: {mae}')


## ElasticNet
# from sklearn.linear_model import ElasticNet
# elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
# elastic_net_model.fit(X_train, y_train)
# pred_elastic_net = elastic_net_model.predict(X_test)
# mae = mean_absolute_error(y_test, pred_elastic_net)
# print(f'Mean Absolute Error: {mae}')


## DecisionTreeRegressor
# from sklearn.tree import DecisionTreeRegressor
# tree_model = DecisionTreeRegressor()
# tree_model.fit(X_train, y_train)
# pred_tree = tree_model.predict(X_test)
# mae = mean_absolute_error(y_test, pred_tree)
# print(f'Mean Absolute Error: {mae}')


## GradientBoostingRegressor
# from sklearn.ensemble import GradientBoostingRegressor
# gb_model = GradientBoostingRegressor(n_estimators=100)
# gb_model.fit(X_train, y_train)
# pred_gb = gb_model.predict(X_test)
# mae = mean_absolute_error(y_test, pred_gb)
# print(f'Mean Absolute Error: {mae}')


## SVR
# from sklearn.svm import SVR
# svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
# svr_model.fit(X_train, y_train)
# pred_svr = svr_model.predict(X_test)
# mae = mean_absolute_error(y_test, pred_svr)
# print(f'Mean Absolute Error: {mae}')


## MLPRegressor
# from sklearn.neural_network import MLPRegressor
# nn_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
# nn_model.fit(X_train, y_train)
# pred_nn = nn_model.predict(X_test)
# mae = mean_absolute_error(y_test, pred_knn)
# print(f'Mean Absolute Error: {mae}')


## SVC
# from sklearn.svm import SVC
# svm_model = SVC(kernel='linear', C=1.0, random_state=42)
# svm_model.fit(X_train, y_train)
# pred_svm = svm_clf.predict(X_test)
# mae = mean_absolute_error(y_test, pred_svm)
# print(f'Mean Absolute Error: {mae}')


## NN
# pip install tensorflow
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
# # Build the model
# model_NN = Sequential()
# model_NN.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
# model_NN.add(Dense(64, activation='relu'))
# model_NN.add(Dense(1))  # Output layer for regression
# # Step 5: Compile the model
# model_NN.compile(optimizer='adam', loss='mean_absolute_error')
# # Step 6: Train the model
# history = model_NN.fit(X_train, y_train, epochs=100, batch_size=100, validation_split=0.3, verbose=1)
# # Step 7: Evaluate the model
# y_pred = model_NN.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# print(f'mean_absolute_error: {mae}')
# Plot training history (optional)
# import matplotlib.pyplot as plt
# plt.plot(history.history['loss'], label='train_loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()




# # # ------------Apply on test_data--------------

test_data = pd.read_csv('test_data.csv')

test_data['row_indx'] = range(1, 60833)

def fill_mode_per_group(group):
    mode_values = group['CurrentGameMode'].mode()
    if not mode_values.empty:
        group['CurrentGameMode'] = group['CurrentGameMode'].fillna(mode_values.iloc[0])
    return group
# Apply the function to each group by 'UserID'
test_data = test_data.groupby('UserID').apply(fill_mode_per_group)

# Reset the index after grouping
test_data.reset_index(drop=True, inplace=True)
mode_value = test_data['CurrentGameMode'].mode()[0]
test_data['CurrentGameMode'].fillna(mode_value, inplace=True)
test_data.drop("QuestionType", axis = 1 , inplace = True)
test_data.drop("CurrentTask", axis = 1 , inplace = True)
test_data.drop("LastTaskCompleted", axis = 1 , inplace = True)
def fill_mean_per_group(group):
    group['LevelProgressionAmount'].fillna(group['LevelProgressionAmount'].mean(), inplace=True)
    return group

# Apply the function to each group by 'UserID'
test_data = test_data.groupby('UserID').apply(fill_mean_per_group)

# Reset the index after grouping
test_data.reset_index(drop=True, inplace=True)
test_data['LevelProgressionAmount'].fillna(test_data['LevelProgressionAmount'].mean(), inplace=True)
from sklearn.preprocessing import MinMaxScaler
test_data['QuestionTiming'] = test_data['QuestionTiming'].map({'System Initiated': 1, 'User Initiated': 0})
test_data['TimeUtc'] = pd.to_datetime(test_data['TimeUtc'], errors='coerce')
test_data['hour'] = test_data['TimeUtc'].dt.hour
test_data['dayofweek'] = test_data['TimeUtc'].dt.dayofweek
test_data['day'] = test_data['TimeUtc'].dt.day
test_data['month'] = test_data['TimeUtc'].dt.month
test_data['year'] = test_data['TimeUtc'].dt.year

# Drop the original TimeUtc column
test_data.drop('TimeUtc', axis=1, inplace=True)

test_data['CurrentGameMode'] = LabelEncoder().fit_transform(test_data['CurrentGameMode'])
test_data['UserID_encoded'] = LabelEncoder().fit_transform(test_data['UserID'])

scaler=MinMaxScaler()
test_data['CurrentSessionLength'] = scaler.fit_transform(test_data[['CurrentSessionLength']])

# left Join
test_data = pd.merge(test_data,unique_data[['UserID','UserResponseMean','UserResponseStd']], on='UserID', how='left')

# fill na
test_data.isnull().mean() * 100

test_data['UserResponseMean'].fillna(test_data['UserResponseMean'].mean(), inplace=True)
test_data['UserResponseStd'].fillna(test_data['UserResponseStd'].mean(), inplace=True)
test_data = test_data.sort_values(by='row_indx')
X_test = test_data.drop(['UserID','row_indx','ResponseValue'], axis=1, errors='ignore')
test_predictions = model.predict(X_test)
    
# Save predictions
test_predictions = np.round(test_predictions, 0)
test_predictions= test_predictions.astype(int)
test_data['ResponseValue'] = test_predictions

# check that the shapes all make sense
print(test_data.shape, test_predictions.shape)

# create the CSV
np.savetxt("predicted.csv", test_predictions)
