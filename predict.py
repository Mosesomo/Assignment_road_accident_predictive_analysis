import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

# Our road accident dataset
data = pd.read_csv('data.csv')
new_data = pd.DataFrame(data)

# Selecting the independent variables
new_data =new_data.loc[:,['Time', 'Weather_conditions', 'Road_surface_conditions', 'Light_conditions', 'Number_of_vehicles_involved', 'Age_band_of_driver', 'Accident_severity']]

# Converting categorical independent variables 
# into numerical format
new_data = pd.get_dummies(new_data, columns=['Weather_conditions', 'Road_surface_conditions', 'Light_conditions', 'Age_band_of_driver'])
# converting accident severity into numericat format
label_encoder = LabelEncoder()
new_data['Accident_severity_encode'] = label_encoder.fit_transform(new_data['Accident_severity'])

# Independent variables
X = new_data[['Number_of_vehicles_involved', 'Weather_conditions_Normal', 'Weather_conditions_Raining', 'Road_surface_conditions_Dry', 'Road_surface_conditions_Wet or damp', 'Light_conditions_Darkness - lights lit', 'Light_conditions_Daylight', 'Age_band_of_driver_18-30', 'Age_band_of_driver_31-50']]

#selcting the dependent variable
y = new_data['Accident_severity_encode']

# spliting the datasets for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# creating training model using linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# making predictions
y_pred = model.predict(X_test)
predicted_labels = label_encoder.inverse_transform(y_pred.astype(int))

results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predicted_labels})
print(results_df.head(50))

# Evaluating the model
print('\nMean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('R-squared:', metrics.r2_score(y_test, y_pred))