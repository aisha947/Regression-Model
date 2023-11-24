import pandas as pd

# Read train data
train_data = pd.read_csv('trainRegression.csv')
# Convert train data to Numpy Array
train_numpydata = train_data.to_numpy()

from sklearn.linear_model import LinearRegression

# Initialize the linear regression model
model = LinearRegression()

# Prepare the train data
X_train = train_numpydata[:, 0].reshape(-1, 1)  # Assuming 'X' is the feature column
y_train = train_numpydata[:, 1]  # Assuming 'R' is the target column

# Fit the model
model.fit(X_train, y_train)

# Display the coefficients
print(f'Coefficient: {model.coef_}')
print(f'Intercept: {model.intercept_}')
