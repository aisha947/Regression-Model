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

from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
model = LinearRegression()
model.coef_ = 2.0  # Replace with the actual coefficient
model.intercept_ = 1.0  # Replace with the actual intercept

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        input_data = request.json['X']  # Assuming the input is a single value

        # Make prediction
        prediction = model.predict([[input_data]])

        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

