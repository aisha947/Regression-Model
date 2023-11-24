import pandas as pd
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify

# Read train data
train_data = pd.read_csv('trainRegression.csv')
# Convert train data to Numpy Array
train_numpydata = train_data.to_numpy()

# Initialize the linear regression model and fit it with the training data
model = LinearRegression()
X_train = train_numpydata[:, 0].reshape(-1, 1)  # Assuming 'X' is the feature column
y_train = train_numpydata[:, 1]  # Assuming 'R' is the target column
model.fit(X_train, y_train)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        input_data = request.json.get('X')  # Assuming the input is a single value

        # Validate input data
        if input_data is None:
            raise ValueError("Input data 'X' is missing in the request.")

        # Make prediction
        prediction = model.predict([[input_data]])

        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        # Log the exception details for debugging
        print(f"Exception: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the request.'})

if __name__ == '__main__':
    app.run(debug=False, port=5002)  # Set the port explicitly

