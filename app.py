from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the saved model
with open('models/rf_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    sex = int(request.form['sex'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = int(request.form['smoker'])
    region = int(request.form['region'])

    # Prepare input data
    input_data = [[age, sex, bmi, children, smoker, region]]

    # Make prediction
    prediction = model.predict(input_data)[0]

    return render_template('op.html', pred=f"Predicted insurance charges based on given data: Rs.{prediction:.2f}")

if __name__ == '__main__':
    app.run(debug=True)