from flask import Flask, render_template, request
import pickle as pkl
import pandas as pd

loaded_model = pkl.load(open('LinearRegressionModel1.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        name = request.form['name']
        company = request.form['company']
        year = int(request.form['year'])
        kms_driven = int(request.form['kms_driven'])
        fuel_type = request.form['fuel_type']

        sample = pd.DataFrame([{
            'name': name,
            'company': company,
            'year': year,
            'kms_driven': kms_driven,
            'fuel_type': fuel_type
        }])

        prediction = loaded_model.predict(sample)[0]

        return render_template('index.html',
                               prediction_text=f"Predicted Price: ₹ {round(prediction,2)} INR")

if __name__ == "__main__":
    app.run(debug=True)
