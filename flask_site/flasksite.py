from flask import Flask, render_template, url_for, flash, redirect, request, jsonify
from forms import DataForm
from NN_testing import test_nn
app = Flask(__name__)

app.config['SECRET_KEY'] = 'd800790f750f387fed926d434e29fede'

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html', title='About')

@app.route("/data", methods=['GET', 'POST'])
def data():
    form = DataForm()
    return render_template('data.html', title='Data', form=form)

@app.route("/downloads")
def downloads():
    return render_template('downloads.html', title='Downloads')

@app.route("/testing", methods=['POST'])
def testing():   
    location = request.form['Location']
    minTemp = request.form['MinTemp']
    maxTemp = request.form['MaxTemp']
    rainfall  = request.form['Rainfall']
    windGustDir = request.form['WindGustDir']
    windGustSpeed = request.form['WindGustSpeed']
    windDir9am = request.form['WindDir9am']
    windDir3pm = request.form['WindDir3pm']
    windSpeed9am = request.form['WindSpeed9am']
    windSpeed3pm = request.form['WindSpeed3pm']
    humidity9am = request.form['Humidity9am']
    humidity3pm = request.form['Humidity3pm']
    pressure9am = request.form['Pressure9am']
    pressure3pm = request.form['Pressure3pm']
    temp9am = request.form['Temp9am']
    temp3pm = request.form['Temp3pm']

    testData = [location, minTemp, maxTemp, rainfall, windGustDir, windGustSpeed, 
    windDir9am, windDir3pm, windSpeed9am, windSpeed3pm, humidity9am, humidity3pm, 
    pressure9am, pressure3pm, temp9am, temp3pm]

    test_nn(testData)

    return jsonify({'res' : 'result!'})

if __name__ == '__main__':
    app.run(debug=True)