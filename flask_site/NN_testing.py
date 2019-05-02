from flask import Flask
from forms import DataForm
from RBF_Backend import get_predict, load_model

def test_nn(TestData):
    print("Loading data.")
    '''dataTest = [
        DataForm.Location
        , DataForm.MinTemp
        , DataForm.MaxTemp
        , DataForm.Rainfall
        , DataForm.WindGustDir
        , DataForm.WindGustSpeed
        , DataForm.WindDir9am
        , DataForm.WindDir3pm
        , DataForm.WindSpeed9am
        , DataForm.WindSpeed3pm
        , DataForm.Humidity9am
        , DataForm.Humidity3pm
        , DataForm.Pressure9am
        , DataForm.Pressure3pm
        , DataForm.Temp9am
        , DataForm.Temp3pm
    ]'''
    
    print(TestData)

    print("1. Loading model.")
    center, delta, w = load_model("messidor_center.txt", "messidor_delta.txt", "messidor_weight.txt")
    print("2. Getting prediction.")
    result = get_predict(TestData, center, delta, w)
    print('result', result)
    print("3. Saving result")
    res = save_predict(result)


    