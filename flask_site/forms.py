from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FloatField
from wtforms.validators import DataRequired


class DataForm(FlaskForm): 
    Location = StringField('Location', validators=[DataRequired()])
    MinTemp = FloatField('Minimal Temperature', validators=[DataRequired()])
    MaxTemp = FloatField('Maximal Temperature', validators=[DataRequired()])
    Rainfall =FloatField('Rainfall', validators=[DataRequired()])
    WindGustDir = FloatField('Wind Gust Direction', validators=[DataRequired()])
    WindGustSpeed = FloatField('Wind Gust Speed', validators=[DataRequired()])
    WindDir9am = FloatField('Wind Direction at 9am', validators=[DataRequired()])
    WindDir3pm = FloatField('Wind Direction at 3pm', validators=[DataRequired()])
    WindSpeed9am = FloatField('Wind Speed at 9am', validators=[DataRequired()])
    WindSpeed3pm = FloatField('Wind Speed at 3pm', validators=[DataRequired()])
    Humidity9am = FloatField('Humidity at 9am', validators=[DataRequired()])
    Humidity3pm = FloatField('Humidity at 3pm', validators=[DataRequired()])
    Pressure9am = FloatField('Pressure at 9am', validators=[DataRequired()])
    Pressure3pm = FloatField('Pressure at 3pm', validators=[DataRequired()])
    Temp9am = FloatField('Temperature at 9am', validators=[DataRequired()])
    Temp3pm = FloatField('Temperature at 3pm', validators=[DataRequired()])
    submit = SubmitField('Submit data')