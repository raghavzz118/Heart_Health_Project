from flask import Flask,request,render_template     
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for our Homepage

@app.route("/")

def index():
    return render_template("index.html")

@app.route("/predictdata",methods=["GET","POST"])

def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            Gender=request.form.get("Gender"),
            age=float(request.form.get("age")),
            education=request.form.get("education"),
            currentSmoker=request.form.get("currentSmoker"),
            cigsPerDay=float(request.form.get("cigsPerDay")),
            BP_Meds=request.form.get("BP_Meds"),
            prevalentStroke=request.form.get("prevalentStroke"),
            prevalentHyp=request.form.get("prevalentHyp"),
            diabetes=request.form.get("diabetes"),
            tot_cholesterol=float(request.form.get("tot_cholesterol")),
            Systolic_BP=float(request.form.get("Systolic_BP")),
            Diastolic_BP=float(request.form.get("Diastolic_BP")),
            BMI=float(request.form.get("BMI")),
            heartRate=float(request.form.get("heartRate")),
            glucose=float(request.form.get("glucose"))
        )

        pred_df=data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline=PredictPipeline()

        results=predict_pipeline.predict(pred_df)
        result_text = "High Risk" if results[0] == 1 else "Low Risk"
        return render_template("home.html", results=result_text)


if __name__=="__main__":
    app.run(host="0.0.0.0")