
from flask import Flask, render_template, request
import pickle
import pandas as pd

# Load model and dataset
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("data/health_data_final.csv")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    details = {}
    if request.method == "POST":
        symptoms_input = request.form["symptoms"]
        symptoms = [s.strip().lower() for s in symptoms_input.split(",")]
        prediction = model.predict([", ".join(symptoms)])[0]

        row = df[df["disease"].str.lower() == prediction.lower()]
        if not row.empty:
            details = {
                "Disease": row["disease"].values[0],
                "Description": row["description"].values[0],
                "Precautions": row["precautions"].values[0],
                "Medications": row["medications"].values[0],
                "Workouts": row["workouts"].values[0],
                "Diets": row["diets"].values[0]
            }

    return render_template("index.html", prediction=prediction, details=details)

if __name__ == "__main__":
    app.run(debug=True)
