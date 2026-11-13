import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"status": "running", "message": "Backend API is active."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # Add your model prediction logic here
        # Example placeholder:
        input_text = data.get("text", "")
        result = {"prediction": "success", "input": input_text}
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
