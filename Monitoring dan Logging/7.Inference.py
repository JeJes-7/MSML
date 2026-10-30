from flask import Flask, request, jsonify
import joblib
import time
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "Membangun_model",
    "model_random_forest",
    "model.pkl"
)

model = joblib.load(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()
        if payload is None or "data" not in payload:
            return jsonify({"error": "Invalid JSON payload"}), 400

        data = payload["data"]

        if not isinstance(data, list):
            return jsonify({"error": "Data must be a list"}), 400

        # FIX JUMLAH FITUR
        if len(data) == 13:
            data.append(0)

        start = time.time()
        prediction = model.predict([data]).tolist()
        latency = time.time() - start

        return jsonify({
            "prediction": prediction,
            "latency": latency
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("SERVER STARTING...")
    app.run(
        host="127.0.0.1",
        port=5050,
        debug=False,
        use_reloader=False
    )
