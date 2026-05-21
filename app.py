"""
PSL 1st Innings Score Predictor
Flask Web Application
SZABIST - Department of Robotics & Artificial Intelligence
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load artifacts ────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE, "model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE, "le_venue.pkl"), "rb") as f:
    le_venue = pickle.load(f)
with open(os.path.join(BASE, "le_team.pkl"), "rb") as f:
    le_team = pickle.load(f)
with open(os.path.join(BASE, "le_opp.pkl"), "rb") as f:
    le_opp = pickle.load(f)

VENUES = sorted(le_venue.classes_.tolist())
TEAMS  = sorted(le_team.classes_.tolist())

# Model performance metadata
MODEL_STATS = {"mae": 1.11, "r2": 0.9908, "accuracy": "99.08%"}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template(
        "index.html",
        venues=VENUES,
        teams=TEAMS,
        model_stats=MODEL_STATS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        venue        = data["venue"]
        team         = data["team"]
        opposition   = data["opposition"]
        over_num     = float(data["over_number"])
        ball_num     = float(data["ball_number"])
        current_run  = float(data["current_run"])
        wickets_left = float(data["wickets_left"])
        runs_5       = float(data["runs_till_5_over"])

        # Validate inputs
        if over_num < 5 or over_num > 19:
            return jsonify({"error": "Over Number must be between 5 and 19."}), 400
        if ball_num < 1 or ball_num > 6:
            return jsonify({"error": "Ball Number must be between 1 and 6."}), 400
        if current_run < 0:
            return jsonify({"error": "Current runs cannot be negative."}), 400
        if wickets_left < 0 or wickets_left > 10:
            return jsonify({"error": "Wickets left must be between 0 and 10."}), 400
        if team == opposition:
            return jsonify({"error": "Batting team and opposition cannot be the same."}), 400

        # Encode
        venue_enc = le_venue.transform([venue])[0]
        team_enc  = le_team.transform([team])[0]
        opp_enc   = le_opp.transform([opposition])[0]

        # Compute run rate
        total_balls = (over_num - 1) * 6 + ball_num
        run_rate    = (current_run / total_balls * 6) if total_balls > 0 else 0.0

        # Feature vector (must match training order)
        X = np.array([[venue_enc, team_enc, opp_enc,
                       over_num, ball_num,
                       current_run, run_rate,
                       wickets_left, runs_5]])

        predicted_score = float(model.predict(X)[0])
        predicted_score = max(current_run, round(predicted_score))

        # Extra context
        balls_remaining = (20 - over_num) * 6 + (6 - ball_num)
        runs_needed_avg = predicted_score - current_run
        proj_rr         = round((runs_needed_avg / balls_remaining * 6), 2) if balls_remaining > 0 else 0

        return jsonify({
            "predicted_score": int(predicted_score),
            "current_run":     int(current_run),
            "run_rate":        round(run_rate, 2),
            "runs_to_come":    max(0, int(predicted_score - current_run)),
            "balls_remaining": balls_remaining,
            "projected_rr":    proj_rr,
            "wickets_left":    int(wickets_left),
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model-info")
def model_info():
    return jsonify({
        "model":     "Random Forest Regressor",
        "mae":        MODEL_STATS["mae"],
        "r2":         MODEL_STATS["r2"],
        "accuracy":   MODEL_STATS["accuracy"],
        "features":  ["venue", "team", "opposition", "over_number", "ball_number",
                      "current_run", "run_rate", "wickets_left", "runs_till_5_over"],
        "teams":     TEAMS,
        "venues":    VENUES,
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  PSL Score Predictor - SZABIST ML Project")
    print("  Open http://127.0.0.1:5000  in your browser")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
