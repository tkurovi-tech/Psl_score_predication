# PSL 1st Innings Score Predictor
**SZABIST · Department of Robotics & Artificial Intelligence · ML Lab Project**

Real-time PSL match score prediction using a Random Forest model trained on 22,000+ ball-by-ball records.

---

## 📁 Project Structure

```
psl_predictor/
├── app.py                  ← Flask web application (backend)
├── train_model.py          ← Standalone training script
├── requirements.txt        ← Python dependencies
├── model.pkl               ← Trained Random Forest model
├── le_venue.pkl            ← LabelEncoder: venue
├── le_team.pkl             ← LabelEncoder: team
├── le_opp.pkl              ← LabelEncoder: opposition
└── templates/
    └── index.html          ← Web UI (dark cricket-themed)
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Retrain the model
```bash
python train_model.py --data PSL_1st_Inning_Clean.csv --out-dir .
```
This generates `model.pkl`, `le_*.pkl`, `training_report.txt`, and `model_performance.png`.

### 3. Start the web app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## 🎯 Model Performance

| Model              | MAE (runs) | R²     |
|--------------------|-----------|--------|
| **Random Forest**  | **1.11**  | **0.9908** |
| Gradient Boosting  | 11.13     | 0.7633 |
| Ridge Regression   | 14.50     | 0.6011 |

---

## 🔢 Features Used

| Feature              | Description                          |
|----------------------|--------------------------------------|
| `venue`              | Stadium / ground                     |
| `team`               | Batting team                         |
| `opposition`         | Bowling team                         |
| `Over Number`        | Current over (5–19)                  |
| `Ball Number`        | Ball within the over (1–6)           |
| `current_run`        | Runs scored so far                   |
| `run_rate`           | Runs per over (computed)             |
| `wickets_left`       | Wickets remaining                    |
| `runs_till_5_over`   | Power-play score (first 5 overs)     |

---

## 🌐 API Endpoint

### `POST /predict`
```json
{
  "venue": "Gaddafi Stadium, Lahore",
  "team": "Lahore Qalandars",
  "opposition": "Karachi Kings",
  "over_number": 12,
  "ball_number": 3,
  "current_run": 95,
  "wickets_left": 7,
  "runs_till_5_over": 42
}
```
**Response:**
```json
{
  "predicted_score": 172,
  "current_run": 95,
  "run_rate": 7.41,
  "runs_to_come": 77,
  "balls_remaining": 45,
  "projected_rr": 10.27,
  "wickets_left": 7
}
```

---

## 👥 Team Members

| Name             | Reg. No.  |
|------------------|-----------|
| Tanveer Abbas    | 23108401  |
| Sajid Ali        | 23108378  |
| Mushaf Ali       | 23108317  |
| Adnan Haider     | 23108285  |
| Muhammad Hasnain | 23108312  |

**Supervisor:** Sir Anas Khalid  
**Course:** Machine Learning Lab · ANN-LabBS (AI)-5A · SZABIST-ISB
