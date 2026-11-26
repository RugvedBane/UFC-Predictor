# UFC Fight Predictor

A machine learning web app that predicts the outcome of UFC fights from each fighter's history before the fight.

**Live app:** [https://ufc-predictor-rugvedbane.streamlit.app](https://ufc-predictor-rugvedbane.streamlit.app)

---

## About the Project

I built this because I wanted to work on something I actually find interesting: UFC. The question was whether a fighter's past results can predict who wins. To a useful extent, they can: the model picks the right winner in about 67.5% of fights it has never seen.

The first version of this project reported 70.3% accuracy, but that number was inflated. It used a career-stats snapshot that included the very fights it was predicting (data leakage), and it split train and test at random. I rebuilt the whole pipeline so that every stat comes only from fights that happened before the one being predicted, and the model is tested on future fights. The honest number is lower, and it comes with a baseline to compare against.

---

## How It Works

1. Select a weight class. The dropdowns only list fighters from that division, and by default only fighters who have fought in the last 3 years.
2. Choose Fighter A and Fighter B.
3. Click Predict. The app shows each fighter's win probability and the predicted winner.
4. A comparison table shows the stats behind the prediction: Elo rating, streak, recent form, KO and submission rates, knockdowns, takedowns and control time.

Order does not matter. Swapping A and B gives exactly the mirrored probabilities.

---

## Results

Tested on 1,015 UFC fights from 2024 onward. The models were trained only on earlier fights, and the test set was not used for tuning.

| Model | Accuracy | Log loss | AUC |
|---|---|---|---|
| Elo rating only (baseline) | 58.9% | 0.6642 | 0.6442 |
| Logistic Regression | 67.6% | 0.6121 | 0.7261 |
| Random Forest | 67.3% | 0.6175 | 0.7217 |
| Gradient Boosting | 67.3% | 0.6166 | 0.7206 |
| LightGBM | 67.2% | 0.6152 | 0.7225 |
| **Blend (average of the four)** | **67.5%** | **0.6139** | **0.7245** |

A coin flip scores 50% accuracy and 0.693 log loss.

What the results say:

- **The model clearly beats the baseline.** It is about 9 points above plain Elo, and Elo here is untuned (K=32), so a tuned Elo would narrow the gap somewhat.
- **The four models are statistically tied.** With 1,015 test fights the margin of error is about ±3 points. Simple Logistic Regression matched the ensemble, so the ensemble added robustness rather than accuracy.
- **Confidence means something.** Accuracy rises with the model's confidence:

| Model confidence | Fights | Actual accuracy |
|---|---|---|
| 50% to 60% | 308 | 56.5% |
| 60% to 70% | 304 | 68.1% |
| 70% to 80% | 295 | 72.9% |
| 80% to 100% | 108 | 82.4% |

The top bin has only 108 fights, so treat its 82% as roughly ±7 points.

---

## How Leakage Is Avoided

- **Point-in-time features.** Every fighter stat is computed from that fighter's earlier fights only (each row's stats exclude its own fight). For example, a fighter's win rate before a 2019 fight uses nothing from 2019 or later. I checked this directly: changing a fight's result and stats does not change that fight's own features.
- **Time-based split.** Training on 2001 to 2022, validation on 2023, test on 2024 onward. The deployed model is retrained on all fights through December 2025.
- **No position bias.** The dataset lists Fighter 1 as the winner about 64% of the time, so a model can cheat by learning positions. I randomly swap sides, train on both orientations, and average each prediction with its mirror image.
- **Small samples are shrunk.** A debutant with one lucky fight is pulled toward the average instead of looking like a superstar.

---

## Features

20 difference features (Fighter A minus Fighter B), all built from earlier fights:

- Elo rating
- Experience (number of UFC fights)
- Win rate
- KO win rate, submission win rate, and finish-loss rate
- Knockdowns landed and absorbed per fight
- Strikes landed and absorbed per fight
- Takedowns landed and absorbed per fight
- Control time landed and absorbed per fight
- Submission attempts per fight
- Significant strike accuracy
- Win rate over the last 3 fights
- Current win or loss streak
- Layoff since the last fight
- Height

---

## Model

The final prediction is the average of four models' probabilities: Logistic Regression, Random Forest, Gradient Boosting and LightGBM. Hyperparameters are deliberately conservative and were not tuned on the test set. I also tried Platt probability calibration, but it made log loss worse, so the raw blend is used.

---

## Data Source

- Dataset: UFC 2025 Dataset: Fights, Fighters and Events
- Source: Kaggle (aminealibi), originally from ufcstats.com
- Coverage: about 8,300 decided fights through December 2025

---

## Limitations

- No betting odds, injuries, camp changes, age or reach. The betting market knows things this model cannot.
- Fighters with few UFC fights are pulled toward the average, so debutants are hard to judge.
- Layoff is measured to the end of the dataset, not to a real booked fight date.
- MMA is high variance. Even an 80% pick loses about one fight in five.
- Accuracy is measured on 2024 and 2025 fights. Future performance can differ.

---

## Tech Stack

- Python
- Pandas and NumPy: data cleaning and feature engineering
- Scikit-learn and LightGBM: modeling
- Streamlit: web app
- GitHub and Streamlit Cloud: deployment

---

## Project Structure

```
UFC-Predictor/
├── app.py
├── ufc_model_v2.pkl
├── fighter_state.csv
└── requirements.txt
```

`fighter_state.csv` holds each fighter's current stats and Elo rating, and `ufc_model_v2.pkl` holds the trained models.

---

## Run Locally

```bash
git clone https://github.com/RugvedBane/UFC-Predictor.git
cd UFC-Predictor
pip install -r requirements.txt
streamlit run app.py
```
