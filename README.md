# UFC Fight Predictor

A machine learning web app that predicts the outcome of UFC fights based on historical fighter statistics.

Live App: [https://rugvedbane-ufc-predictor.streamlit.app](https://ufc-predictor-rugvedbane.streamlit.app)

---

## About the Project

I built this project because I wanted to work on something I actually find interesting — UFC. The idea was to see if historical fighter stats can predict who wins a fight. Turns out, to some extent, they can.

This project covers the full ML pipeline — data collection, cleaning, feature engineering, model selection, and deployment. One of the most important things I learned while building this was how easy it is to accidentally introduce data leakage in sports prediction, and how to avoid it.

---

## How It Works

1. Select a weight class — this filters both dropdowns to the same division so you can only compare fighters who would actually fight each other
2. Choose two fighters from the dropdown
3. Click Predict — the model returns a predicted winner and a confidence score
4. A fighter comparison section shows key stats side by side so you can understand why the model predicted that result

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Gradient Boosting Classifier |
| Accuracy | 70.3% |
| Training Data | 8,294 UFC fights (1994 – 2025) |
| Features | 14 difference features |
| Cross Validation | 5-Fold GridSearchCV |
| Best Params | learning_rate=0.05, max_depth=3, n_estimators=100 |

### Features Used

All features are difference features — Fighter 1 stat minus Fighter 2 stat. This is important to avoid leakage since we can only use stats that are known before the fight happens.

- Win rate difference
- KO rate difference
- Submission rate difference
- Decision rate difference
- Win/Loss count difference
- Average knockdowns difference
- Average takedowns difference
- Control time difference
- Significant strike accuracy difference
- Average fight time difference
- Height difference
- Striker membership difference
- Wrestler membership difference

---

## Data Source

- Dataset: UFC 2025 Dataset — Fights, Fighters and Events
- Source: Kaggle (aminealibi)
- Coverage: UFC 2 through UFC 319 (1994 – 2025)

---

## Model Limitations

I want to be upfront about what this model cannot do:

- It does not account for recent form. A fighter on a 5-fight win streak looks the same as someone returning from a long injury layoff.
- It has an experience bias. Fighters with more career fights tend to be favoured even if a newer fighter has better quality wins against tougher opponents.
- It does not use betting odds data. Prediction markets are much better at capturing intangibles like camp changes, injuries, and motivation.
- MMA is inherently unpredictable. A single punch can end any fight regardless of what the stats say.
- The dataset was last updated April 2025, so very recently signed fighters may not be in the system.

---

## Tech Stack

- Python
- Pandas — data cleaning and feature engineering
- Scikit-learn — model training and evaluation
- Streamlit — web app
- Joblib — model saving
- GitHub and Streamlit Cloud — deployment

---

## Project Structure

```
UFC_Predictor/
├── app.py
├── ufc_model.pkl
├── feature_cols.json
├── fighters_stats_clean.csv
└── requirements.txt
```

---

## Run Locally

```bash
git clone https://github.com/RugvedBane/UFC-Predictor.git
cd UFC-Predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

Rugved Bane
- GitHub: https://github.com/RugvedBane
- LinkedIn: https://linkedin.com/in/rugvedbane
