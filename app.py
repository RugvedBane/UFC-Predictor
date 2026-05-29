import streamlit as st
import pandas as pd
import joblib
import json

# Load model and data
model = joblib.load('ufc_model.pkl')
fighters = pd.read_csv('fighters_stats_clean.csv')
with open('feature_cols.json') as f:
    feature_cols = json.load(f)

st.title("UFC Fight Predictor")

tab1, tab2 = st.tabs(["Predict a Fight", "About"])

with tab1:
    st.subheader("Select two fighters")

    # Weight class filter
    weight_classes = fighters['Weight_Class'].dropna().unique().tolist()
    weight_class = st.selectbox("Select Weight Class", sorted(weight_classes))

    # Filter fighters by weight class
    filtered_fighters = fighters[fighters['Weight_Class'] == weight_class]['Full Name'].dropna().sort_values().tolist()

    col1, col2 = st.columns(2)
    with col1:
        fighter1 = st.selectbox("Fighter 1 (Red Corner)", filtered_fighters)
    with col2:
        fighter2 = st.selectbox("Fighter 2 (Blue Corner)", filtered_fighters)

    if st.button("Predict Winner"):
        f1 = fighters[fighters['Full Name'] == fighter1].iloc[0]
        f2 = fighters[fighters['Full Name'] == fighter2].iloc[0]

        # Build diff features
        features = {
            'KO_rate_diff':     f1['KO Rate'] - f2['KO Rate'],
            'SUB_rate_diff':    f1['SUB Rate'] - f2['SUB Rate'],
            'DEC_rate_diff':    f1['DEC Rate'] - f2['DEC Rate'],
            'win_diff':         f1['W'] - f2['W'],
            'loss_diff':        f1['L'] - f2['L'],
            'win_rate_diff':    (f1['W'] / (f1['W'] + f1['L'] + 1)) - (f2['W'] / (f2['W'] + f2['L'] + 1)),
            'reach_diff':       f1['Ht.'] - f2['Ht.'],
            'KD_avg_diff':      f1['KD'] - f2['KD'],
            'TD_avg_diff':      f1['TD'] - f2['TD'],
            'Ctrl_avg_diff':    f1['Ctrl'] - f2['Ctrl'],
            'sig_str_acc_diff': f1['Sig. Str. %'] - f2['Sig. Str. %'],
            'avg_time_diff':    f1['Avg Fight Time'] - f2['Avg Fight Time'],
            'striker_diff':     f1['Striker_Membership'] - f2['Striker_Membership'],
            'wrestler_diff':    f1['Wrestler_Membership'] - f2['Wrestler_Membership'],
        }

        X = pd.DataFrame([features])
        prob = model.predict_proba(X)[0]
        prediction = model.predict(X)[0]

        winner = fighter1 if prediction == 1 else fighter2
        confidence = prob[1] if prediction == 1 else prob[0]

        st.markdown("---")
        st.metric("Predicted Winner", winner)
        st.metric("Confidence", f"{confidence:.0%}")

        st.markdown("---")
        st.subheader("Fighter Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{fighter1}**")
            st.write(f"Record: {int(f1['W'])}W - {int(f1['L'])}L")
            st.write(f"KO Rate: {f1['KO Rate']:.0%}")
            st.write(f"SUB Rate: {f1['SUB Rate']:.0%}")
            st.write(f"DEC Rate: {f1['DEC Rate']:.0%}")
        with col2:
            st.write(f"**{fighter2}**")
            st.write(f"Record: {int(f2['W'])}W - {int(f2['L'])}L")
            st.write(f"KO Rate: {f2['KO Rate']:.0%}")
            st.write(f"SUB Rate: {f2['SUB Rate']:.0%}")
            st.write(f"DEC Rate: {f2['DEC Rate']:.0%}")

with tab2:
    st.subheader("About this Project")

    st.write("""
    I built this UFC Fight Predictor as a personal project to combine my interest in 
    combat sports with machine learning. The idea was simple — can historical fighter 
    stats predict who wins a fight? Turns out, to some extent, yes.
    """)

    st.write("""
    This project gave me hands-on experience with the full ML pipeline — data collection, 
    cleaning, feature engineering, model selection, and deployment. It also taught me 
    the importance of avoiding data leakage, which is easy to get wrong in sports prediction.
    """)

    st.markdown("---")
    st.subheader("Model Info")
    st.write("**Algorithm:** Gradient Boosting Classifier")
    st.write("**Accuracy:** 70.3%")
    st.write("**Training Data:** 8,294 UFC fights (1994 – 2025)")
    st.write("**Features Used:** 14 difference features (win rate, KO rate, SUB rate, control time, etc.)")

    st.markdown("---")
    st.subheader("Data Source")
    st.write("**Dataset:** UFC 2025 Dataset — Fights, Fighters & Events")
    st.write("**Source:** Kaggle (aminealibi)")
    st.write("**Fighter Stats:** ufcstats.com (via dataset)")

    st.markdown("---")
    st.subheader("Limitations")
    st.write("- Model does not account for recent form — a fighter on a 5 fight win streak vs a fighter returning from injury looks the same to the model")
    st.write("- Experience bias — fighters with more fights are favoured even if a newer fighter has better quality wins")
    st.write("- Upsets are inherently unpredictable — MMA has a high variance nature that no model can fully capture")
    st.write("- Dataset last updated April 2025 — new fighters may not be in the system")