import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="UFC Fight Predictor", page_icon="🥊")


@st.cache_resource
def load_bundle():
    return joblib.load("ufc_model_v2.pkl")


@st.cache_data
def load_state():
    s = pd.read_csv("fighter_state.csv")
    s["last_fight"] = pd.to_datetime(s["last_fight"])
    s = s.dropna(subset=["Full Name"]).copy()

    label = s["Full Name"].astype(str)
    dup = label.duplicated(keep=False)
    label = label.where(~dup, label + " (last fought " + s["last_fight"].dt.year.astype(str) + ")")
    dup = label.duplicated(keep=False)
    label = label.where(~dup, label + " #" + s["Fighter_Id"].astype(str).str[:4])
    s["label"] = label
    return s.set_index("Fighter_Id")


bundle = load_bundle()
state = load_state()
as_of = pd.to_datetime(bundle["as_of"])


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def proba_sym(model, X):
    return 0.5 * (model.predict_proba(X)[:, 1] + 1 - model.predict_proba(-X)[:, 1])


def win_prob(a, b):
    """Probability that fighter A beats fighter B (a, b are rows of fighter_state)."""
    x = np.array([[a[c] - b[c] for c in bundle["base_cols"]] + [a["elo"] - b["elo"]]], dtype=float)
    p = np.mean([proba_sym(m, x)[0] for m in bundle["models"].values()])
    z = bundle["platt_coef"] * logit(p) 
    return float(1 / (1 + np.exp(-z)))


def fmt_ctrl(sec):
    sec = int(round(sec))
    return f"{sec // 60}:{sec % 60:02d}"


def profile(r):
    s = int(round(r["streak"]))
    streak = f"W{s}" if s > 0 else (f"L{-s}" if s < 0 else "-")
    return {
        "Elo rating": f"{r['elo']:.0f}",
        "UFC fights (wins + losses)": f"{np.expm1(r['exp']):.0f}",
        "Current streak": streak,
        "Win rate, last 3 fights": f"{r['last3_win']:.0%}",
        "KO win rate": f"{r['ko_win_rate']:.0%}",
        "Submission win rate": f"{r['sub_win_rate']:.0%}",
        "Knockdowns per fight": f"{r['kd_pf']:.2f}",
        "Takedowns per fight": f"{r['td_pf']:.2f}",
        "Control time per fight": fmt_ctrl(r["ctrl_pf"]),
        "Last fight": r["last_fight"].strftime("%b %Y"),
    }


def show_prediction(id_a, id_b):
    a, b = state.loc[id_a], state.loc[id_b]
    pa = win_prob(a, b)
    pb = 1 - pa
    winner, conf = (a["Full Name"], pa) if pa >= 0.5 else (b["Full Name"], pb)

    st.markdown("---")
    st.metric("Predicted winner", winner)
    c1, c2 = st.columns(2)
    c1.metric(f"{a['Full Name']} wins", f"{pa:.0%}")
    c2.metric(f"{b['Full Name']} wins", f"{pb:.0%}")

    if conf < 0.6:
        st.info("This is close to a coin flip. In testing, picks below 60% confidence were right only about 57% of the time.")

    st.markdown("---")
    st.subheader("Fighter comparison")
    st.caption("Stats use only fights up to " + as_of.strftime("%B %Y") + ". Rates are shrunk toward the average for fighters with few fights.")
    st.table(pd.DataFrame({a["Full Name"]: profile(a), b["Full Name"]: profile(b)}))


tab1, tab2 = st.tabs(["Predict a Fight", "About"])

with tab1:
    st.title("UFC Fight Predictor")
    st.subheader("Select two fighters")

    weight_classes = sorted(state["Weight_Class"].dropna().unique())
    wc = st.selectbox("Weight class", weight_classes, key="wc")

    cutoff = as_of - pd.DateOffset(years=3)
    active_only = st.checkbox(f"Only fighters who fought since {cutoff.year}", value=True, key="active")

    pool = state[state["Weight_Class"] == wc]
    if active_only:
        pool = pool[pool["last_fight"] >= cutoff]
    pool = pool.sort_values("label")
    options = pool["label"].tolist()
    label_to_id = dict(zip(pool["label"], pool.index))

    if len(options) < 2:
        st.warning("Not enough fighters in this view. Untick the filter or pick another weight class.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fighter_a = st.selectbox("Fighter A", options, key="fa")
        with col2:
            fighter_b = st.selectbox("Fighter B", options, index=1, key="fb")

        st.caption("Order does not matter: swapping A and B gives the mirrored result.")

        if st.button("Predict winner"):
            if fighter_a == fighter_b:
                st.warning("Pick two different fighters.")
            else:
                show_prediction(label_to_id[fighter_a], label_to_id[fighter_b])

with tab2:
    st.subheader("About this project")
    st.write(
        "A personal project combining my interest in combat sports with machine learning: "
        "can a fighter's history predict who wins? To some extent, yes."
    )
    st.write(
        "The first version of this app reported 70.3% accuracy, but I later found it used a career-stats "
        "snapshot that included the very fights it was predicting (data leakage). I rebuilt it so every "
        "stat is computed only from a fighter's earlier fights, and tested it on fights it had never seen."
    )

    st.markdown("---")
    st.subheader("Model info")
    st.write("**Model:** average of four models (Logistic Regression, Random Forest, Gradient Boosting, LightGBM)")
    st.write("**Test accuracy:** about 67.5% on 1,015 fights from 2024 onward (roughly ±3 points), "
             "using a model trained only on earlier fights. The Elo-only baseline scored 58.9%.")
    st.write("**Confidence check:** picks at 70-80% confidence were right 73% of the time, and above 80% were right 82% "
             "of the time (108 fights).")
    st.write("**Features:** 20 difference features built from each fighter's earlier fights: Elo rating, win rate, "
             "finish rates, knockdowns, takedowns, control time, striking accuracy, recent form, win streak, "
             "layoff and height.")
    st.write(f"**Data as of:** {as_of.strftime('%B %Y')} (the deployed model is retrained on all fights up to then)")

    st.markdown("---")
    st.subheader("Data source")
    st.write("**Dataset:** UFC 2025 Dataset: Fights, Fighters & Events")
    st.write("**Source:** Kaggle (aminealibi), originally from ufcstats.com")

    st.markdown("---")
    st.subheader("Limitations")
    st.write("- No betting odds, injuries, camp changes, age or reach. The market knows things this model cannot.")
    st.write("- Fighters with few UFC fights are pulled toward the average, so debutants are hard to judge.")
    st.write("- Layoff is measured to the end of the dataset, not to the date of a real booked fight.")
    st.write("- MMA is high variance. Even an 80% pick loses about one fight in five.")
    st.write("- For entertainment and learning, not betting advice.")