from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

from modules.authentication import init_db, register_user, authenticate_user
from modules.explainable_ai import calculate_permutation_importance
from modules.analytics import render_analytics
from modules.chatbot import chatbot_response

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "diabetes_model.joblib"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.joblib"
RESULTS_PATH = BASE_DIR / "models" / "histgradientboosting_results.csv"
DATA_PATH = BASE_DIR / "data" / "diabetes_health_indicators.csv"

st.set_page_config(page_title="Diabetes Awareness", page_icon="💙", layout="wide")

init_db()

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(FEATURE_PATH)

def show_login():
    st.title("💙 Diabetes Awareness")
    st.subheader("Health & Well-Being Platform")
    login_tab, register_tab = st.tabs(["Log in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email").strip().lower()
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            user = authenticate_user(email, password)
            if user:
                st.session_state["user"] = user
                st.rerun()
            else:
                st.error("Login failed. Check your email and password.")

    with register_tab:
        with st.form("register_form"):
            email = st.text_input("Email address", key="register_email").strip().lower()
            password = st.text_input("Password (at least 10 characters)", type="password", key="register_password")
            confirm = st.text_input("Confirm password", type="password", key="register_confirm")
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            if password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, message = register_user(email, password)
                if ok:
                    st.success(message + " You can now log in.")
                else:
                    st.error(message)

def assessment_page(model, features):
    st.header("🩺 Health assessment")
    st.info("Educational awareness only—not a diagnosis or a validated estimate of future diabetes risk. The model was trained on a US survey dataset.")
    with st.form("assessment_form"):
        c1, c2 = st.columns(2)
        with c1:
            high_bp = st.selectbox("Have you ever been told you have high blood pressure?", ["No", "Yes"])
            high_chol = st.selectbox("Have you ever been told you have high cholesterol?", ["No", "Yes"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=25.0, step=0.1)
            smoker = st.selectbox("Have you smoked at least 100 cigarettes in your life?", ["No", "Yes"])
            stroke = st.selectbox("Have you ever had a stroke?", ["No", "Yes"])
            heart_disease = st.selectbox("History of coronary heart disease or heart attack?", ["No", "Yes"])
            activity = st.selectbox("Any physical activity outside your regular job in the past 30 days?", ["No", "Yes"])
        with c2:
            fruits = st.selectbox("Consume fruit at least once per day?", ["No", "Yes"])
            veggies = st.selectbox("Consume vegetables at least once per day?", ["No", "Yes"])
            alcohol = st.selectbox("Heavy alcohol consumption (dataset definition)?", ["No", "Yes"])
            gen_health = st.select_slider("General health", options=[1, 2, 3, 4, 5], value=3,
                                          format_func=lambda x: {1:"Excellent",2:"Very good",3:"Good",4:"Fair",5:"Poor"}[x])
            phys_days = st.slider("Days physical health was not good in the past 30 days", 0, 30, 0)
            age_group = st.selectbox("Age group (dataset-coded)", options=list(range(1, 14)),
                                     format_func=lambda x: {
                                         1:"18–24",2:"25–29",3:"30–34",4:"35–39",5:"40–44",
                                         6:"45–49",7:"50–54",8:"55–59",9:"60–64",10:"65–69",
                                         11:"70–74",12:"75–79",13:"80 or older"
                                     }[x], index=4)
        submitted = st.form_submit_button("Assess", type="primary")

    if submitted:
        # Preserve the dataset's encoded feature order.
        values = {
            "HighBP": int(high_bp == "Yes"),
            "HighChol": int(high_chol == "Yes"),
            "BMI": float(bmi),
            "Smoker": int(smoker == "Yes"),
            "Stroke": int(stroke == "Yes"),
            "HeartDiseaseorAttack": int(heart_disease == "Yes"),
            "PhysActivity": int(activity == "Yes"),
            "Fruits": int(fruits == "Yes"),
            "Veggies": int(veggies == "Yes"),
            "HvyAlcoholConsump": int(alcohol == "Yes"),
            "GenHlth": int(gen_health),
            "PhysHlth": int(phys_days),
            "Age": int(age_group),
        }
        input_df = pd.DataFrame([[values[f] for f in features]], columns=features)
        pred = int(model.predict(input_df)[0])
        score = float(model.predict_proba(input_df)[0, 1])
        st.session_state["last_assessment"] = {
            "values": values, "prediction": pred, "score": score
        }
        st.subheader("Model output")
        st.metric("Positive-class model score", f"{score * 100:.1f}%")
        st.write("Predicted dataset class:", "**Class 1**" if pred == 1 else "**Class 0**")
        st.caption("This score is not a calibrated probability of developing diabetes and must not be interpreted as a diagnosis.")
        st.subheader("Factors to discuss")
        tips = []
        if values["BMI"] >= 25:
            tips.append("BMI is at or above 25. BMI is a screening measure, not a diagnosis; discuss personal goals with a clinician.")
        if values["HighBP"]:
            tips.append("You reported a history of high blood pressure. Follow your clinician's advice for monitoring and management.")
        if values["HighChol"]:
            tips.append("You reported high cholesterol. Follow professional advice on cholesterol checks and management.")
        if not values["PhysActivity"]:
            tips.append("If safe for you, gradually build regular physical activity; choose activities suited to your ability.")
        if not values["Fruits"] or not values["Veggies"]:
            tips.append("Consider adding a variety of fruits and vegetables to meals, according to your needs and access.")
        if not tips:
            tips.append("Continue routine preventive care and healthy habits. This brief form cannot assess every relevant factor.")
        for tip in tips:
            st.write("•", tip)
        st.warning("Seek professional medical advice for personal risk assessment, symptoms, or screening decisions.")

def xai_page(model, features):
    st.header("🧠 Explainable AI")
    st.write("This chart describes global feature importance on a held-out sample. It does not explain causation or guarantee why a particular prediction occurred.")
    if not DATA_PATH.exists():
        st.warning("Training dataset not found, so feature importance cannot be calculated.")
        return
    @st.cache_data
    def get_sample(path_str):
        df = pd.read_csv(path_str)
        cols = features + ["Diabetes_binary"]
        df = df[cols].dropna()
        if len(df) > 5000:
            df = df.sample(5000, random_state=42)
        return df[features], df["Diabetes_binary"].astype(int)
    try:
        X, y = get_sample(str(DATA_PATH))
        with st.spinner("Calculating permutation importance..."):
            importance = calculate_permutation_importance(model, X, y)
        st.plotly_chart(px.bar(importance.sort_values("importance"), x="importance", y="feature",
                               orientation="h", title="Permutation importance (ROC-AUC decrease)"),
                        use_container_width=True)
        st.dataframe(importance, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Could not calculate importance: {exc}")

def main():
    st.sidebar.title("💙 Diabetes Awareness")
    if "user" not in st.session_state:
        show_login()
        st.stop()

    st.sidebar.write(f"Signed in as {st.session_state['user']['email']}")
    page = st.sidebar.radio("Navigate", [
        "Dashboard", "Health assessment", "Explainable AI",
        "Predictive analytics", "Health chatbot", "About & disclaimer"
    ])
    if st.sidebar.button("Log out"):
        st.session_state.pop("user", None)
        st.session_state.pop("last_assessment", None)
        st.rerun()

    model, features = load_model()
    if page != "Dashboard" and page != "About & disclaimer" and model is None:
        st.error("Trained model files not found. Run `python data/train_main.py` first.")
        st.stop()

    if page == "Dashboard":
        st.title("Welcome to your health-awareness dashboard")
        st.write("Understand lifestyle information, explore educational insights, and learn about prevention.")
        a, b, c = st.columns(3)
        a.metric("Core module", "ML assessment")
        b.metric("Explainability", "Permutation importance")
        c.metric("Support", "Health FAQ chatbot")
        st.info("Start with Health assessment, then explore Explainable AI and Predictive analytics.")
        st.caption("This is an educational prototype, not a medical device or diagnosis tool.")
    elif page == "Health assessment":
        if model is None:
            st.error("Model files are missing. Run `python data/train_main.py` first.")
        else:
            assessment_page(model, features)
    elif page == "Explainable AI":
        xai_page(model, features)
    elif page == "Predictive analytics":
        render_analytics(RESULTS_PATH, DATA_PATH)
    elif page == "Health chatbot":
        st.header("💬 Health awareness chatbot")
        st.caption("Rule-based educational chatbot; not for diagnosis or emergencies.")
        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = [
                {"role": "assistant", "content": "Hello! Ask me about diabetes, BMI, healthy habits, or screening."}
            ]
        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        prompt = st.chat_input("Ask a general health question...")
        if prompt:
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})
            response = chatbot_response(prompt)
            st.session_state["chat_messages"].append({"role": "assistant", "content": response})
            st.rerun()
    else:
        st.header("About & disclaimer")
        st.write("This prototype uses the CDC Behavioral Risk Factor Surveillance System Diabetes Health Indicators dataset distributed through the UCI Machine Learning Repository.")
        st.write("The target represents existing diabetes-related status in the source data. The model has not been validated to predict future diabetes onset or calibrated for Indian users.")
        st.write("It is not a diagnosis. Do not use it to start, stop, or change treatment. Consult a qualified healthcare professional for screening or medical advice.")
        st.write("Avoid entering identifying details or information you do not wish to store. The demo stores account email and password hashes locally in SQLite.")

if __name__ == "__main__":
    main()
