from pathlib import Path
from datetime import datetime
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.inspection import permutation_importance


# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Awareness",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# GLOBAL THEME — VISUAL STYLING ONLY
# White, Blue and Teal Theme
# =========================================================

st.markdown(
    """
    <style>
    /* Login page inspired by the supplied HealthLens reference */
    .login-hero { background: linear-gradient(135deg, rgba(8, 35, 53, .96), rgba(31, 137, 133, .90)), radial-gradient(circle at 20% 20%, #2DA99F, #0B2537); color: white; border-radius: 22px 0 0 22px; padding: 44px 38px; min-height: 540px; display: flex; flex-direction: column; justify-content: center; }
    .login-hero h1, .login-hero h2, .login-hero h3, .login-hero p, .login-hero span { color: white !important; }
    .login-brand { font-size: 1.05rem; font-weight: 800; letter-spacing: .3px; margin-bottom: 42px; }
    .login-hero-title { font-size: clamp(1.7rem, 3vw, 2.6rem); line-height: 1.15; font-weight: 800; margin: 0 0 18px; }
    .login-hero-copy { color: #D8F4F1 !important; line-height: 1.7; font-size: 1rem; max-width: 440px; }
    .login-stats { display: flex; gap: 28px; margin-top: 42px; flex-wrap: wrap; }
    .login-stat strong { display: block; font-size: 1.35rem; color: white; }
    .login-stat small { color: #C6E6E5; }
    /* Extra professional polish: soft page glow, glass-like cards and refined controls */
    .stApp {
        background-image: radial-gradient(ellipse at 7% 8%, rgba(45,169,159,.09), transparent 28%),
                          radial-gradient(ellipse at 95% 90%, rgba(37,99,235,.07), transparent 30%);
    }
    [data-testid="stMainBlockContainer"] { padding-top: 1.8rem; padding-bottom: 2.5rem; max-width: 1500px; }
    .login-hero {
        position: relative; overflow: hidden;
        background: radial-gradient(circle at 90% 12%, rgba(160,255,239,.24), transparent 26%),
                    radial-gradient(circle at 5% 95%, rgba(64,151,255,.18), transparent 34%),
                    linear-gradient(140deg, #102B42 0%, #115E68 52%, #218E88 100%);
        box-shadow: 0 22px 55px rgba(14,70,84,.18);
    }
    .login-hero:after {
        content: ""; position: absolute; width: 230px; height: 230px; right: -95px; bottom: -110px;
        border: 1px solid rgba(255,255,255,.22); border-radius: 50%;
        box-shadow: 0 0 0 25px rgba(255,255,255,.035), 0 0 0 52px rgba(255,255,255,.025);
        pointer-events: none;
    }
    .login-brand { display: inline-flex; align-items: center; gap: 8px; }
    .login-stat { min-width: 90px; padding: 12px 14px; border: 1px solid rgba(255,255,255,.15); border-radius: 14px; background: rgba(255,255,255,.075); backdrop-filter: blur(8px); }
    .login-stat small { display: block; margin-top: 4px; font-size: .72rem; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.login-panel-marker) { backdrop-filter: blur(12px); }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.login-panel-marker) [data-testid="stTabs"] button[aria-selected="true"] { color: #167D78 !important; border-bottom-color: #2DA99F !important; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.login-panel-marker) [data-testid="stFormSubmitButton"] button { box-shadow: 0 7px 16px rgba(45,169,159,.20); letter-spacing: .01em; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.login-panel-marker) [data-testid="stFormSubmitButton"] button:hover { transform: translateY(-1px); box-shadow: 0 10px 22px rgba(45,169,159,.25); }
    .stButton > button:focus, .stFormSubmitButton > button:focus { outline: 3px solid rgba(45,169,159,.24); outline-offset: 2px; }
    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] { padding: 1rem .8rem 2rem; }
        .login-stat { padding: 9px 10px; min-width: 0; }
        .login-hero-title { font-size: 1.8rem; }
    }
    /* Keep the two login columns aligned; Streamlit widgets cannot be wrapped
       reliably by opening/closing HTML divs across separate elements. */
    .login-hero { box-sizing: border-box; min-height: 570px; height: 100%; border-radius: 22px; padding: 48px 42px; }
    .login-hero-title { font-size: clamp(1.8rem, 2.7vw, 2.65rem); }
    .login-panel-heading { text-align: center; font-size: 2rem; line-height: 1.2; font-weight: 800; color: #172033; margin: 4px 0 8px; }
    .login-subtitle { text-align: center; color: #718096 !important; font-size: .95rem; margin: 0 0 20px; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.login-panel-marker) { background: #FFFFFF; border: 1px solid #E6ECF4; border-radius: 22px; min-height: 570px; padding: 28px 30px; box-shadow: 0 16px 40px rgba(20,45,80,.06); }
    .login-panel-marker { display: none; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.login-panel-marker) [data-testid="stForm"] { border: 0; box-shadow: none; padding: 8px 0; background: transparent; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.login-panel-marker) [data-testid="stTabs"] [role="tablist"] { justify-content: center; border-bottom: 1px solid #E8EDF5; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.login-panel-marker) [data-testid="stTabs"] button { padding: 10px 20px; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.login-panel-marker) label { font-weight: 600; color: #344256; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.login-panel-marker) input { min-height: 44px; background: #F8FAFC; color: #172033; border: 1px solid #DCE4EF; }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.login-panel-marker) input:focus { border-color: #2DA99F; box-shadow: 0 0 0 2px rgba(45,169,159,.12); }
    @media (max-width: 768px) { .login-hero { min-height: 320px; padding: 28px 22px; } [data-testid="stVerticalBlockBorderWrapper"]:has(.login-panel-marker) { min-height: auto; padding: 22px 16px; } .login-brand { margin-bottom: 22px; } .login-stats { margin-top: 24px; gap: 16px; } }

    :root {
        --primary-blue: #2563EB;
        --primary-teal: #2DA99F;
        --background: #F7F9FC;
        --card-background: #FFFFFF;
        --text-color: #172033;
        --border-color: #E5EAF2;
    }

    .stApp { background-color: var(--background); color: var(--text-color); }
    [data-testid="stHeader"] { background-color: rgba(247, 249, 252, 0.96); }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid var(--border-color); }
    h1, h2, h3, h4 { color: #15233B; font-weight: 700; }
    p, li, label { color: #334155; }
    [data-testid="stMetric"] { background-color: var(--card-background); border: 1px solid var(--border-color); border-radius: 16px; padding: 18px; box-shadow: 0 4px 14px rgba(30, 64, 120, 0.05); }
    div[data-testid="stForm"] { background-color: #FFFFFF; border: 1px solid var(--border-color); border-radius: 16px; padding: 22px; }
    .stButton > button, .stFormSubmitButton > button { background: linear-gradient(100deg, #2DA99F, #3182CE); color: #FFFFFF; border: none; border-radius: 10px; min-height: 42px; font-weight: 600; transition: all 0.2s ease; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { background: linear-gradient(100deg, #238D85, #2563EB); color: #FFFFFF; border: none; box-shadow: 0 4px 12px rgba(45, 169, 159, 0.2); }
    .stTextInput input, .stNumberInput input, .stDateInput input { border-radius: 10px; border-color: #CBD5E1; }
    [data-baseweb="select"] > div { border-radius: 10px; border-color: #CBD5E1; }
    [data-testid="stTabs"] button { font-weight: 600; }
    [data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #2DA99F, #2563EB); }
    [data-testid="stAlert"] { border-radius: 12px; }
    [data-testid="stDataFrame"] { border: 1px solid var(--border-color); border-radius: 12px; overflow: hidden; }
    hr { border-color: var(--border-color); }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: 6px; }
    [data-testid="stSidebar"] [role="radio"] { border-radius: 10px; padding: 8px; }
    [data-testid="stChatMessage"] { border: 1px solid var(--border-color); border-radius: 12px; padding: 12px; }
    @media (max-width: 768px) {
        [data-testid="stMetric"] { padding: 12px; }
        div[data-testid="stForm"] { padding: 14px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "diabetes_health_indicators.csv"
MODEL_PATH = BASE_DIR / "models" / "diabetes_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "feature_names.joblib"
METRICS_PATH = BASE_DIR / "models" / "model_comparison.csv"

RECORDS_DIR = BASE_DIR / "user_records"
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

ACCOUNT_FILE = BASE_DIR / "accounts.json"


# =========================================================
# 2. SESSION STATE
# =========================================================

DEFAULTS = {
    "logged_in": False,
    "user_email": "",
    "chat_history": [],
    "quest_points": 0,
    "quest_completed": [],
    "quest_answers": {},
    "quest_challenge_done": [],
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 3. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_features():
    if not FEATURES_PATH.exists():
        return []
    return list(joblib.load(FEATURES_PATH))


try:
    model = load_model()
    features = load_features()
    MODEL_ERROR = None
except Exception as exc:
    model = None
    features = []
    MODEL_ERROR = str(exc)


# =========================================================
# 4. ACCOUNT FUNCTIONS
# =========================================================

def read_accounts():
    if not ACCOUNT_FILE.exists():
        return {}

    try:
        with open(ACCOUNT_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def write_accounts(accounts):
    with open(ACCOUNT_FILE, "w", encoding="utf-8") as file:
        json.dump(accounts, file, indent=2)


def hash_password(password):
    # Prototype only; not suitable for production authentication.
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_current_user():
    accounts = read_accounts()
    return accounts.get(
        st.session_state.user_email,
        {"name": "User"},
    )


# =========================================================
# 5. ASSESSMENT HISTORY
# =========================================================

def record_file():
    email_hash = hashlib.sha256(
        st.session_state.user_email.encode("utf-8")
    ).hexdigest()[:20]

    return RECORDS_DIR / f"{email_hash}.csv"


def load_records():
    path = record_file()

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def save_record(record):
    path = record_file()
    previous = load_records()

    updated = pd.concat(
        [previous, pd.DataFrame([record])],
        ignore_index=True,
    )

    updated.to_csv(path, index=False)


# =========================================================
# 6. HEALTH FACTOR NAMES AND EXPLANATIONS
# =========================================================

FRIENDLY_NAMES = {
    "HighBP": "High blood pressure history",
    "HighChol": "High cholesterol history",
    "BMI": "Body Mass Index (BMI)",
    "Smoker": "Smoking history",
    "Stroke": "History of stroke",
    "HeartDiseaseorAttack": "Heart disease history",
    "PhysActivity": "Physical activity",
    "Fruits": "Fruit consumption",
    "Veggies": "Vegetable consumption",
    "HvyAlcoholConsump": "Heavy alcohol consumption",
    "GenHlth": "General health rating",
    "PhysHlth": "Days of poor physical health",
    "Age": "Age group",
}

FACTOR_EXPLANATIONS = {
    "HighBP": (
        "Blood pressure history is one of the health indicators "
        "used by the model."
    ),
    "HighChol": (
        "Cholesterol history provides information about "
        "cardiovascular and metabolic health."
    ),
    "BMI": (
        "BMI compares weight with height. It is a screening "
        "measure, not a diagnosis."
    ),
    "Smoker": (
        "Smoking history is one of the lifestyle indicators "
        "in the dataset."
    ),
    "Stroke": (
        "A history of stroke is a health-history indicator "
        "in the dataset."
    ),
    "HeartDiseaseorAttack": (
        "Heart disease history is included as a health indicator."
    ),
    "PhysActivity": (
        "Physical activity is a lifestyle indicator associated "
        "with general health."
    ),
    "Fruits": (
        "Fruit consumption is one of the dietary indicators "
        "in the dataset."
    ),
    "Veggies": (
        "Vegetable consumption is one of the dietary indicators "
        "in the dataset."
    ),
    "HvyAlcoholConsump": (
        "Alcohol consumption is one of the lifestyle indicators "
        "in the dataset."
    ),
    "GenHlth": (
        "This feature represents how respondents rated their "
        "overall health."
    ),
    "PhysHlth": (
        "This feature records the number of days respondents "
        "reported poor physical health."
    ),
    "Age": (
        "Age group is one of the demographic indicators "
        "used by the model."
    ),
}


# =========================================================
# 7. LOGIN AND REGISTRATION
# =========================================================

def authentication_page():
    """Styled login/registration page; authentication behavior is unchanged."""
    # Add comfortable page margins and keep both panels in the same row.
    st.markdown(
        "<div style='height:18px'></div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.02, 0.98], gap="medium", vertical_alignment="center")

    with left:
        st.markdown(
            """
            <div class="login-hero">
              <div class="login-brand">💠 HealthLens <span style="font-weight:500;opacity:.8">/ DIABETES AWARENESS</span></div>
              <div class="login-hero-title">Your AI-Powered<br>Health Assistant</div>
              <p class="login-hero-copy">Track, understand, and improve your health with intelligent health insights and diabetes awareness tools.</p>
              <div class="login-stats">
                <div class="login-stat"><strong>10K+</strong><small>Health insights</small></div>
                <div class="login-stat"><strong>98%</strong><small>User satisfaction</small></div>
                <div class="login-stat"><strong>24/7</strong><small>Access anytime</small></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        # Use a real Streamlit container instead of an HTML wrapper around
        # widgets; this prevents the blank card / misaligned form seen before.
        with st.container(border=True):
            st.markdown('<span class="login-panel-marker"></span>', unsafe_allow_html=True)
            st.markdown('<div class="login-panel-heading">Welcome Back</div>', unsafe_allow_html=True)
            st.markdown('<p class="login-subtitle">Sign in to access your health dashboard</p>', unsafe_allow_html=True)

            login_tab, register_tab = st.tabs(["Sign In", "Create Account"])

            with login_tab:
                with st.form("login_form"):
                    email = st.text_input("Email Address", placeholder="you@example.com")
                    password = st.text_input("Password", type="password", placeholder="Enter your password")
                    login_clicked = st.form_submit_button("Sign In  →", use_container_width=True, type="primary")

                if login_clicked:
                    accounts = read_accounts()
                    email = email.strip().lower()
                    account = accounts.get(email)
                    if account and account.get("password") == hash_password(password):
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.rerun()
                    else:
                        st.error("Invalid email address or password.")

            with register_tab:
                with st.form("register_form"):
                    name = st.text_input("Your Name")
                    email = st.text_input("Email Address", key="register_email", placeholder="you@example.com")
                    password = st.text_input("Create Password", type="password", key="register_password")
                    confirm = st.text_input("Confirm Password", type="password")
                    register_clicked = st.form_submit_button("Create Account  →", use_container_width=True, type="primary")

                if register_clicked:
                    accounts = read_accounts()
                    name = name.strip()
                    email = email.strip().lower()
                    if not name or not email or not password:
                        st.error("Please complete every field.")
                    elif "@" not in email or "." not in email:
                        st.error("Enter a valid email address.")
                    elif len(password) < 8:
                        st.error("Password must contain at least 8 characters.")
                    elif password != confirm:
                        st.error("Passwords do not match.")
                    elif email in accounts:
                        st.error("An account with this email already exists.")
                    else:
                        accounts[email] = {"name": name, "password": hash_password(password)}
                        write_accounts(accounts)
                        st.success("Account created successfully. Please log in.")


# =========================================================
# 8. DASHBOARD
# =========================================================

def dashboard_page():
    user = get_current_user()
    records = load_records()

    st.title("💙 My Health Dashboard")
    st.header(f"Welcome back, {user.get('name', 'User')}!")

    st.write(
        "Your personal space to review assessments and "
        "build healthier habits."
    )

    col1, col2, col3 = st.columns(3)

    latest_date = (
        str(records.iloc[-1]["Date"])
        if not records.empty
        else "No assessments yet"
    )

    col1.metric("My Assessments", len(records))
    col2.metric("Latest Assessment", latest_date)
    col3.metric(
        "Quest Points",
        st.session_state.quest_points,
    )

    st.divider()
    st.subheader("🩺 My Latest Assessment")

    if records.empty:
        st.info(
            "No previous assessments yet. Open Health Assessment "
            "to complete your first assessment."
        )
    else:
        latest = records.iloc[-1]

        if int(latest["Classification"]) == 1:
            st.warning(
                "Your latest information was classified as Class 1 "
                "by the model."
            )
        else:
            st.success(
                "Your latest information was classified as Class 0 "
                "by the model."
            )

        st.write(f"**Assessment date:** {latest['Date']}")

        if (
            "SystolicBP" in latest.index
            and pd.notna(latest["SystolicBP"])
        ):
            st.write(
                f"**Blood pressure:** {latest['SystolicBP']}/"
                f"{latest['DiastolicBP']} mmHg"
            )

        st.caption(
            "This is a model classification, not a diagnosis or "
            "a validated prediction of future diabetes."
        )

    st.divider()
    st.subheader("🎮 Your Prevention Quest")

    completed = len(st.session_state.quest_challenge_done)

    st.write(
        f"You have earned **{st.session_state.quest_points} points** "
        f"and completed **{completed} daily challenges**."
    )

    st.info(
        "Open Prevention Quest in the sidebar to answer health "
        "questions and complete wellness challenges."
    )


# =========================================================
# 9. HEALTH ASSESSMENT
# =========================================================

def assessment_page():
    st.title("🩺 Health Assessment")

    st.write(
        "Enter your health information to receive an educational "
        "model classification and general preventive guidance."
    )

    st.warning(
        "This tool does not diagnose diabetes. Its score is not a "
        "validated probability of developing diabetes."
    )

    if MODEL_ERROR:
        st.error(f"Model loading error: {MODEL_ERROR}")
        return

    if model is None or not features:
        st.error(
            "Model files could not be loaded. Check your models folder."
        )
        return

    with st.form("health_assessment_form"):
        st.subheader("Basic Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.selectbox(
                "Age group",
                options=list(range(1, 14)),
                index=4,
                format_func=lambda x: {
                    1: "18–24",
                    2: "25–29",
                    3: "30–34",
                    4: "35–39",
                    5: "40–44",
                    6: "45–49",
                    7: "50–54",
                    8: "55–59",
                    9: "60–64",
                    10: "65–69",
                    11: "70–74",
                    12: "75–79",
                    13: "80 or older",
                }[x],
            )

            bmi = st.number_input(
                "BMI (kg/m²)",
                min_value=10.0,
                max_value=80.0,
                value=25.0,
                step=0.1,
            )

        with col2:
            systolic = st.number_input(
                "Systolic blood pressure (mmHg)",
                min_value=60,
                max_value=250,
                value=120,
            )

            diastolic = st.number_input(
                "Diastolic blood pressure (mmHg)",
                min_value=30,
                max_value=150,
                value=80,
            )

        st.subheader("Health History")

        col1, col2 = st.columns(2)

        with col1:
            bp_history = st.selectbox(
                "Have you been told you have high blood pressure?",
                ["No", "Yes", "Not sure"],
            )

            high_chol = st.selectbox(
                "High cholesterol history?",
                ["No", "Yes"],
            )

            smoker = st.selectbox(
                "Smoking history?",
                ["No", "Yes"],
            )

            stroke = st.selectbox(
                "History of stroke?",
                ["No", "Yes"],
            )

            heart = st.selectbox(
                "Heart disease or heart attack history?",
                ["No", "Yes"],
            )

        with col2:
            activity = st.selectbox(
                "Regular physical activity?",
                ["Yes", "No"],
            )

            fruits = st.selectbox(
                "Eat fruit regularly?",
                ["Yes", "No"],
            )

            veggies = st.selectbox(
                "Eat vegetables regularly?",
                ["Yes", "No"],
            )

            alcohol = st.selectbox(
                "Heavy alcohol consumption?",
                ["No", "Yes"],
            )

            general_health = st.select_slider(
                "General health",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: {
                    1: "Excellent",
                    2: "Very good",
                    3: "Good",
                    4: "Fair",
                    5: "Poor",
                }[x],
            )

            poor_health_days = st.slider(
                "Days of poor physical health in the last 30 days",
                min_value=0,
                max_value=30,
                value=0,
            )

        submitted = st.form_submit_button(
            "Generate My Assessment",
            use_container_width=True,
            type="primary",
        )

    if not submitted:
        return

    if systolic <= diastolic:
        st.error(
            "Please check the blood pressure values. "
            "The systolic number should normally be higher "
            "than the diastolic number."
        )
        return

    # The existing model uses a binary HighBP feature, not raw BP.
    # This mapping is a prototype approximation, not a validated
    # conversion and not a model trained on numerical BP readings.
    high_bp_model = int(
        bp_history == "Yes"
        or systolic >= 130
        or diastolic >= 80
    )

    values = {
        "HighBP": high_bp_model,
        "HighChol": int(high_chol == "Yes"),
        "BMI": float(bmi),
        "Smoker": int(smoker == "Yes"),
        "Stroke": int(stroke == "Yes"),
        "HeartDiseaseorAttack": int(heart == "Yes"),
        "PhysActivity": int(activity == "Yes"),
        "Fruits": int(fruits == "Yes"),
        "Veggies": int(veggies == "Yes"),
        "HvyAlcoholConsump": int(alcohol == "Yes"),
        "GenHlth": int(general_health),
        "PhysHlth": int(poor_health_days),
        "Age": int(age),
        "SystolicBP": int(systolic),
        "DiastolicBP": int(diastolic),
    }

    missing = [
        feature for feature in features
        if feature not in values
    ]

    if missing:
        st.error(
            "The trained model requires additional inputs: "
            + ", ".join(missing)
        )
        return

    input_df = pd.DataFrame(
        [[values[feature] for feature in features]],
        columns=features,
    )

    try:
        prediction = int(model.predict(input_df)[0])

        score = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if 1 in classes:
                score = float(
                    probabilities[classes.index(1)]
                )

    except Exception as exc:
        st.error(f"Unable to generate assessment: {exc}")
        return

    record = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Classification": prediction,
        "Model_Score_Percent": (
            round(score * 100, 2)
            if score is not None
            else np.nan
        ),
        **values,
    }

    save_record(record)

    st.divider()
    st.header("📋 Your Assessment Results")

    if prediction == 1:
        st.warning(
            "The model classified your information as Class 1 "
            "in the source dataset."
        )
    else:
        st.success(
            "The model classified your information as Class 0 "
            "in the source dataset."
        )

    if score is not None:
        st.metric(
            "Model Score",
            f"{score * 100:.1f}%",
        )

        st.caption(
            "This is the model's score for its positive classification "
            "class, not a validated probability of future diabetes."
        )

        # Plain-language summary displayed below the model score.
        # This explains the model output; it is not a medical diagnosis.
        if score < 0.10:
            summary = (
                "Your result suggests that your chances of having diabetes "
                "may be low based on the answers you entered. Keep up healthy "
                "habits. This app cannot confirm whether you have diabetes, "
                "so speak with a healthcare professional if you have symptoms "
                "or concerns."
            )
            summary_color = "#166534"
            summary_icon = "💚"
        elif score < 0.30:
            summary = (
                "Your result is not in the lowest range. Keep following "
                "healthy habits, and consider asking a healthcare professional "
                "whether a diabetes test would be appropriate for you."
            )
            summary_color = "#1E40AF"
            summary_icon = "💙"
        elif score < 0.60:
            summary = (
                "Your result suggests you should pay attention to your health "
                "and consider discussing diabetes testing with a healthcare "
                "professional. This app cannot tell you for sure whether you "
                "have diabetes."
            )
            summary_color = "#854D0E"
            summary_icon = "🟡"
        else:
            summary = (
                "Your result is in the higher range. Please discuss diabetes "
                "testing with a healthcare professional. Only appropriate "
                "medical tests can determine whether you have diabetes."
            )
            summary_color = "#991B1B"
            summary_icon = "❤️"

        st.markdown(
            f"""
            <div style="
                background-color: {summary_color};
                color: white;
                padding: 20px 22px;
                border-radius: 12px;
                margin: 12px 0 20px 0;
                font-size: 19px;
                line-height: 1.6;
                font-weight: 500;
            ">
                <div style="font-size: 23px; font-weight: 700; margin-bottom: 6px;">
                    {summary_icon} What this score means
                </div>
                {summary}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("🩸 Blood Pressure")

    st.write(f"**Systolic:** {systolic} mmHg")
    st.write(f"**Diastolic:** {diastolic} mmHg")

    if systolic >= 180 or diastolic >= 120:
        st.error(
            "This reading is severely elevated. Repeat the measurement "
            "after resting and seek urgent medical advice if it remains "
            "this high. If you have chest pain, shortness of breath, "
            "weakness, confusion, or vision changes, seek emergency care."
        )
    elif systolic >= 130 or diastolic >= 80:
        st.warning(
            "This reading is elevated under commonly used adult "
            "categories. A single reading does not diagnose "
            "hypertension. Discuss repeated elevated readings "
            "with a healthcare professional."
        )
    else:
        st.info(
            "This reading is below the commonly used 130/80 mmHg "
            "threshold. Interpretation depends on your health history "
            "and repeated measurements."
        )

    st.subheader("🌱 Preventive Recommendations")

    recommendations = [
        "Choose balanced meals with vegetables, whole grains, "
        "and appropriate portions.",
        "Include regular physical activity suitable for your ability.",
        "Avoid tobacco and aim for consistent, sufficient sleep.",
        "Ask a healthcare professional whether diabetes screening "
        "is appropriate for you.",
    ]

    if bmi >= 25:
        recommendations.append(
            "If appropriate, discuss sustainable nutrition and "
            "weight-management goals with a healthcare professional."
        )

    if systolic >= 130 or diastolic >= 80:
        recommendations.append(
            "Record properly measured blood pressure readings and "
            "discuss repeated elevated readings with a professional."
        )

    for recommendation in recommendations:
        st.markdown(f"- {recommendation}")

    st.caption(
        "These recommendations are general educational guidance "
        "and do not replace medical advice or clinical testing."
    )


# =========================================================
# 10. EXPLAINABLE AI
# =========================================================

def explainable_ai_page():
    """Show relative permutation importance for the trained model."""
    st.title("🧠 Understand Health Factors")
    st.subheader("Discover What Influences Your Model")

    st.write(
        "Explore the health factors used by the trained machine learning "
        "model and compare how they influence its performance."
    )

    st.info(
        "💡 The percentages on this page show relative feature importance. "
        "They are NOT percentages of personal diabetes risk."
    )

    if model is None or not features:
        st.error("The trained model or feature list could not be loaded.")
        return

    if not DATA_PATH.exists():
        st.error(f"Dataset not found: {DATA_PATH}")
        return

    try:
        df = pd.read_csv(DATA_PATH)
        target = "Diabetes_binary"
        required_columns = list(features) + [target]
        missing_columns = [c for c in required_columns if c not in df.columns]

        if missing_columns:
            st.error(f"Dataset is missing columns: {missing_columns}")
            return

        df = df[required_columns].dropna()
        if df.empty:
            st.error("No valid rows are available for analysis.")
            return

        # A reproducible sample keeps the page responsive.
        if len(df) > 2000:
            df = df.sample(n=2000, random_state=42)

        X = df[list(features)]
        y = df[target].astype(int)

        if y.nunique() < 2:
            st.error("The sample must contain both target classes.")
            return

        with st.spinner("🔍 Analysing health factors..."):
            result = permutation_importance(
                model,
                X,
                y,
                scoring="roc_auc",
                n_repeats=3,
                random_state=42,
                n_jobs=-1,
            )

        importance_df = pd.DataFrame({
            "Feature": list(features),
            "Importance": result.importances_mean,
            "Variation": result.importances_std,
        })
        importance_df["Health Factor"] = (
            importance_df["Feature"].map(FRIENDLY_NAMES)
            .fillna(importance_df["Feature"])
        )
        importance_df = importance_df.sort_values(
            "Importance", ascending=False
        ).reset_index(drop=True)

        positive_scores = importance_df.loc[
            importance_df["Importance"] > 0, "Importance"
        ]
        max_score = float(positive_scores.max()) if not positive_scores.empty else None

        st.divider()
        st.header("🏆 Top 5 Contributing Health Factors")
        st.write(
            "The highest positive importance score is the 100% reference. "
            "Other positive scores are shown relative to that reference."
        )

        if max_score is None:
            st.warning(
                "No positive importance scores were found. Relative "
                "importance percentages will be shown as 0%."
            )

        for rank, (_, row) in enumerate(importance_df.head(5).iterrows(), start=1):
            feature = row["Feature"]
            name = row["Health Factor"]
            score = float(row["Importance"])
            percentage = (
                min(100.0, (score / max_score) * 100.0)
                if max_score is not None and score > 0 else 0.0
            )

            title_col, badge_col = st.columns([4, 1])
            with title_col:
                st.markdown(f"### {rank}. {name}")
            with badge_col:
                st.markdown(
                    f"""<div style="background:linear-gradient(135deg,#123B64,#167D9A);color:white;padding:10px 5px;border-radius:12px;text-align:center;font-size:20px;font-weight:bold;margin-top:5px;">{percentage:.1f}%</div>""",
                    unsafe_allow_html=True,
                )

            st.progress(percentage / 100.0)
            st.caption(
                f"Relative importance: {percentage:.1f}% of the highest "
                "positive importance score"
            )
            st.markdown(f"**Feature importance score:** `{score:.4f}`")
            st.write(FACTOR_EXPLANATIONS.get(
                feature, "This feature is included in the trained model."
            ))
            st.markdown(
                f"""<div style="background-color:#E8F5F9;color:#123B64;padding:12px 15px;border-radius:10px;border-left:5px solid #167D9A;margin:10px 0 18px 0;"><b>📊 Relative Importance</b><br>{percentage:.1f}% compared with the highest positive importance score.</div>""",
                unsafe_allow_html=True,
            )
            st.divider()

        st.header("📈 Feature Importance Comparison")
        st.write("Compare importance scores for all health factors used by the model.")
        chart_df = importance_df.sort_values("Importance", ascending=True).copy()
        fig = px.bar(
            chart_df,
            x="Importance",
            y="Health Factor",
            orientation="h",
            labels={
                "Importance": "Permutation Importance Score",
                "Health Factor": "Health Factor",
            },
            title="How Features Influence Model Performance",
            hover_data={
                "Feature": True,
                "Importance": ":.4f",
                "Variation": ":.4f",
            },
        )
        fig.update_layout(
            height=max(450, len(chart_df) * 38),
            showlegend=False,
            xaxis_title="Change in ROC-AUC after shuffling",
            yaxis_title="Health Factor",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 View All Health Factor Scores"):
            display_df = importance_df[
                ["Health Factor", "Importance", "Variation"]
            ].copy()
            display_df["Importance"] = display_df["Importance"].round(4)
            display_df["Variation"] = display_df["Variation"].round(4)
            display_df.columns = [
                "Health Factor", "Importance Score", "Score Variation"
            ]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.divider()
        st.header("💡 How to Understand These Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "#### 📊 Relative Importance\n"
                "- **100%** means the highest positive importance score.\n"
                "- **50%** means half of that reference score.\n"
                "- **0%** means the score is not positive."
            )
        with col2:
            st.markdown(
                "#### 🧠 What It Means\n"
                "- Importance measures changes in model performance.\n"
                "- It does not establish cause and effect.\n"
                "- It does not tell you an individual's diabetes risk."
            )

        st.warning(
            "A negative importance score is possible. It means shuffling "
            "that feature did not reduce measured performance on this sample. "
            "Its percentage badge is set to 0%, while its original score remains visible."
        )
        st.caption(
            "This analysis describes overall model behaviour on a sample of "
            "the dataset. It is not a medical diagnosis or a validated prediction "
            "of future diabetes."
        )

    except Exception as exc:
        st.error(f"Unable to calculate Explainable AI results: {exc}")


# =========================================================
# 11. PREDICTIVE ANALYTICS
# =========================================================

def predictive_analytics_page():
    st.title("📊 Health Insights")

    st.write(
        "Understand the model's performance on its held-out test dataset."
    )

    st.info(
        "These results describe dataset performance, not your personal "
        "health risk."
    )

    if not METRICS_PATH.exists():
        st.warning(
            "The model evaluation file was not found. "
            "Run the training script first."
        )
        return

    try:
        df = pd.read_csv(METRICS_PATH)

        if df.empty:
            st.warning("The model evaluation file is empty.")
            return

        if "Model" in df.columns:
            selected = df[
                df["Model"].astype(str).str.contains(
                    "HistGradientBoosting",
                    case=False,
                    na=False,
                )
            ]

            row = (
                selected.iloc[0]
                if not selected.empty
                else df.iloc[0]
            )
        else:
            row = df.iloc[0]

        def read_metric(names):
            for name in names:
                if name in row.index and pd.notna(row[name]):
                    value = float(row[name])
                    return value / 100 if value > 1 else value
            return None

        metric_info = [
            (
                "Accuracy",
                read_metric(["Accuracy", "accuracy"]),
                "How often the model classified test records correctly.",
            ),
            (
                "Precision",
                read_metric(["Precision", "precision"]),
                "Among positive classifications, how many were actually positive.",
            ),
            (
                "Recall",
                read_metric(["Recall", "recall"]),
                "Among actual positive cases, how many the model identified.",
            ),
            (
                "F1-score",
                read_metric(["F1_Score", "F1-score", "F1"]),
                "A combined measure of precision and recall.",
            ),
            (
                "ROC-AUC",
                read_metric(["ROC_AUC", "ROC-AUC", "ROC_AUC_Score"]),
                "How well the model ranks positive records above negative records.",
            ),
        ]

        st.subheader("Model Performance Summary")

        for start in range(0, len(metric_info), 2):
            pair = metric_info[start:start + 2]
            cols = st.columns(len(pair))

            for col, (name, value, description) in zip(cols, pair):
                with col:
                    with st.container(border=True):
                        st.subheader(name)

                        if value is None:
                            st.write("Not available")
                        else:
                            st.metric(name, f"{value * 100:.1f}%")

                        st.write(description)

        recall = read_metric(["Recall", "recall"])

        if recall is not None and recall < 0.5:
            st.warning(
                "Recall is low. The model misses many positive cases "
                "in the evaluation dataset, so it should not be "
                "presented as a reliable screening tool."
            )

        st.subheader("What should you remember?")

        st.markdown(
            """
            - These measurements describe performance on a test dataset.
            - High accuracy alone does not guarantee useful screening.
            - Precision and recall describe different types of errors.
            - Performance may differ in another population.
            - Clinical validation is required before medical use.
            """
        )

    except Exception as exc:
        st.error(f"Unable to display model metrics: {exc}")


# =========================================================
# 12. PREVIOUS RECORDS
# =========================================================

def previous_records_page():
    st.title("📚 My Previous Assessments")

    records = load_records()

    if records.empty:
        st.info(
            "No previous assessments yet. Complete a Health Assessment "
            "to save your first record."
        )
        return

    records = records.sort_values(
        "Date",
        ascending=False,
    )

    # Make binary health/lifestyle values easier to read in this table.
    # Convert only the displayed copy; saved records and the graph stay unchanged.
    records_display = records.copy()

    yes_no_columns = [
        "HighBP",
        "HighChol",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "DiffWalk",
    ]

    for column in yes_no_columns:
        if column in records_display.columns:
            records_display[column] = records_display[column].replace({
                1: "Yes",
                0: "No",
                1.0: "Yes",
                0.0: "No",
                "1": "Yes",
                "0": "No",
                "1.0": "Yes",
                "0.0": "No",
            })

    st.dataframe(
        records_display,
        use_container_width=True,
        hide_index=True,
    )

    if "Model_Score_Percent" in records.columns:
        chart_df = records.dropna(
            subset=["Model_Score_Percent"]
        ).copy()

        if not chart_df.empty:
            st.subheader("Model Scores Over Time")

            fig = px.line(
                chart_df,
                x="Date",
                y="Model_Score_Percent",
                markers=True,
                labels={
                    "Date": "Assessment date",
                    "Model_Score_Percent": "Model score (%)",
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                "Changes in model scores do not prove that your "
                "real-world diabetes risk has changed."
            )


# =========================================================
# 13. DIABETES PREVENTION QUEST
# =========================================================

def prevention_game_page():
    st.title("🎮 Diabetes Prevention Quest")

    st.write(
        "Learn about healthier habits through quizzes, daily "
        "challenges, and achievements."
    )

    st.info(
        "Quest points measure participation, not medical risk. "
        "Completing the game does not guarantee a lower risk score."
    )

    questions = [
        {
            "question": (
                "1. You want to improve your health. "
                "What is a good first step?"
            ),
            "options": [
                "Make one small, realistic change",
                "Change every habit overnight",
                "Wait until you become unwell",
            ],
            "answer": "Make one small, realistic change",
            "explanation": (
                "Small, achievable changes can be easier to maintain."
            ),
        },
        {
            "question": (
                "2. Which is a sensible everyday drink choice "
                "for many people?"
            ),
            "options": [
                "Water instead of a sugary drink",
                "Sugary drinks throughout the day",
                "Avoid drinking water all day",
            ],
            "answer": "Water instead of a sugary drink",
            "explanation": (
                "Choosing water instead of sugary drinks can help "
                "reduce added sugar intake."
            ),
        },
        {
            "question": (
                "3. What can help support general health?"
            ),
            "options": [
                "Regular physical activity suited to you",
                "Sitting all day whenever possible",
                "Sleeping as little as possible",
            ],
            "answer": "Regular physical activity suited to you",
            "explanation": (
                "Regular activity supports general health. "
                "Choose something appropriate for your abilities."
            ),
        },
        {
            "question": (
                "4. If your health assessment concerns you, "
                "what should you do?"
            ),
            "options": [
                "Ignore it completely",
                "Discuss appropriate screening with a healthcare professional",
                "Treat the app result as a diagnosis",
            ],
            "answer": (
                "Discuss appropriate screening with a healthcare professional"
            ),
            "explanation": (
                "A healthcare professional can advise whether "
                "testing is appropriate for you."
            ),
        },
        {
            "question": (
                "5. Which is a useful approach to meals?"
            ),
            "options": [
                "Eat balanced meals with suitable portions",
                "Skip every meal",
                "Eat only sugary foods",
            ],
            "answer": "Eat balanced meals with suitable portions",
            "explanation": (
                "Balanced meals can include vegetables, whole grains, "
                "and appropriate portions of protein-rich foods."
            ),
        },
    ]

    # Quiz state
    if "quest_points" not in st.session_state:
        st.session_state.quest_points = 0

    if "quest_completed" not in st.session_state:
        st.session_state.quest_completed = []

    if "quest_answers" not in st.session_state:
        st.session_state.quest_answers = {}

    if "quest_challenge_done" not in st.session_state:
        st.session_state.quest_challenge_done = []

    # Summary
    completed_count = len(st.session_state.quest_completed)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "⭐ Quest Points",
        st.session_state.quest_points,
    )

    col2.metric(
        "✅ Questions Completed",
        f"{completed_count}/{len(questions)}",
    )

    col3.metric(
        "🏆 Challenges Completed",
        len(st.session_state.quest_challenge_done),
    )

    st.progress(completed_count / len(questions))

    st.divider()
    st.header("🧠 Health Knowledge Challenge")

    for index, question in enumerate(questions):

        with st.container(border=True):
            st.subheader(question["question"])

            if index in st.session_state.quest_completed:

                selected = st.session_state.quest_answers[index]

                if selected == question["answer"]:
                    st.success("Correct answer!")
                else:
                    st.info(f"Your answer: {selected}")
                    st.write(
                        f"Suggested answer: {question['answer']}"
                    )

                st.write(
                    f"💡 Why? {question['explanation']}"
                )

            else:

                with st.form(
                    key=f"quest_form_{index}"
                ):

                    selected = st.radio(
                        "Choose your answer:",
                        question["options"],
                        key=f"quest_radio_{index}",
                    )

                    submitted = st.form_submit_button(
                        "Check Answer",
                        use_container_width=True,
                    )

                    if submitted:

                        st.session_state.quest_answers[index] = selected
                        st.session_state.quest_completed.append(index)

                        # Award participation points.
                        st.session_state.quest_points += 5

                        st.rerun()

    # Daily challenges
    st.divider()
    st.header("🌱 Daily Wellness Challenges")

    challenges = [
        (
            "🚶 Move your body",
            "Try a short walk or another suitable physical activity.",
        ),
        (
            "🥗 Build a balanced plate",
            "Include vegetables and suitable portions of nutritious foods.",
        ),
        (
            "💧 Choose water",
            "Choose water instead of a sugary drink when appropriate.",
        ),
        (
            "😴 Support your sleep",
            "Plan a consistent bedtime that allows enough rest.",
        ),
        (
            "🩺 Plan a health check",
            "Ask a healthcare professional whether screening is appropriate.",
        ),
    ]

    for index, (title, description) in enumerate(challenges):

        with st.container(border=True):

            col1, col2 = st.columns([4, 1])

            with col1:
                st.subheader(title)
                st.write(description)

            with col2:

                if index in st.session_state.quest_challenge_done:
                    st.success("Completed")

                elif st.button(
                    "Complete",
                    key=f"quest_challenge_button_{index}",
                    use_container_width=True,
                ):

                    st.session_state.quest_challenge_done.append(index)
                    st.session_state.quest_points += 10

                    st.rerun()

    # Achievements
    st.divider()
    st.header("🏆 Your Achievements")

    points = st.session_state.quest_points

    if points >= 50:
        st.success(
            "Habit Hero! You have earned 50 or more quest points."
        )
    elif points >= 25:
        st.success(
            "Healthy Habits Explorer! Keep learning and practising."
        )
    elif points >= 10:
        st.info(
            "Good start! Continue with small, realistic steps."
        )
    else:
        st.write(
            "Complete questions and challenges to earn your first badge."
        )

    st.metric("Total Quest Points", points)

    st.divider()
    st.header("💡 Your Next Step")

    st.write(
        "Choose one healthy habit you can realistically practise today. "
        "Consistency matters more than perfection."
    )

    st.warning(
        "Completing this game does not automatically lower your "
        "diabetes risk or change your model score. It is an educational "
        "game, not a diagnostic or treatment tool."
    )

    with st.expander("Reset game progress"):

        st.write(
            "This clears your current points, quiz answers, "
            "and completed challenges."
        )

        if st.button(
            "Reset My Quest",
            key="reset_quest",
        ):

            st.session_state.quest_points = 0
            st.session_state.quest_completed = []
            st.session_state.quest_answers = {}
            st.session_state.quest_challenge_done = []

            st.rerun()


# =========================================================
# 14. ENGLISH AND TAMIL HEALTH CHATBOT
# =========================================================

def chatbot_page():
    st.title("💬 Health Awareness Chatbot")

    st.write(
        "Ask a general health question in English or Tamil. "
        "தமிழ் அல்லது ஆங்கிலத்தில் கேள்விகளைக் கேட்கலாம்."
    )

    st.info(
        "This is a rule-based educational chatbot, not a medical professional."
    )

    def get_answer(question):
        q = question.lower().strip()

        emergency_terms = [
            "chest pain",
            "difficulty breathing",
            "can't breathe",
            "unconscious",
            "மார்பு வலி",
            "மூச்சு விட முடியவில்லை",
        ]

        if any(term in q for term in emergency_terms):
            return (
                "English: These symptoms may be an emergency. "
                "Seek emergency medical care immediately.\n\n"
                "தமிழ்: இவை அவசர அறிகுறிகளாக இருக்கலாம். "
                "உடனடியாக அவசர மருத்துவ உதவியை நாடுங்கள்."
            )

        if any(
            term in q
            for term in ["hello", "hi", "வணக்கம்", "vanakkam"]
        ):
            return (
                "English: Hello! Ask me about diet, exercise, BMI, "
                "blood pressure, diabetes, or screening.\n\n"
                "தமிழ்: வணக்கம்! உணவு, உடற்பயிற்சி, BMI, இரத்த "
                "அழுத்தம் அல்லது நீரிழிவு பரிசோதனை பற்றி கேளுங்கள்."
            )

        if any(
            term in q
            for term in ["diet", "food", "eat", "உணவு", "சாப்பாடு"]
        ):
            return (
                "English: Choose balanced meals with vegetables, "
                "whole grains, suitable portions, and less added sugar.\n\n"
                "தமிழ்: காய்கறிகள், முழுதானியங்கள் மற்றும் சரியான "
                "அளவிலான உணவுகளைத் தேர்வு செய்யுங்கள். கூடுதல் "
                "சர்க்கரையைக் குறைக்க முயற்சிக்கவும்."
            )

        if any(
            term in q
            for term in [
                "exercise", "walking", "activity",
                "உடற்பயிற்சி", "நடைப்பயிற்சி",
            ]
        ):
            return (
                "English: Regular physical activity supports general "
                "health. Choose activities suited to your abilities.\n\n"
                "தமிழ்: வழக்கமான உடற்பயிற்சி உடல்நலத்திற்கு உதவும். "
                "உங்கள் உடல்நிலைக்கு ஏற்ற செயல்பாடுகளைத் தேர்வு செய்யுங்கள்."
            )

        if "bmi" in q:
            return (
                "English: BMI compares weight with height. It is a "
                "screening measure, not a diagnosis.\n\n"
                "தமிழ்: BMI என்பது உயரத்தையும் எடையையும் ஒப்பிடும் "
                "அளவீடு. இது மட்டும் நோயைக் கண்டறியாது."
            )

        if any(
            term in q
            for term in [
                "blood pressure",
                "இரத்த அழுத்தம்",
            ]
        ) or q == "bp":
            return (
                "English: Blood pressure has an upper systolic number "
                "and a lower diastolic number. One reading cannot "
                "diagnose hypertension.\n\n"
                "தமிழ்: இரத்த அழுத்தத்தில் மேல் எண் மற்றும் கீழ் எண் "
                "இருக்கும். ஒரே அளவீட்டால் உயர் இரத்த அழுத்தத்தை "
                "உறுதி செய்ய முடியாது."
            )

        if any(
            term in q
            for term in [
                "screening", "test", "hba1c", "பரிசோதனை",
            ]
        ):
            return (
                "English: A healthcare professional can advise "
                "whether diabetes screening, such as HbA1c or "
                "fasting glucose, is appropriate.\n\n"
                "தமிழ்: HbA1c அல்லது வெறும் வயிற்று சர்க்கரைப் "
                "பரிசோதனை தேவையா என்பதை மருத்துவரிடம் ஆலோசிக்கலாம்."
            )

        if any(
            term in q
            for term in [
                "symptom", "thirst", "urination",
                "அறிகுறி", "தாகம்",
            ]
        ):
            return (
                "English: Possible symptoms include increased thirst, "
                "frequent urination, fatigue, and unexplained weight "
                "loss. Testing is needed for diagnosis.\n\n"
                "தமிழ்: அதிக தாகம், அடிக்கடி சிறுநீர் கழித்தல், "
                "சோர்வு மற்றும் காரணமற்ற எடை குறைதல் போன்றவை "
                "அறிகுறிகளாக இருக்கலாம். உறுதிப்படுத்த பரிசோதனை தேவை."
            )

        if any(
            term in q
            for term in [
                "diabetes", "sugar", "நீரிழிவு", "சர்க்கரை",
            ]
        ):
            return (
                "English: Diabetes involves elevated blood glucose. "
                "A clinical test is needed to determine whether "
                "someone has diabetes.\n\n"
                "தமிழ்: நீரிழிவு என்பது இரத்த சர்க்கரை அளவு "
                "அதிகரிக்கும் நிலை. இதை உறுதிப்படுத்த மருத்துவப் "
                "பரிசோதனை தேவை."
            )

        return (
            "English: I can answer general questions about diabetes, "
            "diet, exercise, BMI, blood pressure, and screening. "
            "For personal medical advice, consult a healthcare professional.\n\n"
            "தமிழ்: நீரிழிவு, உணவு, உடற்பயிற்சி, BMI, இரத்த அழுத்தம் "
            "மற்றும் பரிசோதனை பற்றிய பொதுவான தகவல்களை வழங்க முடியும். "
            "தனிப்பட்ட மருத்துவ ஆலோசனைக்கு மருத்துவரை அணுகவும்."
        )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Type your question / உங்கள் கேள்வியை உள்ளிடுங்கள்"
    )

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        answer = get_answer(question)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
        })

        st.rerun()

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# =========================================================
# 15. ABOUT AND LIMITATIONS
# =========================================================

def about_limitations_page():
    st.title("ℹ️ About This Project")
    st.subheader("Diabetes Risk Predictor — Health Awareness & Education")

    st.markdown(
        """
        ### 🌱 Project Brief

        Diabetes Risk Predictor is a HealthTech project created to make
        diabetes-related health information easier to explore. It combines
        machine learning with clear explanations and interactive learning
        features, helping users reflect on health indicators and build
        awareness. It is an educational prototype, not a medical diagnosis.

        ### ✨ Features Included

        - **Health assessment:** Enter health indicators and view the model's
          result for the supplied information.
        - **Explainable AI:** Explore which input features influence overall
          model performance, with percentage-style comparisons and a chart.
        - **Health insights:** View helpful context about diabetes-related
          indicators and healthier habits.
        - **Prevention awareness:** Explore educational tips and activities
          that encourage healthy lifestyle choices.
        - **Assessment records:** Review saved assessment information where
          record-saving features are enabled.
        - **Model and dataset information:** Learn what data the prototype
          was built with and how to interpret its output.
        - **Simple interface:** Use the app's sidebar to navigate between
          the available sections.

        ### 🔎 How It Works

        1. A user enters the available health indicators.
        2. The application maps inputs to the format expected by its model.
        3. The trained machine-learning model produces an output.
        4. The Explainable AI section presents feature-importance comparisons
           to help users understand model behaviour.

        The model is based on the public CDC Diabetes Health Indicators
        dataset. The dataset contains recorded diabetes-related status and
        health indicators; it is not a live feed of a person's current health.

        ### 📊 Understanding Explainable AI Percentages

        The percentages compare each feature's importance score with the
        highest positive feature-importance score in the analysis. They are
        **relative model-importance comparisons**, not a person's diabetes
        probability, percentage risk, or percentage contribution to a
        diagnosis. They do not need to add up to 100%.

        ### 🔐 Privacy & Data Protection

        - Treat health information as private. Do not enter information that
          you do not want stored or processed by this prototype.
        - Do not share assessment records, screenshots containing personal
          information, passwords, access tokens, or other credentials publicly.
        - Never store passwords in plain text or include passwords, API keys,
          or secret credentials in source code. Use secure password hashing
          and appropriate secret-management practices for any deployment.
        - Keep local datasets, saved user records, model files, and
          configuration files in controlled locations. Do not upload private
          records to public repositories or unrelated websites.
        - Only use datasets you are authorised to access and share. Do not
          expose private or identifiable records when demonstrating the app.
        - This prototype's use of local files does not by itself guarantee
          complete security. Actual storage, access control, backups, and
          deletion behaviour depend on how the app is configured and hosted.
        - Do not assume that data is encrypted, never stored, or automatically
          deleted unless those protections have been implemented and tested.

        ### 🧭 Limitations & Responsible Use

        - **Educational prototype:** The model is not presented as a
          clinically validated diagnostic or screening system.
        - **Dataset scope:** Results depend on the source dataset and the
          population it represents; performance may differ in other settings.
        - **Approximate input mapping:** Some user inputs may be mapped to
          dataset-style categories and may not match the original definitions.
        - **Self-reported indicators:** For example, `HighBP` is a history
          indicator in the dataset, not a measured systolic or diastolic
          blood-pressure value.
        - **Not future-onset prediction:** The model learns patterns in
          recorded diabetes-related status; it has not been established as a
          validated prediction of future diabetes onset.
        - **Importance is not causation:** Feature importance describes
          model behaviour and does not prove that a feature causes diabetes
          or explain an individual's health.
        - **No guaranteed outcome:** Educational tips can support awareness
          but cannot guarantee prevention or a change in model output.

        ### 💚 A Note on Health

        Use this application to learn and explore—not to make medical
        decisions. For personal concerns, speak with a qualified healthcare
        professional. Clinical testing is needed to determine whether someone
        has diabetes.
        """
    )

    st.success(
        "Our goal: make diabetes-related health information more understandable "
        "through an interactive, explainable, and privacy-aware learning tool."
    )
    st.caption(
        "Privacy reminder: protect personal health records and credentials, "
        "and verify security settings before deploying or sharing the app."
    )


# =========================================================
# 15. MAIN APPLICATION AND NAVIGATION
# =========================================================

def main():
    if not st.session_state.logged_in:
        authentication_page()
        return

    user = get_current_user()

    with st.sidebar:
        st.title("💙 Diabetes Awareness")
        st.caption("Health & Well-Being Platform")

        st.write(
            f"Welcome, **{user.get('name', 'User')}**"
        )

        st.caption(st.session_state.user_email)
        st.divider()

        pages = [
            "Dashboard",
            "Health Assessment",
            "Explainable AI",
            "Predictive Analytics",
            "Health Chatbot",
            "Prevention Quest",
            "Previous Records",
            "About & Limitations",
        ]

        selected_page = st.radio(
            "Navigation",
            pages,
            key="navigation",
        )

        st.divider()

        if st.button(
            "Logout",
            use_container_width=True,
        ):
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.session_state.chat_history = []

            # Reset game state when logging out.
            st.session_state.quest_points = 0
            st.session_state.quest_completed = []
            st.session_state.quest_answers = {}
            st.session_state.quest_challenge_done = []

            st.rerun()

    if selected_page == "Dashboard":
        dashboard_page()

    elif selected_page == "Health Assessment":
        assessment_page()

    elif selected_page == "Explainable AI":
        explainable_ai_page()

    elif selected_page == "Predictive Analytics":
        predictive_analytics_page()

    elif selected_page == "Health Chatbot":
        chatbot_page()

    elif selected_page == "Prevention Quest":
        prevention_game_page()

    elif selected_page == "Previous Records":
        previous_records_page()

    elif selected_page == "About & Limitations":
        about_limitations_page()


# =========================================================
# 16. RUN APPLICATION
# =========================================================

if __name__ == "__main__":
    main()
