import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import openai
import json
from datetime import datetime

# ------------------------------------------------------------
# Helper Functions for Reviews
# ------------------------------------------------------------
REVIEWS_FILE = "reviews.csv"

def load_reviews():
    if Path(REVIEWS_FILE).exists():
        return pd.read_csv(REVIEWS_FILE)
    return pd.DataFrame(columns=["name", "role", "rating", "review", "timestamp"])

def save_review(review_data):
    df = load_reviews()
    df = pd.concat([df, pd.DataFrame([review_data])], ignore_index=True)
    df.to_csv(REVIEWS_FILE, index=False)

# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Student Academic Performance Predictor",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Home (Prediction)",
        "📊 Prediction Results",
        "📈 Dashboard",
        "🔥 Feature Importance",
        "🔍 SHAP Explainability",
        "📚 Admin / Lecturer Prompts",
        "⭐ Reviews & Feedback",
        "ℹ️ About"
    ]
)

# ------------------------------------------------------------
# Load Model Artifacts
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "best_model.joblib"
PIPELINE_PATH = BASE_DIR / "outputs" / "models" / "best_pipeline.joblib"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline

model, pipeline = load_artifacts()

# ------------------------------------------------------------
# Global Variables
# ------------------------------------------------------------
prediction = None
probability = None
input_data = None

# ------------------------------------------------------------
# Encodings
# ------------------------------------------------------------
COURSE_MAP = {
    "Agronomy": 33,
    "Design": 171,
    "Nursing": 8014,
    "Social Service": 9070,
    "Management": 9991,
    "Technologies": 9119
}

PARENT_OCCUPATION_MAP = {
    "Unknown": 34,
    "Can't read or write": 35,
    "Basic education (1st cycle)": 37,
    "Basic education (2nd cycle)": 38,
    "Secondary education (12th year)": 1,
    "Higher education – Bachelor’s": 2,
    "Higher education – Degree": 3,
    "Higher education – Master’s": 4,
    "Higher education – Doctorate": 5,
    "Technical-professional course": 22,
    "Technological specialization course": 39
}

TUITION_MAP = {
    "Yes (Up to date)": 1,
    "No (Pending payments)": 0
}

# ============================================================
# HOME PAGE
# ============================================================
if page == "🏠 Home (Prediction)":
    st.title("📘 Student Academic Performance Predictor")

    with st.form("prediction_form"):
        age = st.number_input("Age at Enrollment", 14, 100, 18)
        admission_grade = st.number_input("Admission Grade", 0.0, 200.0)
        prev_grade = st.number_input("Previous Qualification Grade", 0.0, 300.0)

        course = COURSE_MAP[st.selectbox("Course", COURSE_MAP.keys())]
        tuition = TUITION_MAP[st.selectbox("Tuition Fees Status", TUITION_MAP.keys())]

        cu1_enrolled = st.number_input("1st Sem Units Enrolled", 0, 40)
        cu1_approved = st.number_input("1st Sem Units Approved", 0, 40)
        cu1_grade = st.number_input("1st Sem Grade", 0.0, 20.0)
        cu1_eval = st.number_input("1st Sem Evaluations", 0, 100)

        cu2_enrolled = st.number_input("2nd Sem Units Enrolled", 0, 40)
        cu2_approved = st.number_input("2nd Sem Units Approved", 0, 40)
        cu2_grade = st.number_input("2nd Sem Grade", 0.0, 20.0)
        cu2_eval = st.number_input("2nd Sem Evaluations", 0, 100)

        submit = st.form_submit_button("🔍 Predict Performance")

    if submit:
        input_data = pd.DataFrame([{
            "Age_at_enrollment": age,
            "Admission_grade": admission_grade,
            "Previous_qualification_grade": prev_grade,
            "Course": course,
            "Tuition_fees_up_to_date": tuition,
            "Curricular_units_1st_sem_enrolled": cu1_enrolled,
            "Curricular_units_1st_sem_approved": cu1_approved,
            "Curricular_units_1st_sem_grade": cu1_grade,
            "Curricular_units_1st_sem_evaluations": cu1_eval,
            "Curricular_units_2nd_sem_enrolled": cu2_enrolled,
            "Curricular_units_2nd_sem_approved": cu2_approved,
            "Curricular_units_2st_sem_grade": cu2_grade,
            "Curricular_units_2nd_sem_evaluations": cu2_eval
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data).max()

        st.session_state["input_data"] = input_data
        st.session_state["prediction"] = prediction
        st.session_state["probability"] = probability

        label_map = {0: "Dropout 🚫", 1: "Enrolled 📚", 2: "Graduate 🎓"}
        st.success(f"Prediction: **{label_map[prediction]}** ({probability:.2f})")

# ============================================================
# ABOUT PAGE
# ============================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("""
### 📘 Student Academic Performance Predictor

This application predicts student academic outcomes using machine learning.

**Purpose**
- Early risk identification
- Explainable AI for academic decision support
- Ethical, human-in-the-loop deployment

**Key Technologies**
- Streamlit
- scikit-learn
- SHAP
- Pandas / Matplotlib
- OpenAI (assistant)

**Ethics**
This tool is decision-support only and must not replace human academic judgment.
""")

# ============================================================
# REVIEWS
# ============================================================
elif page == "⭐ Reviews & Feedback":
    st.title("⭐ Reviews & Feedback")

    with st.form("review_form"):
        name = st.text_input("Name (optional)")
        role = st.selectbox("Role", ["Student", "Lecturer", "Admin", "Other"])
        rating = st.slider("Rating", 1, 5, 4)
        review = st.text_area("Review")
        submit = st.form_submit_button("Submit")

    if submit and review.strip():
        save_review({
            "name": name or "Anonymous",
            "role": role,
            "rating": rating,
            "review": review,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        st.success("Thank you for your feedback!")

    df = load_reviews()
    for _, r in df.iterrows():
        st.markdown(f"**{r['name']}** ({r['role']}) ⭐{r['rating']}")
        st.write(r["review"])
        st.caption(r["timestamp"])
