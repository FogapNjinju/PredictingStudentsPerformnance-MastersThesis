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
    """Load reviews from CSV file"""
    if Path(REVIEWS_FILE).exists():
        return pd.read_csv(REVIEWS_FILE)
    return pd.DataFrame(columns=["name", "role", "rating", "review", "timestamp"])

def save_review(review_data):
    """Append review to CSV file"""
    df = load_reviews()
    df = pd.concat([df, pd.DataFrame([review_data])], ignore_index=True)
    df.to_csv(REVIEWS_FILE, index=False)


# ------------------------------------------------------------
# Streamlit Page Config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Student Academic Performance Predictor",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------------
# CUSTOM UI THEMING
# ------------------------------------------------------------
st.markdown("""
<style>
/* -----------------------------------------
ROOT VARIABLES (AUTO DARK/LIGHT MODE)
------------------------------------------*/
:root {
    --bg-main: #ffffff;
    --bg-sidebar: #f3f3f3;
    --bg-card: rgba(240,240,240,0.85);
    --text-main: #111111;
    --text-muted: #555555;
    --border-color: #cccccc;
    --accent: #FFD93D;
    --button-bg: #111111;
    --button-text: #ffffff;
}

/* Dark mode overrides */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-main: #0e1117;
        --bg-sidebar: #161b22;
        --bg-card: rgba(30,30,30,0.85);
        --text-main: #f0f0f0;
        --text-muted: #b0b0b0;
        --border-color: #2f2f2f;
        --accent: #FFD93D;
        --button-bg: #FFD93D;
        --button-text: #111111;
    }
}

/* -----------------------------------------
GLOBAL
------------------------------------------*/
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-main);
}

/* -----------------------------------------
SIDEBAR
------------------------------------------*/
section[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    padding: 20px;
}

/* -----------------------------------------
HEADINGS
------------------------------------------*/
h1, h2, h3 {
    color: var(--text-main) !important;
    font-weight: 800 !important;
}

/* -----------------------------------------
MAIN CONTAINER
------------------------------------------*/
.block-container {
    padding-top: 2rem;
}

/* -----------------------------------------
BUTTONS
------------------------------------------*/
.stButton > button {
    background: var(--button-bg) !important;
    color: var(--button-text) !important;
    border-radius: 12px !important;
    border: 3px solid var(--border-color) !important;
    padding: 10px 20px;
    font-weight: 700;
}

.stButton > button:hover {
    background: var(--accent) !important;
    color: #111 !important;
    transition: 0.3s;
}

/* -----------------------------------------
RADIO BUTTONS
------------------------------------------*/
div[role='radiogroup'] > label {
    background: var(--bg-card);
    padding: 10px 15px;
    border-radius: 10px;
    margin-bottom: 8px;
    font-size: 16px;
    font-weight: 600;
    display: inline-block;
    width: 250px;
    text-align: left;
    color: var(--text-main);
    border: 1px solid var(--border-color);
}

div[role='radiogroup'] > label:hover {
    background: rgba(255,255,255,0.15);
}

/* -----------------------------------------
CARDS / PANELS
------------------------------------------*/
.custom-card {
    background: var(--bg-card);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid var(--border-color);
    color: var(--text-main);
}

/* -----------------------------------------
METRICS
------------------------------------------*/
[data-testid="stMetricValue"] {
    color: var(--text-main);
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted);
}

/* -----------------------------------------
DATAFRAMES
------------------------------------------*/
.stDataFrame {
    background-color: var(--bg-main);
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------
st.sidebar.title("📊 Navigation")

# Initialize page in session state if not already
if "current_page" not in st.session_state:
    st.session_state["current_page"] = 0

page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home (Prediction)", "📊 Prediction Results","📈 Dashboard", "🔥 What Influenced This Result?", "🔍 Detailed Explanation (Advanced)","📚 Admin / Lecturer Prompts","⭐ Reviews & Feedback", "ℹ️ About"]
)


# ------------------------------------------------------------
# Load Model Artifacts
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "best_model.joblib"
PIPELINE_PATH = BASE_DIR / "outputs" / "models" / "best_pipeline.joblib"

# Model names and their file paths
AVAILABLE_MODELS = {
    "LightGBM": "lightgbm_model.joblib",
    "RandomForest": "randomforest_model.joblib",
    "XGBoost": "xgboost_model.joblib",
    "DecisionTree": "decisiontree_model.joblib",
    "SVM": "svm_model.joblib"
}

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline

def load_selected_model(model_name):
    """Load a specific model by name"""
    if model_name not in AVAILABLE_MODELS:
        st.error(f"Model '{model_name}' not found.")
        return None
    
    model_file = AVAILABLE_MODELS[model_name]
    model_path = BASE_DIR / "outputs" / "models" / model_file
    
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}")
        return None
    
    return joblib.load(model_path)

model, pipeline = load_artifacts()

# ============================================================
# Helper Functions for Actionable Recommendations
# ============================================================

def get_confidence_interpretation(confidence):
    """Interpret confidence score and return color + interpretation"""
    if confidence >= 0.90:
        return {
            "level": "🟢 Very High Certainty",
            "description": "The model is very confident in this prediction.",
            "color": "green",
            "interpretation": "Use this prediction with high confidence for decision-making."
        }
    elif confidence >= 0.70:
        return {
            "level": "🟡 Moderate Confidence",
            "description": "The model has reasonable confidence in this prediction.",
            "color": "orange",
            "interpretation": "This prediction is reliable but should be considered alongside other factors."
        }
    else:
        return {
            "level": "🔴 Use with Caution",
            "description": "The model's confidence is below typical thresholds.",
            "color": "red",
            "interpretation": "This prediction is uncertain. Verify with additional assessment methods."
        }

def get_actionable_recommendations(prediction_label, confidence, input_data):
    """Generate actionable recommendations based on prediction outcome"""

    if prediction_label.startswith("Dropout"):
        return {
            "status": "🚨 INTERVENTION REQUIRED",
            "status_color": "error",
            "title": "Student At-Risk of Dropout",
            "actions": [
                "📞 **Contact Student Immediately** – Reach out within 48 hours to understand challenges",
                "📋 **Academic Assessment** – Review performance data; identify struggling subjects",
                "💰 **Check Financial Status** – Verify tuition payments and financial aid eligibility",
                "🤝 **Assign Mentor/Advisor** – Pair with academic or peer mentor for support",
                "📚 **Tutoring Referral** – Recommend subject-specific or general tutoring services",
                "🎯 **Create Support Plan** – Develop written action plan with clear milestones",
                "📊 **Monitor Progress** – Schedule regular check-ins (weekly/bi-weekly)",
                "🏫 **Campus Resources** – Connect with counseling, career services, or disability support"
            ],
            "urgency": "HIGH",
            "timeline": "Immediate action required"
        }

    elif prediction_label.startswith("Enrolled"):
        return {
            "status": "⚠️ ONGOING MONITORING",
            "status_color": "warning",
            "title": "Student On Track – Requires Support",
            "actions": [
                "✅ **Positive Reinforcement** – Acknowledge effort and progress made",
                "🔍 **Identify Risk Factors** – Use SHAP analysis to see what could cause dropout",
                "🎯 **Set Academic Goals** – Help student establish semester/year targets",
                "📈 **Monitor Grade Trends** – Track progression to ensure grades don't decline",
                "🤝 **Peer Support** – Encourage study groups and peer collaboration",
                "💪 **Build Resilience** – Teach time management, stress management techniques",
                "🌟 **Challenge & Engage** – Offer opportunities for academic enrichment",
                "📅 **Scheduled Check-ins** – Monthly progress reviews to stay on track"
            ],
            "urgency": "MEDIUM",
            "timeline": "Regular monitoring recommended"
        }

    else: # Graduate
        return {
            "status": "🎓 ON TRACK FOR SUCCESS",
            "status_color": "success",
            "title": "Student Likely to Graduate",
            "actions": [
                "🌟 **Positive Recognition** – Celebrate strong academic performance",
                "🎓 **Graduation Planning** – Begin final degree requirements checklist",
                "💼 **Career Development** – Connect with career services for post-graduation planning",
                "📚 **Advanced Opportunities** – Suggest honors programs, research, or internships",
                "🔗 **Alumni Network** – Prepare for transition to alumni community",
                "💬 **Peer Mentoring** – Encourage student to mentor struggling peers",
                "🎯 **Post-Graduation Goals** – Discuss grad school or employment plans",
                "🏆 **Recognition** – Consider for scholarships, awards, or leadership roles"
            ],
            "urgency": "LOW",
            "timeline": "Supportive monitoring"
        }


# ============================================================
# Plain Language Explanations & Tooltips
# ============================================================

TOOLTIP_PREDICTION_CERTAINTY = """
**Prediction Certainty** (0.0 - 1.0):

* **0.9+** = The model is very sure about this prediction
* **0.7-0.89** = The model is reasonably confident
* **Below 0.7** = The prediction is uncertain; verify with other methods
"""

TOOLTIP_WHAT_INFLUENCED = """
**What Influenced This Result?**
This shows which student factors had the biggest impact on the prediction.
Factors at the top pushed the prediction most strongly.
Think of it like a recipe - this shows which ingredients matter most.
"""

TOOLTIP_DETAILED_EXPLANATION = """
**Detailed Explanation (Advanced)**
This shows exactly HOW each factor influenced the prediction.

* Green factors pushed toward "Graduate"
* Red factors pushed toward "Dropout"
* Length of bar = strength of influence

Example: If "2nd semester grade" has a long red bar,
that low grade is a major reason for the dropout prediction.
"""

TOOLTIP_PREDICTION_RESULT = """
**What's this prediction?**
The model learned patterns from many students to predict three outcomes:

* 🚫 Dropout: Student likely to leave before graduating
* 📚 Enrolled: Student likely to continue but timeline uncertain
* 🎓 Graduate: Student likely to complete degree
"""

def show_tooltip(title, content, color="info"):
    """Display a formatted tooltip"""
    if color == "info":
        st.info(content)
    elif color == "warning":
        st.warning(content)
    elif color == "success":
        st.success(content)


# ============================================================
# Global variables
# ============================================================
prediction = None
probability = None
input_data = None


# ==============================
# MAPPINGS
# ==============================
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

# ------------------------------------------------------------
# ---------------------- HOME PAGE ---------------------------
# ------------------------------------------------------------
help_course = " Numerical identifier representing the students degree programme.Different programmes have distinct academic structures, assessment difficulty, and dropout risks, making this a strong contextual predictor. Degree program code: 33=Agronomy, 171=Design, 8014=Nursing, 9070=Social Service, 9991=Management, 9119=Technologies"
help_prevqual = "Grade obtained in the student's highest previous qualification (e.g. secondary education or equivalent). This reflects prior academic preparedness and baseline learning capacity. (between 0 and 200)"
help_parents_qual = "Parent education level (encoded)."
help_parents_occ = "Parent job (encoded)."
MOTHERS_OCCUPATION = "1 - Secondary Education - 12th Year of Schooling or Eq. 2 - Higher Education - Bachelor's Degree 3 - Higher Education - Degree 4 - Higher Education - Master's 5 - Higher Education - Doctorate 6 - Frequency of Higher Education 9 - 12th Year of Schooling - Not Completed 10 - 11th Year of Schooling - Not Completed 11 - 7th Year (Old) 12 - Other - 11th Year of Schooling 14 - 10th Year of Schooling 18 - General commerce course 19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv. 22 - Technical-professional course 26 - 7th year of schooling 27 - 2nd cycle of the general high school course 29 - 9th Year of Schooling - Not Completed 30 - 8th year of schooling 34 - Unknown 35 - Can't read or write 36 - Can read without having a 4th year of schooling 37 - Basic education 1st cycle (4th/5th year) or equiv. 38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 41 - Specialized higher studies course 42 - Professional higher technical course 43 - Higher Education - Master (2nd cycle) 44 - Higher Education - Doctorate (3rd cycle)"
FATHERS_OCCUPATION = "1 - Secondary Education - 12th Year of Schooling or Eq. 2 - Higher Education - Bachelor's Degree 3 - Higher Education - Degree 4 - Higher Education - Master's 5 - Higher Education - Doctorate 6 - Frequency of Higher Education 9 - 12th Year of Schooling - Not Completed 10 - 11th Year of Schooling - Not Completed 11 - 7th Year (Old) 12 - Other - 11th Year of Schooling 13 - 2nd year complementary high school course 14 - 10th Year of Schooling 18 - General commerce course 19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv. 20 - Complementary High School Course 22 - Technical-professional course 25 - Complementary High School Course - not concluded 26 - 7th year of schooling 27 - 2nd cycle of the general high school course 29 - 9th Year of Schooling - Not Completed 30 - 8th year of schooling 31 - General Course of Administration and Commerce 33 - Supplementary Accounting and Administration 34 - Unknown 35 - Can't read or write 36 - Can read without having a 4th year of schooling 37 - Basic education 1st cycle (4th/5th year) or equiv. 38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv. 39 - Technological specialization course 40 - Higher education - degree (1st cycle) 41 - Specialized higher studies course 42 - Professional higher technical course 43 - Higher Education - Master (2nd cycle) 44 - Higher Education - Doctorate (3rd cycle)"
help_admission_grade = "Final admission score used for university entry (0–200). Reflects prior academic achievement and preparedness for higher education."
help_tuition = "Indicates whether tuition fees are fully paid at the time of evaluation. Financial instability is a known risk factor for dropout and delayed progression. 1 = Up to date | 0 = Behind on tuition fees"
help_age = "Student age at the time of enrolment. Younger or older students may face different academic and social challenges impacting performance."
help_units_enrolled_1 = "Number of curricular units the student enrolled in during the first semester, This reflects initial academic workload and commitment."
help_units_eval_1 = "Number of assessments or evaluations taken in the first semester. This indicates engagement and progress in coursework."
help_units_approved_1 = "Number of curricular units successfully passed in the first semester. This shows academic success and mastery of material."
help_units_grade_1 = "Average grade obtained across approved curricular units in the first semester. This reflects overall academic performance (between 0 and 20)"
help_units_enrolled_2 = "Number of curricular units enrolled in during the second semester. Indicates continued academic engagement."
help_units_eval_2 = "Number of curricular units enrolled in during the second semester. Indicates continued academic engagement."
help_units_approved_2 = "Number of curricular units successfully completed in the second semester. Reflects ongoing academic success."
help_units_grade_2 = "Average grade obtained in the second semester. Reflects academic performance over the latter half of the academic year (between 0 and 20)."
why_demographic_background = (
    "This information provides contextual background that may influence academic pathways, "
    "engagement patterns, and access to support resources. It helps the model interpret "
    "academic performance within a broader student context."
)

why_academic_performance = (
    "These indicators reflect the student’s academic engagement and achievement across semesters. "
    "They are key predictors of progression, persistence, and completion, and support early "
    "identification of students who may benefit from academic intervention."
)


if page == "🏠 Home (Prediction)":
    st.markdown("""
    <h1 style='text-align:center; color:#2C3E50;'>📘 Student Academic Performance Predictor</h1>
    <p style='text-align:center; font-size:18px;'>
    Predict the academic performance category of a student based on academic & demographic factors.
    </p>
    <hr style="border:1px solid #bbb;">
    """, unsafe_allow_html=True)

    st.header("📝 Student Information")
    with st.form("prediction_form"):
        # ===============================
        # STEP 1: DEMOGRAPHICS
        # ===============================
        with st.expander("🧍 Step 1: Demographics & Background", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                Age_at_enrollment = st.number_input("Age at Enrollment", 14, 100, 18, help=help_age)
                mother_label = st.selectbox("Mother's Education Level", PARENT_OCCUPATION_MAP.keys())
                Mothers_occupation = PARENT_OCCUPATION_MAP[mother_label]

            with col2:
                father_label = st.selectbox("Father's Education Level", PARENT_OCCUPATION_MAP.keys())
                Fathers_occupation = PARENT_OCCUPATION_MAP[father_label]
                tuition_label = st.selectbox("Tuition Fees Status", TUITION_MAP.keys(), help=help_tuition)
                Tuition_fees_up_to_date = TUITION_MAP[tuition_label]

            with col3:
                course_label = st.selectbox("Course", COURSE_MAP.keys(), help=help_course)
                Course = COURSE_MAP[course_label]
                Admission_grade = st.number_input("Admission Grade", 0.0, 200.0, help=help_admission_grade)
                Previous_qualification_grade = st.number_input("Previous Qualification Grade", 0.0, 200.0, help=help_prevqual)

        # ===============================
        # STEP 2: SEMESTER 1
        # ===============================
        with st.expander("📘 Step 2: Academic Performance – Semester 1"):
            col4, col5 = st.columns(2)

            with col4:
                Curricular_units_1st_sem_enrolled = st.number_input("Units Enrolled", 0, 20, help=help_units_enrolled_1)
                Curricular_units_1st_sem_evaluations = st.number_input("Evaluations", 0, 20, help=help_units_eval_1)

            with col5:
                Curricular_units_1st_sem_approved = st.number_input("Units Approved", 0, 20, help=help_units_approved_1)
                Curricular_units_1st_sem_grade = st.number_input("Average Grade", 0.0, 20.0, help=help_units_grade_1)

            if Curricular_units_1st_sem_approved > Curricular_units_1st_sem_enrolled:
                st.warning("⚠ Approved units cannot exceed enrolled units.")

            if Curricular_units_1st_sem_grade == 0 and Curricular_units_1st_sem_approved > 0:
                st.warning("⚠ Grade is 0 but units are approved. Please verify.")

        # ===============================
        # STEP 3: SEMESTER 2
        # ===============================
        with st.expander("📗 Step 3: Academic Performance – Semester 2"):
            col6, col7 = st.columns(2)

            with col6:
                Curricular_units_2nd_sem_enrolled = st.number_input("Units Enrolled ", 0, 20, help=help_units_enrolled_2)
                Curricular_units_2nd_sem_evaluations = st.number_input("Evaluations ", 0, 20, help=help_units_eval_2)

            with col7:
                Curricular_units_2nd_sem_approved = st.number_input("Units Approved ", 0, 20, help=help_units_approved_2)
                Curricular_units_2st_sem_grade = st.number_input("Average Grade ", 0.0, 20.0, help=help_units_grade_2)

            if Curricular_units_2nd_sem_approved > Curricular_units_2nd_sem_enrolled:
                st.warning("⚠ Approved units cannot exceed enrolled units.")

            if Curricular_units_2st_sem_grade == 0 and Curricular_units_2nd_sem_approved > 0:
                st.warning("⚠ Grade is 0 but units are approved. Please verify.")

        st.markdown("---")       
        # ======= MODEL SELECTION =======
        st.subheader("🤖 Model Selection")
        col_model = st.columns(1)[0]
        with col_model:
            selected_model_name = st.selectbox(
                "Choose a prediction model:",
                options=list(AVAILABLE_MODELS.keys()),
                index=0,  # Default to LightGBM
                help="Select which ML model to use for predictions. LightGBM is recommended for best performance."
            )
        
        st.markdown("---")
        submitted = st.form_submit_button("🔍 Predict Performance", use_container_width=True)
        if submitted:
            # Load the selected model
            selected_model = load_selected_model(selected_model_name)
            if selected_model is None:
                st.error("Failed to load the selected model. Please try again.")
            else:
                input_data = pd.DataFrame([{
                    "Curricular_units_2nd_sem_approved": Curricular_units_2nd_sem_approved,
                    "Tuition_fees_up_to_date": Tuition_fees_up_to_date,
                    "Curricular_units_2st_sem_grade": Curricular_units_2st_sem_grade,
                    "Admission_grade": Admission_grade,
                    "Previous_qualification_grade": Previous_qualification_grade,
                    "Curricular_units_1st_sem_grade": Curricular_units_1st_sem_grade,
                    "Curricular_units_2nd_sem_enrolled": Curricular_units_2nd_sem_enrolled,
                    "Course": Course,
                    "Age_at_enrollment": Age_at_enrollment,
                    "Curricular_units_1st_sem_evaluations": Curricular_units_1st_sem_evaluations,
                    "Curricular_units_2nd_sem_evaluations": Curricular_units_2nd_sem_evaluations,
                    "Curricular_units_1st_sem_approved": Curricular_units_1st_sem_approved,
                    "Curricular_units_1st_sem_enrolled": Curricular_units_1st_sem_enrolled,
                    "Fathers_occupation": Fathers_occupation,
                    "Mothers_occupation": Mothers_occupation
                }])
                prediction = selected_model.predict(input_data)[0]
                probability = selected_model.predict_proba(input_data).max()
                label_map = {0: "Dropout 🚫🎓", 1: "Enrolled 📚🎓", 2: "Graduate 🎓✨"}
                prediction_label = label_map.get(prediction, "Unknown")
                st.markdown("---")
                st.header("📊 Prediction Results")
                with st.expander("ℹ️ What does this prediction mean?", expanded=False):
                    st.info(TOOLTIP_PREDICTION_RESULT)
                colA, colB = st.columns(2)
                with colA:
                    st.metric("Predicted Category", prediction_label)
                with colB:
                    st.metric("Prediction Certainty", f"{probability:.2f}")

                # ===== CONFIDENCE INTERPRETATION =====
                confidence_info = get_confidence_interpretation(probability)
                col_conf1, col_conf2 = st.columns([1, 2])
                with col_conf1:
                    st.metric("Certainty Level", confidence_info["level"])
                with col_conf2:
                    st.info(f"**{confidence_info['interpretation']}**")

                st.markdown("---")

                # ===== ACTIONABLE RECOMMENDATIONS =====
                recommendations = get_actionable_recommendations(prediction_label, probability, input_data)

                if recommendations["status_color"] == "error":
                    st.error(f"### {recommendations['status']}")
                elif recommendations["status_color"] == "warning":
                    st.warning(f"### {recommendations['status']}")
                else:
                    st.success(f"### {recommendations['status']}")

                st.subheader(f"📋 {recommendations['title']}")
                st.markdown(f"**Urgency Level:** {recommendations['urgency']} | **Timeline:** {recommendations['timeline']}")

                st.markdown("### ✅ Recommended Actions:")
                for i, action in enumerate(recommendations["actions"], 1):
                    st.markdown(f"{i}. {action}")

                st.markdown("---")
                st.success(f"🎯 The student is predicted to **{prediction_label}** with a confidence of **{probability:.2f}**.")
                st.session_state["input_data"] = input_data
                st.session_state["selected_model"] = selected_model
                st.session_state["selected_model_name"] = selected_model_name
            st.session_state["prediction"] = prediction
            st.session_state["probability"] = probability

# ------------------------------------------------------------
# ------------------ 📊 PREDICTION RESULTS TAB ---------------
# ------------------------------------------------------------
elif page == "📊 Prediction Results":
    st.title("📊 Prediction Results")
    st.markdown("Review the prediction outcome for the selected student.")
    st.info("""
### 🧠 How was this prediction made?

The prediction is generated by a trained machine learning model that analyzes the student's
academic performance, engagement indicators, and contextual background.

The **predicted category** corresponds to the class with the highest estimated probability
among the three possible outcomes (Dropout, Enrolled, Graduate).

The **confidence score** represents the model’s estimated probability for the predicted category.
It is extracted directly from the model’s internal probability output and is not a separate
evaluation metric.

To understand *why* the model made this prediction, feature importance and SHAP explanations
are provided in the corresponding tabs.
""")

    # Check if prediction exists
    if "prediction" not in st.session_state or "probability" not in st.session_state:
        st.warning("⚠ No prediction available yet. Please enter inputs in the prediction page.")
    else:
        # Map numeric prediction to category label
        label_map = {0: "Dropout 🚫🎓", 1: "Enrolled 📚🎓", 2: "Graduate 🎓✨"}
        prediction = label_map.get(st.session_state["prediction"], "Unknown")
        confidence = st.session_state["probability"]

        # Display metrics in columns
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="🏷 Predicted Category",
                value=prediction
            )

        with col2:
            st.metric(
                label="📈 Confidence Score",
                value=f"{confidence:.2f}"
            )

        # Nice result card
        st.markdown("---")
        st.markdown(
            f"""
<div style="
    background: rgba(240,240,240,0.7);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #ccc;
">
    <h3 style="margin-bottom: 10px;">🎯 Final Prediction Summary</h3>
    <p style="font-size: 18px;">
    The student is predicted to <b>{prediction}</b> with a confidence score of
    <b>{confidence:.2f}</b>.
    </p>
</div>
""",
            unsafe_allow_html=True
        )

# ------------------------------------------------------------
# ---------------------- DASHBOARD PAGE -----------------------
# ------------------------------------------------------------
elif page == "📈 Dashboard":
    st.title("📈 Student Dashboard")
    if "input_data" not in st.session_state:
        st.warning("⚠ Please make a prediction first on the Home page.")
        st.stop()
    input_data = st.session_state["input_data"]

    # ---------------------- KPIs -----------------------
    st.subheader("📌 Summary KPIs")
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    total_units = input_data["Curricular_units_1st_sem_enrolled"][0] + input_data["Curricular_units_2nd_sem_enrolled"][0]
    avg_grade = (input_data["Curricular_units_1st_sem_grade"][0] + input_data["Curricular_units_2st_sem_grade"][0]) / 2
    fees_status = input_data["Tuition_fees_up_to_date"][0]
    prediction_raw = st.session_state.get("prediction", "Unknown")
    probability = st.session_state.get("probability", 0)
    # Map numeric prediction to a readable label for display and logic
    label_map_short = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
    try:
        prediction_label = label_map_short.get(int(prediction_raw), str(prediction_raw))
    except Exception:
        prediction_label = str(prediction_raw)

    col_kpi1.metric("Total Units Enrolled", total_units)
    col_kpi2.metric("Average Semester Grade", f"{avg_grade:.2f}/20")
    col_kpi3.metric("Tuition Fees Up-to-Date", "✔ Yes" if fees_status==1 else "❌ No")
    col_kpi4.metric("Predicted Category", prediction_label)

    # ---------------------- Benchmarks & Insight Logic ------------------
    RECOMMENDED_AVG_GRADE = 10.0 # benchmark: passing/healthy average out of 20
    # Expect about 75% of enrolled units to be approved as a simple benchmark
    expected_approved_total = int(total_units * 0.75) if total_units > 0 else 0
    units_1_approved = input_data["Curricular_units_1st_sem_approved"][0]
    units_2_approved = input_data["Curricular_units_2nd_sem_approved"][0]
    approved_total = units_1_approved + units_2_approved

    below_recommended_avg = avg_grade < RECOMMENDED_AVG_GRADE
    healthy_academic_load = approved_total >= expected_approved_total and total_units > 0

    # Determine overall academic status
    if prediction_label.startswith("Dropout") or (below_recommended_avg and not healthy_academic_load) or fees_status == 0:
        overall_status = "High Risk"
        status_color = "#E74C3C"
    elif below_recommended_avg or not healthy_academic_load:
        overall_status = "Moderate Risk"
        status_color = "#F39C12"
    else:
        overall_status = "Low Risk"
        status_color = "#27AE60"

    # Key concern detection
    key_concerns = []
    if input_data["Curricular_units_2st_sem_grade"][0] < RECOMMENDED_AVG_GRADE:
        key_concerns.append("Low 2nd semester grade")
    if approved_total < expected_approved_total:
        key_concerns.append("Lower-than-expected approved units")
    if fees_status == 0:
        key_concerns.append("Tuition payments pending")

    key_concern_text = ", ".join(key_concerns) if key_concerns else "None"
    # Recommended focus (simple, actionable suggestions)
    recommended_actions = []
    if "Low 2nd semester grade" in key_concerns:
        recommended_actions.append("Academic tutoring")
    if "Tuition payments pending" in key_concerns:
        recommended_actions.append("Fee follow-up / financial aid check")
    if "Lower-than-expected approved units" in key_concerns:
        recommended_actions.append("Study plan & workload review")
    if not recommended_actions:
        recommended_actions.append("Continue regular monitoring and support")

    # ---------------------- Charts ----------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Semester Grades")
        st.bar_chart(pd.DataFrame({
            "Semester": ["1st", "2nd"],
            "Grade": [input_data["Curricular_units_1st_sem_grade"][0], input_data["Curricular_units_2st_sem_grade"][0]]
        }).set_index("Semester"))
        # contextual annotation for grades
        if input_data["Curricular_units_2st_sem_grade"][0] < RECOMMENDED_AVG_GRADE:
            st.markdown("⚠️ **Below recommended average (≥10/20)** — consider targeted tutoring for low second-semester grades.")
        else:
            st.markdown("✔️ **Average grade meets or exceeds the recommended benchmark (≥10/20).**")
    with col2:
        st.subheader("Curricular Progress")
        st.bar_chart(pd.DataFrame({
            "Status": ["Enrolled", "Evaluated", "Approved"],
            "1st Sem": [
                input_data["Curricular_units_1st_sem_enrolled"][0],
                input_data["Curricular_units_1st_sem_evaluations"][0],
                input_data["Curricular_units_1st_sem_approved"][0]
            ]
        }).set_index("Status"))
        # contextual annotation for academic load
        if healthy_academic_load:
            st.markdown("✔️ **Healthy academic load** — approved units meet expectations.")
        else:
            st.markdown(f"⚠️ **Below expected approved units** (approved {approved_total} vs expected {expected_approved_total}).")
    with col3:
        st.subheader("Tuition Status")
        if fees_status==1:
            st.success("Fees are up-to-date ✔")
        else:
            st.error("Fees NOT up-to-date ❌")

    # ---------------------- Summary Insight Card --------------------
    st.markdown("---")
    st.markdown(
        f"""
<div style="background: rgba(250,250,250,0.9); padding:20px; border-radius:12px; border-left:6px solid {status_color};">
    <h3 style="margin:0;">**Overall Academic Status:** {overall_status}</h3>
    <p style="margin:6px 0 0 0;"><strong>Key Concern:</strong> {key_concern_text}</p>
    <p style="margin:6px 0 0 0;"><strong>Recommended Focus:</strong> {', '.join(recommended_actions)}</p>
</div>
""",
        unsafe_allow_html=True
    )

# ------------------------------------------------------------
# ---------------------- FEATURE IMPORTANCE -------------------
# ------------------------------------------------------------
elif page == "🔥 What Influenced This Result?":
    st.title("🔥 What Influenced This Result?")

    if "input_data" not in st.session_state:
        st.warning("⚠ Please make a prediction first on the Home page.")
        st.stop()

    try:
        final_model = model[-1]
        importances = final_model.feature_importances_

        fi_df = pd.DataFrame({
            "Feature": st.session_state["input_data"].columns,
            "Importance": importances
        }).sort_values("Importance", ascending=True) # ascending for horizontal bars

        st.subheader("📌 Ranking of Factors")
        with st.expander("ℹ️ What does this mean?"):
            st.info(TOOLTIP_WHAT_INFLUENCED)
        st.write("The chart below displays the contribution of each factor to the model's decision.")

        # ----- Nicely formatted horizontal bar chart -----
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(fi_df["Feature"], fi_df["Importance"])
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance Ranking")
        plt.tight_layout()

        st.pyplot(fig)

        # ----- Display table too -----
        st.subheader("📋 Ranking Table - What Influenced the Prediction")
        st.dataframe(fi_df[::-1].reset_index(drop=True)) # highest first

    except Exception as e:
        st.warning("⚠ Factor analysis is not available for this model.")
        st.text(str(e))

# ------------------------------------------------------------
# ---------------------- SHAP EXPLAINABILITY -----------------
# ------------------------------------------------------------
elif page == "🔍 Detailed Explanation (Advanced)":
    st.title("🔍 Detailed Explanation (Advanced)")
    with st.expander("ℹ️ How do I read this?", expanded=False):
        st.info(TOOLTIP_DETAILED_EXPLANATION)
    if "input_data" not in st.session_state:
        st.warning("⚠ Please make a prediction first on the Home page.")
        st.stop()

    input_data = st.session_state["input_data"]
    prediction = st.session_state.get("prediction", 0)
    
    # Get the selected model from session state, or use default model
    selected_model = st.session_state.get("selected_model", model)
    selected_model_name = st.session_state.get("selected_model_name", "Best Model")
    
    # Display which model is being explained
    st.info(f"📊 Showing SHAP explanation for: **{selected_model_name}** model")
    
    try:
        import numpy as np
        
        preprocessor = selected_model[:-1]
        final_model = selected_model[-1]
        X_transformed = preprocessor.transform(input_data)
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_transformed)

        # Handle different SHAP value structures
        # Some models return a list of arrays (one per class), others return a single array
        if isinstance(shap_values, list):
            # Multi-output case (list of arrays for each class)
            if len(shap_values) > prediction:
                shap_vals = shap_values[prediction]
                expected_val = explainer.expected_value[prediction] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                # Fallback to first class if prediction index is out of bounds
                shap_vals = shap_values[0]
                expected_val = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value
        else:
            # Single array case
            shap_vals = shap_values
            expected_val = explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[0]

        st.subheader("Top Feature Contributions")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_vals, X_transformed, feature_names=input_data.columns, plot_type="bar", show=False)
        st.pyplot(fig)

        st.subheader("Detailed SHAP Force Plot")
        try:
            # Prepare single-instance arrays for force_plot
            X_arr = np.asarray(X_transformed)

            # Normalize shap_values structure to a numpy array for the target class
            if isinstance(shap_values, list):
                shap_list = [np.asarray(sv) for sv in shap_values]
                class_idx = int(prediction) if int(prediction) < len(shap_list) else 0
                shap_vals_class = shap_list[class_idx]
                ev = explainer.expected_value
                ev_arr = np.asarray(ev)
                expected_scalar = float(ev_arr[class_idx]) if ev_arr.size > 1 else float(ev_arr.flatten()[0])
            else:
                shap_vals_class = np.asarray(shap_values)
                ev = explainer.expected_value
                try:
                    expected_scalar = float(np.asarray(ev).flatten()[0])
                except Exception:
                    expected_scalar = float(ev)

            # Ensure shap_vals_class is 2D: (n_samples, n_features)
            if shap_vals_class.ndim == 1:
                shap_vals_class = shap_vals_class.reshape(1, -1)

            # Choose first instance (we only explain the predicted student)
            shap_instance = shap_vals_class[0]
            X_instance = X_arr[0] if X_arr.ndim > 1 else X_arr

            # Force plot expects 1D shap array and 1D feature array for a single instance
            force_html = shap.force_plot(expected_scalar, shap_instance, X_instance, feature_names=list(input_data.columns), matplotlib=False).html()
            st.components.v1.html(force_html, height=350)
        except Exception as force_error:
            st.info("Showing feature contribution summary instead:")
            fig2, ax2 = plt.subplots(figsize=(10, 6))

            # Create a custom force plot alternative - show top positive and negative contributors
            try:
                # Ensure we use plain numpy arrays to avoid multi-dimensional indexing errors
                if shap_vals is None:
                    shap_arr = np.zeros(len(input_data.columns))
                else:
                    shap_arr = np.asarray(shap_vals)

                # Collapse to 2D or 1D numpy arrays explicitly
                shap_arr = np.asarray(shap_arr)
                if shap_arr.ndim == 3:
                    # Some explainers can return shape (classes, samples, features) - pick predicted class or first
                    shap_arr = shap_arr[0]

                if shap_arr.ndim == 2:
                    base = np.abs(shap_arr).mean(axis=0)
                    instance_vals = np.asarray(shap_arr[0])
                else:
                    base = np.abs(shap_arr)
                    instance_vals = np.asarray(shap_arr)

                # Ensure base and instance_vals are 1D numpy arrays
                base = np.asarray(base).ravel()
                instance_vals = np.asarray(instance_vals).ravel()

                sorted_idx = np.argsort(base)[::-1][:10]
                top_features = [input_data.columns[int(i)] for i in sorted_idx]
                top_values = instance_vals[sorted_idx]

                colors = ['green' if float(v) > 0 else 'red' for v in top_values]
                ax2.barh(range(len(top_features)), top_values, color=colors)
                ax2.set_yticks(range(len(top_features)))
                ax2.set_yticklabels(top_features)
                ax2.set_xlabel("SHAP Value (Impact on Prediction)")
                ax2.set_title("Top Feature Contributions (Alternative View)")
                ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig2)
            except Exception as e_inner:
                st.warning("⚠ Unable to render SHAP fallback visualization.")
                st.text(str(e_inner))
            
    except Exception as e:
        st.warning(f"⚠ SHAP explanation is not available for the {selected_model_name} model.")
        st.text(f"Error details: {str(e)}")

# ------------------------------------------------------------
# ------------------ ADMIN / LECTURER PROMPTS ----------------
# ------------------------------------------------------------
elif page == "📚 Admin / Lecturer Prompts":
    st.title("📚 Admin / Lecturer Prompts")
    st.markdown(
        "Click a prompt below to send it to the Academic Assistant for professional insights and recommendations."
    )

    # ------------------ Embedded Academic Assistant Chatbot ------------------
    st.subheader("💬 Academic Assistant")
    st.markdown("Use the assistant below for tailored advice. Quick prompts are provided for convenience.")
    with st.expander("Open Academic Assistant", expanded=True):
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        qp1, qp2, qp3 = st.columns([1,1,1])
        quick_prompt = None
        if qp1.button("Explain this prediction", key="admin_qp_explain"):
            quick_prompt = "Explain this prediction for the current student profile."
        if qp2.button("How can this student improve?", key="admin_qp_improve"):
            quick_prompt = "How can this student improve their academic performance? Provide practical steps."
        if qp3.button("What risks should I watch?", key="admin_qp_risks"):
            quick_prompt = "What risks should instructors or advisors watch for with this student?"

        user_input_chat = st.text_area("Message to assistant", key="admin_chat_input", height=80)
        send = st.button("Send to Assistant", key="admin_send")
        reset = st.button("Reset conversation", key="admin_reset")
        if reset:
            st.session_state["chat_history"] = []
            st.success("Conversation reset.")

        # Decide message to send
        message_to_send = None
        if quick_prompt:
            message_to_send = quick_prompt
        elif send and user_input_chat and user_input_chat.strip():
            message_to_send = user_input_chat.strip()

        if message_to_send:
            st.session_state["chat_history"].append({"role":"user","text":message_to_send})
            context = "You are a helpful academic assistant. Explain predictions, SHAP feature importance, and give advice based on student data."
            used_student_data = False
            if "input_data" in st.session_state:
                used_student_data = True
                try:
                    context += f"\nStudent Data:\n{st.session_state['input_data'].to_dict(orient='records')[0]}"
                except Exception:
                    context += f"\nStudent Data:\n{str(st.session_state.get('input_data'))}"
            if "prediction" in st.session_state and "probability" in st.session_state:
                context += f"\nPredicted Category: {st.session_state['prediction']}, Confidence: {st.session_state['probability']:.2f}"
            try:
                openai.api_key = st.secrets["OPENAI_API_KEY"]
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role":"system","content":context},
                        {"role":"user","content":message_to_send}
                    ],
                    temperature=1,
                    max_completion_tokens=400
                )
                answer = response.choices[0].message.content
                st.session_state["chat_history"].append({"role":"assistant","text":answer,"used_student_data":used_student_data})
            except Exception as e:
                err = f"⚠ Error contacting OpenAI API: {e}"
                st.session_state["chat_history"].append({"role":"assistant","text":err,"used_student_data":used_student_data})

        # Display chat history
        if st.session_state["chat_history"]:
            for entry in st.session_state["chat_history"]:
                if entry.get("role") == "user":
                    st.markdown(f"**You:** {entry.get('text')}")
                else:
                    st.markdown(f"**Assistant:** {entry.get('text')}")
                    if entry.get("used_student_data"):
                        st.caption("Based on student data")
                    else:
                        st.caption("Generic advice")

    prompts = [
        " Summarize this student's academic risk profile and propose possible interventions, support actions, or advising strategies an instructor or academic department could use to help the student succeed.",
        " Provide a professional summary of this student's academic risk level based on the prediction, KPIs, and SHAP values. Recommend specific interventions the academic team should consider.",
        " Generate a formal report for academic advisors summarizing the student’s predicted performance, key risk factors, and personalised recommendations for academic support.",
        " What actions can lecturers take to support this student based on their prediction and SHAP feature influence? Include suggestions for classroom support, follow-up checks, and communication strategies.",
        " Analyze which course-related features contributed most to the student's academic risk and suggest course-level adjustments or follow-ups the lecturer can apply.",
        " Interpret this student’s fee compliance and its impact on predicted performance. Suggest finance-related interventions or communications for the admin office.",
        " Create a structured meeting agenda for an advisor–student meeting based on this prediction and the student’s KPIs. Include discussion points and action items.",
        " Develop a short-term and long-term monitoring plan for this at-risk student, based on SHAP importance and their academic indicators.",
        " Based on the student’s profile, recommend which campus support services (counseling, tutoring, advising, financial aid) they should be referred to, with justification.",
        " Based on the student’s profile, recommend which campus support services (counseling, tutoring, advising, financial aid) they should be referred to, with justification."
    ]

    if "prompt_response" not in st.session_state:
        st.session_state["prompt_response"] = ""

    for idx, prompt in enumerate(prompts):
        col1, col2 = st.columns([7, 1])
        with col1:
            st.markdown(f"**Prompt {idx+1}:** {prompt}")
        with col2:
            if st.button(f"Send", key=f"send_prompt_{idx}"):
                # Send to chatbot
                context = "You are a helpful academic assistant. Explain predictions, SHAP feature importance, and give advice based on student data."
                if "input_data" in st.session_state:
                    context += f"\nStudent Data:\n{st.session_state['input_data'].to_dict(orient='records')[0]}"
                if "prediction" in st.session_state and "probability" in st.session_state:
                    context += f"\nPredicted Category: {st.session_state['prediction']}, Confidence: {st.session_state['probability']:.2f}"
                try:
                    openai.api_key = st.secrets["OPENAI_API_KEY"]
                    response = openai.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": context},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=1,
                        max_completion_tokens=400
                    )
                    st.session_state["prompt_response"] = response.choices[0].message.content
                except Exception as e:
                    st.session_state["prompt_response"] = f"⚠ Error contacting OpenAI API: {e}"

        # Display the response
        if st.session_state["prompt_response"]:
            st.markdown("---")
            st.subheader("💡 Assistant Response")
            st.markdown(st.session_state["prompt_response"])

# ------------------------------------------------------------
# ---------------------- Reviews & Feedback---------------------------
# ------------------------------------------------------------

# ------------------------------------------------------------
# REVIEWS & FEEDBACK PAGE
# ------------------------------------------------------------
if page == "⭐ Reviews & Feedback":
    st.title("⭐ User Reviews & Feedback")
    st.markdown("We value your feedback. Please leave a review after using the application.")

    # ---------- Review Form ----------
    with st.form("review_form"):
        st.subheader("✍️ Submit a Review")

        name = st.text_input("Your Name (optional)")
        role = st.selectbox(
            "Your Role",
            ["Student", "Lecturer", "Administrator", "Researcher", "Other"]
        )
        rating = st.slider("Overall Rating", 1, 5, 4)
        review_text = st.text_area("Your Review", height=120)

        submit_review = st.form_submit_button("📨 Submit Review")

        if submit_review:
            if review_text.strip() == "":
                st.warning("⚠ Please write a short review before submitting.")
            else:
                save_review({
                    "name": name if name else "Anonymous",
                    "role": role,
                    "rating": rating,
                    "review": review_text,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                st.success("✅ Thank you! Your review has been submitted.")

    st.markdown("---")

    # ---------- Display Reviews ----------
    st.subheader("📢 What Users Are Saying")

    reviews_df = load_reviews()

    if reviews_df.empty:
        st.info("No reviews yet. Be the first to leave feedback!")
    else:
        reviews_df = reviews_df.sort_values("timestamp", ascending=False)

        for _, row in reviews_df.iterrows():
            st.markdown(
                f"""
<div style="
    background: rgba(245,245,245,0.8);
    padding: 15px;
    margin-bottom: 12px;
    border-radius: 12px;
    border-left: 5px solid #FFD93D;
">
    <strong>{row['name']}</strong> · {row['role']}
    <br>
    {"⭐" * int(row['rating'])}
    <p style="margin-top:8px;">{row['review']}</p>
    <small style="color:#666;">{row['timestamp']}</small>
</div>
""",
                unsafe_allow_html=True
            )

# ------------------------------------------------------------
# ---------------------- ABOUT PAGE ---------------------------
# ------------------------------------------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")

    st.markdown("""
## 📘 Student Academic Performance Predictor

This web application predicts student academic performance using machine learning. It helps educators, administrators, and students understand academic performance patterns and identify at-risk students early.

---

## ✨ Features

- **🎯 Student Performance Prediction** – Predicts if a student will dropout, remain enrolled, or graduate
- **📊 Dashboard Visualization** – Dynamic KPIs showing student metrics and progress
- **🔥 What Influenced This Result (Feature Importance)?** – See which factors most influenced this prediction
- **🔍 Detailed Explanation (SHAP)** – Deep dive into how each factor pushed the prediction higher or lower
- **📈 Prediction Results** – Clear visualization of prediction outcomes with certainty scores
- **📚 Admin/Lecturer Prompts** – Pre-built prompts for institutional staff to generate insights
- **💬 OpenAI-Powered Chatbot** – Interactive academic assistant for real-time guidance
- **⭐ User Reviews** – Community feedback and ratings

---

## 🚀 How to Use This Application

### Step 1: 🏠 Home (Prediction)
1. Navigate to the **Home** page from the sidebar
2. Enter student demographic information:
   - Age at enrollment
   - Parent occupation codes
   - Admission grade
   - Tuition payment status
   - Course ID
3. Fill in academic performance data:
   - 1st semester units enrolled, evaluated, approved, and grades
   - 2nd semester units enrolled, evaluated, approved, and grades
4. Click **"🔍 Predict Performance"** button
5. View the prediction result and certainty score

### Step 2: 📊 Prediction Results
- Review your prediction in a dedicated results page
- See the predicted category (Dropout, Enrolled, or Graduate)
- Check the certainty score (0-1 scale)

### Step 3: 📈 Dashboard
- View comprehensive KPIs for the student
- See semester grades, curricular progress, and tuition status
- Visualize academic metrics in interactive charts

### Step 4: 🔥 What Influenced This Result?
- Understand which features impact the prediction most
- View a ranked bar chart showing which factors influenced the decision
- Access detailed factor ranking in table format

### Step 5: 🔍 Detailed Explanation (Advanced)
- Get detailed explanation showing how each factor affected the decision
- See visual diagrams showing how each factor pushed the prediction
- Understand exactly why the model predicted this outcome

### Step 6: 📚 Admin / Lecturer Prompts
- Access pre-built prompts for institutional users
- Send prompts to the AI assistant for professional insights
- Generate reports on academic risk, interventions, and recommendations

### Step 7: ⭐ Reviews & Feedback
- Share your experience with the application
- Rate the app on a 1-5 scale
- Read feedback from other users

### Step 8: 💬 Academic Assistant
- Use the chatbot in the sidebar for real-time questions
- Ask questions about predictions, SHAP values, or academic strategies
- Get personalized responses based on student data

---

## 📋 Input Guide

- **Course ID**: Department/program codes (e.g., 33=Agronomy, 171=Design, 8014=Nursing)
- **Tuition Fees**: 1 = Up to date, 0 = Behind on payments
- **Semester Grades**: Scale of 0-20
- **Units Enrolled/Approved**: Number of courses
- **Evaluations**: Number of exams taken

---

## 🛠️ Technology Stack

- **Framework**: Streamlit (web app framework)
- **Language**: Python 3.13
- **ML Libraries**: scikit-learn, XGBoost, LightGBM
- **Explainability**: SHAP
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **AI Assistant**: OpenAI GPT-4

---

## 👨‍💼 Author

Built by **Njinju Zilefac Fogap** as part of a Master's thesis project on predicting student academic performance using machine learning.

---

## 📞 Support

For issues or questions, please use the Academic Assistant chatbot or consult with your institution's technical support team.
""")