
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
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home (Prediction)", "📊 Prediction Results","📈 Dashboard", "🔥 Feature Importance", "🔍 SHAP Explainability","📚 Admin / Lecturer Prompts","⭐ Reviews & Feedback", "ℹ️ About"]
)

# ------------------------------------------------------------
# Sidebar OpenAI Chatbot
# ------------------------------------------------------------
st.sidebar.header("💬 Academic Assistant")
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.sidebar.text_input("Ask the assistant:", "")
if user_input:
    st.session_state["chat_history"].append(f"Student: {user_input}")

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
                {"role": "user", "content": user_input}
            ],
            temperature=1,
            max_completion_tokens=300
        )
        answer = response.choices[0].message.content
        st.session_state["chat_history"].append(f"Assistant: {answer}")
    except Exception as e:
        answer = f"⚠ Error contacting OpenAI API: {e}"
        st.session_state["chat_history"].append(f"Assistant: {answer}")

for chat in st.session_state["chat_history"]:
    if chat.startswith("Student:"):
        st.sidebar.markdown(f"**{chat}**")
    else:
        st.sidebar.markdown(chat)

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
# Global variables
# ------------------------------------------------------------
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
        with st.expander("🧍 Step 1: Demographics & Background", expanded=True, help=why_demographic_background):
            col1, col2, col3 = st.columns(3)

            with col1:
                Age_at_enrollment = st.number_input("Age at Enrollment", 14, 100, 18, help=help_age)
                mother_label = st.selectbox("Mother's Education Level", PARENT_OCCUPATION_MAP.keys(), help=help_parents_qual)
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
                Previous_qualification_grade = st.number_input("Previous Qualification Grade", 0.0, 300.0, help=help_prevqual)

        # ===============================
        # STEP 2: SEMESTER 1
        # ===============================
        with st.expander("📘 Step 2: Academic Performance – Semester 1", help=why_academic_performance):
            col4, col5 = st.columns(2)

            with col4:
                Curricular_units_1st_sem_enrolled = st.number_input("Units Enrolled", 0, 40, help=help_units_enrolled_1)
                Curricular_units_1st_sem_evaluations = st.number_input("Evaluations", 0, 100, help=help_units_eval_1)

            with col5:
                Curricular_units_1st_sem_approved = st.number_input("Units Approved", 0, 40, help=help_units_approved_1)
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
                Curricular_units_2nd_sem_enrolled = st.number_input("Units Enrolled ", 0, 40, help=help_units_enrolled_2)
                Curricular_units_2nd_sem_evaluations = st.number_input("Evaluations ", 0, 100, help=help_units_eval_2)

            with col7:
                Curricular_units_2nd_sem_approved = st.number_input("Units Approved ", 0, 40, help=help_units_approved_2)
                Curricular_units_2st_sem_grade = st.number_input("Average Grade ", 0.0, 20.0, help=help_units_grade_2)

            if Curricular_units_2nd_sem_approved > Curricular_units_2nd_sem_enrolled:
                st.warning("⚠ Approved units cannot exceed enrolled units.")

            if Curricular_units_2st_sem_grade == 0 and Curricular_units_2nd_sem_approved > 0:
                st.warning("⚠ Grade is 0 but units are approved. Please verify.")
                
        submitted = st.form_submit_button("🔍 Predict Performance", use_container_width=True)
    if submitted:
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
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data).max()
        label_map = {0: "Dropout 🚫🎓", 1: "Enrolled 📚🎓", 2: "Graduate 🎓✨"}
        prediction_label = label_map.get(prediction, "Unknown")
        st.markdown("---")
        st.header("📊 Prediction Results")
        colA, colB = st.columns(2)
        with colA:
            st.metric("Predicted Category", prediction_label)
        with colB:
            st.metric("Confidence Score", f"{probability:.2f}")
        st.success(f"🎯 The student is predicted to **{prediction_label}** with a confidence of **{probability:.2f}**.")
        st.session_state["input_data"] = input_data
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
    prediction_label = st.session_state.get("prediction", "Unknown")
    probability = st.session_state.get("probability", 0)

    col_kpi1.metric("Total Units Enrolled", total_units)
    col_kpi2.metric("Average Semester Grade", f"{avg_grade:.2f}/20")
    col_kpi3.metric("Tuition Fees Up-to-Date", "✔ Yes" if fees_status==1 else "❌ No")
    col_kpi4.metric("Predicted Category", prediction_label)

    # ---------------------- Charts ----------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Semester Grades")
        st.bar_chart(pd.DataFrame({
            "Semester": ["1st", "2nd"],
            "Grade": [input_data["Curricular_units_1st_sem_grade"][0], input_data["Curricular_units_2st_sem_grade"][0]]
        }).set_index("Semester"))
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
    with col3:
        st.subheader("Tuition Status")
        if fees_status==1:
            st.success("Fees are up-to-date ✔")
        else:
            st.error("Fees NOT up-to-date ❌")

# ------------------------------------------------------------
# ---------------------- FEATURE IMPORTANCE -------------------
# ------------------------------------------------------------
elif page == "🔥 Feature Importance":
    st.title("🔥 Model Feature Importance")

    if "input_data" not in st.session_state:
        st.warning("⚠ Please make a prediction first on the Home page.")
        st.stop()

    try:
        final_model = model[-1]
        importances = final_model.feature_importances_

        fi_df = pd.DataFrame({
            "Feature": st.session_state["input_data"].columns,
            "Importance": importances
        }).sort_values("Importance", ascending=True)  # ascending for horizontal bars

        st.subheader("📌 Ranked Feature Importance")
        st.write("The chart below displays the contribution of each feature to the model's decision.")

        # ----- Nicely formatted horizontal bar chart -----
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(fi_df["Feature"], fi_df["Importance"])
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance Ranking")
        plt.tight_layout()

        st.pyplot(fig)

        # ----- Display table too -----
        st.subheader("📋 Feature Importance Table (Sorted)")
        st.dataframe(fi_df[::-1].reset_index(drop=True))  # highest first

    except Exception as e:
        st.warning("⚠ Feature importance is not available for this model.")
        st.text(str(e))

# ------------------------------------------------------------
# ---------------------- SHAP EXPLAINABILITY -----------------
# ------------------------------------------------------------
elif page == "🔍 SHAP Explainability":
    st.title("🔍 SHAP Explainability")
    if "input_data" not in st.session_state:
        st.warning("⚠ Please make a prediction first on the Home page.")
        st.stop()

    input_data = st.session_state["input_data"]
    prediction = st.session_state.get("prediction", 0)
    try:
        preprocessor = model[:-1]
        final_model = model[-1]
        X_transformed = preprocessor.transform(input_data)
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_transformed)

        st.subheader("Top Feature Contributions")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values[prediction], X_transformed, feature_names=input_data.columns, plot_type="bar", show=False)
        st.pyplot(fig)

        st.subheader("Detailed SHAP Force Plot")
        force_html = shap.force_plot(explainer.expected_value[prediction], shap_values[prediction], X_transformed, feature_names=input_data.columns, matplotlib=False).html()
        st.components.v1.html(force_html, height=350)
    except Exception:
        st.warning("⚠ SHAP explanation is not available for this model.")
    
# ------------------------------------------------------------
# ------------------ ADMIN / LECTURER PROMPTS ----------------
# ------------------------------------------------------------
elif page == "📚 Admin / Lecturer Prompts":
    st.title("📚 Admin / Lecturer Prompts")
    st.markdown(
        "Click a prompt below to send it to the Academic Assistant for professional insights and recommendations."
    )

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
                    ⭐{"⭐" * int(row['rating'])}
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
    - **🔥 Feature Importance** – Understand which factors influence predictions most
    - **🔍 SHAP Explainability** – Detailed model interpretability with SHAP force plots
    - **📈 Prediction Results** – Clear visualization of prediction outcomes with confidence scores
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
    5. View the prediction result and confidence score
    
    ### Step 2: 📊 Prediction Results
    - Review your prediction in a dedicated results page
    - See the predicted category (Dropout, Enrolled, or Graduate)
    - Check the confidence score (0-1 scale)
    
    ### Step 3: 📈 Dashboard
    - View comprehensive KPIs for the student
    - See semester grades, curricular progress, and tuition status
    - Visualize academic metrics in interactive charts
    
    ### Step 4: 🔥 Feature Importance
    - Understand which features impact the prediction most
    - View a ranked bar chart of feature contributions
    - Access detailed importance scores in table format
    
    ### Step 5: 🔍 SHAP Explainability
    - Get detailed model interpretation with SHAP values
    - View force plots showing individual feature contributions
    - Understand why the model made a specific prediction
    
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