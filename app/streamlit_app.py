import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import openai

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
html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* NEOBRUTALISM UI THEME */
section[data-testid="stSidebar"] {
    background-color: #D3D3D3 !important;
    padding: 20px;
}

h1, h2, h3 {
    color: #202020 !important;
    font-weight: 800 !important;
}

.block-container {
    padding-top: 2rem;
}

.stButton>button {
    background: #111 !important;
    color: white !important;
    border-radius: 12px !important;
    border: 3px solid #202020 !important;
    padding: 10px 20px;
    font-weight: 700;
}

.stButton>button:hover {
    background: #FFD93D !important;
    color: #111 !important;
    transition: 0.3s;
}

div[role='radiogroup'] > label {
    background: rgba(255,255,255,0.4);
    padding: 10px 15px;
    border-radius: 10px;
    margin-bottom: 8px;
    font-size: 17px;
    font-weight: 600;

    /* Force same width */
    display: inline-block;
    width: 250px;  /* adjust width as needed */
    text-align: left;
}

div[role='radiogroup'] > label:hover {
    background: rgba(0,0,0,0.15);
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home (Prediction)", "📊 Prediction Results","📈 Dashboard", "🔥 Feature Importance", "🔍 SHAP Explainability","📚 Admin / Lecturer Prompts", "ℹ️ About"]
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

# ------------------------------------------------------------
# ---------------------- HOME PAGE ---------------------------
# ------------------------------------------------------------


help_course = "Degree program code: 33=Agronomy, 171=Design, 8014=Nursing, 9070=Social Service, 9991=Management, 9119=Technologies"
help_prevqual = "Highest previous qualification (encoded)."
help_parents_qual = "Parent education level (encoded)."
help_parents_occ = "Parent job (encoded)."
help_admission_grade = "Score obtained at admission (0–200)."
help_tuition = "1 = Up to date | 0 = Behind on tuition fees"
help_age = "Age at enrollment"
help_units_enrolled_1 = "Number of courses enrolled (1st semester)"
help_units_eval_1 = "Number of evaluations taken (1st semester)"
help_units_approved_1 = "Number of approved units (1st semester)"
help_units_grade_1 = "Average grade (1st semester)"
help_units_enrolled_2 = "Number of courses enrolled (2nd semester)"
help_units_eval_2 = "Number of evaluations taken (2nd semester)"
help_units_approved_2 = "Number of approved units (2nd semester)"
help_units_grade_2 = "Average grade (2nd semester)"

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
        st.subheader("Demographic & Background Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            Age_at_enrollment = st.number_input("Age at Enrollment", 14, 100, 18, help = help_age)
            Mothers_occupation = st.number_input("Mother's Occupation", 0, 100, help = help_parents_occ)
            Fathers_occupation = st.number_input("Father's Occupation", 0, 100, help = help_parents_occ)
        with col2:
            Admission_grade = st.number_input("Admission Grade", 0.0, 200.0, help = help_admission_grade)
            Tuition_fees_up_to_date = st.selectbox("Tuition Fees Up-to-Date?", [0, 1] , help = help_tuition)
            Previous_qualification_grade = st.number_input("Previous Qualification Grade", 0.0, 300.0, help = help_prevqual)
        with col3:
            Course = st.number_input("Course ID", 171, 9999, help = help_course)
            Curricular_units_1st_sem_enrolled = st.number_input("1st Sem Units Enrolled", 0, 40, help = help_units_enrolled_1)
            Curricular_units_1st_sem_approved = st.number_input("1st Sem Units Approved", 0, 40 , help = help_units_approved_1)
        st.markdown("---")
        st.subheader("🎓 Academic Performance Inputs")
        col4, col5 = st.columns(2)
        with col4:
            Curricular_units_1st_sem_evaluations = st.number_input("1st Sem Evaluations", 0, 100, help = help_units_eval_1)
            Curricular_units_1st_sem_grade = st.number_input("1st Sem Grade", 0.0, 20.0, help = help_units_grade_1)
            Curricular_units_2nd_sem_enrolled = st.number_input("2nd Sem Units Enrolled", 0, 40, help = help_units_enrolled_2)
        with col5:
            Curricular_units_2nd_sem_approved = st.number_input("2nd Sem Units Approved", 0, 40, help = help_units_approved_2)
            Curricular_units_2nd_sem_evaluations = st.number_input("2nd Sem Evaluations", 0, 100, help = help_units_eval_2)
            Curricular_units_2st_sem_grade = st.number_input("2nd Sem Grade", 0.0, 20.0, help = help_units_grade_2)
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
# ---------------------- ABOUT PAGE ---------------------------
# ------------------------------------------------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("""
    This web application predicts student academic performance using machine learning.
    
    **Features:**
    - Student performance prediction  
    - Dashboard visualization with dynamic KPIs  
    - Feature importance insight  
    - SHAP model explainability  
    - predicted results display
    - customized prompts for academic staff
    - OpenAI-powered chatbot assistant  

    Built by Njinju Zilefac Fogap using **Streamlit**, **Python**, and **Machine Learning**.
    """)
