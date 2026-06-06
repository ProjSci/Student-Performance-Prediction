import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ==========================
# LOAD FILES
# ==========================

model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

try:
    df = pd.read_csv(
        "online_classroom_data_augmented.csv"
    )
except:
    df = None

# ==========================
# HEADER
# ==========================

st.title("🎓 Student Performance Prediction System")

st.markdown("""
Predict student academic performance using machine learning
based on e-learning activity data.

**Models Used**
- Random Forest
- XGBoost

**Features**
- Student engagement
- Classroom scores
- Learning behavior
""")

st.divider()

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("🎓 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "📊 EDA",
        "⚙️ Feature Engineering",
        "🌡️ Model Performance",
        "🎓 Prediction Demo"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
Machine Learning Final Project

Student Performance Prediction

Algorithms:
- Random Forest
- XGBoost

Includes:
- Feature Engineering
- Early Warning System
- Recommendation System
""")

st.sidebar.info("""
Group 5 - LC01

1. Samuel Christoff
2. Jovin Prasetia Willim
3. Daniel Setiawan
""")

# ==========================
# HOME PAGE
# ==========================

if page == "🏠 Home":

    st.title("🎓 Student Performance Prediction")

    st.markdown("""
    Predict student performance using Machine Learning
    based on e-learning activities.

    ### Models Used
    - Random Forest
    - XGBoost

    ### Features
    - Engagement Score
    - Positive Behavior
    - Negative Behavior
    - Activity Intensity
    - Focus Ratio

    ### Outputs
    - Success Probability
    - Risk Classification
    - Recommendations
    """)

    c1,c2,c3 = st.columns(3)

    c1.metric("Models", "2")
    c2.metric("Engineered Features", "5")
    c3.metric("Prediction", "Success / Failure")

# ==========================
# EDA PAGE
# ==========================

if page == "📊 EDA":

    st.title("📊 Exploratory Data Analysis")

    if df is not None:

        col1,col2,col3,col4 = st.columns(4)

        col1.metric(
            "Students",
            len(df)
        )

        col2.metric(
            "Approved",
            int(df["Approved"].sum())
        )

        col3.metric(
            "Not Approved",
            len(df)-int(df["Approved"].sum())
        )

        col4.metric(
            "Features",
            len(df.columns)-1
        )

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        st.subheader("Approval Distribution")

        fig, ax = plt.subplots()

        df["Approved"].value_counts().plot(
            kind="bar",
            ax=ax
        )

        st.pyplot(fig)

    else:
        st.warning("Dataset not found")

# ==========================
# FEATURE ENGINEERING PAGE
# ==========================

if page == "⚙️ Feature Engineering":

    st.title("⚙️ Feature Engineering")

    st.code("""
engagement_score =
(helpful_post +
nice_code_post +
collaborative_post)/3

positive_behavior =
creative_post +
amazing_post

negative_behavior =
bad_post +
confused_post

activity_intensity =
total_posts +
timeonline

focus_ratio =
helpful_post /
(bad_post+1)
""")

    st.success(
        "Feature engineering improves model performance."
    )

# ==========================
# MODEL PERFORMANCE PAGE
# ==========================

if page == "🌡️ Model Performance":

    st.title("🌡️ Model Performance")

    st.subheader(
        "Confusion Matrix"
    )

    st.image(
        "confusion_matrix.png"
    )

    st.subheader(
        "ROC Curve"
    )

    st.image(
        "roc_curve.png"
    )

    st.subheader("📊 Model Accuracy Comparison")

    comparison_df = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost"],
        "Accuracy": [0.91, 0.94]  # replace with your actual values
    })

    st.dataframe(comparison_df)

    fig, ax = plt.subplots(figsize=(6,4))

    ax.bar(
        comparison_df["Model"],
        comparison_df["Accuracy"]
    )

    ax.set_ylim(0,1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Random Forest vs XGBoost")

    st.pyplot(fig)

# ==========================
# INPUTS
# ==========================

if page == "🎓 Prediction Demo":

    st.subheader("🎯 Example Profiles")

    a,b,c = st.columns(3)

    with a:
        st.info("🔴 High Risk")

    with b:
        st.info("🟡 Medium Risk")

    with c:
        st.info("🟢 Low Risk")

    with st.expander("View Example Inputs"):

        st.code("""
    HIGH RISK

    Helpful Posts: 1
    Code Contributions: 1
    Collaborative Posts: 1
    Creative Posts: 0
    Amazing Posts: 0

    Bad Posts: 4
    Confused Posts: 8

    Total Posts: 5
    Time Online: 1000

    Scores:
    2.0, 2.5, 2.0, 1.8, 2.3


    MEDIUM RISK

    Helpful Posts: 8
    Code Contributions: 7
    Collaborative Posts: 10
    Creative Posts: 6
    Amazing Posts: 5

    Bad Posts: 2
    Confused Posts: 3

    Total Posts: 40
    Time Online: 15000

    Scores:
    6.0, 6.2, 6.5, 6.3, 6.1


    LOW RISK

    Helpful Posts: 25
    Code Contributions: 20
    Collaborative Posts: 20
    Creative Posts: 15
    Amazing Posts: 18

    Bad Posts: 0
    Confused Posts: 1

    Total Posts: 120
    Time Online: 30000

    Scores:
    8.5, 8.8, 9.0, 8.7, 8.9
    """)

    st.header("📊 Student Activity Data")

    col1, col2 = st.columns(2)

    with col1:

        helpful_post = st.slider(
            "Helpful Posts",
            0, 50, 10
        )

        nice_code_post = st.slider(
            "Code Contributions",
            0, 50, 10
        )

        collaborative_post = st.slider(
            "Collaborative Posts",
            0, 50, 10
        )

        creative_post = st.slider(
            "Creative Posts",
            0, 50, 10
        )

        amazing_post = st.slider(
            "Amazing Posts",
            0, 50, 10
        )

    with col2:

        bad_post = st.slider(
            "Bad Posts",
            0, 20, 1
        )

        confused_post = st.slider(
            "Confused Posts",
            0, 20, 1
        )

        total_posts = st.slider(
            "Total Posts",
            0, 200, 50
        )

        timeonline = st.slider(
            "Time Online",
            0, 50000, 20000
        )

    st.header("🏫 Classroom Scores")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        sk1 = st.number_input("Score 1", 0.0, 10.0, 7.0)

    with c2:
        sk2 = st.number_input("Score 2", 0.0, 10.0, 7.0)

    with c3:
        sk3 = st.number_input("Score 3", 0.0, 10.0, 7.0)

    with c4:
        sk4 = st.number_input("Score 4", 0.0, 10.0, 7.0)

    with c5:
        sk5 = st.number_input("Score 5", 0.0, 10.0, 7.0)

    # ==========================
    # RECOMMENDATIONS
    # ==========================

    def recommendation(prob):

        if prob < 0.4:
            return """
            🔴 High Risk

            - Increase participation
            - Spend more time learning online
            - Improve classroom performance
            - Seek academic assistance
            """

        elif prob < 0.7:
            return """
            🟡 Medium Risk

            - Increase engagement
            - Join discussions
            - Improve consistency
            """

        else:
            return """
            🟢 Low Risk

            - Excellent performance
            - Maintain current engagement
            - Continue active participation
            """

    # ==========================
    # PREDICT
    # ==========================

    if st.button("🚀 Predict Performance"):

        user = pd.DataFrame([{

            "helpful_post": helpful_post,
            "nice_code_post": nice_code_post,
            "collaborative_post": collaborative_post,
            "creative_post": creative_post,
            "amazing_post": amazing_post,
            "bad_post": bad_post,
            "confused_post": confused_post,
            "total_posts": total_posts,
            "timeonline": timeonline,
            "sk1_classroom": sk1,
            "sk2_classroom": sk2,
            "sk3_classroom": sk3,
            "sk4_classroom": sk4,
            "sk5_classroom": sk5
        }])

        # ==========================
        # FEATURE ENGINEERING
        # ==========================

        user["engagement_score"] = (
            user["helpful_post"]
            + user["nice_code_post"]
            + user["collaborative_post"]
        ) / 3

        user["positive_behavior"] = (
            user["creative_post"]
            + user["amazing_post"]
        )

        user["negative_behavior"] = (
            user["bad_post"]
            + user["confused_post"]
        )

        user["activity_intensity"] = (
            user["total_posts"]
            + user["timeonline"]
        )

        user["focus_ratio"] = (
            user["helpful_post"] /
            (user["bad_post"] + 1)
        )

        user = user[feature_columns]

        scaled = scaler.transform(user)

        prob = model.predict_proba(scaled)[0][1]

        st.divider()

        st.header("📈 Prediction Result")

        st.metric(
            "Success Probability",
            f"{prob:.2%}"
        )

        if prob < 0.4:
            st.error("⚠️ HIGH RISK")

        elif prob < 0.7:
            st.warning("🟡 MEDIUM RISK")

        else:
            st.success("✅ LOW RISK")

        st.subheader("📋 Recommendation")

        st.write(
            recommendation(prob)
        )