import streamlit as st
import pandas as pd
import time

#from src.pipeline.train_pipeline import TraininingPipeline
from src.pipeline.test_pipeline import PredictionPipeline

# ----------------------------
# ⚙️ Page Config
# ----------------------------
st.set_page_config(
    page_title="Wafer Fault Detection System Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ----------------------------
# 🎨 Custom CSS (IMPORTANT)
# ----------------------------
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
        }
        .card {
            padding: 20px;
            border-radius: 15px;
            background-color: #1c1f26;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# 📌 Sidebar
# ----------------------------
st.sidebar.title("⚡Fault Detection ")
# menu = st.sidebar.radio(
#     "Navigation",
#     ["Dashboard", "Train Model", "Predict"],
#     label_visibility="collapsed"
# )
menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predict"]
)
# ----------------------------
# 🏠 DASHBOARD
# ----------------------------
if menu == "Dashboard":
    st.title("⚡Wafer Fault Detection System")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">📊 <b>Model Status</b><br>Ready</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">📁 <b>Dataset</b><br>Available</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">⚙️ <b>Pipeline</b><br>Idle</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📌 About Project")
    st.write("""
    This system detects faults using machine learning.
    
    - Train model on dataset  
    - Upload new data  
    - Get predictions instantly  
    """)

# ----------------------------
# 🧠 TRAIN MODEL
# ----------------------------
elif menu == "Train Model":
    st.title("🧠 Model Training")

    st.write("Click below to train the model")

    if st.button("🚀 Start Training"):
        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.02)
            progress.progress(i + 1)

        try:
            pipeline = TraininingPipeline()
            pipeline.run_pipeline()
            st.success("✅ Model trained successfully!")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ----------------------------
# 🔍 PREDICTION
# ----------------------------
elif menu == "Predict":
    st.title("🔍 Prediction Center")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        if st.button("⚡ Run Prediction"):
            with st.spinner("Running model..."):
                try:
                    temp_path = "temp.csv"
                    df.to_csv(temp_path, index=False)

                    pipeline = PredictionPipeline(temp_path)
                    result = pipeline.run_pipeline()

                    output_df = pd.read_csv(result.prediction_file_path)

                    st.success("✅ Prediction Complete!")

                    # ----------------------------
                    # 📊 Clean Preview
                    # ----------------------------
                    st.subheader("📊 Result Preview")

                    cols = output_df.columns

                    if len(cols) > 4:
                        preview_df = pd.concat([
                            output_df.iloc[:, :2],
                            pd.DataFrame({"...": ["..."] * len(output_df)}),
                            output_df.iloc[:, -2:]
                        ], axis=1)
                    else:
                        preview_df = output_df

                    st.dataframe(preview_df)

                    # ----------------------------
                    # 📈 Summary
                    # ----------------------------
                    st.subheader("📈 Summary")

                    counts = output_df["quality"].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    col1.metric("Bad Wafers", int(counts.get("bad", 0)))
                    col2.metric("Good Wafers", int(counts.get("good", 0)))
                    
                    st.bar_chart(counts)

                    # ----------------------------
                    # 🔍 Row-wise SHAP View
                    # ----------------------------
                    st.subheader("🔍 Inspect Individual Prediction")

                    idx = st.number_input(
                        "Select row index",
                        0,
                        len(output_df)-1,
                        0
                    )

                    st.write("Prediction:", output_df.iloc[idx]["quality"])
                    st.write("Root Cause:", output_df.iloc[idx]["Root_Cause_Analysis"])

                    # ----------------------------
                    # 📥 Download
                    # ----------------------------
                    with open(result.prediction_file_path, "rb") as f:
                        st.download_button(
                            "📥 Download Full Results",
                            f,
                            file_name="predictions.csv"
                        )

                    # ----------------------------
                    # 📂 Full Data
                    # ----------------------------
                    with st.expander("See full results"):
                        st.dataframe(output_df)

                except Exception as e:
                    st.error(str(e))
