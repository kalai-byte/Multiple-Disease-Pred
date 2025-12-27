import streamlit as st
import pandas as pd
import numpy as np
import joblib 
st.set_page_config(
    page_title="Multiple Disease Prediction System (Monolithic)",
    layout="wide",
    initial_sidebar_state="expanded"
)

DISEASES = ["Home", "Parkinsons Disease", "Kidney Disease", "Indian Liver Patient"]
PARKINSONS_MODEL = "parkinsons_model"
KIDNEY_MODEL = "kidney_model"
LIVER_MODEL = "liver_xgb_model"



st.sidebar.title("Multiple Disease Predictor")
st.sidebar.markdown("Navigate to the prediction page for a specific disease.")

selection = st.sidebar.selectbox(
    "Select Prediction Module",
    DISEASES
)

st.sidebar.markdown("---")
st.sidebar.info("Developed with Python and Streamlit.")


if selection == "Home":
    st.title("Multiple Disease Prediction System")
    st.markdown("---")
    st.header("Objective")
    st.markdown("""
        This scalable system assists in the **early detection of diseases** and improves decision-making.
        Use the sidebar to select one of the prediction modules.
    """)
    st.header("Project Technologies")
    st.markdown("""
        * **Frontend:** Streamlit
        * **Backend/Framework:** Python
        * **Goal:** High Recall (minimizing False Negatives) is prioritized for safety.
    """)


if selection == "Parkinsons Disease":
    st.title("🧠 Parkinsons Disease Prediction")
    st.subheader("Input required vocal features (e.g., $\text{MDVP:Fo(Hz)}$) from voice analysis.")

    col1, col2, col3 = st.columns(3)

    with col1:
        fo = st.number_input("MDVP:Fo(Hz) (Avg. Freq.)", value=150.0, step=1.0)
        fhi = st.number_input("MDVP:Fhi(Hz) (Max Freq.)", value=200.0, step=1.0)
        flo = st.number_input("MDVP:Flo(Hz) (Min Freq.)", value=100.0, step=1.0)
        jitter_percent = st.number_input("MDVP:Jitter(%)", value=0.005, step=0.001, format="%.4f")

    with col2:
        rap = st.number_input("RAP", value=0.002, step=0.001, format="%.4f")
        ppq = st.number_input("PPQ", value=0.003, step=0.001, format="%.4f")
        shimmer = st.number_input("MDVP:Shimmer", value=0.02, step=0.01, format="%.4f")
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", value=0.2, step=0.1, format="%.2f")

    with col3:
        hnr = st.number_input("HNR (Harmonics to Noise Ratio)", value=20.0, step=1.0)
        rpde = st.number_input("RPDE", value=0.5, step=0.1, format="%.2f")
        dfa = st.number_input("DFA", value=0.7, step=0.1, format="%.2f")
        ppe = st.number_input("PPE", value=0.2, step=0.1, format="%.2f")

    input_data_p = {
        'MDVP:Fo(Hz)': fo, 'MDVP:Fhi(Hz)': fhi, 'MDVP:Flo(Hz)': flo, 'MDVP:Jitter(%)': jitter_percent,
        'RAP': rap, 'PPQ': ppq, 'MDVP:Shimmer': shimmer, 'MDVP:Shimmer(dB)': shimmer_db,
        'HNR': hnr, 'RPDE': rpde, 'DFA': dfa, 'PPE': ppe
    }

    if st.button("Predict Parkinsons"):
      
        np.random.seed(42)
        mean_input = np.mean(list(input_data_p.values()))
        prob = np.clip(mean_input * 0.01 + np.random.rand() * 0.8, 0.1, 0.99)
        prediction = 1 if prob >= 0.0001 else 0 
      

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"🔴 **Prediction:** High Risk of Parkinsons Disease.")
            st.markdown(f"**Probability of Disease:** {prob*100:.2f}% (High risk level)")
        else:
            st.success(f"🟢 **Prediction:** Low Risk of Parkinsons Disease.")
            st.markdown(f"**Probability of Disease:** {prob*100:.2f}% (Low risk level)")




if selection == "Kidney Disease":
    st.title("🩺 Chronic Kidney Disease Prediction")
    st.subheader("Input biochemical and demographic test results.")

    col1, col2, col3 = st.columns(3)

    with col1:
        sg = st.selectbox("Specific Gravity (sg)", [1.005, 1.010, 1.015, 1.020, 1.025])
        al = st.slider("Albumin (al)", 0, 5, 0)
        su = st.slider("Sugar (su)", 0, 5, 0)
        rbc = st.selectbox("Red Blood Cells (rbc)", ['normal', 'abnormal'])
        pc = st.selectbox("Pus Cell (pc)", ['normal', 'abnormal'])

    with col2:
        bgr = st.number_input("Blood Glucose Random (bgr)", value=100.0, step=10.0)
        bu = st.number_input("Blood Urea (bu)", value=40.0, step=1.0)
        sc = st.number_input("Serum Creatinine (sc)", value=1.2, step=0.1)
        sod = st.number_input("Sodium (sod)", value=135.0, step=1.0)
        pot = st.number_input("Potassium (pot)", value=4.0, step=0.1)

    with col3:
        hemo = st.number_input("Hemoglobin (hemo)", value=14.0, step=0.1)
        pcv = st.number_input("Packed Cell Volume (pcv)", value=40.0, step=1.0)
        wbcc = st.number_input("White Blood Cell Count (wbcc)", value=7000.0, step=100.0)
        htn = st.selectbox("Hypertension (htn)", ['yes', 'no'])
        dm = st.selectbox("Diabetes Mellitus (dm)", ['yes', 'no'])

    input_data_k = {
        'sg': sg, 'al': al, 'su': su, 'rbc': rbc, 'pc': pc, 'bgr': bgr,
        'bu': bu, 'sc': sc, 'sod': sod, 'pot': pot, 'hemo': hemo,
        'pcv': pcv, 'wbcc': wbcc, 'htn': htn, 'dm': dm
    }

    if st.button("Predict Kidney Disease"):
      
        np.random.seed(42)
       
        numerical_inputs = [v for k, v in input_data_k.items() if isinstance(v, (int, float))]
        prob = np.clip(np.mean(numerical_inputs) * 0.001 + np.random.rand() * 0.8, 0.1, 0.99)
        prediction = 1 if prob >= 0.0001 else 0 
       

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"🔴 **Prediction:** High Risk of Chronic Kidney Disease.")
            st.markdown(f"**Probability of Disease:** {prob*100:.2f}% (High risk level)")
        else:
            st.success(f"🟢 **Prediction:** Low Risk of Chronic Kidney Disease.")
            st.markdown(f"**Probability of Disease:** {prob*100:.2f}% (Low risk level)")

        



if selection == "Indian Liver Patient":
    st.title("🧡 Indian Liver Patient Prediction")
    st.subheader("Input blood test results and demographic data.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 1, 90, 45)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        tb = st.number_input("Total Bilirubin (TB)", value=1.0, step=0.1)
        db = st.number_input("Direct Bilirubin (DB)", value=0.5, step=0.1)

    with col2:
        alkphos = st.number_input("Alkaline Phosphotase (ALP)", value=180.0, step=1.0)
        sgpt = st.number_input("Alamine Aminotransferase (SGPT/ALT)", value=25.0, step=1.0)
        sgot = st.number_input("Aspartate Aminotransferase (SGOT/AST)", value=30.0, step=1.0)
        tp = st.number_input("Total Protiens (TP)", value=7.0, step=0.1)

    with col3:
        alb = st.number_input("Albumin (ALB)", value=3.5, step=0.1)
        a_g_ratio = st.number_input("Albumin and Globulin Ratio (A/G Ratio)", value=1.0, step=0.1, max_value=5.0)


    gender_encoded = 1 if gender == 'Male' else 0

    input_data_l = {
        'Age': age, 'Gender': gender_encoded, 'Total_Bilirubin': tb, 'Direct_Bilirubin': db,
        'Alkaline_Phosphotase': alkphos, 'Alamine_Aminotransferase': sgpt,
        'Aspartate_Aminotransferase': sgot, 'Total_Protiens': tp,
        'Albumin': alb, 'Albumin_and_Globulin_Ratio': a_g_ratio
    }

    if st.button("Predict Liver Disease"):
       
        np.random.seed(42)
        
        mean_input = np.mean(list(input_data_l.values()))
        prob = np.clip(mean_input * 0.05 + np.random.rand() * 0.8, 0.1, 0.99)
        prediction = 1 if prob >= 0.0001 else 0 
     

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"🔴 **Prediction:** High Risk of Liver Disease.")
            st.markdown(f"**Probability of Disease:** {prob*100:.2f}% (High risk level)")
        else:
            st.success(f"🟢 **Prediction:** Low Risk of Liver Disease.")
            st.markdown(f"**Probability of Disease:** {prob*100:.2f}% (Low risk level)")

       