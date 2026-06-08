import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import google.generativeai as genai
from predict_nutrient import predict_nutrient
import os
from dotenv import load_dotenv
# ------------------- 1. PAGE CONFIGURATION -------------------
st.set_page_config(
    page_title="Rose Doctor AI Dashboard",
    page_icon="🌹",
    layout="wide",  # Full desktop width
    initial_sidebar_state="expanded"
)

# ------------------- 2. API SETUP -------------------
# 🔴 PASTE YOUR API KEY HERE 🔴


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") #here you have to paste your API key generated

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ------------------- 3. CUSTOM CSS -------------------
st.markdown("""
    <style>
    /* Clean Desktop Layout */
    .main { background-color: #FAFAFA; }
    h1 { color: #880e4f; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    
    /* Result Cards */
    .diagnosis-container {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 8px solid #880e4f;
        margin-bottom: 25px;
        text-align: center;
    }
    .diagnosis-title { color: #666; font-size: 1.2rem; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px; }
    .diagnosis-result { color: #880e4f; font-size: 3rem; font-weight: 800; margin: 0; }
    
    /* AI Advice Box */
    .advice-box {
        background-color: #f0f7ff;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #cce5ff;
        font-size: 1.1rem; /* Slightly larger for readability */
        line-height: 1.6;
    }
    .stProgress > div > div > div > div { background-color: #880e4f; }
    </style>
""", unsafe_allow_html=True)

# ------------------- 4. SMART MODEL SELECTOR -------------------
def get_ai_advice(pred_class):
    """
    Selects the best model from your available list.
    """
    model_priority = [
        'gemini-2.5-flash',       # Latest & Fastest
        'gemini-2.0-flash',       # Very stable next-gen
        'gemini-1.5-flash',       # Standard fallback
        'gemini-pro'              # Legacy fallback
    ]

    # UPDATED PROMPT: Strictly 2 points
    prompt = f"""
    Act as an expert agronomist. A rose leaf has been diagnosed with **{pred_class} deficiency**.
    
    Provide exactly two solid, concise points in Markdown:
    1. Fertilizers Suggested: (List specific organic and chemical options)
    leave a linespace
    2. How to Prevent it: (Provide one clear prevention strategy)
    
    Do not add introductions or extra fluff. Go straight to the points.
    """

    for model_name in model_priority:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text, model_name
        except Exception:
            continue
            
    return "Could not connect to AI. Please check internet.", "None"

# ------------------- 5. SIDEBAR -------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/rose.png", width=80)
    st.title("Rose Doctor AI")
    st.markdown("---")
    
    if GEMINI_API_KEY:
        st.success("✅ AI System: Online")
    else:
        st.error("❌ AI System: Offline")
        st.caption("Please paste API Key in app.py")
        
    st.markdown("### 📝 Quick Guide")
    st.info("1. **Upload** a leaf photo.\n2. **Wait** for analysis.\n3. **Review** the treatment plan.")
    st.markdown("---")
    st.caption("v2.6.0 | Powered by Gemini 2.5 Flash")

# ------------------- 6. MAIN DASHBOARD -------------------
st.title("🌹 Rose Leaf Nutrient Deficiency Analysis")
st.markdown("#### Precision Agriculture Diagnostic Tool")

uploaded_file = st.file_uploader("Drop your leaf image here (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner("running deep learning model & consulting AI..."):
        pred_class, symptom_activation, scores, cam_img = predict_nutrient(img)

    # ------------------- DESKTOP GRID LAYOUT -------------------
    # Split screen: 45% Image | 55% Analysis
    col_img, col_data = st.columns([0.8, 1], gap="large")

    # --- LEFT PANEL: IMAGES (UPDATED: SIDE-BY-SIDE) ---
    with col_img:
        st.subheader("📷 Leaf Analysis")
        
        # Direct Side-by-Side Layout
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            st.image(img, caption="Original Input", use_container_width=True)
        with sub_c2:
            st.image(cam_img, caption="AI Overlay (Grad-CAM)", use_container_width=True)
            
        st.info("ℹ️ The **AI Overlay** highlights (in red) exactly where the deficiency symptoms were detected.")

    # --- RIGHT PANEL: RESULTS ---
    with col_data:
        st.subheader("📊 Diagnostic Report")
        
        # 1. Main Diagnosis Card
        st.markdown(f"""
            <div class="diagnosis-container">
                <div class="diagnosis-title">Condition Detected</div>
                <div class="diagnosis-result">{pred_class}</div>
            </div>
        """, unsafe_allow_html=True)

        # 2. Confidence Metrics
        st.markdown("**Confidence Scores**")
        class_names = ["Healthy", "Iron", "Magnesium", "Phosphorus"]
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        
        for i, name in enumerate(class_names):
            if i < len(scores):
                c1, c2 = st.columns([1, 4])
                with c1:
                    if name == pred_class: st.markdown(f"**{name}**")
                    else: st.markdown(f"{name}")
                with c2:
                    st.progress(float(norm_scores[i]))

        # ------------------- SYMPTOM TABLE SECTION -------------------
        st.divider()
        st.subheader("🔢 Raw Symptom Data")
        
        df_symptom = pd.DataFrame(
            [symptom_activation], 
            columns=["Feature 1", "Feature 2", "Feature 3", "Feature 4"]
        )
        
        with st.expander("View Symptom Activation Table (Normalized)", expanded=True):
            st.dataframe(
                df_symptom.style.format("{:.4f}").background_gradient(axis=1, cmap="Reds"),
                use_container_width=True
            )
        # -------------------------------------------------------------

        st.divider()

        # 3. Gemini AI Advice (UPDATED: 2 POINTS ONLY)
        st.subheader("💡 Expert Agronomist Advice")
        
        if not GEMINI_API_KEY:
            st.warning("⚠️ API Key missing. Please edit app.py line 19.")
        else:
            if pred_class == "Healthy":
                st.success("✅ **Plant is Healthy!** No treatment needed. Keep up the good work!")
            else:
                with st.spinner("Generating prescription..."):
                    advice, model_used = get_ai_advice(pred_class)
                
                st.markdown(f"""<div class="advice-box">{advice}</div>""", unsafe_allow_html=True)
                st.caption(f"Treatment plan generated by: `{model_used}`")

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown("<center style='color:#888;'>Research Project | Developed by Akash K</center>", unsafe_allow_html=True)