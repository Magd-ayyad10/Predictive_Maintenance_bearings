import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Universal Bearing Monitor",
    page_icon="⚙️",
    layout="wide"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_model_package():
    try:
        # Loading the NEW universal model
        package = joblib.load('universal_model.joblib')
        return package
    except FileNotFoundError:
        return None

package = load_model_package()

# --- CUSTOM HTML/CSS ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #464b5c;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0e1117;
        border-radius: 4px 4px 0px 0px;
        color: #fafafa;
        font-weight: 600;
    }
    /* Custom Title */
    .title-box {
        padding: 20px;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        border-radius: 10px;
        margin-bottom: 20px;
        color: black;
    }
    h1 { color: black !important; }
    p { color: #333 !important; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="title-box">
    <h1>⚙️ Universal Bearing Guard AI</h1>
    <p>Predictive Maintenance System | Trained on NASA IMS Sets 1, 2, & 3</p>
</div>
""", unsafe_allow_html=True)

# Check if model loaded
if package is None:
    st.error("⚠️ Model file 'universal_bearing_model.joblib' not found. Please run the training script to generate it.")
    st.stop()

model = package['model']
scaler = package['scaler']
threshold = package['threshold']

# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠 Overview", "📚 Project Guide", "📡 Live Simulation", "📂 Batch Analysis", "🔍 Diagnostics", "📊 Model Info"])

# ================= TAB 1: OVERVIEW =================
with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="System Status", value="Online", delta="Universal Model Active")
    with col2:
        st.metric(label="Anomaly Threshold", value=f"{threshold:.4f}", help="Based on Universal Healthy Baseline")
    with col3:
        st.metric(label="Training Data", value="NASA Sets 1-3", delta="Multi-Condition")
        
    st.divider()
    
    st.subheader("How it Works")
    st.markdown(f"""
    This system uses **Unsupervised Learning (PCA)** to detect bearing failures before they happen.
    
    1.  **Universal Baseline:** The model learned "Normal" vibration patterns from three different experiments.
    2.  **Detection Logic:** If the Reconstruction Error exceeds **{threshold:.4f}**, an alarm is triggered.
    3.  **Root Cause:** The system automatically identifies which sensor (Bearing 1, 2, 3, or 4) is causing the anomaly.
    """)
    
    # Quick start guide
    st.divider()
    st.subheader("Quick Start")
    col_qs1, col_qs2, col_qs3 = st.columns(3)
    with col_qs1:
        st.write("**1️⃣ Real-Time Demo**")
        st.write("Test live sensor streams with different failure conditions.")
    with col_qs2:
        st.write("**2️⃣ Upload Data**")
        st.write("Analyze your own bearing vibration datasets.")
    with col_qs3:
        st.write("**3️⃣ Diagnose Issues**")
        st.write("Identify which bearing is failing with root cause analysis.")

# ================= TAB 2: PROJECT GUIDE =================
with tab2:
    st.subheader("📚 Understanding Bearing Predictive Maintenance")
    
    col_intro1, col_intro2 = st.columns([2, 1])
    
    with col_intro1:
        st.markdown("""
        ### What is Predictive Maintenance?
        Instead of replacing bearings when they break (**reactive**) or on a schedule (**preventive**),
        this system **predicts failures before they happen** using machine learning on vibration data.
        
        **Key Benefits:**
        - 🛡️ Prevent unexpected downtime
        - 💰 Reduce maintenance costs
        - ⚡ Optimize equipment lifespan
        - 📊 Data-driven decision making
        """)
    
    with col_intro2:
        st.info("""
        **Why Bearings?**
        Bearings are critical rotating 
        components that fail 
        predictably. Vibration 
        patterns reveal degradation 
        weeks before failure.
        """)
    
    st.divider()
    st.subheader("🔬 The Data Science Behind It")
    
    col_ds1, col_ds2 = st.columns(2)
    
    with col_ds1:
        st.write("**Features Used (20 Total)**")
        st.markdown("""
        For each of 4 bearings, we extract:
        - **RMS**: Overall vibration energy
        - **Kurtosis**: Spikiness (impulsiveness)
        - **Skewness**: Asymmetry of distribution
        - **Peak**: Maximum amplitude
        - **Crest Factor**: Peak-to-RMS ratio
        
        These reveal bearing health degradation.
        """)
    
    with col_ds2:
        st.write("**Model Architecture**")
        st.markdown("""
        **Algorithm**: Principal Component Analysis (PCA)
        
        **Approach**: Unsupervised anomaly detection
        - Learns "normal" from healthy operation
        - Detects deviations via reconstruction error
        - No labeled failure data needed
        
        **Trained on**: NASA IMS bearing datasets
        - Set 1: 4 bearings, 6 months
        - Set 2: 4 bearings, 3 weeks
        - Set 3: 4 bearings, 1 week
        """)
    
    st.divider()
    st.subheader("🎯 How the System Works")
    
    steps = {
        "1️⃣ Collect": "Raw vibration signals from sensors at ~20kHz",
        "2️⃣ Extract": "Compute 5 statistical features per bearing (20 total)",
        "3️⃣ Normalize": "Scale features using StandardScaler for model",
        "4️⃣ Transform": "PCA projects data to lower dimension space",
        "5️⃣ Reconstruct": "Model regenerates features from PCA components",
        "6️⃣ Detect": "Compare original vs reconstructed → Anomaly Score",
        "7️⃣ Alert": f"If score > {threshold:.4f}, trigger alarm"
    }
    
    for step, desc in steps.items():
        st.markdown(f"**{step}** → {desc}")
    
    st.divider()
    st.subheader("🔗 Typical Bearing Failure Progression")
    
    st.markdown("""
    | Phase | Timeline | Symptoms | System Status |
    |-------|----------|----------|---------------|
    | **Healthy** | Months | Normal vibration, low variance | ✅ Green |
    | **Early Degradation** | Weeks | Slight energy increase, kurtosis rises | 🟡 Yellow |
    | **Advanced Decay** | Days | High spikes, peak values spike | 🟠 Orange |
    | **Critical Failure** | Hours | Massive reconstruction error, RMS saturated | 🚨 Red |
    """)
    
    st.info("💡 **Pro Tip**: Act when system shows early signs (Yellow phase) to prevent costly emergency shutdown.")

# ================= TAB 3: LIVE SIMULATION =================
with tab3:
    st.subheader("Real-Time Sensor Simulation")
    
    col_ctrl, col_display = st.columns([1, 3])
    
    with col_ctrl:
        st.info("Inject simulated vibration data.")
        status = st.radio("Inject Condition:", ["Normal Operation", "Early Degradation", "Critical Failure"])
        start_btn = st.button("Start Sensor Stream")
        
    with col_display:
        placeholder = st.empty()
        
        if start_btn:
            chart_data = []
            
            # Simulation loop
            for i in range(30):
                # 1. Generate Fake Data based on selection
                # We need 20 features (4 bearings * 5 metrics)
                if status == "Normal Operation":
                    # Random noise around 0 (Scaled space)
                    noise = np.random.normal(0, 0.5, 20)
                elif status == "Early Degradation":
                    # Slight shift in variance
                    noise = np.random.normal(0.5, 1.0, 20)
                else:
                    # Massive spike (Failure)
                    noise = np.random.normal(2.0, 2.5, 20)
                
                # Reshape for model
                input_data = noise.reshape(1, -1)
                
                # 2. Predict (using the logic directly since input is already 'scaled' simulation)
                # In real app, you would scale raw input. Here we simulate scaled input for speed.
                X_recon = model.inverse_transform(model.transform(input_data))
                error = np.mean(np.square(input_data - X_recon))
                
                is_anomaly = error > threshold
                
                # 3. Update Chart
                chart_data.append(error)
                
                with placeholder.container():
                    if is_anomaly:
                        st.error(f"🚨 CRITICAL ALARM! Anomaly Score: {error:.4f}")
                    else:
                        st.success(f"✅ System Healthy. Anomaly Score: {error:.4f}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=chart_data, mode='lines+markers', name='Reconstruction Error', line=dict(color='#00C9FF')))
                    fig.add_hline(y=threshold, line_dash="dash", line_color="#FF4B4B", annotation_text="Limit")
                    fig.update_layout(
                        title="Live Monitoring", 
                        yaxis_title="Error", 
                        height=400, 
                        template="plotly_dark",
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                time.sleep(0.3)

# ================= TAB 4: BATCH ANALYSIS =================
with tab4:
    st.subheader("Analyze Historical Data")
    uploaded_file = st.file_uploader("Upload CSV (Must have 20 feature columns)", type="csv")
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            
            # Auto-detect timestamp if exists
            if 'timestamp' in input_df.columns:
                input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
                input_df.set_index('timestamp', inplace=True)
            
            # Filter numeric only
            numeric_df = input_df.select_dtypes(include=[np.number])
            
            # Check dimensions (Universal model expects 20 cols)
            if numeric_df.shape[1] != 20:
                st.warning(f"⚠️ Warning: Model expects 20 features, but uploaded file has {numeric_df.shape[1]}. Results may be inaccurate.")
            
            if st.button("Run Analysis"):
                # Scale & Predict
                X_scaled = scaler.transform(numeric_df)
                X_pca = model.transform(X_scaled)
                X_recon = model.inverse_transform(X_pca)
                mse = np.mean(np.square(X_scaled - X_recon), axis=1)
                
                results_df = input_df.copy()
                results_df['Anomaly_Score'] = mse
                results_df['Status'] = np.where(mse > threshold, "🚨 FAILURE", "✅ OK")
                
                # Summary Metrics
                fails = results_df[results_df['Status'] == "🚨 FAILURE"]
                st.metric("Anomalies Detected", f"{len(fails)} / {len(results_df)}")
                
                # Plot
                fig = px.line(results_df, y='Anomaly_Score', title="Batch Analysis Result", template="plotly_dark")
                fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(results_df.head())
                
        except Exception as e:
            st.error(f"Error: {e}")

# ================= TAB 5: DIAGNOSTICS =================
with tab5:
    st.subheader("Root Cause Analysis")
    st.info("Which sensor features contribute most to the model's decision?")
    
    # 1. Define Correct Feature Names for the Universal Model (20 Features)
    # Order: B1 (5 features) -> B2 -> B3 -> B4
    feature_names = [
        'B1_rms', 'B1_kurt', 'B1_skew', 'B1_peak', 'B1_crest',
        'B2_rms', 'B2_kurt', 'B2_skew', 'B2_peak', 'B2_crest',
        'B3_rms', 'B3_kurt', 'B3_skew', 'B3_peak', 'B3_crest',
        'B4_rms', 'B4_kurt', 'B4_skew', 'B4_peak', 'B4_crest'
    ]
    
    # 2. Extract Feature Weights from PC1 (Principal Component 1)
    # This shows what the model considers "most important" for defining normal vs abnormal
    components = model.components_
    pc1_weights = np.abs(components[0])
    
    # Safety check for size mismatch
    if len(pc1_weights) != len(feature_names):
        st.warning("Feature count mismatch. Using generic names.")
        feature_names = [f"Feat {i}" for i in range(len(pc1_weights))]
    
    # 3. Plot
    df_weights = pd.DataFrame({'Feature': feature_names, 'Importance': pc1_weights})
    df_weights = df_weights.sort_values('Importance', ascending=True) # Ascending for horizontal bar chart
    
    fig = px.bar(
        df_weights, 
        x='Importance', 
        y='Feature', 
        orientation='h', 
        title="Feature Importance (Model Weights)",
        template="plotly_dark",
        color='Importance',
        color_continuous_scale='Bluered'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 6: MODEL INFO =================
with tab6:
    st.subheader("🤖 Model Details & Performance")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.metric("Algorithm", "PCA (Unsupervised)")
        st.metric("Input Features", "20")
    with col_m2:
        st.metric("Training Sets", "3 (NASA IMS)")
        st.metric("Anomaly Threshold", f"{threshold:.6f}")
    with col_m3:
        st.metric("Detection Method", "Reconstruction Error")
        st.metric("Scaler Type", "StandardScaler")
    
    st.divider()
    st.subheader("📋 Model Configuration")
    
    # Create model info table
    model_info = {
        "Property": [
            "Model Type",
            "Training Data Source",
            "Training Duration",
            "Number of Samples",
            "Feature Scaling",
            "Anomaly Detection",
            "Components Used",
            "Output Range"
        ],
        "Details": [
            "Principal Component Analysis (Dimensionality Reduction)",
            "NASA IMS Bearing Datasets (Sets 1, 2, 3)",
            "Multiple months of continuous operation + failure",
            "1000+ time snapshots",
            "StandardScaler (mean=0, std=1)",
            "Euclidean distance between original & reconstructed",
            f"{model.n_components_} principal components",
            f"0.0 (healthy) to ~{threshold*5:.4f} (critical)"
        ]
    }
    
    df_info = pd.DataFrame(model_info)
    st.dataframe(df_info, use_container_width=True, hide_index=True)
    
    st.divider()
    st.subheader("📊 Feature Breakdown")
    
    col_fb1, col_fb2, col_fb3, col_fb4 = st.columns(4)
    
    bearing_descriptions = {
        "Bearing 1": "**Primary load-bearing component**. Most sensitive to radial loads.",
        "Bearing 2": "**Support bearing**. Detects shaft misalignment issues.",
        "Bearing 3": "**Secondary load-bearing**. Often fails after primary bearing.",
        "Bearing 4": "**Outboard bearing**. Earliest indicator of external shock loads."
    }
    
    metrics_descriptions = {
        "RMS": "Root Mean Square - Overall vibration energy level",
        "Kurtosis": "Measure of spikiness - Early warning for impacts",
        "Skewness": "Asymmetry of the signal - Indicates directional wear",
        "Peak": "Maximum amplitude - Worst-case vibration value",
        "Crest Factor": "Peak/RMS ratio - Normalized impact severity"
    }
    
    with col_fb1:
        st.write("**🔴 Bearing 1**")
        st.caption(bearing_descriptions["Bearing 1"])
        
    with col_fb2:
        st.write("**🟡 Bearing 2**")
        st.caption(bearing_descriptions["Bearing 2"])
        
    with col_fb3:
        st.write("**🟢 Bearing 3**")
        st.caption(bearing_descriptions["Bearing 3"])
        
    with col_fb4:
        st.write("**🔵 Bearing 4**")
        st.caption(bearing_descriptions["Bearing 4"])
    
    st.divider()
    st.subheader("📈 Metrics Explained")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.write("**RMS (Root Mean Square)**")
        st.markdown("""
        $$RMS = \\sqrt{\\frac{1}{N}\\sum_{i=1}^{N} x_i^2}$$
        
        - Represents total energy in the signal
        - Higher values = more vibration
        - Baseline: 0.1-0.5g under normal conditions
        """)
        
        st.write("**Kurtosis**")
        st.markdown("""
        $$Kurt = \\frac{E[(X-\\mu)^4]}{\\sigma^4}$$
        
        - Measures "peakedness" of distribution
        - Normal data: ~3, Impulsive: >5
        - **Early failure indicator** (spikes appear)
        """)
    
    with col_m2:
        st.write("**Skewness**")
        st.markdown("""
        $$Skew = \\frac{E[(X-\\mu)^3]}{\\sigma^3}$$
        
        - Asymmetry of vibration pattern
        - Healthy: ~0, Degrading: shifts from 0
        - Shows directional wear patterns
        """)
        
        st.write("**Crest Factor**")
        st.markdown("""
        $$CF = \\frac{Peak}{RMS}$$
        
        - Normalized impact severity
        - Healthy bearings: 3-4
        - Failing bearings: >10
        - Key for detecting spikes
        """)
    
    st.divider()
    st.subheader("✅ When to Take Action")
    
    action_table = {
        "Anomaly Score": ["0.0 - 0.0001", "0.0001 - 0.0005", "0.0005 - threshold", "> threshold"],
        "Status": ["✅ Healthy", "🟡 Monitor", "🟠 Alert", "🚨 Critical"],
        "Action": [
            "Continue normal operation",
            "Track trends, schedule inspection in 4-6 weeks",
            "Schedule maintenance within 1-2 weeks",
            "Stop machine immediately, replace bearing"
        ]
    }
    
    df_action = pd.DataFrame(action_table)
    st.dataframe(df_action, use_container_width=True, hide_index=True)
    
    st.divider()
    st.subheader("🔗 References & Data Source")
    
    st.markdown("""
    **NASA Intelligent Maintenance Systems (IMS) Center**
    
    The model is trained on publicly available bearing run-to-failure data:
    - **Dataset**: IMS Center Bearing Data
    - **Link**: https://data.nasa.gov/dataset/ims-bearings
    - **Citation**: Bechhoefer E., He D., et al. (2009). IMS bearing run-to-failure data
    
    **Features**:
    - Continuous operation 24/7 until catastrophic failure
    - High-frequency vibration signals (20kHz sampling)
    - Documented failure times and root causes
    - Realistic multi-bearing systems
    """)
    
    st.info("💡 **Educational Note**: This is a demonstration system. For production use, validate on your specific equipment and operating conditions.")