import streamlit as st
import numpy as np
import xgboost as xgb
import mne
import tempfile
import os
import time
import pandas as pd
import plotly.express as px
import gdown
from eeg_features import butter_bandpass_filter, process_emd_cycles, auto_select_active_segment

# 1. Map predictions back to readable labels
REVERSE_MAPPING = {
    0: 'Seizure',
    1: 'LPD (Lateralized Periodic Discharges)',
    2: 'GPD (Generalized Periodic Discharges)',
    3: 'LRDA (Lateralized Rhythmic Delta Activity)',
    4: 'GRDA (Generalized Rhythmic Delta Activity)',
    5: 'Other'
}

# 2. Load the trained model (cached)

@st.cache_resource
def load_model():
    model_path = 'eeg_xgb_model.ubj'
    
    # 如果雲端伺服器上沒有這個檔案，就自動去 Google Drive 下載
    if not os.path.exists(model_path):
        with st.spinner('正在從雲端下載 AI 模型，請稍候...'):
            
            file_id = '1BS_VW8qfabgSy_OsWlTpH6qJN5xHizyH' 
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
            
    # 下載完成後，載入 XGBoost 模型
    model = xgb.Booster()
    model.load_model(model_path)
    return model
model = load_model()

# 3. Build the User Interface
st.title("Clinical EEG Pattern Classifier")
st.write("Upload a standard 19-channels EEG **.edf** file to predict brain activity patterns.")

uploaded_file = st.file_uploader("Upload Medical EEG Data (.edf format)", type=['edf'])

if uploaded_file is not None:
    
    # Define our layout containers immediately
    results_container = st.container()
    st.divider()
    plot_container = st.container()
    
    # Initialize the animated progress bar
    progress_bar = st.progress(0, text="Initializing analysis...")

    # --- PROCESSING PIPELINE (Hidden behind progress bar) ---
    
    # Step 1: Loading & Resampling
    progress_bar.progress(15, text="Loading EDF file...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        raw = mne.io.read_raw_edf(tmp_file_path, preload=True, verbose=False)
        
        expected_channels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'EKG'
        ]
        
        # 1. CLEAN CHANNEL NAMES: Strip common EDF prefixes/suffixes
        mapping = {}
        for ch in raw.ch_names:
            clean_name = ch.replace('EEG ', '').replace('-REF', '').replace('-LE', '').strip()
            if clean_name in expected_channels:
                mapping[ch] = clean_name
                
        if mapping:
            raw.rename_channels(mapping)
            
        # 2. IDENTIFY MISSING CHANNELS
        missing_channels = [ch for ch in expected_channels if ch not in raw.ch_names]
        
        # 3. IMPUTE MISSING CHANNELS (Create flatlines)
        if missing_channels:
            # Alert the user but don't stop the app
            st.warning(f"⚠️ **Missing Channels Imputed:** The file was missing **{', '.join(missing_channels)}**. Dummy flatline channels were created so the model can run, but this may slightly affect prediction accuracy.")
            
            # Create a zero-filled array: shape (number of missing channels, number of timepoints)
            dummy_data = np.zeros((len(missing_channels), raw.n_times))
            
            # Create MNE info for the dummy channels
            dummy_info = mne.create_info(ch_names=missing_channels, 
                                         sfreq=raw.info['sfreq'], 
                                         ch_types='eeg')
            
            # Create a RawArray and add it to the original raw object
            dummy_raw = mne.io.RawArray(dummy_data, dummy_info, verbose=False)
            raw.add_channels([dummy_raw], force_update_info=True)
        
        # 4. FILTER AND REORDER (Now guaranteed to have all 20 channels)
        raw.pick_channels(expected_channels)
        raw.reorder_channels(expected_channels)
        
        original_sfreq = raw.info['sfreq']
        if original_sfreq != 200:
            raw.resample(200.0)
        
        # 5. CONVERT VOLTS TO MICROVOLTS
        raw_data = raw.get_data().T * 1e6 
        
    finally:
        os.remove(tmp_file_path)

    # Step 2: Cropping & Filtering
    progress_bar.progress(40, text="Filtering EEG signals...")
    cropped_data = auto_select_active_segment(raw_data, fs=200, window_sec=50)
    filtered_data = butter_bandpass_filter(cropped_data)

    # Step 3: Feature Extraction
    progress_bar.progress(70, text="Extracting EMD features...")
    features = process_emd_cycles(filtered_data)
    features_2d = features.reshape(1, -1) 
        
    # Step 4: Prediction
    progress_bar.progress(90, text="Running XGBoost prediction model...")
    dmatrix_features = xgb.DMatrix(features_2d)
    probabilities = model.predict(dmatrix_features)[0]
    
    predicted_class_idx = np.argmax(probabilities)
    predicted_label = REVERSE_MAPPING[predicted_class_idx]
    confidence = probabilities[predicted_class_idx] * 100

    # Wrap up the loading animation
    progress_bar.progress(100, text="Analysis Complete!")
    time.sleep(0.5) # Brief pause so the user actually sees it hit 100%
    progress_bar.empty() # Removes the progress bar to clean up the UI


    # --- 1. RENDER RESULTS AT THE TOP ---
    with results_container:
        st.write("### Prediction Results")
        
    # Top-line metrics displayed in a 2x2 matrix
        col1, col2 = st.columns(2)

        
        # Row 1
        with col1:
        
            st.markdown(
            f"""<div style="display: flex; flex-direction: column;">
            <span style="font-size: 14px; color: inherit ">Predicted Primary Pattern</span>
            <span style="font-size: 2.25rem; color: #C85632; font-weight: 600; padding-bottom: 0.25rem;">{predicted_label}</span>
            </div>
            """, 
            unsafe_allow_html=True)
        
  
        with col2:
            st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
            
        st.write("") # Optional: adds a tiny bit of vertical spacing between rows
        
        # Row 2
        col3, col4 = st.columns(2)
        with col3:
            # Example: Hardcoded here, but you could pull this from cropped_data.shape
            st.metric(label="Analyzed Window", value="50 Seconds") 
        with col4:
            # Example: Dynamically check if your imputation logic had to run
            missing_status = f"{len(missing_channels)} Imputed" if missing_channels else "None"
            st.metric(label="Missing Channels", value=missing_status)
            
        st.divider()
        
        # 2. Visual Probability Breakdown
        st.write("#### Probability Distribution")
        
        # Build a DataFrame for the chart
        df_probs = pd.DataFrame({
            "Pattern": [REVERSE_MAPPING[i] for i in range(len(probabilities))],
            "Probability (%)": [p * 100 for p in probabilities]
        })
        
        # Sort ascending so the highest probability sits at the top of the horizontal chart
        df_probs = df_probs.sort_values(by="Probability (%)", ascending=True)
        
        # Create a clean clinical bar chart
        fig = px.bar(
            df_probs, 
            x="Probability (%)", 
            y="Pattern", 
            orientation='h',
            text_auto='.1f', # Displays the exact percentage on the bars
            color="Probability (%)",
            color_continuous_scale="fall" # A professional, clinical color palette
        )
        
        # Tweak the layout to make it look native to Streamlit
        fig.update_layout(
            xaxis_title="Probability (%)",
            yaxis_title=None,
            xaxis_range=[0, 100],
            showlegend=False,
            height=350,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        
        # Render the chart
        st.plotly_chart(fig, use_container_width=True)

# --- 2. RENDER THE PLOTS AT THE BOTTOM ---
    plt.rcParams["figure.figsize"] = (15, 8)
    with plot_container:
        st.write("### Clinical EEG Preview (Raw Data)")
        
        # Plot 1: The Raw Data
        fig_raw = raw.plot(
            duration=10.0,      
            n_channels=min(20, len(raw.ch_names)),     
            scalings='auto',   
            show=False,        
            bgcolor='#f5f5dc', 
            color='gray'       # Set to gray to distinguish from the filtered signal
        )
        st.pyplot(fig_raw)

        st.divider()

        st.write("### Clinical EEG Preview (Filtered: 0.5 - 30 Hz)")
        
        # Plot 2: The Filtered Data
        filtered_mne_data = filtered_data.T
        filtered_raw = mne.io.RawArray(filtered_mne_data, raw.info, verbose=False)
        
        fig_filtered = filtered_raw.plot(
            duration=10.0,      
            n_channels=min(20, filtered_data.shape[1]), 
            scalings='auto',   
            show=False,        
            bgcolor='#f5f5dc', 
            color='darkblue'   
        )
        st.pyplot(fig_filtered)
