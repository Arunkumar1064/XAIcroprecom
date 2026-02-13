import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics.pairwise import euclidean_distances
import time

# --- 1. SETUP ---
st.set_page_config(
    page_title="RPCAU CBSH - Crop Recommender",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Mobile & Aesthetics
st.markdown("""
<style>
    /* Global Background: Deep Night Nature Gradient */
    .stApp {
        background-image: linear-gradient(to right, #000000, #0f2027, #203a43);
        background-attachment: fixed;
        color: #ffffff;
    }
    
    /* Global Text Boldness for Visibility */
    p, label, .stMarkdown, .stText, .stWidgetLabel {
        font-weight: 600 !important;
        color: #e0e0e0 !important;
        font-size: 1.05rem !important;
    }
    
    /* Glassmorphism Cards for Containers */
    div[data-testid="stExpander"], div[data-testid="stMetric"], .stDataFrame, .stForm {
        background-color: rgba(20, 30, 40, 0.85); /* Dark Glass */
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Headers with Shadow */
    h1, h2, h3 {
        color: #00ff88 !important; /* Neon Green */
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        font-weight: 800 !important;
    }
    
    /* Custom Button Style */
    .stButton>button {
        background: linear-gradient(to right, #00b09b, #96c93d); /* Bright Green */
        color: white;
        border: 2px solid #ffffff33;
        border-radius: 12px;
        height: 3.5em; 
        margin-top: 10px; 
        font-weight: 900; 
        font-size: 1.2rem;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(0, 255, 136, 0.6);
    }

    /* Mobile-first adjustments */
    @media (max-width: 768px) {
        h1 { font-size: 1.8rem !important; }
        h2 { font-size: 1.5rem !important; }
        .block-container { padding-top: 3rem !important; padding-left: 1rem !important; padding-right: 1rem !important; }
        .stPlotlyChart { width: 100% !important; }
        p, label { font-size: 1.15rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# --- 2. HEADER ---
st.title("🌾 Intelligent Crop Recommendation System")
st.markdown("### 🎓 **RPCAU - College of Basic Sciences and Humanities (CBSH)**")
st.markdown("---")

# --- 3. DATA LOADING & SAMPLING ---
@st.cache_data
def load_and_prep_data():
    try:
        df = pd.read_csv('crop_remmendation_dataset.csv')
        
        # Sampling (as requested: 500 rows for speed)
        if len(df) > 500:
            df = df.sample(n=500, random_state=42).reset_index(drop=True)
            
        # Feature Selection (Numerical only for simplicity & robustness)
        feature_cols = ['N', 'P', 'K', 'Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 
                       'Electrical_Conductivity', 'Temperature', 'Humidity', 'Rainfall', 
                       'Sunlight_Hours', 'Wind_Speed', 'Altitude', 'Fertilizer_Used']
        
        # Verify columns exist
        available_cols = [c for c in feature_cols if c in df.columns]
        
        X = df[available_cols]
        y = df['Recommended_Crop']
        
        # Encode Target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        return df, X, y_encoded, le, available_cols
    except Exception as e:
        # Fallback for column names if dataset differs
        return None, None, None, None, None

df, X, y, le, feature_cols = load_and_prep_data()

if df is None:
    st.error("Error loading data. Please check 'crop_remmendation_dataset.csv'.")
    st.stop()

# --- 2. TRAIN MODELS (RF & XGBoost) ---
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {}
    scores = {}
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    scores['Random Forest'] = accuracy_score(y_test, rf.predict(X_test))
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=50, max_depth=6, eval_metric='mlogloss', random_state=42)
    xgb.fit(X_train, y_train)
    models['XGBoost'] = xgb
    scores['XGBoost'] = accuracy_score(y_test, xgb.predict(X_test))
    
    return models, scores, X_train, X_test, y_test

models, scores, X_train, X_test, y_test = train_models(X, y)

# --- 3. UI LAYOUT ---
st.title("🌾 XAI Crop Recommender")
st.caption(f"🚀 Optimization: Training on {len(df)} samples for instant deployment.")

# Model Selection
selected_model_name = st.radio("🤖 Select Model", list(models.keys()), horizontal=True)
st.success(f"**Model Accuracy: {scores[selected_model_name]:.1%}** ✅")

model = models[selected_model_name]

# Input Form
with st.form("prediction_form"):
    st.header("📝 Input Soil & Environmental Data")
    col1, col2 = st.columns(2)
    inputs = {}
    
    # Split features into 2 columns
    mid = len(feature_cols) // 2
    
    with col1:
        for feat in feature_cols[:mid]:
            min_val = float(df[feat].min())
            max_val = float(df[feat].max())
            mean_val = float(df[feat].mean())
            inputs[feat] = st.slider(f"{feat}", min_val, max_val, mean_val)
    with col2:
        for feat in feature_cols[mid:]:
            min_val = float(df[feat].min())
            max_val = float(df[feat].max())
            mean_val = float(df[feat].mean())
            inputs[feat] = st.slider(f"{feat}", min_val, max_val, mean_val)
            
    submitted = st.form_submit_button("🔍 Predict Crop Recommendation")

input_df = pd.DataFrame([inputs])

# --- 4. PREDICTION ---
# --- 4. PREDICTION ---
if submitted:
    with st.spinner("Running AI analysis..."):
        # Prediction
        pred = model.predict(input_df)[0]
        pred_prob = model.predict_proba(input_df).max()
        crop_name = le.inverse_transform([pred])[0]
        
        # Save to session state to persist across reruns
        st.session_state['prediction_result'] = {
            'pred': pred,
            'pred_prob': pred_prob,
            'crop_name': crop_name,
            'input_df': input_df
        }

# Display Results if available
if 'prediction_result' in st.session_state:
    result = st.session_state['prediction_result']
    pred = result['pred']
    pred_prob = result['pred_prob']
    crop_name = result['crop_name']
    # Use saved input_df for consistency (though form values persist)
    input_df = result['input_df'] 
    
    st.success(f"### 🌱 Recommended Crop: **{crop_name}**")
    st.write(f"- Confidence: **{pred_prob:.1%}**")
    
    # --- Probability Chart (Top 5) ---
    probs = model.predict_proba(input_df)[0]
    top_5_idx = probs.argsort()[-5:][::-1]
    top_5_probs = probs[top_5_idx]
    top_5_crops = le.inverse_transform(top_5_idx)
    
    fig_prob = px.bar(
        x=top_5_probs, y=top_5_crops, orientation='h', 
        text=[f"{p:.1%}" for p in top_5_probs],
        labels={'x': 'Probability', 'y': 'Crop'},
        title="Top 5 Prediction Probabilities"
    )
    fig_prob.update_layout(yaxis={'categoryorder': 'total ascending'}, dragmode=False)
    st.plotly_chart(fig_prob, use_container_width=True, config={'displayModeBar': True})
    
    # --- Radar Chart (Input vs Average) ---
    st.subheader("📊 Feature Comparison")
    # Get average for predicted crop
    mask = y == pred
    avg_data = X.loc[mask].mean()
    
    # Prepare data for radar chart
    scale_cols = feature_cols
    # Normalize for visualization (0-1 scale relative to max in dataset to keep radar readable)
    # Using simple normalization: value / max_value * 100
    
    categories = list(input_df.columns)
    input_vals = []
    avg_vals = []
    
    # Normalize relative to max in dataset to make chart readable
    max_vals = X.max()
    
    for cat in categories:
        input_vals.append(input_df[cat].iloc[0] / max_vals[cat])
        avg_vals.append(avg_data[cat] / max_vals[cat])
        
    fig_radar = px.line_polar(r=input_vals, theta=categories, line_close=True, title="Input vs Average (Normalized)")
    fig_radar.update_traces(fill='toself', name='Input')
    fig_radar.add_trace(px.line_polar(r=avg_vals, theta=categories, line_close=True).data[0])
    fig_radar.data[1].name = f"Average {crop_name}"
    fig_radar.data[1].line.dash = 'dot'
    fig_radar.update_layout(dragmode=False)
    st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': True})
    st.subheader("🧠 Explainable AI Insights")
    tabs = st.tabs(["⚡ SHAP Analysis", "🔍 LIME Explanation", "🔄 What-if (Counterfactual)"])
    
    # SHAP
    with tabs[0]:
        st.info("SHAP shows how each feature pushed the prediction towards (Red) or away (Blue) from this crop.")
        try:
            # Use standard TreeExplainer (works for RF, XGB)
            explainer = shap.TreeExplainer(model)
            shap_obj = explainer(input_df)
            
            # Check shape: (samples, features, classes) or (samples, features)
            if len(shap_obj.values.shape) == 3:
                # Multi-class: select predicted class
                # explanation for 0th sample, all features, pred class
                explanation = shap_obj[0, :, pred]
            else: 
                # Binary or regression
                explanation = shap_obj[0]
            
            # Plot
            fig, ax = plt.subplots()
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig, bbox_inches='tight')
            
        except Exception as e:
            st.warning(f"SHAP Error: {e}")
            
            # Summary Plot (Global context)
            # st.write("Global Feature Importance:")
            # st.pyplot(shap.summary_plot(shap_values, X_test, plot_type="bar", show=False))
            
        except Exception as e:
            st.warning(f"SHAP visualization limitation: {e}")

    # LIME
    with tabs[1]:
        st.info("LIME explains this specific prediction by approximating the model locally.")
        with st.spinner("⚡ Computing LIME explanation..."):
            try:
                # Create LIME Explainer
                explainer_lime = lime.lime_tabular.LimeTabularExplainer(
                    X_train.values,
                    feature_names=feature_cols,
                    class_names=list(le.classes_), 
                    verbose=False, 
                    mode='classification'
                )
                
                # Get explanation for predicted class (reduced to 3 features for speed)
                exp = explainer_lime.explain_instance(
                    input_df.values[0], 
                    model.predict_proba, 
                    num_features=3, 
                    labels=[pred]
                )
                
                # Plot
                fig = exp.as_pyplot_figure(label=pred)
                st.pyplot(fig, bbox_inches='tight')
                
            except Exception as e:
                st.warning(f"LIME calculation error: {e}")
    
    # Counterfactuals (Nearest Neighbor)
    with tabs[2]:
        st.info("Finding the closest existing crop condition that would yield a different result.")
        
        with st.spinner("⚡ Analyzing alternatives..."):
            # Get 2nd best prediction
            probs = model.predict_proba(input_df)[0]
            sorted_indices = probs.argsort()[::-1]
            second_best_idx = sorted_indices[1] if len(sorted_indices) > 1 else sorted_indices[0]
            
            if second_best_idx == pred:
                st.write("Model is extremely confident. No close alternative found.")
            else:
                target_class = second_best_idx
                target_crop = le.inverse_transform([target_class])[0]
                
                st.write(f"Closest Alternative: **{target_crop}** ({probs[target_class]:.1%})")
                
                # Find nearest neighbor of that class
                mask = y == target_class
                if mask.sum() > 0:
                    X_target = X[mask]
                    dists = euclidean_distances(input_df, X_target)
                    min_idx = dists.argmin()
                    closest_row = X_target.iloc[min_idx]
                    
                    # Show delta
                    delta = closest_row - input_df.iloc[0]
                    changes = delta[abs(delta) > 0.01].sort_values(ascending=False, key=abs)
                    
                    if not changes.empty:
                        st.write("**Changes required:**")
                        for feat, change in changes.head(5).items():
                            arrow = "⬆️ Increase" if change > 0 else "⬇️ Decrease"
                            st.write(f"- **{feat}**: {input_df[feat].iloc[0]:.1f} → {closest_row[feat]:.1f} ({arrow} by {abs(change):.1f})")
                    else:
                        st.write("Conditions are very similar.")
                else:
                    st.write("No sufficient data for alternative crop.")
    
    # --- 6. GLOBAL INSIGHTS ---
    st.divider()
    st.subheader("🌍 Global Model Insights")
    with st.expander("Show Feature Importance (Global Summary)", expanded=False):
        st.markdown("This shows the most important features for crop prediction across all data.")
        
        # Use model's native feature importance (instant, no computation needed)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fi_df = pd.DataFrame({
            'Feature': [feature_cols[i] for i in indices],
            'Importance': importances[indices]
        })
        
        # Create bar chart
        fig_global = px.bar(
            fi_df, 
            x='Importance', 
            y='Feature', 
            orientation='h', 
            title="Global Feature Importance",
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig_global.update_layout(
            yaxis={'categoryorder': 'total ascending'}, 
            dragmode=False,
            height=500
        )
        st.plotly_chart(fig_global, use_container_width=True, config={'displayModeBar': True})

    # --- 7. ADVANCED ANALYTICS ---
    with st.expander("📈 Advanced Analytics (Model Performance & Distributions)"):
        # confusion matrix
        from sklearn.metrics import confusion_matrix
        import plotly.figure_factory as ff

        st.subheader("Model Performance (Confusion Matrix)")
        y_pred_test = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_test)
        
        # Plotly Heatmap
        x_labels = list(le.classes_)
        y_labels = list(le.classes_)
        
        # Invert y for standard matrix view
        # cm = cm[::-1] 
        # y_labels = y_labels[::-1]

        fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=x_labels, y=y_labels, title="Confusion Matrix")
        fig_cm.update_layout(dragmode=False)
        st.plotly_chart(fig_cm, use_container_width=True, config={'displayModeBar': True})
        
        # Feature Importance (Model based)
        st.subheader("🌲 Model Feature Importance")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fi_df = pd.DataFrame({
            'Feature': [feature_cols[i] for i in indices],
            'Importance': importances[indices]
        })
        
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title="Feature Importance (Model Native)")
        fig_fi.update_layout(yaxis={'categoryorder': 'total ascending'}, dragmode=False)
        st.plotly_chart(fig_fi, use_container_width=True, config={'displayModeBar': True})
        
        # Distribution Plot
        st.subheader("📊 Distribution Analysis")
        selected_feat = st.selectbox("Select Feature to Compare", feature_cols)
        
        fig_dist = px.box(df, x='Recommended_Crop', y=selected_feat, color='Recommended_Crop', 
                          title=f"Distribution of {selected_feat} by Crop")
        
        # Add user input line
        user_val = input_df[selected_feat].iloc[0]
        fig_dist.add_hline(y=user_val, line_dash="dash", line_color="red", annotation_text=f"Your Input: {user_val}")
        fig_dist.update_layout(dragmode=False)
        
        st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': True})

# --- FOOTER ---
st.divider()
if st.checkbox("Show Dataset Sample"):
    st.write("Sample of the dataset used for training:")
    st.dataframe(df.head())
