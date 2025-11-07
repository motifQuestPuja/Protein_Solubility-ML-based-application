import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# --------------------------
# Feature extraction functions
# --------------------------
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
hydro_scores = {
    'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,
    'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,
    'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3
}
aa_weights = {
    'A':89.1,'C':121.2,'D':133.1,'E':147.1,'F':165.2,'G':75.1,'H':155.2,'I':131.2,
    'K':146.2,'L':131.2,'M':149.2,'N':132.1,'P':115.1,'Q':146.2,'R':174.2,'S':105.1,
    'T':119.1,'V':117.1,'W':204.2,'Y':181.2
}

def extract_features(seq):
    seq = seq.upper().replace(" ", "")
    length = len(seq)
    if length == 0:
        return None
    mw = sum(aa_weights.get(a,0) for a in seq)
    aromaticity = sum(seq.count(a) for a in "FWY") / length
    hydrophobicity = np.mean([hydro_scores.get(a,0) for a in seq])
    charge = (seq.count("K")+seq.count("R") - seq.count("D")-seq.count("E")) / length
    features = {
        'length': length,
        'molecular_weight': mw,
        'aromaticity': aromaticity,
        'hydrophobicity': hydrophobicity,
        'charge': charge
    }
    for a in amino_acids:
        features[f'freq_{a}'] = seq.count(a) / length
    return features

# --------------------------
# Load and prepare dataset
# --------------------------
@st.cache_data
def load_data():
    path = "/mnt/c/Users/swamy/Downloads/ML_1/Protein Solubility Prediction Benchmark datasets/Protein Solubility Prediction Benchmark datasets/esol_train.csv"
    df = pd.read_csv("ecoli_train.csv")
    seq_col = [c for c in df.columns if "seq" in c.lower()][0]
    label_col = [c for c in df.columns if "sol" in c.lower() or "label" in c.lower()][0]
    df = df[[seq_col, label_col]].dropna()
    df.columns = ["sequence", "solubility"]
    df['features'] = df['sequence'].apply(extract_features)
    df = df.dropna(subset=['features'])
    feature_df = pd.DataFrame(df['features'].tolist())
    X = feature_df
    y = df['solubility'].astype(str)
    return X, y, df

# --------------------------
# Train model
# --------------------------
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return model, acc, cm

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Protein Solubility Predictor", layout="wide")

st.title("🧬 Protein Solubility Prediction App")
st.markdown("""
### 🌟 Welcome to the Protein Property Predictor!
This ML app predicts whether a protein is **soluble** or **insoluble** based on its amino acid sequence and biochemical features.
""")

tab1, tab2, tab3 = st.tabs(["🏠 Home", "🔬 Predict", "📊 Dataset Insights"])

# --------------------------
# Tab 1: Home
# --------------------------
with tab1:
    st.header("About This Project")
    st.markdown("""
    Proteins are vital biomolecules whose **solubility** affects their function, purification, and industrial usability.
    This app uses machine learning to predict solubility from amino acid sequences.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/5a/Protein_structure.png",
             caption="Protein structure representation", use_container_width=True)
    st.info("Developed with Streamlit + scikit-learn")

# --------------------------
# Tab 2: Prediction
# --------------------------
with tab2:
    seq = st.text_area("Enter Protein Sequence (A-Z)", "MKTAYIAKQRQISFVKSHFSRQDILD...")
    if st.button("Predict Solubility"):
        features = extract_features(seq)
        if features:
            df_input = pd.DataFrame([features])
            X, y, data = load_data()
            model, acc, cm = train_model(X, y)
            pred = model.predict(df_input)[0]
            st.success(f"Predicted Solubility: **{pred}**")
            st.subheader("Computed Features:")
            st.dataframe(pd.DataFrame([features]).T)
            st.subheader(f"Model Accuracy: {acc*100:.2f}%")
            st.bar_chart(df_input.T)
        else:
            st.error("Invalid or empty sequence. Please enter a valid amino acid sequence.")

# --------------------------
# Tab 3: Dataset Insights
# --------------------------
with tab3:
    X, y, data = load_data()
    model, acc, cm = train_model(X, y)
    st.subheader("Sample Data:")
    st.dataframe(data.head(), use_container_width=True)
    st.subheader("Feature Importance")
    feat_imp = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).head(10)
    st.bar_chart(feat_imp.set_index('feature'))
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    st.metric(label="Accuracy", value=f"{acc*100:.2f}%")
