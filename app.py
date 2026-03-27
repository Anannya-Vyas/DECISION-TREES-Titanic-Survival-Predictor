# ═══════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, roc_curve, auc)
import seaborn as sns


# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Titanic Survival | Decision Tree Classifier",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ═══════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    .stApp {
        background: linear-gradient(180deg, #0a0e27 0%, #141b2d 50%, #0a0e27 100%);
        font-family: 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255,255,255,0.03);
        padding: 8px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        color: rgba(255,255,255,0.6);
        padding: 12px 24px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00b4d8, #0077b6) !important;
        color: white !important;
        font-weight: 600;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 28px;
        margin: 12px 0;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(0,180,216,0.3);
        box-shadow: 0 8px 32px rgba(0,180,216,0.1);
    }

    /* ── Metric Card ── */
    .metric-card {
        background: linear-gradient(135deg, rgba(0,180,216,0.08), rgba(0,119,182,0.08));
        border: 1px solid rgba(0,180,216,0.2);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.8em;
        font-weight: 800;
        background: linear-gradient(135deg, #00b4d8, #90e0ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.85em;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
    }

    /* ── Hero ── */
    .hero-title {
        font-size: 3.2em;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff, #00b4d8, #90e0ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        line-height: 1.2;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 1.2em;
        color: rgba(255,255,255,0.45);
        text-align: center;
        font-weight: 300;
        margin-bottom: 35px;
    }

    /* ── Section Header ── */
    .section-header {
        font-size: 1.8em;
        font-weight: 700;
        color: #ffffff;
        margin: 20px 0 10px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(0,180,216,0.3);
    }

    /* ── Info / Warn / Success Boxes ── */
    .info-box {
        background: rgba(0,180,216,0.08);
        border-left: 4px solid #00b4d8;
        border-radius: 0 12px 12px 0;
        padding: 18px 22px;
        margin: 15px 0;
        color: rgba(255,255,255,0.85);
        line-height: 1.7;
    }
    .warn-box {
        background: rgba(255,215,0,0.08);
        border-left: 4px solid #ffd700;
        border-radius: 0 12px 12px 0;
        padding: 18px 22px;
        margin: 15px 0;
        color: rgba(255,255,255,0.85);
        line-height: 1.7;
    }
    .success-box {
        background: rgba(0,200,83,0.08);
        border-left: 4px solid #00c853;
        border-radius: 0 12px 12px 0;
        padding: 18px 22px;
        margin: 15px 0;
        color: rgba(255,255,255,0.85);
        line-height: 1.7;
    }

    /* ── Tags ── */
    .tag {
        display: inline-block;
        background: rgba(0,180,216,0.15);
        color: #00b4d8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
        margin: 2px 4px;
        border: 1px solid rgba(0,180,216,0.3);
    }
    .tag-gold {
        background: rgba(255,215,0,0.12);
        color: #ffd700;
        border-color: rgba(255,215,0,0.3);
    }
    .tag-green {
        background: rgba(0,200,83,0.12);
        color: #00c853;
        border-color: rgba(0,200,83,0.3);
    }

    /* ── Feature Table ── */
    .feature-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        margin: 15px 0;
    }
    .feature-table th {
        background: rgba(0,180,216,0.12);
        color: #00b4d8;
        padding: 14px 18px;
        text-align: left;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.85em;
    }
    .feature-table td {
        padding: 12px 18px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        color: rgba(255,255,255,0.8);
        font-size: 0.9em;
    }
    .feature-table tr:hover td {
        background: rgba(255,255,255,0.03);
    }

    /* ── Formula ── */
    .formula {
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 14px 18px;
        font-family: 'Courier New', monospace;
        color: #90e0ef;
        margin: 10px 0;
        text-align: center;
        font-size: 1.05em;
    }

    /* ── Concept Card ── */
    .concept-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 22px;
        margin: 8px 0;
        height: 100%;
    }
    .concept-title {
        font-size: 1.05em;
        font-weight: 700;
        color: #00b4d8;
        margin-bottom: 10px;
    }
    .concept-body {
        color: rgba(255,255,255,0.65);
        font-size: 0.88em;
        line-height: 1.65;
    }

    /* ── Step Indicator ── */
    .step {
        display: flex;
        align-items: flex-start;
        margin: 14px 0;
    }
    .step-number {
        background: linear-gradient(135deg, #00b4d8, #0077b6);
        color: white;
        min-width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85em;
        margin-right: 15px;
    }
    .step-content {
        color: rgba(255,255,255,0.8);
        font-size: 0.92em;
        line-height: 1.6;
        padding-top: 4px;
    }
    .step-content strong { color: #00b4d8; }

    /* ── Result Cards ── */
    .result-survived {
        background: linear-gradient(135deg, rgba(0,200,83,0.1), rgba(0,200,83,0.03));
        border: 2px solid rgba(0,200,83,0.4);
        border-radius: 16px;
        padding: 35px;
        text-align: center;
    }
    .result-died {
        background: linear-gradient(135deg, rgba(255,82,82,0.1), rgba(255,82,82,0.03));
        border: 2px solid rgba(255,82,82,0.4);
        border-radius: 16px;
        padding: 35px;
        text-align: center;
    }
    .result-emoji { font-size: 4em; }
    .result-text { font-size: 1.8em; font-weight: 700; margin: 10px 0; color: white; }
    .result-confidence { font-size: 1.05em; color: rgba(255,255,255,0.55); }

    /* ── Custom Divider ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,180,216,0.3), transparent);
        margin: 30px 0;
    }

    /* ── Button Override ── */
    .stButton > button {
        background: linear-gradient(135deg, #00b4d8, #0077b6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 30px !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0,180,216,0.3) !important;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING & MODEL TRAINING (CACHED)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_and_train():
    # Load raw data
    raw_df = sns.load_dataset("titanic")

    # Process
    df = raw_df.copy()
    df.drop(["deck", "embark_town", "alive", "class", "who", "adult_male"],
            axis=1, inplace=True)
    df["age"].fillna(df["age"].mean(), inplace=True)
    df.dropna(subset=["embarked"], inplace=True)

    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'])        # female=0, male=1
    df["embarked"] = le.fit_transform(df["embarked"])  # C=0, Q=1, S=2
    df = df.astype(int)

    X = df.drop("survived", axis=1)
    y = df["survived"]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    return (raw_df, df, X, y, X_train, X_test, y_train, y_test,
            X_train_scaled, X_test_scaled,
            model, scaler, feature_names,
            y_pred, y_proba, acc, cm, cr, fpr, tpr, roc_auc)


(raw_df, df, X, y, X_train, X_test, y_train, y_test,
 X_train_scaled, X_test_scaled,
 model, scaler, feature_names,
 y_pred, y_proba, acc, cm, cr, fpr, tpr, roc_auc) = load_and_train()


# ═══════════════════════════════════════════════════════════════
#  PLOTLY THEME
# ═══════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='rgba(255,255,255,0.8)', family='Inter'),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.06)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.06)'),
)
COLORS = ['#00b4d8', '#0077b6', '#90e0ef', '#caf0f8',
          '#ffd700', '#ff6b6b', '#48cae4', '#023e8a']


# ═══════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════

# Hero
st.markdown('<div class="hero-title">🚢 Titanic Survival Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">A Machine Learning Case Study using Decision Tree Classifier</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview", "📊 Dataset Explorer", "🌳 Decision Trees",
    "📈 Model Performance", "🔮 Predict Survival"
])


# ───────────────────────────────────────────
#  TAB 1 — OVERVIEW
# ───────────────────────────────────────────
with tab1:
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Key metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{acc*100:.1f}%</div>
            <div class="metric-label">Model Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Passengers</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(feature_names)}</div>
            <div class="metric-label">Features Used</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        survived_pct = (df['survived'].sum() / len(df)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{survived_pct:.1f}%</div>
            <div class="metric-label">Survival Rate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("""
        <div class="glass-card">
            <div class="section-header">About This Project</div>
            <p style="color:rgba(255,255,255,0.75); line-height:1.8; font-size:0.95em;">
                The sinking of the <strong style="color:#00b4d8;">RMS Titanic</strong> on
                April 15, 1912, is one of the deadliest maritime disasters in history.
                Of the estimated 2,224 passengers and crew, more than 1,500 perished.<br><br>
                This project uses a <strong style="color:#ffd700;">Decision Tree Classifier</strong>
                to predict whether a passenger survived based on features like ticket class,
                gender, age, and fare. The goal is not only to build an accurate model but also
                to <strong style="color:#00c853;">understand the decision-making process</strong>
                of the algorithm — exploring which factors most influenced survival.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="glass-card">
            <div class="section-header">Tech Stack</div>
            <div style="margin-top:15px;">
                <span class="tag">Python</span>
                <span class="tag">Scikit-learn</span>
                <span class="tag">Streamlit</span>
                <span class="tag">Pandas</span>
                <span class="tag">Plotly</span>
                <span class="tag">NumPy</span>
                <span class="tag tag-gold">Decision Tree</span>
                <span class="tag tag-gold">StandardScaler</span>
                <span class="tag tag-green">LabelEncoder</span>
            </div>
            <div style="margin-top:25px;">
                <div class="section-header" style="font-size:1.2em;">Workflow</div>
                <div class="step"><div class="step-number">1</div><div class="step-content">Load & clean the Titanic dataset</div></div>
                <div class="step"><div class="step-number">2</div><div class="step-content">Encode categorical features</div></div>
                <div class="step"><div class="step-number">3</div><div class="step-content">Scale features with StandardScaler</div></div>
                <div class="step"><div class="step-number">4</div><div class="step-content">Train Decision Tree Classifier</div></div>
                <div class="step"><div class="step-number">5</div><div class="step-content">Evaluate & deploy the model</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ───────────────────────────────────────────
#  TAB 2 — DATASET EXPLORER
# ───────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">📊 The Titanic Dataset</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        The dataset contains information about <strong>891 passengers</strong> from the
        Titanic. After cleaning (removing columns with too many nulls, filling missing
        ages with the mean, and dropping 2 rows with missing embarkation ports), we
        retain <strong>889 records</strong> and <strong>8 features</strong>.
    </div>
    """, unsafe_allow_html=True)

    # Feature Description Table
    st.markdown("""
    <table class="feature-table">
        <tr><th>Feature</th><th>Description</th><th>Type</th><th>Values</th></tr>
        <tr><td><strong>pclass</strong></td><td>Ticket class (proxy for socio-economic status)</td><td>Ordinal</td><td>1, 2, 3</td></tr>
        <tr><td><strong>sex</strong></td><td>Gender of the passenger</td><td>Binary</td><td>0 = Female, 1 = Male</td></tr>
        <tr><td><strong>age</strong></td><td>Age in years</td><td>Continuous</td><td>0 – 80</td></tr>
        <tr><td><strong>sibsp</strong></td><td>Number of siblings / spouses aboard</td><td>Discrete</td><td>0 – 8</td></tr>
        <tr><td><strong>parch</strong></td><td>Number of parents / children aboard</td><td>Discrete</td><td>0 – 6</td></tr>
        <tr><td><strong>fare</strong></td><td>Passenger fare in British pounds</td><td>Continuous</td><td>0 – 512</td></tr>
        <tr><td><strong>embarked</strong></td><td>Port of embarkation</td><td>Categorical</td><td>0 = Cherbourg, 1 = Queenstown, 2 = Southampton</td></tr>
        <tr><td><strong>alone</strong></td><td>Whether the passenger was travelling alone</td><td>Binary</td><td>0 = No, 1 = Yes</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Show sample data
    st.markdown('<div class="section-header" style="font-size:1.3em;">🔍 Sample Data (First 10 Rows)</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, height=400)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ── EDA Charts ──
    st.markdown('<div class="section-header" style="font-size:1.3em;">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    eda1, eda2 = st.columns(2)

    with eda1:
        # Survival Distribution
        surv_counts = df['survived'].value_counts().reset_index()
        surv_counts.columns = ['Survived', 'Count']
        surv_counts['Survived'] = surv_counts['Survived'].map({0: 'Did Not Survive', 1: 'Survived'})
        fig = px.pie(surv_counts, values='Count', names='Survived',
                     color_discrete_sequence=['#ff6b6b', '#00c853'],
                     title='Survival Distribution', hole=0.45)
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=True, height=400)
        fig.update_traces(textinfo='percent+value', textfont_size=14)
        st.plotly_chart(fig, use_container_width=True)

    with eda2:
        # Age Distribution
        fig = px.histogram(df, x='age', nbins=30, title='Age Distribution',
                           color_discrete_sequence=['#00b4d8'])
        fig.update_layout(**PLOTLY_LAYOUT, height=400, bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    eda3, eda4 = st.columns(2)

    with eda3:
        # Survival by Sex
        sex_surv = df.groupby('sex')['survived'].mean().reset_index()
        sex_surv['sex'] = sex_surv['sex'].map({0: 'Female', 1: 'Male'})
        sex_surv['survived'] = (sex_surv['survived'] * 100).round(1)
        fig = px.bar(sex_surv, x='sex', y='survived',
                     color='sex', color_discrete_sequence=['#ff6b9d', '#48cae4'],
                     title='Survival Rate by Sex (%)',
                     text='survived')
        fig.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_yaxes(title_text='Survival Rate (%)', range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    with eda4:
        # Survival by Class
        class_surv = df.groupby('pclass')['survived'].mean().reset_index()
        class_surv['survived'] = (class_surv['survived'] * 100).round(1)
        class_surv['pclass'] = class_surv['pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
        fig = px.bar(class_surv, x='pclass', y='survived',
                     color='pclass', color_discrete_sequence=['#ffd700', '#c0c0c0', '#cd7f32'],
                     title='Survival Rate by Passenger Class (%)',
                     text='survived')
        fig.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        fig.update_yaxes(title_text='Survival Rate (%)', range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap
    st.markdown('<div class="section-header" style="font-size:1.3em;">🔗 Feature Correlation Matrix</div>', unsafe_allow_html=True)
    corr = df.corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                    aspect='auto', zmin=-1, zmax=1,
                    title='')
    fig.update_layout(**PLOTLY_LAYOUT, height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-box">
        <strong>Key Observations:</strong><br>
        • <strong>Sex</strong> has the strongest negative correlation with survival (males less likely to survive)<br>
        • <strong>Fare</strong> and <strong>pclass</strong> show notable correlations — higher fare passengers (1st class) survived more<br>
        • <strong>alone</strong> is positively correlated with <strong>sibsp</strong>/<strong>parch</strong> as expected
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────
#  TAB 3 — DECISION TREES
# ───────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🌳 Decision Tree Classifier — Explained</div>', unsafe_allow_html=True)

    # What is it?
    st.markdown("""
    <div class="glass-card">
        <div class="concept-title" style="font-size:1.3em;">What is a Decision Tree?</div>
        <div class="concept-body" style="font-size:0.95em; line-height:1.8;">
            A <strong style="color:#00b4d8;">Decision Tree</strong> is a supervised machine learning algorithm
            that makes predictions by learning a series of <strong style="color:#ffd700;">if-then-else rules</strong>
            from the training data. Think of it like a flowchart — at each step, the algorithm asks a question
            about one feature and branches left or right based on the answer.<br><br>
            For <strong style="color:#00c853;">classification tasks</strong> (like predicting survival),
            each leaf node represents a class label, and the path from root to leaf represents the
            combination of conditions that lead to that prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # How does it work — step by step
    st.markdown('<div class="section-header" style="font-size:1.3em;">⚙️ How Does It Work?</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="step">
            <div class="step-number">1</div>
            <div class="step-content"><strong>Start with all data</strong> at the root node. The algorithm looks at every feature and every possible split point.</div>
        </div>
        <div class="step">
            <div class="step-number">2</div>
            <div class="step-content"><strong>Find the best split</strong> — the feature and threshold that most cleanly separates the classes. This is measured using <strong>Gini Impurity</strong> or <strong>Entropy</strong>.</div>
        </div>
        <div class="step">
            <div class="step-number">3</div>
            <div class="step-content"><strong>Create two child nodes</strong> — data points that satisfy the condition go left; the rest go right.</div>
        </div>
        <div class="step">
            <div class="step-number">4</div>
            <div class="step-content"><strong>Repeat recursively</strong> for each child node, finding the next best split within that subset of data.</div>
        </div>
        <div class="step">
            <div class="step-number">5</div>
            <div class="step-content"><strong>Stop</strong> when a stopping criterion is met (max depth reached, minimum samples, or node is pure — all one class).</div>
        </div>
        <div class="step">
            <div class="step-number">6</div>
            <div class="step-content"><strong>Assign predictions</strong> — each leaf node gets the majority class label of the data points that reached it.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Key Concepts
    st.markdown('<div class="section-header" style="font-size:1.3em;">🧠 Key Concepts</div>', unsafe_allow_html=True)

    kc1, kc2 = st.columns(2)

    with kc1:
        st.markdown("""
        <div class="concept-card">
            <div class="concept-title">📐 Gini Impurity</div>
            <div class="concept-body">
                Measures the probability of incorrectly classifying a randomly
                chosen element. A Gini of <strong>0</strong> means the node is
                <strong>pure</strong> (all one class). A Gini of <strong>0.5</strong>
                means maximum impurity (50/50 split).
            </div>
            <div class="formula">Gini = 1 − Σ (pᵢ)²</div>
            <div class="concept-body">
                where pᵢ is the proportion of class i samples in the node.
                <br><br><em style="color:#ffd700;">This is the default criterion used by
                scikit-learn's DecisionTreeClassifier.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kc2:
        st.markdown("""
        <div class="concept-card">
            <div class="concept-title">📊 Entropy & Information Gain</div>
            <div class="concept-body">
                <strong>Entropy</strong> measures the randomness or disorder in a node.
                A pure node has entropy <strong>0</strong>; a 50/50 split has entropy <strong>1</strong>.
            </div>
            <div class="formula">Entropy = − Σ pᵢ · log₂(pᵢ)</div>
            <div class="concept-body">
                <strong style="color:#00b4d8;">Information Gain</strong> is the reduction in
                entropy after a split:
            </div>
            <div class="formula">IG = Entropy(parent) − Weighted Avg Entropy(children)</div>
            <div class="concept-body">
                The algorithm picks the split that <strong>maximizes Information Gain</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    kc3, kc4 = st.columns(2)

    with kc3:
        st.markdown("""
        <div class="concept-card">
            <div class="concept-title">🌿 Max Depth</div>
            <div class="concept-body">
                Controls how deep the tree can grow. A deeper tree captures more
                patterns but risks <strong style="color:#ff6b6b;">overfitting</strong>.
                A shallow tree may <strong style="color:#ffd700;">underfit</strong>.
                <br><br>
                In our model, we used <code>max_depth=None</code> (no limit),
                allowing the tree to grow until all leaves are pure or contain
                fewer samples than <code>min_samples_split</code>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kc4:
        st.markdown("""
        <div class="concept-card">
            <div class="concept-title">🔀 Feature Selection at Each Node</div>
            <div class="concept-body">
                At every node, the algorithm evaluates <strong>all features</strong> and
                all possible thresholds. It selects the one that produces the best
                split (lowest Gini or highest Information Gain).
                <br><br>
                This is why decision trees can capture <strong style="color:#00c853;">
                non-linear relationships</strong> and <strong style="color:#00c853;">
                feature interactions</strong> naturally.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Pros and Cons
    st.markdown('<div class="section-header" style="font-size:1.3em;">⚖️ Advantages vs Disadvantages</div>', unsafe_allow_html=True)

    pc1, pc2 = st.columns(2)

    with pc1:
        st.markdown("""
        <div class="concept-card" style="border-color: rgba(0,200,83,0.2);">
            <div class="concept-title" style="color:#00c853;">✅ Advantages</div>
            <div style="color:rgba(255,255,255,0.75); line-height:2; font-size:0.9em;">
                ✓ Easy to understand and interpret (white-box model)<br>
                ✓ Handles both numerical and categorical data<br>
                ✓ Requires little data preprocessing<br>
                ✓ No need for feature scaling (though we used it)<br>
                ✓ Can capture non-linear relationships<br>
                ✓ Handles feature interactions automatically<br>
                ✓ Fast prediction time (O(log n))
            </div>
        </div>
        """, unsafe_allow_html=True)

    with pc2:
        st.markdown("""
        <div class="concept-card" style="border-color: rgba(255,82,82,0.2);">
            <div class="concept-title" style="color:#ff5252;">❌ Disadvantages</div>
            <div style="color:rgba(255,255,255,0.75); line-height:2; font-size:0.9em;">
                ✗ Prone to <strong>overfitting</strong> (especially deep trees)<br>
                ✗ Sensitive to small changes in data (high variance)<br>
                ✗ Can create <strong>biased trees</strong> with imbalanced datasets<br>
                ✗ Greedy algorithm — doesn't guarantee global optimum<br>
                ✗ Unstable — small data changes can alter the tree completely<br>
                ✗ Generally less accurate than ensemble methods<br>
                ✗ Can struggle with XOR-like relationships
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Actual tree visualization
    st.markdown('<div class="section-header" style="font-size:1.3em;">🌲 Our Trained Decision Tree (Top 4 Levels)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Below is a visual representation of the first 4 levels of our trained Decision Tree.
        Each node shows the <strong>splitting condition</strong>, <strong>Gini impurity</strong>,
        <strong>number of samples</strong>, and the <strong>majority class</strong>
        (blue = survived, orange = did not survive).
    </div>
    """, unsafe_allow_html=True)

    fig_tree, ax_tree = plt.subplots(figsize=(28, 10))
    fig_tree.patch.set_facecolor('#0a0e27')
    ax_tree.set_facecolor('#0a0e27')

    plot_tree(model, max_depth=4, filled=True, rounded=True,
              feature_names=feature_names,
              class_names=['Died', 'Survived'],
              fontsize=8, ax=ax_tree,
              impurity=True, proportion=False)
    ax_tree.set_title("Decision Tree Visualization (max_depth=4 shown)",
                      fontsize=16, color='white', pad=20)
    st.pyplot(fig_tree, use_container_width=True)
    plt.close()

    # Decision rules
    with st.expander("📜 View Text-Based Decision Rules (Top 5 Levels)"):
        rules = export_text(model, feature_names=feature_names, max_depth=5)
        st.code(rules, language="text")

    st.markdown("""
    <div class="warn-box">
        <strong>💡 Why did we still use StandardScaler?</strong><br>
        Decision Trees are <em>mathematically invariant</em> to feature scaling — the splits
        remain the same. However, we used StandardScaler because: (1) it's part of a
        consistent ML pipeline, (2) it enables fair comparison with other models like
        Logistic Regression or SVM, and (3) it's good practice for production systems.
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────
#  TAB 4 — MODEL PERFORMANCE
# ───────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">📈 Model Performance Analysis</div>', unsafe_allow_html=True)

    # Metric cards
    precision_0 = cr['0']['precision']
    precision_1 = cr['1']['precision']
    recall_0 = cr['0']['recall']
    recall_1 = cr['1']['recall']
    f1_0 = cr['0']['f1-score']
    f1_1 = cr['1']['f1-score']

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{acc*100:.1f}%</div>
            <div class="metric-label">Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{cr['macro avg']['precision']*100:.1f}%</div>
            <div class="metric-label">Avg Precision</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{cr['macro avg']['recall']*100:.1f}%</div>
            <div class="metric-label">Avg Recall</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{roc_auc:.3f}</div>
            <div class="metric-label">AUC-ROC</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    perf1, perf2 = st.columns(2)

    with perf1:
        # Confusion Matrix
        st.markdown('<div class="section-header" style="font-size:1.2em;">🎯 Confusion Matrix</div>', unsafe_allow_html=True)

        fig_cm = px.imshow(cm,
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Died (0)', 'Survived (1)'],
                           y=['Died (0)', 'Survived (1)'],
                           text_auto=True,
                           color_continuous_scale=[[0, '#0a0e27'], [1, '#00b4d8']])
        fig_cm.update_layout(**PLOTLY_LAYOUT, height=420, title='')
        fig_cm.update_traces(textfont_size=20)
        st.plotly_chart(fig_cm, use_container_width=True)

        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
        <div class="info-box">
            <strong>Reading the Matrix:</strong><br>
            • <strong>True Negatives (Correctly predicted died):</strong> {tn}<br>
            • <strong>True Positives (Correctly predicted survived):</strong> {tp}<br>
            • <strong>False Positives (Predicted survived, actually died):</strong> {fp}<br>
            • <strong>False Negatives (Predicted died, actually survived):</strong> {fn}
        </div>
        """, unsafe_allow_html=True)

    with perf2:
        # ROC Curve
        st.markdown('<div class="section-header" style="font-size:1.2em;">📉 ROC Curve</div>', unsafe_allow_html=True)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                      name=f'Decision Tree (AUC = {roc_auc:.3f})',
                                      line=dict(color='#00b4d8', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                      name='Random Classifier',
                                      line=dict(color='rgba(255,255,255,0.2)', dash='dash')))
        fig_roc.update_layout(**PLOTLY_LAYOUT, height=420,
                               xaxis_title='False Positive Rate',
                               yaxis_title='True Positive Rate',
                               showlegend=True,
                               legend=dict(x=0.4, y=0.1, bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig_roc, use_container_width=True)

        st.markdown(f"""
        <div class="info-box">
            <strong>AUC-ROC = {roc_auc:.3f}</strong><br>
            The Area Under the ROC Curve measures the model's ability to
            distinguish between classes. An AUC of <strong>1.0</strong> is perfect;
            <strong>0.5</strong> is random guessing. Our model achieves
            <strong>{roc_auc:.3f}</strong>.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Classification Report
    st.markdown('<div class="section-header" style="font-size:1.2em;">📋 Detailed Classification Report</div>', unsafe_allow_html=True)

    report_data = {
        'Class': ['Died (0)', 'Survived (1)', 'Macro Avg', 'Weighted Avg'],
        'Precision': [f"{precision_0:.3f}", f"{precision_1:.3f}",
                      f"{cr['macro avg']['precision']:.3f}", f"{cr['weighted avg']['precision']:.3f}"],
        'Recall': [f"{recall_0:.3f}", f"{recall_1:.3f}",
                   f"{cr['macro avg']['recall']:.3f}", f"{cr['weighted avg']['recall']:.3f}"],
        'F1-Score': [f"{f1_0:.3f}", f"{f1_1:.3f}",
                     f"{cr['macro avg']['f1-score']:.3f}", f"{cr['weighted avg']['f1-score']:.3f}"],
        'Support': [int(cr['0']['support']), int(cr['1']['support']),
                    int(cr['macro avg']['support']), int(cr['weighted avg']['support'])]
    }
    st.dataframe(pd.DataFrame(report_data).set_index('Class'), use_container_width=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Feature Importance
    st.markdown('<div class="section-header" style="font-size:1.2em;">🏆 Feature Importance</div>', unsafe_allow_html=True)

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)

    fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale=[[0, '#0077b6'], [1, '#90e0ef']],
                     text=feat_imp['Importance'].apply(lambda x: f'{x:.3f}'))
    fig_imp.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False,
                           coloraxis_showscale=False)
    fig_imp.update_traces(textposition='outside')
    st.plotly_chart(fig_imp, use_container_width=True)

    top_feat = feat_imp.iloc[-1]['Feature']
    top_val = feat_imp.iloc[-1]['Importance']
    st.markdown(f"""
    <div class="success-box">
        <strong>Most important feature:</strong> <code>{top_feat}</code> with an importance
        score of <strong>{top_val:.3f}</strong>. This means the Decision Tree relied most
        heavily on this feature when making splits.
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────
#  TAB 5 — PREDICT SURVIVAL
# ───────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">🔮 Will You Survive the Titanic?</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Enter the passenger details below and click <strong>Predict</strong> to see
        whether the model predicts survival. The model uses the same Decision Tree
        trained on 80% of the Titanic dataset.
    </div>
    """, unsafe_allow_html=True)

    # Input form
    inp1, inp2, inp3 = st.columns(3)

    with inp1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        pclass = st.selectbox("🎫 Passenger Class", [1, 2, 3],
                               format_func=lambda x: f"{x}{'st' if x==1 else 'nd' if x==2 else 'rd'} Class")
        sex = st.selectbox("👤 Sex", ["Female", "Male"])
        age = st.slider("🎂 Age", min_value=0, max_value=80, value=28)
        st.markdown('</div>', unsafe_allow_html=True)

    with inp2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        sibsp = st.number_input("👫 Siblings/Spouses Aboard", 0, 10, 0)
        parch = st.number_input("👨‍👩‍👧 Parents/Children Aboard", 0, 10, 0)
        fare = st.number_input("💰 Fare Paid (£)", 0.0, 600.0, 32.0, step=1.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with inp3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        embarked = st.selectbox("⚓ Port of Embarkation",
                                 ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"])
        alone = st.selectbox("🧍 Travelling Alone?", ["No", "Yes"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # Encode
    sex_enc = 0 if sex == "Female" else 1
    emb_map = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}
    emb_enc = emb_map[embarked]
    alone_enc = 0 if alone == "No" else 1

    # Predict button
    if st.button("🚀  Predict Survival", use_container_width=True):
        features = np.array([[pclass, sex_enc, int(age), int(sibsp),
                               int(parch), int(fare), emb_enc, alone_enc]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        res1, res2 = st.columns([2, 1])

        with res1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-survived">
                    <div class="result-emoji">🎉</div>
                    <div class="result-text" style="color:#00c853;">SURVIVED!</div>
                    <div class="result-confidence">
                        Survival Probability: <strong>{proba[1]*100:.1f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class="result-died">
                    <div class="result-emoji">💔</div>
                    <div class="result-text" style="color:#ff5252;">DID NOT SURVIVE</div>
                    <div class="result-confidence">
                        Death Probability: <strong>{proba[0]*100:.1f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with res2:
            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba[1] * 100,
                number={'suffix': '%', 'font': {'color': 'white', 'size': 36}},
                title={'text': 'Survival Chance', 'font': {'color': 'rgba(255,255,255,0.7)', 'size': 14}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'rgba(255,255,255,0.3)'},
                    'bar': {'color': '#00b4d8'},
                    'bgcolor': 'rgba(255,255,255,0.05)',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(255,82,82,0.15)'},
                        {'range': [30, 60], 'color': 'rgba(255,215,0,0.15)'},
                        {'range': [60, 100], 'color': 'rgba(0,200,83,0.15)'}
                    ],
                    'threshold': {
                        'line': {'color': 'white', 'width': 2},
                        'thickness': 0.8,
                        'value': proba[1] * 100
                    }
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                     font=dict(color='white'),
                                     height=280, margin=dict(l=30, r=30, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Input summary
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="font-size:1.1em;">📋 Your Passenger Profile</div>', unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="font-size:1.8em;">{pclass}{'st' if pclass==1 else 'nd' if pclass==2 else 'rd'}</div>
                <div class="metric-label">Class</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="font-size:1.8em;">{sex}</div>
                <div class="metric-label">Sex</div>
            </div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="font-size:1.8em;">{age}</div>
                <div class="metric-label">Age</div>
            </div>""", unsafe_allow_html=True)
        with s4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="font-size:1.8em;">£{fare:.0f}</div>
                <div class="metric-label">Fare</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:rgba(255,255,255,0.25); font-size:0.85em; padding:20px;">
    Built with ❤️ using Streamlit & Scikit-learn  •  Decision Tree Classifier
    •  Titanic Survival Prediction
</div>
""", unsafe_allow_html=True)