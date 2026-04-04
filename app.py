import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

# App Configuration
st.set_page_config(page_title="Olympic ML System", layout="wide", page_icon="🏅")
st.title("🏅 Olympic Games: End-to-End ML Prediction System")
st.markdown("An implementation app for analyzing numerical Olympic sports data and predicting country performance categories.")

@st.cache_data
def load_and_preprocess_data():
    try:
        # Load dataset from the user's downloads folder directly
        df = pd.read_csv('/Users/raj/Downloads/athlete_events.csv')
    except Exception as e:
        st.error(f"Error loading dataset: {e}. Please ensure athlete_events.csv is located in /Users/raj/Downloads/")
        return pd.DataFrame(), pd.DataFrame()
        
    st.sidebar.success("Dataset loaded successfully!")
    
    # 1. Cleaning
    df['Medal'] = df['Medal'].fillna('None')
    
    # 2. Extract Dummy features for Medals
    medal_dummies = pd.get_dummies(df['Medal'], prefix='Medal', dtype=int)
    if 'Medal_None' in medal_dummies.columns:
        medal_dummies = medal_dummies.drop('Medal_None', axis=1)
        
    df = pd.concat([df, medal_dummies], axis=1)
    
    # 3. Aggregation (Country-wise performance)
    country_year_data = df.groupby(['NOC', 'Year']).agg(
        Athlete_Count=('ID', 'nunique'),
        Event_Count=('Event', 'nunique'),
        Sport_Count=('Sport', 'nunique'),
        Total_Gold=('Medal_Gold', 'sum'),
        Total_Silver=('Medal_Silver', 'sum'),
        Total_Bronze=('Medal_Bronze', 'sum')
    ).reset_index()
    
    # Total medals outcome
    country_year_data['Total_Medals'] = (
        country_year_data['Total_Gold'] + 
        country_year_data['Total_Silver'] + 
        country_year_data['Total_Bronze']
    )
    
    # 4. Target Classification
    def categorize_performance(medals):
        if medals >= 16:
            return 'High'
        elif medals >= 3:
            return 'Medium'
        else:
            return 'Low'
            
    country_year_data['Performance_Class'] = country_year_data['Total_Medals'].apply(categorize_performance)
    
    # 5. Normalization / Transformation
    numeric_cols = ['Athlete_Count', 'Event_Count', 'Sport_Count']
    scaler = StandardScaler()
    country_year_data_scaled = country_year_data.copy()
    country_year_data_scaled[numeric_cols] = scaler.fit_transform(country_year_data_scaled[numeric_cols])
    
    return country_year_data, country_year_data_scaled

# Load data into frontend
raw_df, scaled_df = load_and_preprocess_data()

if raw_df.empty:
    st.stop()

# Build GUI Navigation Tabs
tabs = st.tabs(["1️⃣ Preprocessing", "2️⃣ EDA & Viz", "3️⃣ Feature Selection", "4️⃣ ML Modeling & Evaluation"])

with tabs[0]:
    st.header("Phase 1: Data Preprocessing")
    st.markdown("Raw data normalized, missing values cleaned, and features aggregated to form a stable statistical dataset representing country performance across years.")
    
    st.dataframe(scaled_df.head(50))
    st.write(f"**Final Dataset Shape:** {scaled_df.shape[0]} rows, {scaled_df.shape[1]} columns")
    st.info("Performance Class is categorised as: High (>=16 medals), Medium (3-15 medals), Low (<3 medals).")

with tabs[1]:
    st.header("Phase 2: Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Distributions (Bar Chart)")
        fig, ax = plt.subplots(figsize=(6,4))
        target_counts = raw_df['Performance_Class'].value_counts()
        sns.barplot(x=target_counts.index, y=target_counts.values, ax=ax, palette="plasma", hue=target_counts.index, legend=False)
        ax.set_ylabel("Number of Country-Year Occurrences")
        # Fixed layout issue
        fig.tight_layout()
        st.pyplot(fig)
        
    with col2:
        st.subheader("Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(6,4))
        corr_cols = ['Athlete_Count', 'Event_Count', 'Sport_Count', 'Total_Medals', 'Total_Gold']
        sns.heatmap(raw_df[corr_cols].corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        fig.tight_layout()
        st.pyplot(fig)
        
    st.subheader("Global Participation Trends Over Time (Line Plot)")
    trend_df = raw_df.groupby('Year')['Athlete_Count'].sum().reset_index()
    st.line_chart(trend_df.set_index('Year'), use_container_width=True)

with tabs[2]:
    st.header("Phase 3: Feature Engineering & Selection")
    
    features = ['Athlete_Count', 'Event_Count', 'Sport_Count']
    X = scaled_df[features]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Principal Component Analysis (PCA)")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        st.metric("Total Variance Explained", f"{sum(pca.explained_variance_ratio_):.2f}")
        
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=scaled_df['Performance_Class'], ax=ax, palette='deep')
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        fig.tight_layout()
        st.pyplot(fig)
        
    with col2:
        st.subheader("Forward Selection")
        st.markdown("Identifies the most statistically valuable attributes subset using Logistic Regression iterations:")
        
        y_encode = scaled_df['Performance_Class'].map({'Low':0, 'Medium':1, 'High':2})
        
        lr = LogisticRegression(max_iter=1000)
        sfs = SequentialFeatureSelector(lr, n_features_to_select=2, direction='forward')
        
        with st.spinner("Running Forward Selection Algorithms..."):
            sfs.fit(X, y_encode)
            
        selected_features = list(np.array(features)[sfs.get_support()])
        
        for f in features:
            if f in selected_features:
                st.success(f"✅ {f} (Selected)")
            else:
                st.error(f"❌ {f} (Discarded)")

with tabs[3]:
    st.header("Phase 4 & 5: Classification Modeling & Evaluation")
    
    st.markdown("Using numerical metrics to predict if an Olympic Country's outcome will be **High, Medium, or Low** Performance.")
    
    # Train test split
    X = scaled_df[['Athlete_Count', 'Event_Count', 'Sport_Count']]
    y = scaled_df['Performance_Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    with st.spinner('Training Random Forest Classifier...'):
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predictive Evaluation (Metrics)")
        # Display key f1 scores
        f1_w = f1_score(y_test, y_pred, average='weighted')
        st.metric("F1 Score (Weighted)", f"{f1_w:.4f}")
        
        st.text("Detailed Classification Report:")
        st.code(classification_report(y_test, y_pred, digits=4))
        
    with col2:
        st.subheader("Confusion Matrix")
        labels = ['Low', 'Medium', 'High']
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        
        fig, ax = plt.subplots(figsize=(6,4.5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues', ax=ax)
        ax.set_ylabel('Actual Category')
        ax.set_xlabel('Predicted Category')
        fig.tight_layout()
        st.pyplot(fig)
        
    st.success("End-to-End Analytics and Modeling Pipeline Completed Automatically!")
