import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize analyzer
analyzer = SentimentIntensityAnalyzer()

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Pro",
    page_icon="😊",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }
    .stTextArea textarea {
        background-color: #f0f8ff;
        border: 2px solid #1e90ff;
        border-radius: 10px;
        color: #000080;
    }
    .stButton button {
        background-color: #1e90ff;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1e90ff;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎭 Sentiment Analysis Pro")
st.markdown("Analyze the emotional tone of your text using advanced VADER sentiment analysis")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Enter Your Text")
    user_input = st.text_area(
        "Type or paste your text below:",
        height=150,
        placeholder="Enter your text here to analyze sentiment...",
        label_visibility="collapsed"
    )
    
    analyze_btn = st.button("🚀 Analyze Sentiment", use_container_width=True)

with col2:
    st.subheader("📊 Quick Stats")
    if user_input and analyze_btn:
        score = analyzer.polarity_scores(user_input)
        
        # Metrics in cards
        col2_1, col2_2, col2_3 = st.columns(3)
        
        with col2_1:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #28a745; margin:0;">Positive</h4>
                <h2 style="color: #28a745; margin:0;">{score['pos']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2_2:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #dc3545; margin:0;">Negative</h4>
                <h2 style="color: #dc3545; margin:0;">{score['neg']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2_3:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #6c757d; margin:0;">Neutral</h4>
                <h2 style="color: #6c757d; margin:0;">{score['neu']:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)

# Analysis Results
if user_input and analyze_btn:
    score = analyzer.polarity_scores(user_input)
    
    st.markdown("---")
    st.subheader("🎯 Analysis Results")
    
    # Overall Sentiment
    compound_score = score['compound']
    
    if compound_score >= 0.05:
        st.success(f"😊 Positive Sentiment (Score: {compound_score:.3f})")
    elif compound_score <= -0.05:
        st.error(f"😠 Negative Sentiment (Score: {compound_score:.3f})")
    else:
        st.info(f"😐 Neutral Sentiment (Score: {compound_score:.3f})")
    
    # Visualizations
    st.subheader("📈 Visual Analysis")
    
    # Bar chart
    sentiment_df = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'Score': [score['pos'], score['neg'], score['neu']]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = px.bar(
            sentiment_df,
            x='Sentiment',
            y='Score',
            color='Sentiment',
            color_discrete_map={
                'Positive': '#28a745',
                'Negative': '#dc3545', 
                'Neutral': '#6c757d'
            },
            title="Sentiment Score Distribution"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        fig_pie = px.pie(
            sentiment_df,
            values='Score',
            names='Sentiment',
            title="Sentiment Proportion",
            color='Sentiment',
            color_discrete_map={
                'Positive': '#28a745',
                'Negative': '#dc3545',
                'Neutral': '#6c757d'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d;'>"
    "Powered by VADER Sentiment Analysis | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
