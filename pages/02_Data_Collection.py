import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from utils.visualizations import create_data_sources_chart, create_data_quality_metrics

st.set_page_config(page_title="Data Collection", page_icon="📊", layout="wide")

st.title("📊 Data Collection for LLM Training")
st.markdown("### The Foundation of Successful Language Model Development")

# Overview
st.header("🎯 Overview")
st.markdown("""
Data collection is the critical first step in LLM training. The quality, diversity, and scale of your training data 
directly impacts the capabilities and performance of your final model. This stage involves gathering, curating, 
and preparing massive text datasets from various sources.
""")

# Key concepts
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔑 Key Concepts")
    concepts = [
        ("Data Scale", "Modern LLMs require terabytes of text data"),
        ("Data Quality", "Clean, well-formatted text improves training efficiency"),
        ("Data Diversity", "Varied sources prevent overfitting to specific domains"),
        ("Licensing", "Ensuring legal compliance for data usage"),
        ("Deduplication", "Removing duplicate content to improve model generalization")
    ]
    
    for concept, description in concepts:
        st.markdown(f"**{concept}**: {description}")

with col2:
    st.subheader("📈 Data Scale Requirements")
    
    # Data scale visualization
    scale_data = {
        'Model Size': ['Small (125M)', 'Medium (1.3B)', 'Large (7B)', 'Very Large (13B)', 'Ultra Large (70B+)'],
        'Training Tokens (Billions)': [100, 300, 1000, 1500, 3000],
        'Raw Data (TB)': [0.2, 0.6, 2.0, 3.0, 6.0]
    }
    
    df_scale = pd.DataFrame(scale_data)
    
    fig = px.bar(df_scale, x='Model Size', y='Training Tokens (Billions)', 
                title="Training Data Requirements by Model Size")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Data sources
st.header("🌐 Data Sources")

tabs = st.tabs(["📚 Common Sources", "⚖️ Source Comparison", "📋 Collection Methods"])

with tabs[0]:
    st.subheader("Popular Data Sources for LLM Training")
    
    sources = [
        {
            "name": "Common Crawl",
            "description": "Web crawl data containing billions of web pages",
            "size": "~3.5TB per monthly crawl",
            "quality": "Variable - requires significant filtering",
            "license": "Open access with attribution"
        },
        {
            "name": "Wikipedia",
            "description": "High-quality encyclopedic content",
            "size": "~20GB compressed",
            "quality": "High - well-edited and structured",
            "license": "CC BY-SA 3.0"
        },
        {
            "name": "Books Corpus",
            "description": "Collection of digitized books",
            "size": "~4GB",
            "quality": "High - professionally edited content",
            "license": "Varies - check individual works"
        },
        {
            "name": "Academic Papers",
            "description": "Scientific literature and research papers",
            "size": "Variable",
            "quality": "Very high - peer-reviewed content",
            "license": "Varies by publisher"
        },
        {
            "name": "News Articles",
            "description": "News content from various publications",
            "size": "Variable",
            "quality": "High - professionally written",
            "license": "Usually requires licensing"
        },
        {
            "name": "Social Media",
            "description": "Posts, comments, and discussions",
            "size": "Massive - billions of posts",
            "quality": "Variable - requires heavy filtering",
            "license": "Platform-specific terms"
        }
    ]
    
    for source in sources:
        with st.expander(f"📖 {source['name']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Description:** {source['description']}")
                st.markdown(f"**Size:** {source['size']}")
            with col2:
                st.markdown(f"**Quality:** {source['quality']}")
                st.markdown(f"**License:** {source['license']}")

with tabs[1]:
    st.subheader("📊 Data Source Comparison")
    
    # Create comparison chart
    comparison_data = {
        'Source': ['Common Crawl', 'Wikipedia', 'Books', 'Academic Papers', 'News', 'Social Media'],
        'Quality Score': [6, 9, 9, 10, 8, 5],
        'Size (Relative)': [10, 3, 2, 4, 5, 10],
        'Diversity': [10, 7, 6, 5, 7, 9],
        'Accessibility': [10, 10, 4, 6, 3, 7]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    fig = go.Figure()
    
    for metric in ['Quality Score', 'Size (Relative)', 'Diversity', 'Accessibility']:
        fig.add_trace(go.Scatter(
            x=df_comparison['Source'],
            y=df_comparison[metric],
            mode='lines+markers',
            name=metric,
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Data Source Comparison (1-10 Scale)",
        xaxis_title="Data Source",
        yaxis_title="Score",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("🔧 Data Collection Methods")
    
    methods = [
        {
            "method": "Web Scraping",
            "description": "Automated extraction of content from websites",
            "tools": ["Scrapy", "BeautifulSoup", "Selenium"],
            "pros": ["Large scale", "Real-time content", "Customizable"],
            "cons": ["Legal issues", "Rate limiting", "Quality varies"],
            "code_example": """
import requests
from bs4 import BeautifulSoup

def scrape_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract text content
    text = soup.get_text()
    
    # Clean and process text
    cleaned_text = ' '.join(text.split())
    
    return cleaned_text
"""
        },
        {
            "method": "API Integration",
            "description": "Using official APIs to access platform data",
            "tools": ["Twitter API", "Reddit API", "News APIs"],
            "pros": ["Legal compliance", "Structured data", "Rate limiting"],
            "cons": ["Limited access", "Cost", "Platform restrictions"],
            "code_example": """
import tweepy

# Twitter API example
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

def collect_tweets(query, count=100):
    tweets = tweepy.Cursor(api.search_tweets, 
                          q=query, 
                          lang="en").items(count)
    
    return [tweet.text for tweet in tweets]
"""
        },
        {
            "method": "Dataset Aggregation",
            "description": "Combining existing open datasets",
            "tools": ["Hugging Face Datasets", "Kaggle", "GitHub"],
            "pros": ["Pre-processed", "Legal clarity", "Quality assured"],
            "cons": ["Limited customization", "Potential overlap", "Size constraints"],
            "code_example": """
from datasets import load_dataset

# Load pre-existing dataset
dataset = load_dataset("wikipedia", "20220301.en")

# Access training split
train_data = dataset['train']

# Process text
texts = [example['text'] for example in train_data]
"""
        }
    ]
    
    for method in methods:
        with st.expander(f"🛠️ {method['method']}"):
            st.markdown(method['description'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Advantages:**")
                for pro in method['pros']:
                    st.markdown(f"• {pro}")
                
                st.markdown("**🛠️ Common Tools:**")
                for tool in method['tools']:
                    st.markdown(f"• {tool}")
            
            with col2:
                st.markdown("**❌ Challenges:**")
                for con in method['cons']:
                    st.markdown(f"• {con}")
            
            if 'code_example' in method:
                st.markdown("**💻 Code Example:**")
                st.code(method['code_example'], language='python')

# Data quality considerations
st.header("🎯 Data Quality Considerations")

quality_tabs = st.tabs(["📊 Quality Metrics", "🧹 Cleaning Steps", "⚠️ Common Issues"])

with quality_tabs[0]:
    st.subheader("Key Quality Metrics to Track")
    
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.metric("Duplicate Rate", "15%", "-5%")
        st.metric("Language Purity", "98.5%", "+1.2%")
        st.metric("Average Doc Length", "2,847 tokens", "+124")
    
    with metrics_col2:
        st.metric("Vocabulary Size", "2.1M words", "+50K")
        st.metric("Error Rate", "0.3%", "-0.1%")
        st.metric("Coverage Score", "87%", "+3%")

with quality_tabs[1]:
    st.subheader("Essential Data Cleaning Steps")
    
    cleaning_steps = [
        ("Language Detection", "Filter content to target language(s)"),
        ("Deduplication", "Remove exact and near-duplicate content"),
        ("Format Normalization", "Standardize text encoding and formatting"),
        ("Content Filtering", "Remove low-quality, spam, or inappropriate content"),
        ("Length Filtering", "Remove extremely short or long documents"),
        ("Privacy Scrubbing", "Remove personal information and sensitive data")
    ]
    
    for i, (step, description) in enumerate(cleaning_steps, 1):
        st.markdown(f"**{i}. {step}**: {description}")

with quality_tabs[2]:
    st.subheader("Common Data Quality Issues")
    
    issues = [
        {
            "issue": "Duplicate Content",
            "impact": "Model memorization and poor generalization",
            "solution": "Use fuzzy matching and hash-based deduplication"
        },
        {
            "issue": "Language Contamination",
            "impact": "Reduced performance on target language",
            "solution": "Robust language detection and filtering"
        },
        {
            "issue": "Low-Quality Text",
            "impact": "Model learns poor writing patterns",
            "solution": "Quality scoring and threshold-based filtering"
        },
        {
            "issue": "Biased Content",
            "impact": "Model reproduces harmful biases",
            "solution": "Bias detection tools and diverse source selection"
        }
    ]
    
    for issue in issues:
        st.warning(f"**{issue['issue']}**: {issue['impact']}")
        st.info(f"💡 **Solution**: {issue['solution']}")

# Best practices
st.header("✨ Best Practices")

practices = [
    "Start with high-quality, curated datasets before adding web-scraped content",
    "Implement robust deduplication at multiple levels (exact, near-duplicate, semantic)",
    "Maintain detailed provenance tracking for all data sources",
    "Regular quality audits throughout the collection process",
    "Legal review of all data sources and usage rights",
    "Version control for datasets with clear change tracking",
    "Balanced representation across domains, languages, and demographics",
    "Privacy-preserving techniques for sensitive content handling"
]

for practice in practices:
    st.markdown(f"✅ {practice}")

# Interactive exercises
st.header("🎮 Interactive Exercises")

exercise_tabs = st.tabs(["📊 Data Audit", "🔍 Quality Check", "📈 Scale Calculator"])

with exercise_tabs[0]:
    st.subheader("Data Source Audit Tool")
    st.markdown("Evaluate your data sources using this interactive tool:")
    
    source_name = st.text_input("Data Source Name", placeholder="e.g., Common Crawl")
    source_size = st.number_input("Size (GB)", min_value=0.0, step=0.1)
    quality_score = st.slider("Quality Score (1-10)", 1, 10, 5)
    license_type = st.selectbox("License Type", 
                               ["Open Source", "Academic Use", "Commercial", "Custom", "Unknown"])
    
    if st.button("Evaluate Source"):
        # Simple scoring algorithm
        score = (quality_score * 0.4) + (min(source_size/100, 10) * 0.3) + \
                ({"Open Source": 10, "Academic Use": 8, "Commercial": 6, 
                  "Custom": 7, "Unknown": 3}[license_type] * 0.3)
        
        if score >= 8:
            st.success(f"✅ Excellent source! Score: {score:.1f}/10")
        elif score >= 6:
            st.warning(f"⚠️ Good source with room for improvement. Score: {score:.1f}/10")
        else:
            st.error(f"❌ Consider alternative sources. Score: {score:.1f}/10")

with exercise_tabs[1]:
    st.subheader("Text Quality Checker")
    
    sample_text = st.text_area("Paste sample text to analyze:", 
                              placeholder="Enter text to check quality metrics...")
    
    if sample_text and st.button("Analyze Quality"):
        # Basic quality metrics
        word_count = len(sample_text.split())
        char_count = len(sample_text)
        avg_word_length = sum(len(word) for word in sample_text.split()) / word_count if word_count > 0 else 0
        sentence_count = sample_text.count('.') + sample_text.count('!') + sample_text.count('?')
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Word Count", word_count)
        with col2:
            st.metric("Avg Word Length", f"{avg_word_length:.1f}")
        with col3:
            st.metric("Character Count", char_count)
        with col4:
            st.metric("Sentences", sentence_count)

with exercise_tabs[2]:
    st.subheader("Training Data Scale Calculator")
    
    target_model_size = st.selectbox("Target Model Size", 
                                   ["125M", "350M", "1.3B", "7B", "13B", "30B", "70B+"])
    
    efficiency_factor = st.slider("Training Efficiency (1.0 = standard)", 0.5, 2.0, 1.0, 0.1)
    
    # Data requirements mapping
    size_to_tokens = {
        "125M": 100, "350M": 200, "1.3B": 300, "7B": 1000, 
        "13B": 1500, "30B": 2000, "70B+": 3000
    }
    
    base_tokens = size_to_tokens[target_model_size]
    adjusted_tokens = int(base_tokens / efficiency_factor)
    estimated_size_tb = adjusted_tokens / 500  # Rough conversion
    
    st.markdown("### 📊 Estimated Requirements")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Tokens", f"{adjusted_tokens}B")
    with col2:
        st.metric("Raw Data Size", f"{estimated_size_tb:.1f}TB")
    with col3:
        st.metric("Collection Time", f"{estimated_size_tb*2:.0f} weeks")
