import streamlit as st
import plotly.graph_objects as go
from utils.visualizations import create_detailed_pipeline_diagram, create_stage_timeline
from utils.pipeline_data import get_pipeline_stages

st.set_page_config(page_title="Pipeline Overview", page_icon="📊", layout="wide")

st.title("📊 LLM Training Pipeline Overview")
st.markdown("### Complete Visual Guide to Large Language Model Development")

# Main pipeline visualization
st.header("🔄 Interactive Pipeline Diagram")

fig = create_detailed_pipeline_diagram()
st.plotly_chart(fig, use_container_width=True)

# Timeline view
st.header("⏱️ Development Timeline")
timeline_fig = create_stage_timeline()
st.plotly_chart(timeline_fig, use_container_width=True)

# Pipeline stages overview
st.header("📋 All Pipeline Stages")

stages = get_pipeline_stages()

# Create cards for each stage
for i, stage in enumerate(stages):
    with st.expander(f"Stage {i+1}: {stage['name']}", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(stage['description'])
            
            # Key deliverables
            if 'deliverables' in stage:
                st.markdown("**📦 Key Deliverables:**")
                for deliverable in stage['deliverables']:
                    st.markdown(f"• {deliverable}")
        
        with col2:
            # Stage metrics
            if 'metrics' in stage:
                st.markdown("**📊 Key Metrics:**")
                for metric in stage['metrics']:
                    st.metric(metric['name'], metric['value'], metric.get('delta', None))
            
            # Duration estimate
            if 'duration' in stage:
                st.markdown(f"**⏱️ Estimated Duration:** {stage['duration']}")
            
            # Difficulty level
            if 'difficulty' in stage:
                difficulty_map = {
                    'Beginner': '🟢',
                    'Intermediate': '🟡', 
                    'Advanced': '🔴',
                    'Expert': '🟣'
                }
                difficulty_icon = difficulty_map.get(stage['difficulty'], '⚪')
                st.markdown(f"**📈 Difficulty:** {difficulty_icon} {stage['difficulty']}")

# Quick navigation to detailed pages
st.header("🧭 Quick Navigation")
st.markdown("Jump directly to any stage for detailed exploration:")

# Core Pipeline Navigation (first 7 stages)
st.subheader("📊 Core Training Pipeline")
cols = st.columns(4)
core_stages = stages[:7]  # First 7 stages are the core pipeline

for i, stage in enumerate(core_stages):
    with cols[i % 4]:
        page_map = {
            0: "pages/02_Data_Collection.py",
            1: "pages/03_Data_Preprocessing.py", 
            2: "pages/04_Model_Architecture.py",
            3: "pages/05_Training.py",
            4: "pages/06_Fine_Tuning.py",
            5: "pages/07_Evaluation.py",
            6: "pages/08_Deployment.py"
        }
        
        if st.button(f"🚀 {stage['name']}", key=f"nav_button_{i}"):
            st.switch_page(page_map[i])

# Advanced Techniques Navigation 
st.subheader("🚀 Advanced LLM Techniques")
advanced_cols = st.columns(3)
advanced_pages = [
    ("Prompt Engineering", "pages/09_Prompt_Engineering.py"),
    ("In-Context Learning", "pages/10_In_Context_Learning.py"),
    ("Embeddings", "pages/11_Embeddings.py"),
    ("RAG Systems", "pages/12_RAG.py"),
    ("AI Agents", "pages/13_Agents.py"),
    ("RLHF", "pages/16_RLHF.py")
]

for i, (name, page) in enumerate(advanced_pages):
    with advanced_cols[i % 3]:
        if st.button(f"⚡ {name}", key=f"advanced_nav_{i}"):
            st.switch_page(page)

# Infrastructure & Ethics Navigation
st.subheader("🔧 Infrastructure & Ethics")
infra_cols = st.columns(3)
infra_pages = [
    ("Data Engineering", "pages/14_Data_Engineering.py"),
    ("Model Selection", "pages/15_Model_Selection.py"),
    ("Explainability", "pages/17_Explainability.py"),
    ("Bias Mitigation", "pages/18_Bias_Mitigation.py")
]

for i, (name, page) in enumerate(infra_pages):
    with infra_cols[i % 3]:
        if st.button(f"🛠️ {name}", key=f"infra_nav_{i}"):
            st.switch_page(page)
