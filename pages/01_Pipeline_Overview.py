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

cols = st.columns(4)
for i, stage in enumerate(stages):
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
            st.switch_page(page_map.get(i, "app.py"))
