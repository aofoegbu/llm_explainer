import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from utils.pipeline_data import get_pipeline_stages, get_stage_details
from utils.visualizations import create_pipeline_flowchart, create_progress_chart

# Configure page
st.set_page_config(
    page_title="LLM Training Pipeline Guide",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for progress tracking
if 'completed_stages' not in st.session_state:
    st.session_state.completed_stages = []
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = 0

def main():
    st.title("🤖 Complete LLM Training Pipeline Guide")
    st.markdown("### An Interactive Journey Through Large Language Model Development")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🗺️ Navigation")
        
        stages = get_pipeline_stages()
        
        # Progress indicator
        progress = len(st.session_state.completed_stages) / len(stages)
        st.progress(progress)
        st.write(f"Progress: {len(st.session_state.completed_stages)}/{len(stages)} stages completed")
        
        st.markdown("---")
        st.subheader("Pipeline Stages")
        
        for i, stage in enumerate(stages):
            # Status indicator
            if i in st.session_state.completed_stages:
                status = "✅"
            elif i == st.session_state.current_stage:
                status = "🔄"
            else:
                status = "⏳"
            
            if st.button(f"{status} {stage['name']}", key=f"nav_{i}"):
                st.session_state.current_stage = i
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Pipeline Overview")
        
        # Interactive pipeline flowchart
        fig = create_pipeline_flowchart()
        st.plotly_chart(fig, use_container_width=True)
        
        # Current stage details
        stages = get_pipeline_stages()
        current_stage = stages[st.session_state.current_stage]
        
        st.markdown("---")
        st.header(f"🎯 Current Focus: {current_stage['name']}")
        
        # Stage description
        st.markdown(current_stage['description'])
        
        # Key concepts
        st.subheader("🔑 Key Concepts")
        for concept in current_stage['key_concepts']:
            st.markdown(f"• **{concept['name']}**: {concept['description']}")
        
        # Mark stage as completed button
        if st.session_state.current_stage not in st.session_state.completed_stages:
            if st.button("✅ Mark this stage as completed"):
                st.session_state.completed_stages.append(st.session_state.current_stage)
                if st.session_state.current_stage < len(stages) - 1:
                    st.session_state.current_stage += 1
                st.rerun()
    
    with col2:
        st.header("📈 Learning Progress")
        
        # Progress chart
        progress_fig = create_progress_chart(st.session_state.completed_stages, len(stages))
        st.plotly_chart(progress_fig, use_container_width=True)
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📊 Quick Stats")
        stages = get_pipeline_stages()
        total_stages = len(stages)
        completed = len(st.session_state.completed_stages)
        
        st.metric("Total Stages", total_stages)
        st.metric("Completed", completed)
        st.metric("Remaining", total_stages - completed)
        
        # Next steps
        if st.session_state.current_stage < len(stages) - 1:
            next_stage = stages[st.session_state.current_stage + 1]
            st.markdown("---")
            st.subheader("🔮 Next Up")
            st.info(f"**{next_stage['name']}**\n\n{next_stage['description'][:100]}...")
            
            # Next up button
            if st.button("▶️ Go to Next Stage", key="next_stage_btn"):
                st.session_state.current_stage += 1
                st.rerun()
        else:
            st.markdown("---")
            st.success("🎉 Congratulations! You've completed the entire pipeline!")
    
    # Bottom section - detailed exploration
    st.markdown("---")
    st.header("🔍 Detailed Exploration")
    
    tabs = st.tabs([
        "📋 Stage Details",
        "⚖️ Alternatives Comparison", 
        "🛠️ Tools & Technologies",
        "📚 Resources"
    ])
    
    with tabs[0]:
        display_stage_details(current_stage)
    
    with tabs[1]:
        display_alternatives_comparison(current_stage)
    
    with tabs[2]:
        display_tools_technologies(current_stage)
    
    with tabs[3]:
        display_resources(current_stage)

def display_stage_details(stage):
    """Display detailed information about the current stage"""
    st.subheader(f"📖 Detailed Guide: {stage['name']}")
    
    # Step-by-step process
    if 'steps' in stage:
        st.markdown("### 📝 Step-by-Step Process")
        for i, step in enumerate(stage['steps'], 1):
            with st.expander(f"Step {i}: {step['title']}"):
                st.markdown(step['description'])
                if 'code_example' in step:
                    st.code(step['code_example'], language='python')
    
    # Best practices
    if 'best_practices' in stage:
        st.markdown("### ✨ Best Practices")
        for practice in stage['best_practices']:
            st.markdown(f"• {practice}")
    
    # Common challenges
    if 'challenges' in stage:
        st.markdown("### ⚠️ Common Challenges")
        for challenge in stage['challenges']:
            st.warning(f"**{challenge['issue']}**: {challenge['solution']}")

def display_alternatives_comparison(stage):
    """Display comparison of alternative approaches"""
    if 'alternatives' not in stage:
        st.info("No alternatives available for this stage.")
        return
    
    st.subheader(f"⚖️ Alternative Approaches for {stage['name']}")
    
    # Create comparison table
    alternatives = stage['alternatives']
    
    comparison_data = []
    for alt in alternatives:
        comparison_data.append({
            'Approach': alt['name'],
            'Pros': ', '.join(alt['pros']),
            'Cons': ', '.join(alt['cons']),
            'Complexity': alt['complexity'],
            'Cost': alt['cost']
        })
    
    if comparison_data:
        import pandas as pd
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Detailed comparison
        for alt in alternatives:
            with st.expander(f"🔍 Deep Dive: {alt['name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Advantages:**")
                    for pro in alt['pros']:
                        st.markdown(f"• {pro}")
                
                with col2:
                    st.markdown("**❌ Disadvantages:**")
                    for con in alt['cons']:
                        st.markdown(f"• {con}")
                
                st.markdown(f"**📊 Complexity:** {alt['complexity']}")
                st.markdown(f"**💰 Cost:** {alt['cost']}")
                if 'use_cases' in alt:
                    st.markdown("**🎯 Best for:**")
                    for use_case in alt['use_cases']:
                        st.markdown(f"• {use_case}")

def display_tools_technologies(stage):
    """Display tools and technologies for the current stage"""
    if 'tools' not in stage:
        st.info("No specific tools listed for this stage.")
        return
    
    st.subheader(f"🛠️ Tools & Technologies for {stage['name']}")
    
    tools = stage['tools']
    
    # Group tools by category
    categories = {}
    for tool in tools:
        category = tool.get('category', 'General')
        if category not in categories:
            categories[category] = []
        categories[category].append(tool)
    
    for category, cat_tools in categories.items():
        st.markdown(f"### {category}")
        
        cols = st.columns(min(3, len(cat_tools)))
        for i, tool in enumerate(cat_tools):
            with cols[i % 3]:
                st.markdown(f"**{tool['name']}**")
                st.markdown(tool['description'])
                if 'link' in tool:
                    st.markdown(f"[Learn More]({tool['link']})")

def display_resources(stage):
    """Display learning resources for the current stage"""
    if 'resources' not in stage:
        st.info("No specific resources listed for this stage.")
        return
    
    st.subheader(f"📚 Learning Resources for {stage['name']}")
    
    resources = stage['resources']
    
    # Group resources by type
    resource_types = {}
    for resource in resources:
        res_type = resource.get('type', 'General')
        if res_type not in resource_types:
            resource_types[res_type] = []
        resource_types[res_type].append(resource)
    
    for res_type, type_resources in resource_types.items():
        st.markdown(f"### {res_type}")
        
        for resource in type_resources:
            with st.expander(f"📖 {resource['title']}"):
                st.markdown(resource['description'])
                if 'link' in resource:
                    st.markdown(f"**[Access Resource]({resource['link']})**")
                if 'difficulty' in resource:
                    st.markdown(f"**Difficulty:** {resource['difficulty']}")

if __name__ == "__main__":
    main()
