"""
Visualization utilities for the LLM training pipeline Streamlit app.
Contains functions to create interactive charts and diagrams using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_pipeline_flowchart():
    """Create an interactive flowchart showing the main LLM training pipeline stages."""
    
    stages = [
        "Data Collection",
        "Data Preprocessing", 
        "Model Architecture",
        "Training",
        "Fine-Tuning",
        "Evaluation",
        "Deployment"
    ]
    
    # Create flowchart using Plotly
    fig = go.Figure()
    
    # Define positions for stages
    x_positions = list(range(len(stages)))
    y_position = [0] * len(stages)
    
    # Add nodes for each stage
    for i, stage in enumerate(stages):
        fig.add_trace(go.Scatter(
            x=[x_positions[i]],
            y=[y_position[i]],
            mode='markers+text',
            marker=dict(
                size=60,
                color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF'][i],
                line=dict(width=2, color='white')
            ),
            text=stage,
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            name=stage,
            hovertemplate=f"<b>{stage}</b><br>Click to learn more<extra></extra>"
        ))
    
    # Add arrows between stages
    for i in range(len(stages) - 1):
        fig.add_annotation(
            x=x_positions[i] + 0.4,
            y=y_position[i],
            ax=x_positions[i+1] - 0.4,
            ay=y_position[i+1],
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor="#2C3E50",
            showarrow=True
        )
    
    fig.update_layout(
        title={
            'text': "LLM Training Pipeline Overview",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2C3E50'}
        },
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-0.5, len(stages) - 0.5]
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-0.8, 0.8]
        ),
        plot_bgcolor='white',
        height=300,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_progress_chart(completed_stages, total_stages):
    """Create a progress visualization showing learning journey completion."""
    
    # Handle both list and integer inputs for completed_stages
    if isinstance(completed_stages, list):
        completed_count = len(completed_stages)
    else:
        completed_count = completed_stages
    
    completion_percentage = (completed_count / total_stages) * 100
    
    # Create a donut chart for progress
    fig = go.Figure(data=[go.Pie(
        labels=['Completed', 'Remaining'],
        values=[completed_count, total_stages - completed_count],
        hole=0.6,
        marker_colors=['#4ECDC4', '#E8E8E8'],
        textinfo='none',
        hovertemplate='<b>%{label}</b><br>Stages: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    # Add progress text in the center
    fig.add_annotation(
        text=f"{completion_percentage:.0f}%<br><span style='font-size:14px'>Complete</span>",
        x=0.5, y=0.5,
        font_size=24,
        font_color='#2C3E50',
        showarrow=False
    )
    
    fig.update_layout(
        title={
            'text': "Learning Progress",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2C3E50'}
        },
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_detailed_pipeline_diagram():
    """Create a detailed pipeline diagram with sub-processes and decision points."""
    
    # Define the detailed pipeline with sub-steps
    pipeline_data = {
        'Stage': ['Data Collection', 'Data Collection', 'Data Collection',
                 'Preprocessing', 'Preprocessing', 'Preprocessing',
                 'Architecture', 'Architecture', 
                 'Training', 'Training', 'Training',
                 'Fine-Tuning', 'Evaluation', 'Deployment'],
        'Substage': ['Source Selection', 'Web Scraping', 'Quality Assessment',
                    'Cleaning', 'Tokenization', 'Formatting',
                    'Design', 'Parameter Sizing',
                    'Pre-training', 'Hyperparameter Tuning', 'Validation',
                    'Task Adaptation', 'Benchmarking', 'Production Serving'],
        'Duration_Days': [3, 5, 2, 4, 3, 2, 3, 2, 21, 7, 3, 7, 5, 10],
        'Complexity': ['Medium', 'High', 'Medium', 'Medium', 'High', 'Low',
                      'High', 'Medium', 'High', 'High', 'Medium',
                      'Medium', 'Medium', 'High'],
        'Start_Day': [0, 3, 8, 10, 14, 17, 19, 22, 24, 45, 52, 55, 62, 67]
    }
    
    df = pd.DataFrame(pipeline_data)
    
    # Create Gantt chart
    fig = go.Figure()
    
    # Color mapping for complexity
    color_map = {'Low': '#4ECDC4', 'Medium': '#FECA57', 'High': '#FF6B6B'}
    
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Start_Day'], row['Start_Day'] + row['Duration_Days']],
            y=[i, i],
            mode='lines',
            line=dict(width=20, color=color_map[row['Complexity']]),
            name=f"{row['Stage']}: {row['Substage']}",
            text=f"{row['Substage']}<br>{row['Duration_Days']} days",
            textposition="middle center",
            hovertemplate=f"<b>{row['Stage']}: {row['Substage']}</b><br>" +
                         f"Duration: {row['Duration_Days']} days<br>" +
                         f"Complexity: {row['Complexity']}<extra></extra>"
        ))
    
    fig.update_layout(
        title={
            'text': "Detailed LLM Training Timeline",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        xaxis_title="Days",
        yaxis_title="Process Steps",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(df))),
            ticktext=[f"{row['Stage']}: {row['Substage']}" for _, row in df.iterrows()],
            autorange="reversed"
        ),
        showlegend=False,
        height=600,
        plot_bgcolor='white',
        margin=dict(l=200, r=50, t=80, b=50)
    )
    
    return fig

def create_stage_timeline():
    """Create a timeline view of the training stages with milestones."""
    
    stages_timeline = [
        {'stage': 'Data Collection', 'start': 0, 'duration': 10, 'milestone': 'Dataset Ready'},
        {'stage': 'Data Preprocessing', 'start': 8, 'duration': 9, 'milestone': 'Clean Data'},
        {'stage': 'Model Architecture', 'start': 15, 'duration': 5, 'milestone': 'Architecture Finalized'},
        {'stage': 'Training', 'start': 20, 'duration': 35, 'milestone': 'Base Model Trained'},
        {'stage': 'Fine-Tuning', 'start': 50, 'duration': 12, 'milestone': 'Model Specialized'},
        {'stage': 'Evaluation', 'start': 58, 'duration': 9, 'milestone': 'Performance Validated'},
        {'stage': 'Deployment', 'start': 65, 'duration': 12, 'milestone': 'Production Ready'}
    ]
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']
    
    for i, stage_info in enumerate(stages_timeline):
        # Add timeline bar
        fig.add_trace(go.Scatter(
            x=[stage_info['start'], stage_info['start'] + stage_info['duration']],
            y=[i, i],
            mode='lines',
            line=dict(width=25, color=colors[i]),
            name=stage_info['stage'],
            text=f"{stage_info['stage']}<br>{stage_info['duration']} days",
            textposition="middle center",
            hovertemplate=f"<b>{stage_info['stage']}</b><br>" +
                         f"Duration: {stage_info['duration']} days<br>" +
                         f"Milestone: {stage_info['milestone']}<extra></extra>"
        ))
        
        # Add milestone marker
        fig.add_trace(go.Scatter(
            x=[stage_info['start'] + stage_info['duration']],
            y=[i],
            mode='markers',
            marker=dict(size=15, color='white', line=dict(width=3, color=colors[i])),
            name=f"{stage_info['milestone']}",
            text=stage_info['milestone'],
            textposition="top center",
            showlegend=False,
            hovertemplate=f"<b>Milestone: {stage_info['milestone']}</b><extra></extra>"
        ))
    
    fig.update_layout(
        title={
            'text': "LLM Training Project Timeline",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        xaxis_title="Project Days",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(stages_timeline))),
            ticktext=[stage['stage'] for stage in stages_timeline],
            autorange="reversed"
        ),
        showlegend=False,
        height=400,
        plot_bgcolor='white',
        margin=dict(l=150, r=50, t=80, b=50)
    )
    
    return fig

def create_data_sources_chart():
    """Create a visualization comparing different data sources for LLM training."""
    
    data_sources = {
        'Source': ['Common Crawl', 'Wikipedia', 'Books', 'Academic Papers', 'News', 'Social Media', 'Forums'],
        'Size_TB': [3500, 0.02, 4, 2, 1.5, 500, 50],
        'Quality_Score': [6, 9, 9, 10, 8, 5, 6],
        'Accessibility': [10, 10, 4, 6, 5, 7, 8],
        'Legal_Complexity': [7, 2, 8, 5, 6, 9, 7],
        'Processing_Difficulty': [9, 3, 4, 5, 4, 8, 7]
    }
    
    df = pd.DataFrame(data_sources)
    
    # Create bubble chart
    fig = px.scatter(
        df,
        x='Quality_Score',
        y='Accessibility',
        size='Size_TB',
        color='Legal_Complexity',
        hover_name='Source',
        size_max=60,
        color_continuous_scale='RdYlBu_r',
        title="Data Sources Comparison: Quality vs Accessibility",
        labels={
            'Quality_Score': 'Data Quality (1-10)',
            'Accessibility': 'Accessibility (1-10)',
            'Size_TB': 'Dataset Size (TB)',
            'Legal_Complexity': 'Legal Complexity (1-10)'
        }
    )
    
    # Add source labels
    for i, row in df.iterrows():
        fig.add_annotation(
            x=row['Quality_Score'],
            y=row['Accessibility'],
            text=row['Source'],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="black",
            ax=20,
            ay=-20,
            font=dict(size=10)
        )
    
    fig.update_layout(height=500)
    
    return fig

def create_data_quality_metrics():
    """Create visualization showing data quality metrics over time."""
    
    # Simulate data quality metrics over processing pipeline
    processing_steps = ['Raw Data', 'Language Filter', 'Deduplication', 'Quality Filter', 'Final Clean']
    
    metrics_data = {
        'Step': processing_steps * 4,
        'Metric': ['Volume (GB)'] * 5 + ['Quality Score'] * 5 + ['Diversity Score'] * 5 + ['Error Rate (%)'] * 5,
        'Value': [
            # Volume
            1000, 800, 600, 450, 400,
            # Quality Score (1-10)
            4.2, 5.8, 6.5, 7.8, 8.2,
            # Diversity Score (1-10)
            6.5, 6.8, 6.9, 7.2, 7.5,
            # Error Rate (%)
            15, 8, 5, 2, 1
        ]
    }
    
    df = pd.DataFrame(metrics_data)
    
    # Create subplots for different metrics
    fig = go.Figure()
    
    metrics = df['Metric'].unique()
    colors = ['#FF6B6B', '#4ECDC4', '#FECA57', '#96CEB4']
    
    for i, metric in enumerate(metrics):
        metric_data = df[df['Metric'] == metric]
        
        fig.add_trace(go.Scatter(
            x=metric_data['Step'],
            y=metric_data['Value'],
            mode='lines+markers',
            name=metric,
            line=dict(width=3, color=colors[i]),
            marker=dict(size=8),
            yaxis=f'y{i+1}' if i > 0 else 'y'
        ))
    
    fig.update_layout(
        title={
            'text': "Data Quality Metrics Through Processing Pipeline",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2C3E50'}
        },
        xaxis_title="Processing Step",
        height=400,
        legend=dict(x=0.7, y=1),
        plot_bgcolor='white'
    )
    
    return fig

def create_model_architecture_comparison():
    """Create comparison chart for different model architectures."""
    
    architectures = {
        'Architecture': ['GPT-3.5', 'GPT-4', 'Claude-2', 'LLaMA-2-7B', 'LLaMA-2-13B', 'PaLM-2'],
        'Parameters_B': [175, 1000, 137, 7, 13, 340],
        'Training_Cost_M': [4.6, 100, 20, 2, 5, 25],
        'Performance_Score': [85, 95, 88, 70, 75, 82],
        'Inference_Speed': [50, 25, 45, 80, 65, 40]
    }
    
    df = pd.DataFrame(architectures)
    
    # Create scatter plot with size representing parameters
    fig = px.scatter(
        df,
        x='Training_Cost_M',
        y='Performance_Score',
        size='Parameters_B',
        color='Inference_Speed',
        hover_name='Architecture',
        size_max=50,
        color_continuous_scale='Viridis',
        title="Model Architecture Comparison: Cost vs Performance",
        labels={
            'Training_Cost_M': 'Training Cost (Million $)',
            'Performance_Score': 'Performance Score (1-100)',
            'Parameters_B': 'Parameters (Billions)',
            'Inference_Speed': 'Inference Speed (tokens/sec)'
        }
    )
    
    # Add architecture labels
    for i, row in df.iterrows():
        fig.add_annotation(
            x=row['Training_Cost_M'],
            y=row['Performance_Score'],
            text=row['Architecture'],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            ax=15,
            ay=-15,
            font=dict(size=9)
        )
    
    fig.update_layout(height=500)
    
    return fig

def create_training_metrics_dashboard():
    """Create a comprehensive training metrics dashboard."""
    
    # Simulate training metrics over time
    epochs = list(range(1, 101))
    
    # Generate realistic training curves
    train_loss = [4.0 * np.exp(-epoch/25) + 0.2 + 0.05*np.random.randn() for epoch in epochs]
    val_loss = [loss + 0.1 + 0.02*epoch/100 + 0.05*np.random.randn() for epoch, loss in zip(epochs, train_loss)]
    learning_rate = [0.0001 * (0.95 ** (epoch//10)) for epoch in epochs]
    gradient_norm = [2.0 + 0.5*np.random.randn() for _ in epochs]
    
    # Create subplot figure
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Loss', 'Learning Rate', 'Gradient Norm', 'Loss Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Training loss
    fig.add_trace(
        go.Scatter(x=epochs, y=train_loss, name='Training Loss', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Learning rate
    fig.add_trace(
        go.Scatter(x=epochs, y=learning_rate, name='Learning Rate', line=dict(color='green')),
        row=1, col=2
    )
    
    # Gradient norm
    fig.add_trace(
        go.Scatter(x=epochs, y=gradient_norm, name='Gradient Norm', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Loss comparison
    fig.add_trace(
        go.Scatter(x=epochs, y=train_loss, name='Train Loss', line=dict(color='blue')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=val_loss, name='Validation Loss', line=dict(color='red')),
        row=2, col=2
    )
    
    fig.update_layout(
        title={
            'text': "Training Metrics Dashboard",
            'x': 0.5,
            'xanchor': 'center'
        },
        height=600,
        showlegend=False
    )
    
    return fig

def create_deployment_architecture_diagram():
    """Create a deployment architecture visualization."""
    
    # Define deployment components
    components = {
        'Component': ['Load Balancer', 'API Gateway', 'Model Server 1', 'Model Server 2', 
                     'Model Server 3', 'Database', 'Monitoring', 'Cache'],
        'X': [2, 2, 1, 2, 3, 4, 4, 3],
        'Y': [4, 3, 2, 2, 2, 2, 3, 1],
        'Type': ['Infrastructure', 'Infrastructure', 'Compute', 'Compute', 
                'Compute', 'Storage', 'Monitoring', 'Cache'],
        'Status': ['Active', 'Active', 'Active', 'Active', 'Active', 'Active', 'Active', 'Active']
    }
    
    df = pd.DataFrame(components)
    
    # Color mapping for component types
    color_map = {
        'Infrastructure': '#FF6B6B',
        'Compute': '#4ECDC4', 
        'Storage': '#FECA57',
        'Monitoring': '#96CEB4',
        'Cache': '#FF9FF3'
    }
    
    fig = go.Figure()
    
    # Add components
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['X']],
            y=[row['Y']],
            mode='markers+text',
            marker=dict(
                size=80,
                color=color_map[row['Type']],
                line=dict(width=2, color='white')
            ),
            text=row['Component'],
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            name=row['Type'],
            showlegend=row['Component'] == df[df['Type'] == row['Type']]['Component'].iloc[0],
            hovertemplate=f"<b>{row['Component']}</b><br>Type: {row['Type']}<br>Status: {row['Status']}<extra></extra>"
        ))
    
    # Add connections (simplified)
    connections = [
        (2, 4, 2, 3),  # Load Balancer to API Gateway
        (2, 3, 1, 2),  # API Gateway to Model Server 1
        (2, 3, 2, 2),  # API Gateway to Model Server 2
        (2, 3, 3, 2),  # API Gateway to Model Server 3
        (2, 2, 4, 2),  # Model Servers to Database
        (3, 2, 3, 1),  # Model Server 3 to Cache
    ]
    
    for x1, y1, x2, y2 in connections:
        fig.add_trace(go.Scatter(
            x=[x1, x2], y=[y1, y2],
            mode='lines',
            line=dict(width=2, color='gray', dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title={
            'text': "Deployment Architecture Overview",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2C3E50'}
        },
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='white',
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_cost_breakdown_chart(compute_cost=1000, storage_cost=200, network_cost=150, 
                               maintenance_cost=100, monitoring_cost=50):
    """Create a cost breakdown visualization."""
    
    categories = ['Compute', 'Storage', 'Network', 'Maintenance', 'Monitoring']
    costs = [compute_cost, storage_cost, network_cost, maintenance_cost, monitoring_cost]
    
    # Create pie chart with cost breakdown
    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=costs,
        textinfo='label+percent+value',
        texttemplate='%{label}<br>$%{value}<br>(%{percent})',
        marker_colors=['#FF6B6B', '#4ECDC4', '#FECA57', '#96CEB4', '#FF9FF3'],
        hole=0.3
    )])
    
    total_cost = sum(costs)
    fig.add_annotation(
        text=f"Total<br>${total_cost:,}",
        x=0.5, y=0.5,
        font_size=16,
        font_color='#2C3E50',
        showarrow=False
    )
    
    fig.update_layout(
        title={
            'text': "Monthly Deployment Cost Breakdown",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2C3E50'}
        },
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_performance_comparison_radar():
    """Create a radar chart comparing different aspects of model performance."""
    
    models = ['GPT-3.5', 'Claude-2', 'LLaMA-2-7B']
    metrics = ['Accuracy', 'Speed', 'Cost Efficiency', 'Safety', 'Versatility', 'Reliability']
    
    # Performance scores (0-10 scale)
    scores = {
        'GPT-3.5': [8.5, 7.0, 6.0, 7.5, 9.0, 8.0],
        'Claude-2': [8.8, 6.5, 7.0, 9.0, 8.5, 8.5],
        'LLaMA-2-7B': [7.0, 9.0, 9.5, 7.0, 7.5, 7.0]
    }
    
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#FECA57']
    
    for i, model in enumerate(models):
        fig.add_trace(go.Scatterpolar(
            r=scores[model],
            theta=metrics,
            fill='toself',
            name=model,
            line=dict(color=colors[i], width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont_size=10
            ),
            angularaxis=dict(
                tickfont_size=12
            )
        ),
        title={
            'text': "Model Performance Comparison",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2C3E50'}
        },
        showlegend=True,
        legend=dict(x=0.8, y=0.8),
        height=500
    )
    
    return fig
