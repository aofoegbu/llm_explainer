import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Model Training", page_icon="🏋️", layout="wide")

st.title("🏋️ Model Training")
st.markdown("### The Core Learning Process for Language Models")

# Overview
st.header("🎯 Overview")
st.markdown("""
Model training is where your architecture learns from the preprocessed data. This stage involves 
optimizing billions of parameters through gradient descent, requiring careful tuning of hyperparameters, 
monitoring of training dynamics, and efficient use of computational resources.
""")

# Training process visualization
st.header("🔄 Training Process Flow")

# Create training flow diagram
training_steps = [
    "Data Loading", "Forward Pass", "Loss Calculation", "Backward Pass", 
    "Gradient Clipping", "Optimizer Step", "Learning Rate Update", "Validation"
]

fig = go.Figure()

# Create circular flow
angles = np.linspace(0, 2*np.pi, len(training_steps), endpoint=False)
x_positions = np.cos(angles) * 2
y_positions = np.sin(angles) * 2

for i, (step, x, y) in enumerate(zip(training_steps, x_positions, y_positions)):
    # Add step boxes
    fig.add_shape(
        type="circle",
        x0=x-0.4, y0=y-0.2, x1=x+0.4, y1=y+0.2,
        fillcolor="lightblue" if i % 2 == 0 else "lightgreen",
        line=dict(color="black", width=1)
    )
    
    fig.add_annotation(
        x=x, y=y,
        text=step,
        showarrow=False,
        font=dict(size=10)
    )
    
    # Add arrows connecting to next step
    next_i = (i + 1) % len(training_steps)
    next_x, next_y = x_positions[next_i], y_positions[next_i]
    
    # Calculate distance and direction
    dx = next_x - x
    dy = next_y - y
    distance = np.sqrt(dx**2 + dy**2)
    
    if distance > 0:  # Avoid division by zero
        # Calculate arrow positions from edge to edge of circles
        arrow_start_x = x + 0.4 * dx / distance
        arrow_start_y = y + 0.2 * dy / distance
        arrow_end_x = next_x - 0.4 * dx / distance
        arrow_end_y = next_y - 0.2 * dy / distance
        
        fig.add_annotation(
            x=arrow_end_x, y=arrow_end_y,
            ax=arrow_start_x, ay=arrow_start_y,
            arrowhead=2, arrowsize=1.5, arrowwidth=2,
            arrowcolor="darkblue",
            showarrow=True
        )

fig.update_layout(
    title="Training Loop Flow",
    showlegend=False,
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-3, 3]),
    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-3, 3]),
    height=500,
    plot_bgcolor='white'
)

st.plotly_chart(fig, use_container_width=True)

# Training stages
st.header("📚 Training Stages")

training_tabs = st.tabs([
    "🚀 Pre-training", 
    "⚙️ Hyperparameters", 
    "📊 Monitoring & Debugging",
    "🔧 Optimization Techniques"
])

with training_tabs[0]:
    st.subheader("🚀 Pre-training Process")
    
    st.markdown("""
    Pre-training is the initial phase where the model learns general language understanding 
    from large amounts of unlabeled text data using self-supervised objectives.
    """)
    
    pretraining_aspects = [
        {
            "name": "Training Objectives",
            "description": "Loss functions that guide the learning process",
            "details": [
                ("Causal Language Modeling (CLM)", "Predict next token given previous tokens", "GPT family"),
                ("Masked Language Modeling (MLM)", "Predict masked tokens using bidirectional context", "BERT family"),
                ("Prefix Language Modeling", "Predict tokens in the suffix given prefix", "GLM, PaLM"),
                ("Span Corruption", "Predict corrupted spans of text", "T5, UL2")
            ]
        },
        {
            "name": "Training Dynamics",
            "description": "How the model learns over time",
            "details": [
                ("Loss Curves", "Training and validation loss should decrease steadily", "Monitor for overfitting"),
                ("Learning Phases", "Initial rapid learning, then slower refinement", "Adjust learning rate accordingly"),
                ("Gradient Norms", "Track gradient magnitudes to detect instabilities", "Use gradient clipping"),
                ("Parameter Updates", "Monitor how much parameters change each step", "Indicates learning rate appropriateness")
            ]
        },
        {
            "name": "Data Efficiency",
            "description": "Maximizing learning from available data",
            "details": [
                ("Curriculum Learning", "Start with easier examples, progress to harder", "Improves convergence"),
                ("Data Ordering", "Random vs sequential vs difficulty-based", "Affects learning dynamics"),
                ("Epoch Management", "Multiple passes through data", "Balance overfitting vs underfitting"),
                ("Dynamic Batching", "Vary batch size during training", "Optimize for hardware utilization")
            ]
        }
    ]
    
    for aspect in pretraining_aspects:
        with st.expander(f"📖 {aspect['name']}"):
            st.markdown(aspect['description'])
            
            for name, description, note in aspect['details']:
                st.markdown(f"**{name}**: {description}")
                st.markdown(f"*{note}*")
                st.markdown("---")

with training_tabs[1]:
    st.subheader("⚙️ Hyperparameter Tuning")
    
    st.markdown("Critical hyperparameters that determine training success:")
    
    hyperparam_tabs = st.tabs(["🎛️ Core Parameters", "📈 Learning Rate", "🔄 Optimization", "💾 Regularization"])
    
    with hyperparam_tabs[0]:
        st.markdown("### 🎛️ Core Training Parameters")
        
        core_params = [
            {
                "parameter": "Batch Size",
                "description": "Number of examples processed together",
                "typical_values": "256-4096 sequences",
                "considerations": [
                    "Larger batches: More stable gradients, better parallelization",
                    "Smaller batches: More updates, better generalization",
                    "Memory constraints limit maximum batch size",
                    "Gradient accumulation can simulate larger batches"
                ],
                "tuning_tips": "Start with largest batch size that fits in memory, then experiment"
            },
            {
                "parameter": "Sequence Length",
                "description": "Maximum length of input sequences",
                "typical_values": "512-4096 tokens",
                "considerations": [
                    "Longer sequences: Better context, more memory usage",
                    "Shorter sequences: Faster training, less memory",
                    "Must match intended use case",
                    "Attention complexity scales quadratically"
                ],
                "tuning_tips": "Match to your downstream task requirements"
            },
            {
                "parameter": "Vocabulary Size",
                "description": "Number of unique tokens in vocabulary",
                "typical_values": "30K-100K tokens",
                "considerations": [
                    "Larger vocab: Better coverage, more parameters",
                    "Smaller vocab: Faster training, more subword splits",
                    "Language-dependent optimal size",
                    "Affects embedding layer size significantly"
                ],
                "tuning_tips": "Balance coverage with parameter efficiency"
            }
        ]
        
        for param in core_params:
            with st.expander(f"🎛️ {param['parameter']}"):
                st.markdown(param['description'])
                st.markdown(f"**Typical Values:** {param['typical_values']}")
                
                st.markdown("**Key Considerations:**")
                for consideration in param['considerations']:
                    st.markdown(f"• {consideration}")
                
                st.info(f"💡 **Tuning Tip:** {param['tuning_tips']}")
    
    with hyperparam_tabs[1]:
        st.markdown("### 📈 Learning Rate Optimization")
        
        # Learning rate scheduler visualization
        st.markdown("#### Learning Rate Schedules")
        
        steps = np.arange(0, 10000, 10)
        warmup_steps = 1000
        max_lr = 1e-4
        
        # Different LR schedules
        schedules = {
            "Constant": np.full_like(steps, max_lr),
            "Linear Warmup + Decay": np.where(
                steps < warmup_steps,
                max_lr * steps / warmup_steps,
                max_lr * np.maximum(0, (1 - (steps - warmup_steps) / np.maximum(1, len(steps) - warmup_steps)))
            ),
            "Cosine Annealing": max_lr * 0.5 * (1 + np.cos(np.pi * steps / len(steps))),
            "Exponential Decay": max_lr * np.exp(-steps / 2000)
        }
        
        fig = go.Figure()
        
        for schedule_name, lr_values in schedules.items():
            fig.add_trace(go.Scatter(
                x=steps,
                y=lr_values,
                mode='lines',
                name=schedule_name,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Learning Rate Schedule Comparison",
            xaxis_title="Training Steps",
            yaxis_title="Learning Rate",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Learning rate guidelines
        lr_guidelines = [
            ("Start Small", "Begin with learning rates around 1e-4 to 1e-5"),
            ("Warmup Period", "Gradually increase LR for first 1-10% of training"),
            ("Peak Learning Rate", "Find maximum stable LR through LR range tests"),
            ("Decay Schedule", "Gradually reduce LR to improve final performance"),
            ("Model Size Scaling", "Larger models often need smaller learning rates"),
            ("Batch Size Scaling", "Scale LR proportionally with batch size")
        ]
        
        for guideline, description in lr_guidelines:
            st.markdown(f"**{guideline}:** {description}")
    
    with hyperparam_tabs[2]:
        st.markdown("### 🔄 Optimization Algorithms")
        
        optimizers = [
            {
                "name": "AdamW",
                "description": "Adam with decoupled weight decay",
                "pros": ["Adaptive learning rates", "Good default choice", "Handles sparse gradients"],
                "cons": ["Higher memory usage", "Can be unstable with large LR"],
                "hyperparams": {
                    "learning_rate": "1e-4 to 1e-5",
                    "beta1": "0.9 (momentum)",
                    "beta2": "0.999 (RMSprop factor)",
                    "weight_decay": "0.01 to 0.1",
                    "eps": "1e-8"
                },
                "best_for": "Most LLM training scenarios"
            },
            {
                "name": "SGD with Momentum",
                "description": "Stochastic gradient descent with momentum",
                "pros": ["Memory efficient", "Well understood", "Good final performance"],
                "cons": ["Requires careful tuning", "Sensitive to learning rate"],
                "hyperparams": {
                    "learning_rate": "0.1 to 1.0",
                    "momentum": "0.9 to 0.99",
                    "weight_decay": "1e-4 to 1e-2"
                },
                "best_for": "When memory is constrained"
            },
            {
                "name": "Lion",
                "description": "Evolved optimizer with sign updates",
                "pros": ["Memory efficient", "Strong empirical results", "Simple updates"],
                "cons": ["Newer, less tested", "Requires different hyperparams"],
                "hyperparams": {
                    "learning_rate": "1e-4 to 3e-4",
                    "beta1": "0.9",
                    "beta2": "0.99",
                    "weight_decay": "0.01 to 0.1"
                },
                "best_for": "Large scale training with memory constraints"
            }
        ]
        
        for optimizer in optimizers:
            with st.expander(f"🔧 {optimizer['name']}"):
                st.markdown(optimizer['description'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Advantages:**")
                    for pro in optimizer['pros']:
                        st.markdown(f"• {pro}")
                with col2:
                    st.markdown("**❌ Limitations:**")
                    for con in optimizer['cons']:
                        st.markdown(f"• {con}")
                
                st.markdown("**🎛️ Key Hyperparameters:**")
                for param, value in optimizer['hyperparams'].items():
                    st.markdown(f"• **{param}:** {value}")
                
                st.info(f"💡 **Best for:** {optimizer['best_for']}")
    
    with hyperparam_tabs[3]:
        st.markdown("### 💾 Regularization Techniques")
        
        regularization_methods = [
            {
                "method": "Weight Decay",
                "description": "L2 penalty on model parameters",
                "implementation": "Add λ||θ||² to loss function",
                "typical_values": "0.01 to 0.1",
                "benefits": ["Prevents overfitting", "Improves generalization"],
                "considerations": ["Don't apply to biases", "Scale with model size"]
            },
            {
                "method": "Dropout",
                "description": "Randomly zero out neurons during training",
                "implementation": "Apply after attention and FFN layers",
                "typical_values": "0.1 to 0.3",
                "benefits": ["Reduces overfitting", "Improves robustness"],
                "considerations": ["Disable during inference", "Modern models use less dropout"]
            },
            {
                "method": "Gradient Clipping",
                "description": "Limit gradient norms to prevent instability",
                "implementation": "Clip gradients to maximum norm",
                "typical_values": "0.5 to 5.0",
                "benefits": ["Prevents gradient explosion", "Stabilizes training"],
                "considerations": ["Monitor gradient norms", "Adjust based on model size"]
            },
            {
                "method": "Label Smoothing",
                "description": "Soften target distributions",
                "implementation": "Replace hard targets with smoothed distributions",
                "typical_values": "0.1 to 0.3",
                "benefits": ["Reduces overconfidence", "Better calibration"],
                "considerations": ["Less common in LLM pretraining", "Useful for fine-tuning"]
            }
        ]
        
        for method in regularization_methods:
            with st.expander(f"🛡️ {method['method']}"):
                st.markdown(method['description'])
                st.markdown(f"**Implementation:** {method['implementation']}")
                st.markdown(f"**Typical Values:** {method['typical_values']}")
                
                st.markdown("**Benefits:**")
                for benefit in method['benefits']:
                    st.markdown(f"• {benefit}")
                
                st.markdown("**Considerations:**")
                for consideration in method['considerations']:
                    st.markdown(f"• {consideration}")

with training_tabs[2]:
    st.subheader("📊 Training Monitoring & Debugging")
    
    monitoring_tabs = st.tabs(["📈 Metrics", "🐛 Common Issues", "🔍 Debugging Tools"])
    
    with monitoring_tabs[0]:
        st.markdown("### 📈 Key Training Metrics")
        
        # Sample training metrics visualization
        steps = np.arange(0, 10000, 100)
        
        # Simulate realistic training curves
        base_loss = 4.0
        train_loss = base_loss * np.exp(-steps/3000) + 0.1 * np.random.normal(0, 0.01, len(steps))
        val_loss = train_loss + 0.05 + 0.02 * (steps/10000) + 0.1 * np.random.normal(0, 0.02, len(steps))
        
        learning_rate = 1e-4 * np.where(steps < 1000, steps/1000, 
                                       np.cos(np.pi * (steps-1000)/(len(steps)-1000))**2)
        
        gradient_norm = 2.0 + 0.5 * np.random.normal(0, 0.1, len(steps))
        
        # Create subplots for different metrics
        metric_fig = go.Figure()
        
        # Loss curves
        metric_fig.add_trace(go.Scatter(
            x=steps, y=train_loss,
            mode='lines', name='Training Loss',
            line=dict(color='blue', width=2)
        ))
        metric_fig.add_trace(go.Scatter(
            x=steps, y=val_loss,
            mode='lines', name='Validation Loss',
            line=dict(color='red', width=2)
        ))
        
        metric_fig.update_layout(
            title="Training Progress",
            xaxis_title="Training Steps",
            yaxis_title="Loss",
            height=400
        )
        
        st.plotly_chart(metric_fig, use_container_width=True)
        
        # Metrics explanation
        metrics_info = [
            ("Training Loss", "Primary objective being optimized", "Should decrease steadily"),
            ("Validation Loss", "Performance on held-out data", "Should track training loss"),
            ("Learning Rate", "Current step size for optimization", "Follows schedule"),
            ("Gradient Norm", "Magnitude of gradients", "Indicates training stability"),
            ("Perplexity", "Exponential of loss", "More interpretable metric"),
            ("Throughput", "Tokens processed per second", "Hardware utilization metric")
        ]
        
        for metric, description, expectation in metrics_info:
            st.markdown(f"**{metric}**: {description} - *{expectation}*")
    
    with monitoring_tabs[1]:
        st.markdown("### 🐛 Common Training Issues")
        
        training_issues = [
            {
                "issue": "Loss Not Decreasing",
                "symptoms": ["Flat loss curves", "No learning progress", "High final loss"],
                "causes": [
                    "Learning rate too low",
                    "Learning rate too high (loss diverges)",
                    "Poor data quality",
                    "Architecture issues",
                    "Optimizer problems"
                ],
                "solutions": [
                    "Learning rate sweep/grid search",
                    "Check data preprocessing pipeline",
                    "Verify model architecture implementation",
                    "Try different optimizers",
                    "Reduce model complexity initially"
                ]
            },
            {
                "issue": "Training Instability",
                "symptoms": ["Loss spikes", "NaN values", "Gradient explosion"],
                "causes": [
                    "Learning rate too high",
                    "Gradient clipping not used",
                    "Numerical precision issues",
                    "Batch size too small",
                    "Poor initialization"
                ],
                "solutions": [
                    "Reduce learning rate",
                    "Enable gradient clipping",
                    "Use mixed precision training carefully",
                    "Increase batch size",
                    "Check parameter initialization"
                ]
            },
            {
                "issue": "Overfitting",
                "symptoms": ["Validation loss increases", "Large train-val gap", "Poor generalization"],
                "causes": [
                    "Model too large for data",
                    "Training too long",
                    "Insufficient regularization",
                    "Data quality issues"
                ],
                "solutions": [
                    "Add regularization (dropout, weight decay)",
                    "Early stopping",
                    "Reduce model size",
                    "Get more training data",
                    "Data augmentation"
                ]
            },
            {
                "issue": "Slow Training",
                "symptoms": ["Low throughput", "Long epoch times", "Poor GPU utilization"],
                "causes": [
                    "Inefficient data loading",
                    "Small batch sizes",
                    "CPU bottlenecks",
                    "Memory issues"
                ],
                "solutions": [
                    "Optimize data pipeline",
                    "Increase batch size",
                    "Use multiple data workers",
                    "Profile and optimize bottlenecks"
                ]
            }
        ]
        
        for issue in training_issues:
            with st.expander(f"⚠️ {issue['issue']}"):
                st.markdown("**Symptoms:**")
                for symptom in issue['symptoms']:
                    st.markdown(f"• {symptom}")
                
                st.markdown("**Common Causes:**")
                for cause in issue['causes']:
                    st.markdown(f"• {cause}")
                
                st.markdown("**Solutions:**")
                for solution in issue['solutions']:
                    st.markdown(f"• {solution}")
    
    with monitoring_tabs[2]:
        st.markdown("### 🔍 Debugging Tools and Techniques")
        
        debugging_tools = [
            {
                "tool": "Learning Rate Finder",
                "description": "Systematically test learning rates to find optimal range",
                "when_to_use": "Before starting training",
                "implementation": "Train for short period with exponentially increasing LR"
            },
            {
                "tool": "Gradient Analysis",
                "description": "Monitor gradient norms, distributions, and flow",
                "when_to_use": "When training is unstable",
                "implementation": "Log gradient statistics per layer"
            },
            {
                "tool": "Activation Analysis",
                "description": "Check activation distributions and dead neurons",
                "when_to_use": "When model isn't learning",
                "implementation": "Hook into forward pass to collect activations"
            },
            {
                "tool": "Loss Landscaping",
                "description": "Visualize loss surface around current parameters",
                "when_to_use": "To understand optimization challenges",
                "implementation": "Sample nearby parameters and compute loss"
            },
            {
                "tool": "Data Inspection",
                "description": "Verify data quality and preprocessing",
                "when_to_use": "When loss plateaus early",
                "implementation": "Sample and manually inspect training examples"
            }
        ]
        
        for tool in debugging_tools:
            with st.expander(f"🔧 {tool['tool']}"):
                st.markdown(tool['description'])
                st.markdown(f"**When to use:** {tool['when_to_use']}")
                st.markdown(f"**Implementation:** {tool['implementation']}")

with training_tabs[3]:
    st.subheader("🔧 Advanced Optimization Techniques")
    
    optimization_tabs = st.tabs(["⚡ Efficiency", "🎯 Stability", "📈 Performance"])
    
    with optimization_tabs[0]:
        st.markdown("### ⚡ Training Efficiency Techniques")
        
        efficiency_techniques = [
            {
                "technique": "Mixed Precision Training",
                "description": "Use FP16 for forward/backward pass, FP32 for parameter updates",
                "benefits": ["2x memory reduction", "1.5-2x speedup", "Maintains numerical stability"],
                "implementation": "Use automatic mixed precision (AMP) frameworks",
                "considerations": ["May need loss scaling", "Some operations stay in FP32"],
                "code_example": """
import torch
from torch.cuda.amp import autocast, GradScaler

# Initialize gradient scaler
scaler = GradScaler()

# Training loop with mixed precision
for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass with autocast
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    # Backward pass with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
"""
            },
            {
                "technique": "Gradient Accumulation",
                "description": "Simulate large batch sizes by accumulating gradients",
                "benefits": ["Large effective batch sizes", "Memory efficient", "Stable training"],
                "implementation": "Accumulate gradients over multiple mini-batches",
                "considerations": ["Scale learning rate appropriately", "Normalize accumulated gradients"],
                "code_example": """
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
"""
            },
            {
                "technique": "Gradient Checkpointing",
                "description": "Trade computation for memory by recomputing activations",
                "benefits": ["Significant memory reduction", "Enable larger models/batches"],
                "implementation": "Recompute forward pass during backward pass",
                "considerations": ["~25% slower training", "Memory-compute tradeoff"],
                "code_example": """
import torch.utils.checkpoint as checkpoint

class TransformerLayer(nn.Module):
    def forward(self, x):
        # Use checkpointing for memory efficiency
        return checkpoint.checkpoint(self._forward_impl, x)
    
    def _forward_impl(self, x):
        # Actual forward computation
        x = self.attention(x)
        x = self.feed_forward(x)
        return x
"""
            }
        ]
        
        for technique in efficiency_techniques:
            with st.expander(f"⚡ {technique['technique']}"):
                st.markdown(technique['description'])
                
                st.markdown("**Benefits:**")
                for benefit in technique['benefits']:
                    st.markdown(f"• {benefit}")
                
                st.markdown("**Considerations:**")
                for consideration in technique['considerations']:
                    st.markdown(f"• {consideration}")
                
                st.code(technique['code_example'], language='python')
    
    with optimization_tabs[1]:
        st.markdown("### 🎯 Training Stability Techniques")
        
        stability_techniques = [
            ("Layer Normalization Placement", "Pre-norm vs post-norm affects training stability"),
            ("Parameter Initialization", "Proper weight initialization prevents gradient issues"),
            ("Warmup Schedules", "Gradual learning rate increase improves stability"),
            ("Loss Scaling", "Prevent underflow in mixed precision training"),
            ("EMA (Exponential Moving Average)", "Stabilize parameter updates"),
            ("Spectral Normalization", "Control Lipschitz constant of layers")
        ]
        
        for technique, description in stability_techniques:
            st.markdown(f"**{technique}**: {description}")
    
    with optimization_tabs[2]:
        st.markdown("### 📈 Performance Optimization")
        
        performance_tips = [
            ("Data Loading", "Use multiple workers and prefetching"),
            ("Batch Size Tuning", "Find optimal batch size for your hardware"),
            ("Sequence Packing", "Pack multiple sequences per batch"),
            ("Dynamic Batching", "Group sequences of similar length"),
            ("Distributed Training", "Scale across multiple GPUs/nodes"),
            ("Compiler Optimizations", "Use torch.compile() for PyTorch 2.0+"),
            ("Memory Management", "Optimize memory allocation patterns"),
            ("Profiling", "Identify and eliminate bottlenecks")
        ]
        
        for tip, description in performance_tips:
            st.markdown(f"**{tip}**: {description}")

# Interactive training simulator
st.header("🎮 Interactive Training Simulator")

sim_tabs = st.tabs(["🎛️ Hyperparameter Explorer", "📊 Loss Predictor", "⚖️ Trade-off Analyzer"])

with sim_tabs[0]:
    st.subheader("🎛️ Hyperparameter Impact Explorer")
    
    # Hyperparameter controls
    col1, col2 = st.columns(2)
    
    with col1:
        sim_lr = st.selectbox("Learning Rate", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
        sim_batch_size = st.selectbox("Batch Size", [64, 128, 256, 512, 1024])
        sim_model_size = st.selectbox("Model Size", ["125M", "350M", "1.3B", "7B"])
    
    with col2:
        sim_optimizer = st.selectbox("Optimizer", ["AdamW", "SGD", "Lion"])
        sim_schedule = st.selectbox("LR Schedule", ["Constant", "Cosine", "Linear"])
        sim_regularization = st.slider("Weight Decay", 0.0, 0.2, 0.01, 0.01)
    
    # Simulate training outcome
    if st.button("Simulate Training"):
        # Simple heuristic for training outcome
        model_factor = {"125M": 1.0, "350M": 0.9, "1.3B": 0.8, "7B": 0.7}[sim_model_size]
        lr_factor = 1.0 - abs(np.log10(sim_lr) + 4) * 0.1  # Optimal around 1e-4
        batch_factor = min(sim_batch_size / 256, 2.0) * 0.9
        
        final_loss = 2.0 * model_factor * lr_factor * batch_factor + np.random.normal(0, 0.1)
        training_time = {"125M": 2, "350M": 8, "1.3B": 24, "7B": 168}[sim_model_size]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Final Loss", f"{final_loss:.2f}")
        with col2:
            st.metric("Training Time", f"{training_time} hours")
        with col3:
            st.metric("Estimated Cost", f"${training_time * 3:.0f}")

with sim_tabs[1]:
    st.subheader("📊 Training Loss Predictor")
    
    # Parameters for loss prediction
    pred_col1, pred_col2 = st.columns(2)
    
    with pred_col1:
        target_steps = st.number_input("Training Steps", 1000, 100000, 10000, 1000)
        data_quality = st.slider("Data Quality (1-10)", 1, 10, 8)
    
    with pred_col2:
        starting_loss = st.number_input("Initial Loss", 1.0, 10.0, 4.0, 0.1)
        learning_efficiency = st.slider("Learning Efficiency", 0.5, 2.0, 1.0, 0.1)
    
    # Generate prediction curve
    steps = np.linspace(0, target_steps, 100)
    predicted_loss = starting_loss * np.exp(-steps * learning_efficiency / target_steps) * (data_quality / 10)
    
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(
        x=steps, y=predicted_loss,
        mode='lines', name='Predicted Loss',
        line=dict(color='green', width=3)
    ))
    
    pred_fig.update_layout(
        title="Predicted Training Loss Curve",
        xaxis_title="Training Steps",
        yaxis_title="Loss",
        height=400
    )
    
    st.plotly_chart(pred_fig, use_container_width=True)

with sim_tabs[2]:
    st.subheader("⚖️ Resource Trade-off Analyzer")
    
    # Trade-off analysis
    st.markdown("Analyze trade-offs between different training configurations:")
    
    configs = [
        {"name": "Fast Training", "time": 1, "cost": 1, "quality": 0.7},
        {"name": "Balanced", "time": 3, "cost": 2, "quality": 0.85},
        {"name": "High Quality", "time": 10, "cost": 8, "quality": 0.95},
        {"name": "Research Grade", "time": 30, "cost": 25, "quality": 0.98}
    ]
    
    config_df = pd.DataFrame(configs)
    
    # Scatter plot of trade-offs
    trade_fig = px.scatter(
        config_df, x='time', y='cost', size='quality', hover_name='name',
        title="Training Configuration Trade-offs",
        labels={'time': 'Training Time (relative)', 'cost': 'Cost (relative)', 'quality': 'Quality'}
    )
    
    st.plotly_chart(trade_fig, use_container_width=True)

# Training checklist
st.header("✅ Pre-Training Checklist")

checklist_items = [
    "✅ Data preprocessing pipeline validated and optimized",
    "✅ Model architecture implemented and tested",
    "✅ Training infrastructure set up and tested",
    "✅ Hyperparameters tuned through systematic search",
    "✅ Monitoring and logging systems configured",
    "✅ Checkpoint saving and loading implemented",
    "✅ Distributed training setup if using multiple GPUs",
    "✅ Mixed precision training configured",
    "✅ Gradient clipping and stability measures in place",
    "✅ Validation dataset prepared and evaluation metrics defined",
    "✅ Early stopping criteria established",
    "✅ Resource budgets and timelines confirmed"
]

for item in checklist_items:
    st.markdown(item)

# Best practices summary
st.header("🎯 Training Best Practices")

best_practices = [
    "Start small: Begin with a smaller model to validate your pipeline",
    "Monitor closely: Watch loss curves, gradient norms, and hardware utilization",
    "Save frequently: Checkpoint regularly to recover from failures",
    "Log everything: Comprehensive logging aids debugging and reproduction",
    "Validate early: Check on validation set to catch overfitting",
    "Profile performance: Identify and eliminate training bottlenecks",
    "Plan for failure: Have recovery strategies for common issues",
    "Document decisions: Keep detailed records of hyperparameter choices",
    "Test incrementally: Validate each optimization before combining",
    "Measure what matters: Focus on metrics relevant to your end goal"
]

for practice in best_practices:
    st.markdown(f"• {practice}")
