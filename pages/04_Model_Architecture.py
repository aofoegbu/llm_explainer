import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Model Architecture", page_icon="🏗️", layout="wide")

st.title("🏗️ Model Architecture Design")
st.markdown("### Building the Foundation of Your Language Model")

# Overview
st.header("🎯 Overview")
st.markdown("""
Model architecture defines the neural network structure that will learn from your training data. 
The choice of architecture significantly impacts model capabilities, training efficiency, and final performance. 
This section covers popular architectures, design decisions, and implementation considerations.
""")

# Architecture comparison
st.header("🏛️ Popular LLM Architectures")

architecture_tabs = st.tabs([
    "🤖 Transformer-based", 
    "📊 Architecture Comparison", 
    "🔧 Design Components",
    "📏 Scaling Decisions"
])

with architecture_tabs[0]:
    st.subheader("🤖 Transformer-based Architectures")
    
    architectures = [
        {
            "name": "GPT (Generative Pre-trained Transformer)",
            "type": "Decoder-only",
            "description": "Autoregressive language model for text generation",
            "key_features": [
                "Causal self-attention (unidirectional)",
                "Layer normalization",
                "Positional encoding",
                "Feed-forward networks"
            ],
            "use_cases": [
                "Text generation",
                "Completion tasks", 
                "Creative writing",
                "Code generation"
            ],
            "variants": ["GPT-1", "GPT-2", "GPT-3", "GPT-4", "ChatGPT"],
            "pros": [
                "Excellent generation quality",
                "Scales well with size",
                "Simple architecture",
                "Fast inference"
            ],
            "cons": [
                "Unidirectional context",
                "Less suitable for understanding tasks",
                "Cannot see future tokens"
            ]
        },
        {
            "name": "BERT (Bidirectional Encoder Representations)",
            "type": "Encoder-only", 
            "description": "Bidirectional model for understanding tasks",
            "key_features": [
                "Bidirectional self-attention",
                "Masked language modeling",
                "Next sentence prediction",
                "Position embeddings"
            ],
            "use_cases": [
                "Text classification",
                "Question answering",
                "Named entity recognition",
                "Sentiment analysis"
            ],
            "variants": ["BERT-Base", "BERT-Large", "RoBERTa", "DeBERTa"],
            "pros": [
                "Bidirectional context",
                "Excellent for understanding",
                "Good for classification",
                "Pretrained models available"
            ],
            "cons": [
                "Cannot generate text",
                "Requires task-specific heads",
                "Fixed sequence length"
            ]
        },
        {
            "name": "T5 (Text-to-Text Transfer Transformer)",
            "type": "Encoder-Decoder",
            "description": "Unified framework treating all tasks as text-to-text",
            "key_features": [
                "Full encoder-decoder architecture",
                "Relative position encoding",
                "Text-to-text framework",
                "Multi-task training"
            ],
            "use_cases": [
                "Machine translation",
                "Summarization",
                "Question answering",
                "Multi-task learning"
            ],
            "variants": ["T5-Small", "T5-Base", "T5-Large", "T5-3B", "T5-11B"],
            "pros": [
                "Handles any text task",
                "Bidirectional encoder",
                "Flexible output length",
                "Strong transfer learning"
            ],
            "cons": [
                "More complex architecture",
                "Slower inference",
                "Requires more memory"
            ]
        }
    ]
    
    for arch in architectures:
        with st.expander(f"🏗️ {arch['name']} ({arch['type']})"):
            st.markdown(arch['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔑 Key Features:**")
                for feature in arch['key_features']:
                    st.markdown(f"• {feature}")
                
                st.markdown("**🎯 Use Cases:**")
                for use_case in arch['use_cases']:
                    st.markdown(f"• {use_case}")
            
            with col2:
                st.markdown("**✅ Advantages:**")
                for pro in arch['pros']:
                    st.markdown(f"• {pro}")
                
                st.markdown("**❌ Limitations:**")
                for con in arch['cons']:
                    st.markdown(f"• {con}")
            
            st.markdown(f"**🔄 Popular Variants:** {', '.join(arch['variants'])}")

with architecture_tabs[1]:
    st.subheader("📊 Architecture Comparison Matrix")
    
    # Create comparison data
    comparison_data = {
        'Architecture': ['GPT', 'BERT', 'T5', 'PaLM', 'LLaMA'],
        'Type': ['Decoder-only', 'Encoder-only', 'Encoder-Decoder', 'Decoder-only', 'Decoder-only'],
        'Generation Quality': [9, 3, 8, 10, 9],
        'Understanding Quality': [7, 10, 9, 8, 8],
        'Training Efficiency': [8, 9, 6, 7, 9],
        'Inference Speed': [9, 8, 6, 7, 8],
        'Memory Efficiency': [7, 8, 5, 6, 9],
        'Scalability': [9, 7, 8, 10, 8]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.dataframe(df_comparison.set_index('Architecture'), use_container_width=True)
    
    # Radar chart comparison
    st.markdown("### 📈 Performance Radar Chart")
    
    selected_archs = st.multiselect(
        "Select architectures to compare:",
        ['GPT', 'BERT', 'T5', 'PaLM', 'LLaMA'],
        default=['GPT', 'BERT', 'T5']
    )
    
    if selected_archs:
        metrics = ['Generation Quality', 'Understanding Quality', 'Training Efficiency', 
                  'Inference Speed', 'Memory Efficiency', 'Scalability']
        
        fig = go.Figure()
        
        for arch in selected_archs:
            arch_data = df_comparison[df_comparison['Architecture'] == arch]
            values = [arch_data[metric].iloc[0] for metric in metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=arch,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title="Architecture Performance Comparison",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

with architecture_tabs[2]:
    st.subheader("🔧 Core Architecture Components")
    
    component_tabs = st.tabs([
        "🧠 Attention Mechanisms",
        "📊 Feed-Forward Networks", 
        "📍 Position Encoding",
        "🔄 Normalization"
    ])
    
    with component_tabs[0]:
        st.markdown("### 🧠 Attention Mechanisms")
        
        attention_types = [
            {
                "name": "Multi-Head Self-Attention",
                "description": "Core attention mechanism allowing tokens to attend to each other",
                "formula": "Attention(Q,K,V) = softmax(QK^T/√d_k)V",
                "parameters": "d_model, num_heads, dropout",
                "variations": ["Scaled dot-product", "Multi-query", "Grouped-query"]
            },
            {
                "name": "Causal (Masked) Attention",
                "description": "Prevents attention to future tokens in autoregressive models",
                "formula": "Same as self-attention but with causal mask",
                "parameters": "mask_type, sequence_length",
                "variations": ["Lower triangular mask", "Sliding window"]
            },
            {
                "name": "Cross-Attention",
                "description": "Allows decoder to attend to encoder representations",
                "formula": "Q from decoder, K,V from encoder",
                "parameters": "encoder_dim, decoder_dim",
                "variations": ["Standard cross", "Memory-efficient"]
            }
        ]
        
        for attention in attention_types:
            with st.expander(f"⚡ {attention['name']}"):
                st.markdown(attention['description'])
                st.latex(attention['formula'])
                st.markdown(f"**Key Parameters:** {attention['parameters']}")
                st.markdown(f"**Variations:** {', '.join(attention['variations'])}")
        
        # Interactive attention visualization
        st.markdown("### 🎮 Interactive Attention Heatmap")
        
        sentence = st.text_input("Enter a sentence:", value="The cat sat on the mat")
        tokens = sentence.split()
        
        if len(tokens) > 1:
            # Simulate attention weights (random for demo)
            np.random.seed(42)
            attention_weights = np.random.rand(len(tokens), len(tokens))
            attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
            
            fig = px.imshow(
                attention_weights,
                labels=dict(x="Key Tokens", y="Query Tokens", color="Attention Weight"),
                x=tokens,
                y=tokens,
                title="Attention Weight Heatmap"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with component_tabs[1]:
        st.markdown("### 📊 Feed-Forward Networks")
        
        st.markdown("""
        Feed-forward networks (FFNs) process each position independently, providing 
        non-linear transformations between attention layers.
        """)
        
        ffn_details = [
            ("Architecture", "Two linear transformations with activation function"),
            ("Formula", "FFN(x) = max(0, xW₁ + b₁)W₂ + b₂"),
            ("Hidden Size", "Typically 4x the model dimension"),
            ("Activation", "ReLU, GELU, SwiGLU variants"),
            ("Parameters", "~2/3 of total model parameters"),
            ("Purpose", "Provide model expressiveness and non-linearity")
        ]
        
        for detail, explanation in ffn_details:
            st.markdown(f"**{detail}:** {explanation}")
        
        # FFN size calculator
        st.markdown("### 🧮 FFN Parameter Calculator")
        
        d_model = st.slider("Model Dimension (d_model)", 256, 8192, 768, 256)
        ffn_multiplier = st.slider("FFN Multiplier", 2, 8, 4, 1)
        
        ffn_hidden = d_model * ffn_multiplier
        ffn_params = d_model * ffn_hidden * 2  # W1 and W2
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("FFN Hidden Size", f"{ffn_hidden:,}")
        with col2:
            st.metric("FFN Parameters", f"{ffn_params:,}")
        with col3:
            st.metric("Memory (FP16)", f"{ffn_params * 2 / 1e6:.1f}MB")
    
    with component_tabs[2]:
        st.markdown("### 📍 Position Encoding")
        
        position_types = [
            {
                "name": "Sinusoidal Position Encoding",
                "description": "Fixed trigonometric functions encoding absolute positions",
                "formula": "PE(pos,2i) = sin(pos/10000^(2i/d_model))",
                "pros": ["No learned parameters", "Extrapolates to longer sequences"],
                "cons": ["Fixed patterns", "Limited expressiveness"],
                "used_in": ["Original Transformer", "GPT-1"]
            },
            {
                "name": "Learned Position Embeddings",
                "description": "Trainable embeddings for each position",
                "formula": "PE = Embedding_table[position]",
                "pros": ["More flexible", "Task-adaptive"],
                "cons": ["Fixed max length", "More parameters"],
                "used_in": ["BERT", "GPT-2"]
            },
            {
                "name": "Relative Position Encoding",
                "description": "Encodes relative distances between tokens",
                "formula": "Attention weights modified by relative distances",
                "pros": ["Length generalization", "Translation invariant"],
                "cons": ["More complex", "Computational overhead"],
                "used_in": ["T5", "DeBERTa"]
            },
            {
                "name": "Rotary Position Embedding (RoPE)",
                "description": "Rotates queries and keys based on position",
                "formula": "Complex rotation in embedding space",
                "pros": ["Excellent extrapolation", "Relative position awareness"],
                "cons": ["Complex implementation", "Limited to certain dimensions"],
                "used_in": ["GPT-J", "LLaMA", "PaLM"]
            }
        ]
        
        for pos_type in position_types:
            with st.expander(f"📐 {pos_type['name']}"):
                st.markdown(pos_type['description'])
                st.markdown(f"**Formula:** {pos_type['formula']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Advantages:**")
                    for pro in pos_type['pros']:
                        st.markdown(f"• {pro}")
                with col2:
                    st.markdown("**❌ Limitations:**")
                    for con in pos_type['cons']:
                        st.markdown(f"• {con}")
                
                st.markdown(f"**🏗️ Used in:** {pos_type['used_in']}")
    
    with component_tabs[3]:
        st.markdown("### 🔄 Normalization Techniques")
        
        norm_techniques = [
            {
                "name": "Layer Normalization",
                "description": "Normalizes across the feature dimension",
                "placement": "Pre-norm (before attention) or Post-norm (after)",
                "benefits": ["Stabilizes training", "Enables deeper networks"],
                "variants": ["Standard LayerNorm", "RMSNorm", "DeepNorm"]
            },
            {
                "name": "Pre-Norm vs Post-Norm",
                "description": "Different placement strategies for normalization",
                "placement": "Pre-norm generally preferred for training stability",
                "benefits": ["Pre-norm: Better gradient flow", "Post-norm: Better representation"],
                "variants": ["Pre-LN", "Post-LN", "Sandwich-LN"]
            }
        ]
        
        for norm in norm_techniques:
            with st.expander(f"🔧 {norm['name']}"):
                st.markdown(norm['description'])
                st.markdown(f"**Placement:** {norm['placement']}")
                st.markdown(f"**Benefits:** {norm['benefits']}")
                st.markdown(f"**Variants:** {', '.join(norm['variants'])}")

with architecture_tabs[3]:
    st.subheader("📏 Scaling Decisions")
    
    scaling_tabs = st.tabs(["📊 Model Scaling", "💰 Cost Analysis", "🎯 Performance Prediction"])
    
    with scaling_tabs[0]:
        st.markdown("### 📊 Model Scaling Laws")
        
        st.markdown("""
        Scaling laws help predict model performance based on model size, dataset size, and compute budget.
        Understanding these relationships is crucial for efficient resource allocation.
        """)
        
        # Scaling parameters
        st.markdown("### 🎛️ Scaling Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Size Scaling:**")
            
            # Model size options
            model_sizes = {
                "Small": {"params": 125e6, "layers": 12, "d_model": 768, "heads": 12},
                "Medium": {"params": 350e6, "layers": 24, "d_model": 1024, "heads": 16},
                "Large": {"params": 1.3e9, "layers": 24, "d_model": 2048, "heads": 32},
                "XL": {"params": 7e9, "layers": 32, "d_model": 4096, "heads": 32},
                "XXL": {"params": 13e9, "layers": 40, "d_model": 5120, "heads": 40},
                "XXXL": {"params": 70e9, "layers": 80, "d_model": 8192, "heads": 64}
            }
            
            selected_size = st.selectbox("Select Model Size:", list(model_sizes.keys()))
            size_config = model_sizes[selected_size]
            
            st.metric("Parameters", f"{size_config['params']/1e9:.1f}B")
            st.metric("Layers", size_config['layers'])
            st.metric("Model Dimension", size_config['d_model'])
            st.metric("Attention Heads", size_config['heads'])
        
        with col2:
            st.markdown("**Training Scale Estimation:**")
            
            # Estimate training requirements
            params = size_config['params']
            
            # Chinchilla-optimal data scaling
            optimal_tokens = params * 20  # Chinchilla ratio
            
            # Compute estimates (rough approximations)
            flops_per_token = params * 6  # Forward + backward pass
            total_flops = optimal_tokens * flops_per_token
            
            # Hardware estimates
            gpu_flops = 312e12  # A100 BF16 peak FLOPS
            training_hours = total_flops / gpu_flops / 3600
            
            st.metric("Optimal Training Tokens", f"{optimal_tokens/1e9:.0f}B")
            st.metric("Total FLOPs", f"{total_flops/1e21:.1f} ZFLOPs")
            st.metric("Training Time (A100)", f"{training_hours/24:.0f} days")
            st.metric("Estimated Cost", f"${training_hours*3:.0f}K")
        
        # Scaling law visualization
        st.markdown("### 📈 Scaling Law Visualization")
        
        # Generate scaling data
        param_range = np.logspace(6, 11, 50)  # 1M to 100B parameters
        
        # Kaplan scaling law approximation: Loss ~ N^(-0.076)
        loss_scaling = 2.0 + 0.5 * (param_range / 1e9) ** (-0.076)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=param_range / 1e9,
            y=loss_scaling,
            mode='lines',
            name='Predicted Loss',
            line=dict(width=3, color='blue')
        ))
        
        # Add model size markers
        for name, config in model_sizes.items():
            loss_est = 2.0 + 0.5 * (config['params'] / 1e9) ** (-0.076)
            fig.add_trace(go.Scatter(
                x=[config['params'] / 1e9],
                y=[loss_est],
                mode='markers',
                name=name,
                marker=dict(size=10)
            ))
        
        fig.update_layout(
            title="Model Scaling Law: Loss vs Parameters",
            xaxis_title="Model Size (Billions of Parameters)",
            yaxis_title="Training Loss",
            xaxis_type="log",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with scaling_tabs[1]:
        st.markdown("### 💰 Training Cost Analysis")
        
        # Cost calculator
        st.markdown("#### 🧮 Training Cost Calculator")
        
        calc_col1, calc_col2 = st.columns(2)
        
        with calc_col1:
            model_params = st.number_input("Model Parameters (Billions)", 0.1, 200.0, 7.0, 0.1)
            training_tokens = st.number_input("Training Tokens (Billions)", 10, 5000, 1000, 10)
            gpu_type = st.selectbox("GPU Type", ["A100 (80GB)", "H100 (80GB)", "V100 (32GB)"])
            gpu_cost_per_hour = st.number_input("GPU Cost per Hour ($)", 1.0, 10.0, 3.0, 0.1)
        
        with calc_col2:
            # Calculate requirements
            params_total = model_params * 1e9
            
            # Memory requirements (rough estimates)
            model_memory = params_total * 4 / 1e9  # 4 bytes per param (FP32)
            optimizer_memory = model_memory * 2  # Adam states
            activation_memory = model_memory * 0.5  # Activations
            total_memory = model_memory + optimizer_memory + activation_memory
            
            # GPU memory capacities
            gpu_memory = {"A100 (80GB)": 80, "H100 (80GB)": 80, "V100 (32GB)": 32}[gpu_type]
            gpus_needed = max(1, int(np.ceil(total_memory / gpu_memory)))
            
            # Training time estimates
            gpu_flops = {"A100 (80GB)": 312e12, "H100 (80GB)": 600e12, "V100 (32GB)": 125e12}[gpu_type]
            flops_per_token = params_total * 6
            total_flops = training_tokens * 1e9 * flops_per_token
            training_hours = total_flops / (gpu_flops * gpus_needed) / 3600
            
            total_cost = training_hours * gpus_needed * gpu_cost_per_hour
            
            st.metric("GPUs Needed", gpus_needed)
            st.metric("Training Time", f"{training_hours/24:.1f} days")
            st.metric("Total Cost", f"${total_cost/1000:.1f}K")
            st.metric("Cost per Parameter", f"${total_cost/params_total*1e6:.2f}/M params")
    
    with scaling_tabs[2]:
        st.markdown("### 🎯 Performance Prediction")
        
        st.markdown("""
        Use these tools to estimate model performance based on scaling laws and 
        empirical observations from similar models.
        """)
        
        # Performance predictor
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            target_params = st.slider("Target Model Size (B params)", 0.1, 100.0, 7.0, 0.1)
            target_data = st.slider("Training Data (B tokens)", 50, 3000, 1000, 50)
            
            # Baseline performance (rough estimates)
            base_loss = 3.0
            param_scaling = (target_params / 1.0) ** (-0.076)
            data_scaling = (target_data / 100.0) ** (-0.026)
            
            predicted_loss = base_loss * param_scaling * data_scaling
            
            # Convert to approximate perplexity and other metrics
            perplexity = np.exp(predicted_loss)
            
        with pred_col2:
            st.metric("Predicted Training Loss", f"{predicted_loss:.2f}")
            st.metric("Predicted Perplexity", f"{perplexity:.1f}")
            
            # Rough capability estimates
            if target_params < 1:
                capability = "Basic text completion"
            elif target_params < 10:
                capability = "Good text generation"
            elif target_params < 50:
                capability = "Strong reasoning abilities"
            else:
                capability = "Advanced reasoning and coding"
            
            st.info(f"**Expected Capabilities:** {capability}")

# Architecture design tool
st.header("🛠️ Interactive Architecture Designer")

design_tabs = st.tabs(["🎨 Design Your Model", "📊 Parameter Calculator", "🔍 Architecture Validator"])

with design_tabs[0]:
    st.subheader("🎨 Custom Architecture Designer")
    
    design_col1, design_col2 = st.columns(2)
    
    with design_col1:
        st.markdown("**📐 Architecture Parameters**")
        
        arch_type = st.selectbox("Architecture Type", 
                                ["Decoder-only (GPT-like)", "Encoder-only (BERT-like)", "Encoder-Decoder (T5-like)"])
        
        num_layers = st.slider("Number of Layers", 1, 100, 24, 1)
        d_model = st.slider("Model Dimension", 256, 8192, 768, 64)
        num_heads = st.selectbox("Number of Attention Heads", [4, 8, 12, 16, 20, 24, 32, 40, 48, 64])
        ffn_multiplier = st.slider("FFN Multiplier", 1, 8, 4, 1)
        vocab_size = st.number_input("Vocabulary Size", 1000, 200000, 50000, 1000)
        max_seq_length = st.selectbox("Max Sequence Length", [512, 1024, 2048, 4096, 8192, 16384])
    
    with design_col2:
        st.markdown("**📊 Model Statistics**")
        
        # Calculate parameters
        # Embedding parameters
        embed_params = vocab_size * d_model
        pos_embed_params = max_seq_length * d_model if arch_type != "Decoder-only (GPT-like)" else 0
        
        # Transformer layer parameters
        # Attention: Q, K, V projections + output projection
        attention_params = 4 * d_model * d_model
        # FFN: two linear layers
        ffn_hidden = d_model * ffn_multiplier
        ffn_params = d_model * ffn_hidden + ffn_hidden * d_model
        # Layer norm (2 per layer)
        ln_params = 2 * d_model
        
        layer_params = attention_params + ffn_params + ln_params
        total_layer_params = layer_params * num_layers
        
        # Output head
        output_params = d_model * vocab_size
        
        # Total parameters
        total_params = embed_params + pos_embed_params + total_layer_params + output_params
        
        st.metric("Total Parameters", f"{total_params/1e6:.1f}M")
        st.metric("Embedding Parameters", f"{embed_params/1e6:.1f}M")
        st.metric("Transformer Parameters", f"{total_layer_params/1e6:.1f}M")
        st.metric("Parameters per Layer", f"{layer_params/1e6:.1f}M")
        
        # Memory estimates
        model_memory_gb = total_params * 4 / 1e9  # FP32
        training_memory_gb = model_memory_gb * 4  # Including optimizer states and activations
        
        st.metric("Model Memory (FP32)", f"{model_memory_gb:.1f}GB")
        st.metric("Training Memory Est.", f"{training_memory_gb:.1f}GB")

with design_tabs[1]:
    st.subheader("📊 Detailed Parameter Breakdown")
    
    # Create parameter breakdown
    param_breakdown = {
        'Component': [
            'Token Embeddings',
            'Position Embeddings', 
            'Attention Layers',
            'Feed-Forward Networks',
            'Layer Normalization',
            'Output Head'
        ],
        'Parameters (M)': [
            embed_params / 1e6,
            pos_embed_params / 1e6,
            (attention_params * num_layers) / 1e6,
            (ffn_params * num_layers) / 1e6,
            (ln_params * num_layers) / 1e6,
            output_params / 1e6
        ],
        'Percentage': [
            embed_params / total_params * 100,
            pos_embed_params / total_params * 100,
            (attention_params * num_layers) / total_params * 100,
            (ffn_params * num_layers) / total_params * 100,
            (ln_params * num_layers) / total_params * 100,
            output_params / total_params * 100
        ]
    }
    
    df_breakdown = pd.DataFrame(param_breakdown)
    st.dataframe(df_breakdown, use_container_width=True)
    
    # Pie chart of parameter distribution
    fig = px.pie(df_breakdown, values='Parameters (M)', names='Component',
                title="Parameter Distribution by Component")
    st.plotly_chart(fig, use_container_width=True)

with design_tabs[2]:
    st.subheader("🔍 Architecture Validation")
    
    # Validation checks
    validations = []
    
    # Check head dimension
    head_dim = d_model // num_heads
    if head_dim < 32:
        validations.append(("⚠️", "Warning", f"Head dimension ({head_dim}) is quite small. Consider reducing number of heads."))
    elif head_dim > 128:
        validations.append(("⚠️", "Warning", f"Head dimension ({head_dim}) is large. Consider increasing number of heads."))
    else:
        validations.append(("✅", "Good", f"Head dimension ({head_dim}) is in optimal range (32-128)."))
    
    # Check model dimension
    if d_model % num_heads != 0:
        validations.append(("❌", "Error", "Model dimension must be divisible by number of heads."))
    else:
        validations.append(("✅", "Good", "Model dimension is divisible by number of heads."))
    
    # Check FFN size
    if ffn_multiplier < 2:
        validations.append(("⚠️", "Warning", "FFN multiplier is quite small. May limit model expressiveness."))
    elif ffn_multiplier > 6:
        validations.append(("⚠️", "Warning", "Large FFN multiplier increases parameter count significantly."))
    else:
        validations.append(("✅", "Good", f"FFN multiplier ({ffn_multiplier}) is in typical range."))
    
    # Check parameter efficiency
    if total_params > 10e9:
        validations.append(("💰", "Cost", f"Large model ({total_params/1e9:.1f}B params) will require significant compute resources."))
    
    # Display validations
    for icon, level, message in validations:
        if level == "Error":
            st.error(f"{icon} {message}")
        elif level == "Warning":
            st.warning(f"{icon} {message}")
        elif level == "Cost":
            st.info(f"{icon} {message}")
        else:
            st.success(f"{icon} {message}")
    
    # Architecture summary
    st.markdown("### 📋 Architecture Summary")
    
    summary_code = f"""
# Model Configuration
model_config = {{
    'architecture_type': '{arch_type}',
    'num_layers': {num_layers},
    'd_model': {d_model},
    'num_heads': {num_heads},
    'head_dim': {head_dim},
    'ffn_hidden_dim': {ffn_hidden},
    'vocab_size': {vocab_size},
    'max_seq_length': {max_seq_length},
    'total_parameters': {total_params:,},
    'parameter_breakdown': {{
        'embeddings': {embed_params:,},
        'transformer_layers': {total_layer_params:,},
        'output_head': {output_params:,}
    }}
}}
"""
    
    st.code(summary_code, language='python')
