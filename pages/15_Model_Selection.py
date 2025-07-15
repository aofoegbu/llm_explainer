import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Model Selection & Architecture", page_icon="🏗️", layout="wide")

st.title("🏗️ Model Selection & Architecture")
st.markdown("### Choosing the Right LLM Architecture for Your Use Case")

# Overview
st.header("🎯 Overview")
st.markdown("""
Model selection for LLMs involves choosing the right architecture, size, and configuration based on 
your specific requirements, constraints, and use cases. This decision impacts performance, cost, 
latency, and deployment complexity significantly.
""")

# Core concepts
st.header("🧠 Core Concepts")

concept_tabs = st.tabs([
    "🏗️ Architecture Types",
    "📏 Model Scaling", 
    "⚖️ Trade-off Analysis",
    "🎯 Selection Framework"
])

with concept_tabs[0]:
    st.subheader("🏗️ LLM Architecture Types")
    
    st.markdown("""
    Different LLM architectures have unique characteristics that make them suitable 
    for different types of tasks and deployment scenarios.
    """)
    
    architectures = [
        {
            "name": "Transformer (Decoder-Only)",
            "description": "Auto-regressive models that generate text token by token",
            "examples": ["GPT-4", "ChatGPT", "Claude", "LLaMA", "PaLM"],
            "strengths": [
                "Excellent text generation capabilities",
                "Strong few-shot learning performance",
                "Versatile across many NLP tasks",
                "Scales well with size and data"
            ],
            "weaknesses": [
                "High computational requirements",
                "Sequential generation (slow inference)",
                "Limited bidirectional context",
                "Memory intensive for long sequences"
            ],
            "best_for": [
                "Conversational AI",
                "Content generation",
                "Creative writing",
                "General-purpose language tasks"
            ],
            "architecture_details": """
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff=4*d_model)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None):
        # Token and position embeddings
        x = self.embedding(x) + self.pos_encoding(x)
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        # Final normalization and projection
        x = self.norm(x)
        logits = self.output_proj(x)
        
        return logits

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out = self.self_attn(x, x, x, mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x
"""
        },
        {
            "name": "Encoder-Decoder",
            "description": "Models with separate encoder and decoder components",
            "examples": ["T5", "BART", "UL2", "Flan-T5"],
            "strengths": [
                "Excellent for sequence-to-sequence tasks",
                "Bidirectional encoder context",
                "Flexible input-output lengths",
                "Good for structured generation tasks"
            ],
            "weaknesses": [
                "More complex architecture",
                "Higher memory requirements",
                "Potentially slower inference",
                "Less effective for pure generation"
            ],
            "best_for": [
                "Translation tasks",
                "Summarization",
                "Question answering",
                "Text-to-text transformation"
            ],
            "architecture_details": """
class EncoderDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, n_heads, n_layers)
        self.decoder = TransformerDecoder(vocab_size, d_model, n_heads, n_layers)
        
    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        # Encode input sequence
        encoder_out = self.encoder(src_ids, mask=src_mask)
        
        # Decode with cross-attention to encoder output
        decoder_out = self.decoder(
            tgt_ids, 
            encoder_out=encoder_out,
            self_mask=tgt_mask,
            cross_mask=src_mask
        )
        
        return decoder_out

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, mask=None):
        x = self.embedding(x) + self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask=mask)
            
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, 4*d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Bidirectional self-attention
        attn_out = self.self_attn(x, x, x, mask=mask)
        x = self.norm1(x + attn_out)
        
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x
"""
        },
        {
            "name": "Mixture of Experts (MoE)",
            "description": "Sparse models that activate only a subset of parameters",
            "examples": ["Switch Transformer", "GLaM", "PaLM-2", "LaMDA"],
            "strengths": [
                "Efficient scaling to very large sizes",
                "Lower inference cost per parameter",
                "Specialized expert modules",
                "Better parameter utilization"
            ],
            "weaknesses": [
                "Complex training dynamics",
                "Load balancing challenges",
                "Communication overhead in distributed setting",
                "Harder to optimize and debug"
            ],
            "best_for": [
                "Large-scale deployment",
                "Multi-domain applications",
                "Efficient scaling",
                "Specialized task requirements"
            ],
            "architecture_details": """
class MixtureOfExperts(nn.Module):
    def __init__(self, d_model, n_experts, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        # Router network to select experts
        self.router = nn.Linear(d_model, n_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            FeedForward(d_model, 4*d_model)
            for _ in range(n_experts)
        ])
        
        # Gating mechanism
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Compute router scores
        router_logits = self.router(x)  # [B, L, n_experts]
        router_probs = self.softmax(router_logits)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        outputs = torch.zeros_like(x)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, :, i]  # [B, L]
            expert_probs = top_k_probs[:, :, i:i+1]  # [B, L, 1]
            
            # Route inputs to selected experts
            for expert_id in range(self.n_experts):
                mask = (expert_indices == expert_id)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    
                    # Apply gating weights
                    weighted_output = expert_output * expert_probs[mask]
                    outputs[mask] += weighted_output
        
        return outputs

# Load balancing loss for training stability
def load_balancing_loss(router_probs, expert_indices, n_experts):
    # Encourage balanced usage of experts
    batch_size, seq_len = expert_indices.shape
    
    # Count tokens assigned to each expert
    expert_counts = torch.zeros(n_experts, device=expert_indices.device)
    for i in range(n_experts):
        expert_counts[i] = (expert_indices == i).float().sum()
    
    # Compute load balancing loss
    total_tokens = batch_size * seq_len
    target_load = total_tokens / n_experts
    
    load_loss = torch.sum((expert_counts - target_load) ** 2)
    return load_loss / total_tokens
"""
        },
        {
            "name": "Retrieval-Augmented",
            "description": "Models that incorporate external knowledge through retrieval",
            "examples": ["RAG", "REALM", "FiD", "Atlas"],
            "strengths": [
                "Access to up-to-date information",
                "Factual accuracy improvements",
                "Smaller model requirements",
                "Explainable knowledge sources"
            ],
            "weaknesses": [
                "Retrieval system dependency",
                "Increased inference latency",
                "Complex system architecture",
                "Knowledge base maintenance"
            ],
            "best_for": [
                "Question answering systems",
                "Fact-checking applications",
                "Domain-specific knowledge tasks",
                "Information-intensive applications"
            ],
            "architecture_details": """
class RetrievalAugmentedModel(nn.Module):
    def __init__(self, base_model, retriever, knowledge_encoder):
        super().__init__()
        self.base_model = base_model
        self.retriever = retriever
        self.knowledge_encoder = knowledge_encoder
        self.fusion_layer = CrossAttentionFusion(d_model=768)
        
    def forward(self, input_ids, attention_mask=None):
        # Retrieve relevant documents
        query_embedding = self.encode_query(input_ids)
        retrieved_docs = self.retriever.retrieve(query_embedding, top_k=5)
        
        # Encode retrieved knowledge
        knowledge_embeddings = []
        for doc in retrieved_docs:
            doc_embedding = self.knowledge_encoder.encode(doc)
            knowledge_embeddings.append(doc_embedding)
        
        knowledge_tensor = torch.stack(knowledge_embeddings)
        
        # Get base model hidden states
        base_hidden = self.base_model.get_hidden_states(input_ids, attention_mask)
        
        # Fuse knowledge with base model representations
        fused_hidden = self.fusion_layer(base_hidden, knowledge_tensor)
        
        # Generate final output
        output = self.base_model.generate_from_hidden(fused_hidden)
        
        return output, retrieved_docs

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads=8)
        self.norm = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, 4*d_model)
        
    def forward(self, query_hidden, knowledge_hidden):
        # Cross-attention: query attends to knowledge
        attended_knowledge, attn_weights = self.cross_attention(
            query_hidden, knowledge_hidden, knowledge_hidden
        )
        
        # Residual connection and normalization
        fused = self.norm(query_hidden + attended_knowledge)
        
        # Feed-forward layer
        output = self.feed_forward(fused)
        
        return output
"""
        },
        {
            "name": "Multimodal Architectures",
            "description": "Models that process multiple modalities (text, images, audio)",
            "examples": ["CLIP", "DALL-E", "Flamingo", "GPT-4V", "LLaVA"],
            "strengths": [
                "Rich multimodal understanding",
                "Cross-modal reasoning capabilities",
                "Unified representation learning",
                "Broader application domains"
            ],
            "weaknesses": [
                "Increased complexity",
                "Higher computational requirements",
                "Modality alignment challenges",
                "Limited training data for some modalities"
            ],
            "best_for": [
                "Image captioning",
                "Visual question answering",
                "Multimodal search",
                "Creative applications"
            ],
            "architecture_details": """
class MultimodalTransformer(nn.Module):
    def __init__(self, text_vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        
        # Text processing components
        self.text_embedding = nn.Embedding(text_vocab_size, d_model)
        self.text_pos_encoding = PositionalEncoding(d_model)
        
        # Vision processing components  
        self.vision_encoder = VisionTransformer(
            patch_size=16, d_model=d_model, n_heads=n_heads
        )
        
        # Shared multimodal layers
        self.multimodal_layers = nn.ModuleList([
            MultimodalLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        # Modality-specific projections
        self.text_projection = nn.Linear(d_model, d_model)
        self.vision_projection = nn.Linear(d_model, d_model)
        
    def forward(self, text_ids=None, images=None, attention_mask=None):
        embeddings = []
        modality_types = []
        
        # Process text input
        if text_ids is not None:
            text_emb = self.text_embedding(text_ids)
            text_emb = text_emb + self.text_pos_encoding(text_emb)
            text_emb = self.text_projection(text_emb)
            
            embeddings.append(text_emb)
            modality_types.extend(['text'] * text_emb.shape[1])
        
        # Process image input
        if images is not None:
            vision_emb = self.vision_encoder(images)
            vision_emb = self.vision_projection(vision_emb)
            
            embeddings.append(vision_emb)
            modality_types.extend(['vision'] * vision_emb.shape[1])
        
        # Concatenate multimodal embeddings
        if embeddings:
            combined_emb = torch.cat(embeddings, dim=1)
            
            # Apply multimodal layers
            for layer in self.multimodal_layers:
                combined_emb = layer(combined_emb, modality_types, attention_mask)
            
            return combined_emb
        
        return None

class MultimodalLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, 4*d_model)
        
    def forward(self, x, modality_types, mask=None):
        # Cross-modal attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x
"""
        }
    ]
    
    for arch in architectures:
        with st.expander(f"🏗️ {arch['name']}"):
            st.markdown(arch['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Examples:**")
                for example in arch['examples']:
                    st.markdown(f"• {example}")
                
                st.markdown("**Strengths:**")
                for strength in arch['strengths']:
                    st.markdown(f"• {strength}")
            
            with col2:
                st.markdown("**Best For:**")
                for use_case in arch['best_for']:
                    st.markdown(f"• {use_case}")
                
                st.markdown("**Weaknesses:**")
                for weakness in arch['weaknesses']:
                    st.markdown(f"• {weakness}")
            
            st.markdown("**Architecture Implementation:**")
            st.code(arch['architecture_details'], language='python')

with concept_tabs[1]:
    st.subheader("📏 Model Scaling Considerations")
    
    st.markdown("""
    Model size significantly impacts performance, cost, and deployment characteristics.
    Understanding scaling laws helps in making informed decisions about model size.
    """)
    
    # Scaling comparison chart
    model_sizes = pd.DataFrame({
        'Model Size': ['125M', '350M', '1.3B', '6.7B', '13B', '30B', '70B', '175B'],
        'Parameters (B)': [0.125, 0.35, 1.3, 6.7, 13, 30, 70, 175],
        'Training Cost ($K)': [5, 15, 50, 200, 500, 1500, 5000, 15000],
        'Inference Cost ($/1K tokens)': [0.0001, 0.0003, 0.001, 0.005, 0.01, 0.03, 0.07, 0.15],
        'Memory Required (GB)': [0.5, 1.4, 5.2, 26.8, 52, 120, 280, 700],
        'Quality Score': [3.5, 4.2, 5.8, 7.1, 7.8, 8.4, 8.7, 9.0]
    })
    
    # Create scaling visualization
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=model_sizes['Parameters (B)'],
        y=model_sizes['Quality Score'],
        mode='markers+lines',
        name='Quality Score',
        text=model_sizes['Model Size'],
        textposition="top center",
        marker=dict(
            size=model_sizes['Training Cost ($K)'] / 100,
            sizemode='diameter',
            sizeref=2,
            color='blue',
            opacity=0.7
        ),
        hovertemplate='Size: %{text}<br>Parameters: %{x}B<br>Quality: %{y}<br>Training Cost: $%{marker.size}K'
    ))
    
    fig.update_layout(
        title="Model Scaling: Quality vs Parameters (Bubble size = Training Cost)",
        xaxis_title="Parameters (Billions)",
        yaxis_title="Quality Score (1-10)",
        xaxis_type="log",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Scaling laws
    st.markdown("### 📊 Scaling Laws")
    
    scaling_laws = [
        {
            "law": "Chinchilla Scaling Law",
            "description": "Optimal training involves scaling compute, model size, and dataset size together",
            "formula": "Optimal parameters ≈ 20 × (Training tokens)",
            "implications": [
                "Most models are undertrained for their size",
                "Smaller, well-trained models can outperform larger undertrained ones",
                "Data quality becomes more important at scale",
                "Training efficiency improves with proper scaling"
            ],
            "example": "A 70B parameter model should be trained on ~1.4T tokens for optimal performance"
        },
        {
            "law": "Power Law Scaling",
            "description": "Model performance follows power laws with respect to compute, data, and parameters",
            "formula": "Performance ∝ (Compute)^α × (Data)^β × (Parameters)^γ",
            "implications": [
                "Predictable performance improvements with scale",
                "Diminishing returns at very large scales",
                "Different tasks have different scaling exponents",
                "Architectural improvements can change scaling curves"
            ],
            "example": "Doubling compute typically improves performance by 10-20%"
        },
        {
            "law": "Emergence Threshold",
            "description": "Certain capabilities emerge suddenly at specific model scales",
            "formula": "Capability = 0 if scale < threshold, else f(scale)",
            "implications": [
                "Some capabilities require minimum model size",
                "Investment threshold for certain applications",
                "Unpredictable capability emergence",
                "Different capabilities emerge at different scales"
            ],
            "example": "In-context learning emerges around 1B parameters"
        }
    ]
    
    for law in scaling_laws:
        with st.expander(f"📊 {law['law']}"):
            st.markdown(law['description'])
            st.markdown(f"**Formula:** {law['formula']}")
            st.markdown(f"**Example:** {law['example']}")
            
            st.markdown("**Key Implications:**")
            for implication in law['implications']:
                st.markdown(f"• {implication}")

with concept_tabs[2]:
    st.subheader("⚖️ Trade-off Analysis")
    
    st.markdown("""
    Model selection involves balancing multiple competing factors. Understanding 
    these trade-offs is crucial for making optimal decisions.
    """)
    
    # Trade-off radar chart
    tradeoff_data = {
        'Metric': ['Performance', 'Speed', 'Cost', 'Memory', 'Scalability', 'Flexibility'],
        'Small Model (1B)': [6, 9, 9, 9, 8, 7],
        'Medium Model (13B)': [8, 6, 6, 6, 7, 8],
        'Large Model (70B)': [9, 3, 3, 3, 5, 9],
        'MoE Model (500B)': [9, 7, 5, 4, 9, 8]
    }
    
    tradeoff_df = pd.DataFrame(tradeoff_data)
    
    fig = go.Figure()
    
    models = ['Small Model (1B)', 'Medium Model (13B)', 'Large Model (70B)', 'MoE Model (500B)']
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, model in enumerate(models):
        fig.add_trace(go.Scatterpolar(
            r=tradeoff_df[model],
            theta=tradeoff_df['Metric'],
            fill='toself',
            name=model,
            line_color=colors[i]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True,
        title="Model Size Trade-offs Analysis",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed trade-off analysis
    tradeoff_categories = [
        {
            "category": "Performance vs Cost",
            "analysis": "Larger models generally provide better performance but at exponentially higher costs",
            "considerations": [
                "Training costs scale super-linearly with model size",
                "Inference costs scale with model size and usage",
                "Performance improvements follow power law (diminishing returns)",
                "Break-even point depends on use case value"
            ],
            "decision_framework": """
def cost_benefit_analysis(model_sizes, use_case_value, traffic_volume):
    results = []
    
    for size in model_sizes:
        # Estimate performance (simplified)
        performance = 1 - np.exp(-size / 10)  # Diminishing returns
        
        # Estimate costs
        training_cost = size ** 1.5 * 1000  # Super-linear scaling
        inference_cost_per_token = size * 0.001
        monthly_inference_cost = inference_cost_per_token * traffic_volume * 30
        
        # Calculate value
        monthly_value = performance * use_case_value
        monthly_profit = monthly_value - monthly_inference_cost
        payback_months = training_cost / monthly_profit if monthly_profit > 0 else float('inf')
        
        results.append({
            'size': size,
            'performance': performance,
            'training_cost': training_cost,
            'monthly_cost': monthly_inference_cost,
            'monthly_value': monthly_value,
            'payback_months': payback_months
        })
    
    return results

# Example usage
model_sizes = [1, 7, 13, 30, 70]  # Billion parameters
use_case_value = 50000  # Monthly value from improved performance
traffic_volume = 1000000  # Tokens per day

analysis = cost_benefit_analysis(model_sizes, use_case_value, traffic_volume)
optimal_model = min(analysis, key=lambda x: x['payback_months'])
"""
        },
        {
            "category": "Latency vs Quality",
            "analysis": "There's often an inverse relationship between response quality and speed",
            "considerations": [
                "User experience requirements for response time",
                "Task criticality and quality requirements",
                "Caching and optimization opportunities",
                "Hybrid approaches (fast + slow models)"
            ],
            "decision_framework": """
def latency_quality_optimizer(quality_threshold, latency_threshold, models):
    # Pareto frontier analysis
    feasible_models = []
    
    for model in models:
        if (model['quality'] >= quality_threshold and 
            model['latency'] <= latency_threshold):
            feasible_models.append(model)
    
    if not feasible_models:
        # Relax constraints or use hybrid approach
        return suggest_hybrid_approach(quality_threshold, latency_threshold)
    
    # Select model with best quality within constraints
    optimal_model = max(feasible_models, key=lambda x: x['quality'])
    return optimal_model

def suggest_hybrid_approach(quality_threshold, latency_threshold):
    # Use fast model for initial response, slow model for refinement
    return {
        'strategy': 'hybrid',
        'fast_model': 'small_model_for_speed',
        'quality_model': 'large_model_for_accuracy',
        'routing_logic': 'complexity_based'
    }
"""
        },
        {
            "category": "Specialization vs Generalization",
            "analysis": "Specialized models excel in specific domains but lack versatility",
            "considerations": [
                "Domain-specific requirements and constraints",
                "Maintenance overhead of multiple models",
                "Data availability for specialization",
                "Future use case expansion plans"
            ],
            "decision_framework": """
def specialization_decision(use_cases, domain_overlap, maintenance_capacity):
    # Analyze use case similarity
    similarity_matrix = calculate_use_case_similarity(use_cases)
    
    # Clustering analysis
    clusters = cluster_use_cases(similarity_matrix, threshold=0.7)
    
    recommendations = []
    
    for cluster in clusters:
        if len(cluster) == 1:
            # Single use case - consider specialized model
            use_case = cluster[0]
            if (use_case['criticality'] > 0.8 and 
                use_case['data_availability'] > 0.7):
                recommendations.append({
                    'type': 'specialized',
                    'use_cases': cluster,
                    'rationale': 'High criticality with sufficient data'
                })
            else:
                recommendations.append({
                    'type': 'general',
                    'use_cases': cluster,
                    'rationale': 'Low criticality or insufficient data'
                })
        else:
            # Multiple similar use cases - general model likely better
            recommendations.append({
                'type': 'general',
                'use_cases': cluster,
                'rationale': 'Multiple similar use cases benefit from shared model'
            })
    
    return recommendations
"""
        }
    ]
    
    for category in tradeoff_categories:
        with st.expander(f"⚖️ {category['category']}"):
            st.markdown(category['analysis'])
            
            st.markdown("**Key Considerations:**")
            for consideration in category['considerations']:
                st.markdown(f"• {consideration}")
            
            st.markdown("**Decision Framework:**")
            st.code(category['decision_framework'], language='python')

with concept_tabs[3]:
    st.subheader("🎯 Model Selection Framework")
    
    st.markdown("""
    A systematic approach to model selection considers all relevant factors 
    and provides a structured decision-making process.
    """)
    
    # Decision tree visualization
    st.markdown("### 🌳 Decision Framework")
    
    framework_steps = [
        {
            "step": "1. Requirements Analysis",
            "description": "Define functional and non-functional requirements",
            "questions": [
                "What is the primary use case and task type?",
                "What are the quality/accuracy requirements?", 
                "What are the latency and throughput requirements?",
                "What is the budget for training and inference?",
                "What are the deployment constraints?"
            ],
            "outputs": ["Task specification", "Performance targets", "Constraint matrix"]
        },
        {
            "step": "2. Architecture Screening",
            "description": "Filter architectures based on requirements",
            "questions": [
                "Which architectures are suitable for the task type?",
                "Which architectures meet the performance requirements?",
                "Which architectures fit within the constraints?",
                "Are there any deal-breaker limitations?"
            ],
            "outputs": ["Candidate architectures", "Feasibility assessment"]
        },
        {
            "step": "3. Size Optimization",
            "description": "Determine optimal model size within chosen architecture",
            "questions": [
                "What is the minimum size for acceptable performance?",
                "What is the maximum size within budget constraints?",
                "How does performance scale with size for this task?",
                "What is the optimal size given cost-benefit analysis?"
            ],
            "outputs": ["Size recommendations", "Cost-performance curve"]
        },
        {
            "step": "4. Validation Testing",
            "description": "Empirical validation of model selection",
            "questions": [
                "How do candidate models perform on representative data?",
                "Are there unexpected failure modes or biases?",
                "How do real-world constraints affect performance?",
                "Does the model meet all requirements in practice?"
            ],
            "outputs": ["Performance benchmarks", "Validation report"]
        },
        {
            "step": "5. Final Selection",
            "description": "Make final decision based on comprehensive analysis",
            "questions": [
                "Which model best balances all requirements?",
                "Are there any remaining risks or concerns?",
                "What is the deployment and maintenance plan?",
                "How will performance be monitored and improved?"
            ],
            "outputs": ["Final model selection", "Implementation plan"]
        }
    ]
    
    for step_info in framework_steps:
        with st.expander(f"🎯 {step_info['step']}: {step_info['description']}"):
            st.markdown("**Key Questions:**")
            for question in step_info['questions']:
                st.markdown(f"• {question}")
            
            st.markdown("**Expected Outputs:**")
            for output in step_info['outputs']:
                st.markdown(f"• {output}")
    
    # Interactive model selector
    st.markdown("### 🔧 Interactive Model Selector")
    
    st.markdown("Answer the following questions to get personalized model recommendations:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_case = st.selectbox(
            "Primary Use Case",
            ["Conversational AI", "Content Generation", "Question Answering", 
             "Text Classification", "Translation", "Summarization", "Code Generation"]
        )
        
        quality_requirement = st.slider("Quality Requirement (1-10)", 1, 10, 7)
        latency_requirement = st.selectbox("Latency Requirement", 
                                         ["Real-time (<100ms)", "Interactive (<1s)", "Batch (>1s)"])
        
    with col2:
        budget_range = st.selectbox("Budget Range", 
                                  ["Low (<$1K/month)", "Medium ($1K-$10K/month)", 
                                   "High ($10K-$100K/month)", "Enterprise (>$100K/month)"])
        
        deployment_type = st.selectbox("Deployment Type", 
                                     ["Cloud API", "On-premise", "Edge device", "Hybrid"])
        
        traffic_volume = st.selectbox("Expected Traffic", 
                                    ["Low (<1M tokens/day)", "Medium (1M-100M tokens/day)", 
                                     "High (100M-1B tokens/day)", "Very High (>1B tokens/day)"])
    
    if st.button("🎯 Get Model Recommendations"):
        # Model recommendation logic
        recommendations = []
        
        # Simple rule-based recommendation system
        if use_case in ["Conversational AI", "Content Generation"]:
            if quality_requirement >= 8:
                if budget_range in ["High", "Enterprise"]:
                    recommendations.append({
                        "model": "Large Model (70B+)",
                        "reason": "High quality conversational AI requires large models",
                        "considerations": ["High cost", "Significant compute requirements"]
                    })
                else:
                    recommendations.append({
                        "model": "Medium Model (13B-30B)",
                        "reason": "Good balance of quality and cost for conversations",
                        "considerations": ["May need fine-tuning", "Consider fine-tuned variants"]
                    })
            else:
                recommendations.append({
                    "model": "Small-Medium Model (1B-7B)",
                    "reason": "Sufficient for basic conversational tasks",
                    "considerations": ["Lower quality", "May need task-specific training"]
                })
        
        elif use_case in ["Text Classification", "Question Answering"]:
            if latency_requirement == "Real-time (<100ms)":
                recommendations.append({
                    "model": "Small Specialized Model (100M-1B)",
                    "reason": "Fast inference for real-time classification",
                    "considerations": ["Requires task-specific training", "Limited generalization"]
                })
            else:
                recommendations.append({
                    "model": "Medium General Model (7B-13B)",
                    "reason": "Good performance with reasonable latency",
                    "considerations": ["In-context learning capable", "Few-shot performance"]
                })
        
        elif use_case == "Code Generation":
            recommendations.append({
                "model": "Code-Specialized Model (1B-70B)",
                "reason": "Specialized models perform better for code tasks",
                "considerations": ["Consider CodeT5, CodeLlama, or similar", "Domain-specific training important"]
            })
        
        # Display recommendations
        st.markdown("### 🎯 Recommended Models")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**Recommendation {i}: {rec['model']}**")
            st.markdown(f"*Reason:* {rec['reason']}")
            st.markdown("*Key Considerations:*")
            for consideration in rec['considerations']:
                st.markdown(f"• {consideration}")
            st.markdown("---")

# Implementation guide
st.header("🛠️ Implementation Guide")

implementation_code = """
# Comprehensive Model Selection Implementation
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

class TaskType(Enum):
    CONVERSATIONAL = "conversational"
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    QA = "qa"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"

class DeploymentType(Enum):
    CLOUD_API = "cloud_api"
    ON_PREMISE = "on_premise"
    EDGE = "edge"
    HYBRID = "hybrid"

@dataclass
class ModelRequirements:
    task_type: TaskType
    quality_threshold: float  # 0-1
    max_latency_ms: int
    max_monthly_cost: float
    min_throughput: int  # tokens per second
    deployment_type: DeploymentType
    
@dataclass
class ModelSpec:
    name: str
    architecture: str
    parameters_b: float
    memory_gb: float
    latency_ms: int
    cost_per_1k_tokens: float
    quality_score: float
    supported_tasks: List[TaskType]

class ModelSelector:
    def __init__(self):
        self.available_models = self._load_model_catalog()
        self.benchmark_data = self._load_benchmarks()
    
    def _load_model_catalog(self) -> List[ModelSpec]:
        return [
            ModelSpec("GPT-3.5-turbo", "transformer-decoder", 20, 40, 200, 0.002, 0.85, 
                     [TaskType.CONVERSATIONAL, TaskType.GENERATION, TaskType.QA]),
            ModelSpec("GPT-4", "transformer-decoder", 175, 350, 800, 0.06, 0.95,
                     [TaskType.CONVERSATIONAL, TaskType.GENERATION, TaskType.QA]),
            ModelSpec("Claude-2", "transformer-decoder", 52, 104, 500, 0.008, 0.90,
                     [TaskType.CONVERSATIONAL, TaskType.GENERATION, TaskType.QA]),
            ModelSpec("LLaMA-2-7B", "transformer-decoder", 7, 14, 100, 0.001, 0.75,
                     [TaskType.CONVERSATIONAL, TaskType.GENERATION]),
            ModelSpec("LLaMA-2-70B", "transformer-decoder", 70, 140, 400, 0.015, 0.88,
                     [TaskType.CONVERSATIONAL, TaskType.GENERATION, TaskType.QA]),
            ModelSpec("T5-Large", "encoder-decoder", 0.77, 3, 80, 0.0008, 0.70,
                     [TaskType.TRANSLATION, TaskType.SUMMARIZATION, TaskType.QA]),
            ModelSpec("BERT-Large", "encoder", 0.34, 1.3, 20, 0.0002, 0.82,
                     [TaskType.CLASSIFICATION, TaskType.QA]),
        ]
    
    def select_model(self, requirements: ModelRequirements) -> List[Tuple[ModelSpec, float]]:
        # Filter models by task compatibility
        compatible_models = [
            model for model in self.available_models
            if requirements.task_type in model.supported_tasks
        ]
        
        # Score each model based on requirements
        scored_models = []
        for model in compatible_models:
            score = self._score_model(model, requirements)
            if score > 0:  # Only include viable models
                scored_models.append((model, score))
        
        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        return scored_models[:3]  # Return top 3 recommendations
    
    def _score_model(self, model: ModelSpec, req: ModelRequirements) -> float:
        score = 0.0
        
        # Quality score (40% weight)
        if model.quality_score >= req.quality_threshold:
            quality_bonus = min(1.0, model.quality_score / req.quality_threshold)
            score += 0.4 * quality_bonus
        else:
            return 0.0  # Doesn't meet minimum quality
        
        # Latency score (25% weight)
        if model.latency_ms <= req.max_latency_ms:
            latency_bonus = max(0, 1 - (model.latency_ms / req.max_latency_ms))
            score += 0.25 * latency_bonus
        else:
            return 0.0  # Doesn't meet latency requirement
        
        # Cost score (25% weight)
        monthly_cost = self._estimate_monthly_cost(model, req.min_throughput)
        if monthly_cost <= req.max_monthly_cost:
            cost_bonus = max(0, 1 - (monthly_cost / req.max_monthly_cost))
            score += 0.25 * cost_bonus
        else:
            return 0.0  # Too expensive
        
        # Deployment compatibility (10% weight)
        deployment_bonus = self._deployment_compatibility(model, req.deployment_type)
        score += 0.1 * deployment_bonus
        
        return score
    
    def _estimate_monthly_cost(self, model: ModelSpec, throughput: int) -> float:
        # Estimate monthly cost based on throughput and pricing
        monthly_tokens = throughput * 60 * 60 * 24 * 30  # tokens per month
        cost_per_month = (monthly_tokens / 1000) * model.cost_per_1k_tokens
        
        # Add infrastructure costs for on-premise deployment
        if model.parameters_b > 30:
            infrastructure_cost = 2000  # High-end GPU costs
        elif model.parameters_b > 7:
            infrastructure_cost = 800   # Mid-range GPU costs
        else:
            infrastructure_cost = 200   # Basic GPU costs
        
        return cost_per_month + infrastructure_cost
    
    def _deployment_compatibility(self, model: ModelSpec, deployment: DeploymentType) -> float:
        # Score model compatibility with deployment type
        if deployment == DeploymentType.EDGE:
            # Prefer smaller models for edge deployment
            if model.parameters_b < 1:
                return 1.0
            elif model.parameters_b < 7:
                return 0.7
            else:
                return 0.2
        
        elif deployment == DeploymentType.CLOUD_API:
            # Most models work well with cloud APIs
            return 1.0
        
        elif deployment == DeploymentType.ON_PREMISE:
            # Consider memory requirements
            if model.memory_gb < 50:
                return 1.0
            elif model.memory_gb < 200:
                return 0.8
            else:
                return 0.5
        
        return 0.8  # Default compatibility

# Usage example
def demonstrate_model_selection():
    selector = ModelSelector()
    
    # Define requirements for a chatbot application
    requirements = ModelRequirements(
        task_type=TaskType.CONVERSATIONAL,
        quality_threshold=0.8,
        max_latency_ms=1000,
        max_monthly_cost=5000,
        min_throughput=100,  # tokens per second
        deployment_type=DeploymentType.CLOUD_API
    )
    
    # Get recommendations
    recommendations = selector.select_model(requirements)
    
    print("Model Recommendations:")
    for i, (model, score) in enumerate(recommendations, 1):
        print(f"{i}. {model.name} (Score: {score:.2f})")
        print(f"   Parameters: {model.parameters_b}B")
        print(f"   Quality: {model.quality_score}")
        print(f"   Latency: {model.latency_ms}ms")
        print(f"   Cost: ${model.cost_per_1k_tokens}/1K tokens")
        print()

if __name__ == "__main__":
    demonstrate_model_selection()
"""

st.code(implementation_code, language='python')

# Best practices
st.header("💡 Model Selection Best Practices")

best_practices = [
    "**Start with Requirements**: Clearly define functional and non-functional requirements before evaluation",
    "**Benchmark Systematically**: Test models on representative data and realistic conditions",
    "**Consider Total Cost**: Include training, inference, infrastructure, and maintenance costs",
    "**Plan for Scale**: Consider how requirements may change with increased usage",
    "**Validate Assumptions**: Test performance claims with your specific data and use cases",
    "**Monitor Continuously**: Track performance and costs in production to validate decisions",
    "**Iterate and Improve**: Be prepared to reassess model selection as requirements evolve",
    "**Document Decisions**: Keep records of selection criteria and rationale for future reference",
    "**Consider Alternatives**: Evaluate fine-tuning smaller models vs using larger general models",
    "**Think Long-term**: Consider model lifecycle, vendor dependencies, and future migration paths"
]

for practice in best_practices:
    st.markdown(f"• {practice}")

# Resources
st.header("📚 Learning Resources")

resources = [
    {
        "title": "Scaling Laws for Neural Language Models",
        "type": "Research Paper",
        "description": "Foundational paper on LLM scaling relationships",
        "difficulty": "Advanced"
    },
    {
        "title": "Training Compute-Optimal Large Language Models",
        "type": "Research Paper",
        "description": "Chinchilla paper on optimal training scaling",
        "difficulty": "Advanced"
    },
    {
        "title": "Model Selection for Production ML",
        "type": "Guide",
        "description": "Practical guide to model selection in production",
        "difficulty": "Intermediate"
    },
    {
        "title": "LLM Architecture Comparison",
        "type": "Blog Post",
        "description": "Comprehensive comparison of modern LLM architectures",
        "difficulty": "Beginner"
    }
]

for resource in resources:
    with st.expander(f"📖 {resource['title']}"):
        st.markdown(f"**Type:** {resource['type']}")
        st.markdown(f"**Description:** {resource['description']}")
        st.markdown(f"**Difficulty:** {resource['difficulty']}")