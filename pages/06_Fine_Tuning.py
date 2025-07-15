import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fine-Tuning", page_icon="🎯", layout="wide")

st.title("🎯 Fine-Tuning & Specialization")
st.markdown("### Adapting Pre-trained Models for Specific Tasks")

# Overview
st.header("🎯 Overview")
st.markdown("""
Fine-tuning takes a pre-trained language model and adapts it for specific tasks or domains. 
This process is more efficient than training from scratch and often achieves better performance 
with less data. Modern fine-tuning techniques range from full parameter updates to efficient 
parameter-efficient methods.
""")

# Fine-tuning approaches
st.header("🔧 Fine-Tuning Approaches")

approach_tabs = st.tabs([
    "🔄 Full Fine-Tuning", 
    "⚡ Parameter-Efficient", 
    "🎭 Instruction Tuning",
    "🤝 Human Feedback"
])

with approach_tabs[0]:
    st.subheader("🔄 Full Fine-Tuning")
    
    st.markdown("""
    Full fine-tuning updates all model parameters for the target task. While effective, 
    it requires significant computational resources and can lead to catastrophic forgetting.
    """)
    
    full_ft_aspects = [
        {
            "aspect": "Supervised Fine-Tuning (SFT)",
            "description": "Train on labeled examples for specific tasks",
            "use_cases": [
                "Text classification",
                "Named entity recognition", 
                "Question answering",
                "Summarization",
                "Translation"
            ],
            "advantages": [
                "Maximum flexibility",
                "Can achieve best performance",
                "Simple to implement",
                "Well-understood process"
            ],
            "disadvantages": [
                "Computationally expensive",
                "Risk of catastrophic forgetting",
                "Requires large datasets",
                "May overfit to training data"
            ]
        },
        {
            "aspect": "Multi-Task Fine-Tuning",
            "description": "Train on multiple related tasks simultaneously",
            "use_cases": [
                "Cross-domain adaptation",
                "Multi-objective optimization",
                "Preventing catastrophic forgetting",
                "Improving generalization"
            ],
            "advantages": [
                "Better generalization",
                "Reduced overfitting",
                "Knowledge sharing across tasks",
                "More robust representations"
            ],
            "disadvantages": [
                "Complex setup",
                "Task interference possible",
                "Balancing multiple objectives",
                "Requires diverse datasets"
            ]
        },
        {
            "aspect": "Domain Adaptation",
            "description": "Adapt to specific domains or styles",
            "use_cases": [
                "Medical text processing",
                "Legal document analysis",
                "Scientific literature",
                "Social media content"
            ],
            "advantages": [
                "Domain-specific knowledge",
                "Better performance on specialized tasks",
                "Handles domain vocabulary",
                "Adapts to writing styles"
            ],
            "disadvantages": [
                "May lose general capabilities",
                "Requires domain expertise",
                "Limited transferability",
                "Evaluation challenges"
            ]
        }
    ]
    
    for aspect in full_ft_aspects:
        with st.expander(f"📚 {aspect['aspect']}"):
            st.markdown(aspect['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Use Cases:**")
                for use_case in aspect['use_cases']:
                    st.markdown(f"• {use_case}")
                
                st.markdown("**✅ Advantages:**")
                for advantage in aspect['advantages']:
                    st.markdown(f"• {advantage}")
            
            with col2:
                st.markdown("**❌ Disadvantages:**")
                for disadvantage in aspect['disadvantages']:
                    st.markdown(f"• {disadvantage}")
    
    # Full fine-tuning implementation example
    st.markdown("### 💻 Implementation Example")
    
    st.code("""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

# Load pre-trained model and tokenizer
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare your fine-tuning dataset
def preprocess_function(examples):
    # Tokenize inputs and targets
    inputs = tokenizer(
        examples['text'], 
        truncation=True, 
        padding=True, 
        max_length=512,
        return_tensors="pt"
    )
    
    # For causal LM, labels are the same as input_ids
    inputs['labels'] = inputs['input_ids'].clone()
    return inputs

# Create dataset
train_dataset = Dataset.from_dict({'text': your_training_texts})
train_dataset = train_dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./fine-tuned-model',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    push_to_hub=False
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Your validation dataset
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model()
""", language='python')

with approach_tabs[1]:
    st.subheader("⚡ Parameter-Efficient Fine-Tuning")
    
    st.markdown("""
    Parameter-efficient methods update only a small subset of parameters while keeping 
    most of the pre-trained model frozen. These approaches are more practical for 
    many use cases due to their efficiency.
    """)
    
    peft_methods = [
        {
            "method": "LoRA (Low-Rank Adaptation)",
            "description": "Adds trainable low-rank matrices to existing layers",
            "parameters": "0.1-1% of original model",
            "memory": "Minimal increase",
            "performance": "90-95% of full fine-tuning",
            "advantages": [
                "Very efficient training",
                "Easy to deploy multiple adapters",
                "Preserves original model",
                "Fast switching between tasks"
            ],
            "best_for": "Most fine-tuning scenarios",
            "implementation": """
from peft import LoraConfig, get_peft_model, TaskType

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,  # Low-rank dimension
    lora_alpha=32,  # LoRA scaling parameter
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 774,030,080 || trainable%: 0.54%
"""
        },
        {
            "method": "Prefix Tuning",
            "description": "Adds trainable prefix tokens to input sequences",
            "parameters": "0.1-3% of original model",
            "memory": "Low",
            "performance": "80-90% of full fine-tuning",
            "advantages": [
                "Preserves original parameters",
                "Task-specific prefixes",
                "Good for generation tasks",
                "Compositional capabilities"
            ],
            "best_for": "Text generation tasks",
            "implementation": """
from peft import PrefixTuningConfig, get_peft_model

# Prefix tuning configuration
prefix_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    num_virtual_tokens=20,  # Number of prefix tokens
    token_dim=768,  # Hidden dimension
    num_transformer_submodules=2,
    num_attention_heads=12,
    num_layers=12
)

model = get_peft_model(model, prefix_config)
"""
        },
        {
            "method": "Adapters",
            "description": "Inserts small neural networks between existing layers",
            "parameters": "0.5-5% of original model",
            "memory": "Moderate increase",
            "performance": "85-95% of full fine-tuning",
            "advantages": [
                "Modular design",
                "Can stack multiple adapters",
                "Good task specialization",
                "Interpretable modifications"
            ],
            "best_for": "Multi-task scenarios",
            "implementation": """
from transformers import AutoAdapterModel

# Load model with adapter support
model = AutoAdapterModel.from_pretrained("bert-base-uncased")

# Add task-specific adapter
model.add_adapter("task_adapter", config="pfeiffer")

# Activate adapter for training
model.set_active_adapters("task_adapter")

# Train only the adapter parameters
model.train_adapter("task_adapter")
"""
        },
        {
            "method": "QLoRA (Quantized LoRA)",
            "description": "Combines LoRA with 4-bit quantization for extreme efficiency",
            "parameters": "0.1-1% of original model",
            "memory": "75% reduction",
            "performance": "95-98% of full fine-tuning",
            "advantages": [
                "Extreme memory efficiency",
                "Maintains high performance",
                "Enables large model fine-tuning",
                "4-bit quantization"
            ],
            "best_for": "Large models with limited resources",
            "implementation": """
from transformers import BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Add LoRA
model = get_peft_model(model, lora_config)
"""
        }
    ]
    
    for method in peft_methods:
        with st.expander(f"⚡ {method['method']}"):
            st.markdown(method['description'])
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Parameters", method['parameters'])
            with col2:
                st.metric("Memory", method['memory'])
            with col3:
                st.metric("Performance", method['performance'])
            with col4:
                st.markdown(f"**Best for:** {method['best_for']}")
            
            st.markdown("**Advantages:**")
            for advantage in method['advantages']:
                st.markdown(f"• {advantage}")
            
            st.markdown("**Implementation:**")
            st.code(method['implementation'], language='python')
    
    # PEFT comparison chart
    st.markdown("### 📊 PEFT Methods Comparison")
    
    peft_comparison = {
        'Method': ['LoRA', 'Prefix Tuning', 'Adapters', 'QLoRA'],
        'Efficiency': [95, 85, 80, 98],
        'Performance': [92, 85, 88, 95],
        'Ease of Use': [90, 75, 85, 80],
        'Memory Usage': [95, 90, 85, 98]
    }
    
    peft_df = pd.DataFrame(peft_comparison)
    
    fig = px.radar(
        peft_df, r='Efficiency', theta='Method',
        title="PEFT Methods Radar Comparison",
        range_r=[0, 100]
    )
    
    for i, metric in enumerate(['Performance', 'Ease of Use', 'Memory Usage']):
        fig.add_trace(go.Scatterpolar(
            r=peft_df[metric],
            theta=peft_df['Method'],
            fill='toself',
            name=metric
        ))
    
    st.plotly_chart(fig, use_container_width=True)

with approach_tabs[2]:
    st.subheader("🎭 Instruction Tuning")
    
    st.markdown("""
    Instruction tuning trains models to follow natural language instructions, 
    making them more useful for interactive applications and general-purpose tasks.
    """)
    
    instruction_tabs = st.tabs(["📝 Instruction Datasets", "🔧 Training Process", "📊 Evaluation"])
    
    with instruction_tabs[0]:
        st.markdown("### 📝 Instruction Dataset Design")
        
        dataset_components = [
            {
                "component": "Instruction",
                "description": "Clear, specific task description",
                "examples": [
                    "Summarize the following article in 3 sentences.",
                    "Translate this text from English to French.",
                    "Answer the question based on the given context."
                ],
                "best_practices": [
                    "Be specific and unambiguous",
                    "Include context when necessary",
                    "Vary instruction phrasings",
                    "Cover diverse task types"
                ]
            },
            {
                "component": "Input",
                "description": "The data or context for the task",
                "examples": [
                    "Article text for summarization",
                    "Source text for translation", 
                    "Context paragraph and question"
                ],
                "best_practices": [
                    "Ensure input quality",
                    "Vary input lengths",
                    "Include diverse topics",
                    "Handle edge cases"
                ]
            },
            {
                "component": "Output",
                "description": "High-quality expected response",
                "examples": [
                    "Well-written summary",
                    "Accurate translation",
                    "Correct and helpful answer"
                ],
                "best_practices": [
                    "Ensure response quality",
                    "Maintain consistency",
                    "Follow instruction exactly",
                    "Include reasoning when helpful"
                ]
            }
        ]
        
        for component in dataset_components:
            with st.expander(f"📋 {component['component']}"):
                st.markdown(component['description'])
                
                st.markdown("**Examples:**")
                for example in component['examples']:
                    st.markdown(f"• {example}")
                
                st.markdown("**Best Practices:**")
                for practice in component['best_practices']:
                    st.markdown(f"• {practice}")
        
        # Instruction format example
        st.markdown("### 📄 Standard Instruction Format")
        
        instruction_format = """
### Instruction:
Summarize the main points of the following research paper abstract in 2-3 sentences.

### Input:
Large language models have shown remarkable capabilities in various natural language processing tasks. However, their performance can be significantly improved through careful fine-tuning on task-specific datasets. This paper presents a comprehensive study of different fine-tuning strategies and their effectiveness across multiple domains.

### Response:
This research paper examines how fine-tuning can enhance large language model performance on specific tasks. The study provides a comprehensive analysis of various fine-tuning approaches and evaluates their effectiveness across different domains. The findings demonstrate that task-specific fine-tuning significantly improves model capabilities compared to using pre-trained models alone.
"""
        
        st.code(instruction_format, language='text')
    
    with instruction_tabs[1]:
        st.markdown("### 🔧 Instruction Tuning Process")
        
        process_steps = [
            {
                "step": "Dataset Preparation",
                "description": "Collect and format instruction-following examples",
                "details": [
                    "Gather diverse instruction types",
                    "Ensure high-quality responses",
                    "Format in consistent template",
                    "Split into train/validation sets"
                ]
            },
            {
                "step": "Model Selection",
                "description": "Choose appropriate pre-trained base model",
                "details": [
                    "Consider model size vs. resources",
                    "Evaluate base model capabilities",
                    "Check licensing and usage terms",
                    "Test baseline performance"
                ]
            },
            {
                "step": "Training Configuration",
                "description": "Set up training parameters for instruction tuning",
                "details": [
                    "Lower learning rate than pre-training",
                    "Shorter training duration",
                    "Careful monitoring of overfitting",
                    "Use parameter-efficient methods"
                ]
            },
            {
                "step": "Safety Filtering",
                "description": "Ensure model learns safe and helpful behaviors",
                "details": [
                    "Filter harmful instructions",
                    "Include safety demonstrations",
                    "Test edge cases",
                    "Monitor for biased outputs"
                ]
            }
        ]
        
        for i, step in enumerate(process_steps, 1):
            with st.expander(f"Step {i}: {step['step']}"):
                st.markdown(step['description'])
                
                for detail in step['details']:
                    st.markdown(f"• {detail}")
    
    with instruction_tabs[2]:
        st.markdown("### 📊 Instruction Following Evaluation")
        
        eval_metrics = [
            {
                "metric": "Instruction Following Accuracy",
                "description": "How well the model follows specific instructions",
                "measurement": "Human evaluation or automated rule-based checks",
                "example": "Does the summary have exactly 3 sentences as requested?"
            },
            {
                "metric": "Response Quality",
                "description": "Overall quality and helpfulness of responses",
                "measurement": "Human ratings on scales (1-5 or 1-10)",
                "example": "Rate the helpfulness and accuracy of the answer"
            },
            {
                "metric": "Safety Compliance",
                "description": "Adherence to safety guidelines and policies",
                "measurement": "Safety-specific evaluation prompts",
                "example": "Does the model refuse harmful requests appropriately?"
            },
            {
                "metric": "Factual Accuracy",
                "description": "Correctness of factual claims in responses",
                "measurement": "Fact-checking against reliable sources",
                "example": "Are the dates, names, and numbers correct?"
            }
        ]
        
        for metric in eval_metrics:
            with st.expander(f"📈 {metric['metric']}"):
                st.markdown(metric['description'])
                st.markdown(f"**Measurement:** {metric['measurement']}")
                st.markdown(f"**Example:** {metric['example']}")

with approach_tabs[3]:
    st.subheader("🤝 Reinforcement Learning from Human Feedback (RLHF)")
    
    st.markdown("""
    RLHF uses human preferences to train a reward model, which then guides the language model 
    to generate responses that align better with human values and preferences.
    """)
    
    rlhf_tabs = st.tabs(["🔄 RLHF Process", "🏆 Reward Modeling", "🎯 Policy Optimization"])
    
    with rlhf_tabs[0]:
        st.markdown("### 🔄 RLHF Pipeline")
        
        # RLHF process visualization
        rlhf_steps = ["Base Model", "SFT", "Reward Model Training", "PPO Training", "Aligned Model"]
        
        fig = go.Figure()
        
        # Create flowchart
        for i, step in enumerate(rlhf_steps):
            fig.add_shape(
                type="rect",
                x0=i*2, y0=0, x1=i*2+1.5, y1=1,
                fillcolor=["lightblue", "lightgreen", "orange", "red", "gold"][i],
                line=dict(color="black", width=2)
            )
            
            fig.add_annotation(
                x=i*2+0.75, y=0.5,
                text=step,
                showarrow=False,
                font=dict(size=12, color="black")
            )
            
            # Add arrows
            if i < len(rlhf_steps) - 1:
                fig.add_annotation(
                    x=i*2+1.75, y=0.5,
                    text="→",
                    showarrow=False,
                    font=dict(size=20, color="darkblue")
                )
        
        fig.update_layout(
            title="RLHF Pipeline Overview",
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, len(rlhf_steps)*2]),
            yaxis=dict(showgrid=False, showticklabels=False, range=[-0.2, 1.2]),
            height=200
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # RLHF stages explanation
        rlhf_stages = [
            {
                "stage": "Supervised Fine-Tuning (SFT)",
                "purpose": "Initial instruction following capability",
                "process": "Train on high-quality instruction-response pairs",
                "output": "Model that can follow instructions but may not align with preferences"
            },
            {
                "stage": "Reward Model Training",
                "purpose": "Learn human preferences",
                "process": "Train model to predict human preference rankings",
                "output": "Reward model that scores responses based on human preferences"
            },
            {
                "stage": "Reinforcement Learning",
                "purpose": "Optimize for human preferences",
                "process": "Use PPO to maximize reward model scores",
                "output": "Model aligned with human preferences and values"
            }
        ]
        
        for stage in rlhf_stages:
            with st.expander(f"🎯 {stage['stage']}"):
                st.markdown(f"**Purpose:** {stage['purpose']}")
                st.markdown(f"**Process:** {stage['process']}")
                st.markdown(f"**Output:** {stage['output']}")
    
    with rlhf_tabs[1]:
        st.markdown("### 🏆 Reward Model Training")
        
        st.markdown("""
        The reward model learns to predict human preferences by training on comparison data 
        where humans rank different model responses.
        """)
        
        reward_components = [
            {
                "component": "Preference Data Collection",
                "description": "Gather human rankings of model responses",
                "process": [
                    "Generate multiple responses to prompts",
                    "Present pairs to human annotators",
                    "Collect preference rankings",
                    "Ensure annotation quality and consistency"
                ]
            },
            {
                "component": "Reward Model Architecture",
                "description": "Typically uses the same architecture as the base model",
                "process": [
                    "Start with pre-trained language model",
                    "Replace language modeling head with scalar reward head",
                    "Fine-tune on preference comparison data",
                    "Output single reward score per response"
                ]
            },
            {
                "component": "Training Objective",
                "description": "Maximize likelihood of human preferences",
                "process": [
                    "Use Bradley-Terry model for pairwise comparisons",
                    "Minimize ranking loss between preferred/dispreferred responses",
                    "Regularize to prevent overfitting to annotation artifacts",
                    "Validate on held-out preference data"
                ]
            }
        ]
        
        for component in reward_components:
            with st.expander(f"🏆 {component['component']}"):
                st.markdown(component['description'])
                
                for step in component['process']:
                    st.markdown(f"• {step}")
        
        # Reward model training code
        st.markdown("### 💻 Reward Model Implementation")
        
        st.code("""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RewardModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        # Use last token representation for reward
        last_token_repr = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_token_repr)
        return reward.squeeze(-1)

def ranking_loss(rewards_preferred, rewards_dispreferred):
    \"\"\"Bradley-Terry ranking loss\"\"\"
    return -torch.log(torch.sigmoid(rewards_preferred - rewards_dispreferred)).mean()

# Training loop
def train_reward_model(model, dataloader, optimizer):
    model.train()
    for batch in dataloader:
        # Get rewards for preferred and dispreferred responses
        preferred_rewards = model(batch['preferred_input_ids'], batch['preferred_attention_mask'])
        dispreferred_rewards = model(batch['dispreferred_input_ids'], batch['dispreferred_attention_mask'])
        
        # Compute ranking loss
        loss = ranking_loss(preferred_rewards, dispreferred_rewards)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
""", language='python')
    
    with rlhf_tabs[2]:
        st.markdown("### 🎯 Policy Optimization with PPO")
        
        st.markdown("""
        Proximal Policy Optimization (PPO) is used to fine-tune the language model 
        to maximize rewards from the reward model while maintaining reasonable outputs.
        """)
        
        ppo_components = [
            {
                "component": "Policy Network",
                "description": "The language model being optimized",
                "details": [
                    "Starts from SFT model",
                    "Generates responses to prompts",
                    "Updated to maximize reward",
                    "Maintains language generation capability"
                ]
            },
            {
                "component": "Value Network",
                "description": "Estimates expected future rewards",
                "details": [
                    "Often shares parameters with policy",
                    "Helps reduce variance in policy gradients",
                    "Trained alongside policy network",
                    "Critical for stable RL training"
                ]
            },
            {
                "component": "PPO Objective",
                "description": "Balances reward maximization with stability",
                "details": [
                    "Clips policy updates to prevent large changes",
                    "Includes KL divergence penalty vs. SFT model",
                    "Maximizes advantage-weighted likelihood",
                    "Prevents policy collapse"
                ]
            },
            {
                "component": "KL Regularization",
                "description": "Prevents the model from deviating too far from SFT",
                "details": [
                    "Adds KL penalty between current and SFT policy",
                    "Maintains coherent language generation",
                    "Prevents reward hacking",
                    "Balances alignment with capability"
                ]
            }
        ]
        
        for component in ppo_components:
            with st.expander(f"🎯 {component['component']}"):
                st.markdown(component['description'])
                
                for detail in component['details']:
                    st.markdown(f"• {detail}")
        
        # PPO algorithm overview
        st.markdown("### 🔄 PPO Algorithm Steps")
        
        ppo_steps = [
            "Sample prompts from dataset",
            "Generate responses using current policy",
            "Score responses with reward model",
            "Compute advantages using value network",
            "Update policy using clipped PPO objective",
            "Update value network to predict rewards",
            "Apply KL penalty to maintain stability"
        ]
        
        for i, step in enumerate(ppo_steps, 1):
            st.markdown(f"{i}. {step}")

# Fine-tuning strategies
st.header("📊 Fine-Tuning Strategy Selection")

strategy_tabs = st.tabs(["🎯 Task-Based Selection", "💰 Resource Constraints", "📈 Performance Trade-offs"])

with strategy_tabs[0]:
    st.subheader("🎯 Task-Based Strategy Selection")
    
    task_strategies = {
        "Text Classification": {
            "recommended": "LoRA or Full Fine-tuning",
            "alternatives": ["Adapters", "Prefix Tuning"],
            "considerations": ["Dataset size", "Number of classes", "Performance requirements"],
            "typical_performance": "90-95% of full fine-tuning with LoRA"
        },
        "Question Answering": {
            "recommended": "Full Fine-tuning or LoRA",
            "alternatives": ["QLoRA for large models"],
            "considerations": ["Context length", "Answer format", "Domain specificity"],
            "typical_performance": "Best with full fine-tuning, 85-90% with LoRA"
        },
        "Text Generation": {
            "recommended": "Instruction Tuning + RLHF",
            "alternatives": ["LoRA", "Prefix Tuning"],
            "considerations": ["Quality requirements", "Safety concerns", "Computational budget"],
            "typical_performance": "RLHF provides best alignment with human preferences"
        },
        "Summarization": {
            "recommended": "Full Fine-tuning or LoRA",
            "alternatives": ["Instruction Tuning"],
            "considerations": ["Summary length", "Domain adaptation", "Evaluation metrics"],
            "typical_performance": "LoRA achieves 90-95% of full fine-tuning performance"
        },
        "Code Generation": {
            "recommended": "Instruction Tuning",
            "alternatives": ["LoRA", "Full Fine-tuning"],
            "considerations": ["Programming languages", "Code quality", "Safety"],
            "typical_performance": "Instruction tuning often outperforms traditional fine-tuning"
        }
    }
    
    selected_task = st.selectbox("Select your task:", list(task_strategies.keys()))
    
    if selected_task:
        strategy = task_strategies[selected_task]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**🎯 Recommended Approach:**")
            st.success(strategy['recommended'])
            
            st.markdown(f"**🔄 Alternative Methods:**")
            for alt in strategy['alternatives']:
                st.markdown(f"• {alt}")
        
        with col2:
            st.markdown(f"**🤔 Key Considerations:**")
            for consideration in strategy['considerations']:
                st.markdown(f"• {consideration}")
            
            st.markdown(f"**📈 Expected Performance:**")
            st.info(strategy['typical_performance'])

with strategy_tabs[1]:
    st.subheader("💰 Resource-Based Recommendations")
    
    # Resource calculator
    st.markdown("### 🧮 Resource Requirements Calculator")
    
    calc_col1, calc_col2 = st.columns(2)
    
    with calc_col1:
        model_size = st.selectbox("Model Size", ["125M", "350M", "1.3B", "7B", "13B", "70B"])
        method = st.selectbox("Fine-tuning Method", ["Full FT", "LoRA", "QLoRA", "Prefix Tuning"])
        dataset_size = st.number_input("Dataset Size (examples)", 1000, 1000000, 10000)
    
    with calc_col2:
        gpu_type = st.selectbox("Available GPU", ["RTX 3090", "RTX 4090", "A100", "H100"])
        training_hours = st.slider("Available Training Time (hours)", 1, 168, 24)
    
    # Calculate feasibility
    if st.button("Check Feasibility"):
        # Simplified feasibility calculation
        model_memory = {
            "125M": 0.5, "350M": 1.4, "1.3B": 5.2, 
            "7B": 28, "13B": 52, "70B": 280
        }[model_size]
        
        method_multiplier = {
            "Full FT": 3.5, "LoRA": 1.2, "QLoRA": 0.8, "Prefix Tuning": 1.1
        }[method]
        
        gpu_memory = {
            "RTX 3090": 24, "RTX 4090": 24, "A100": 80, "H100": 80
        }[gpu_type]
        
        required_memory = model_memory * method_multiplier
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Required Memory", f"{required_memory:.1f}GB")
        with col2:
            st.metric("Available Memory", f"{gpu_memory}GB")
        with col3:
            feasible = required_memory <= gpu_memory
            st.metric("Feasible", "✅ Yes" if feasible else "❌ No")
        
        if feasible:
            st.success("✅ This configuration should work with your available resources!")
        else:
            st.error("❌ This configuration requires more memory than available.")
            st.info("💡 Try: Smaller model, QLoRA, or gradient checkpointing")

with strategy_tabs[2]:
    st.subheader("📈 Performance vs. Efficiency Trade-offs")
    
    # Trade-off visualization
    methods_data = {
        'Method': ['Full Fine-tuning', 'LoRA', 'QLoRA', 'Prefix Tuning', 'Adapters'],
        'Performance (%)': [100, 92, 95, 85, 88],
        'Efficiency (%)': [20, 95, 98, 90, 85],
        'Memory Usage (%)': [100, 25, 15, 30, 40]
    }
    
    methods_df = pd.DataFrame(methods_data)
    
    fig = px.scatter(
        methods_df, 
        x='Efficiency (%)', 
        y='Performance (%)',
        size='Memory Usage (%)',
        hover_name='Method',
        title="Fine-tuning Methods: Performance vs. Efficiency Trade-off",
        labels={'size': 'Memory Usage (%)'}
    )
    
    # Add annotations for each method
    for i, row in methods_df.iterrows():
        fig.add_annotation(
            x=row['Efficiency (%)'],
            y=row['Performance (%)'],
            text=row['Method'],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1
        )
    
    st.plotly_chart(fig, use_container_width=True)

# Interactive fine-tuning planner
st.header("🛠️ Interactive Fine-Tuning Planner")

planner_tabs = st.tabs(["🎯 Goal Setting", "📋 Method Selection", "⏱️ Timeline Planning"])

with planner_tabs[0]:
    st.subheader("🎯 Define Your Fine-Tuning Goals")
    
    # Goal setting form
    primary_goal = st.selectbox(
        "Primary Goal:",
        ["Improve task performance", "Adapt to domain", "Align with preferences", "Reduce model size"]
    )
    
    performance_target = st.slider("Target Performance (% of full fine-tuning)", 70, 100, 90)
    
    constraints = st.multiselect(
        "Resource Constraints:",
        ["Limited GPU memory", "Limited training time", "Limited dataset", "Cost sensitivity"]
    )
    
    safety_requirements = st.checkbox("Requires safety alignment")
    deployment_requirements = st.multiselect(
        "Deployment Requirements:",
        ["Fast inference", "Small model size", "Multiple tasks", "Easy updates"]
    )

with planner_tabs[1]:
    st.subheader("📋 Recommended Method")
    
    # Method recommendation logic
    if primary_goal == "Align with preferences":
        recommended_method = "RLHF"
        explanation = "Human preference alignment requires RLHF for best results"
    elif "Limited GPU memory" in constraints:
        recommended_method = "QLoRA"
        explanation = "QLoRA provides excellent efficiency for memory-constrained environments"
    elif performance_target >= 95:
        recommended_method = "Full Fine-tuning"
        explanation = "High performance targets typically require full parameter updates"
    else:
        recommended_method = "LoRA"
        explanation = "LoRA offers good balance of performance and efficiency"
    
    st.success(f"🎯 **Recommended Method:** {recommended_method}")
    st.info(f"💡 **Reasoning:** {explanation}")
    
    # Implementation checklist
    st.markdown("### ✅ Implementation Checklist")
    
    checklist_items = [
        "Dataset prepared and validated",
        "Base model selected and tested",
        "Training infrastructure set up",
        "Evaluation metrics defined",
        "Hyperparameters configured",
        "Monitoring and logging enabled",
        "Checkpoint saving implemented",
        "Evaluation pipeline ready"
    ]
    
    for item in checklist_items:
        st.checkbox(item)

with planner_tabs[2]:
    st.subheader("⏱️ Training Timeline")
    
    # Timeline estimation
    estimated_days = {
        "Full Fine-tuning": 7,
        "LoRA": 2,
        "QLoRA": 3,
        "RLHF": 21,
        "Prefix Tuning": 2
    }.get(recommended_method, 3)
    
    phases = [
        ("Setup & Preparation", 1),
        ("Initial Training", estimated_days),
        ("Evaluation & Tuning", 2),
        ("Final Validation", 1)
    ]
    
    total_days = sum(days for _, days in phases)
    
    st.markdown(f"**Estimated Total Time:** {total_days} days")
    
    # Timeline visualization
    timeline_fig = go.Figure()
    
    cumulative_days = 0
    colors = ['lightblue', 'lightgreen', 'orange', 'lightcoral']
    
    for i, (phase, days) in enumerate(phases):
        timeline_fig.add_trace(go.Bar(
            x=[days],
            y=[phase],
            orientation='h',
            name=phase,
            marker_color=colors[i],
            text=f"{days} days",
            textposition="middle center"
        ))
    
    timeline_fig.update_layout(
        title="Fine-tuning Timeline",
        xaxis_title="Days",
        showlegend=False,
        height=300
    )
    
    st.plotly_chart(timeline_fig, use_container_width=True)

# Best practices
st.header("🎯 Fine-Tuning Best Practices")

best_practices = [
    "**Start with strong baselines**: Use well-established pre-trained models",
    "**Validate your data**: Ensure high-quality, relevant training examples",
    "**Monitor overfitting**: Use validation sets and early stopping",
    "**Experiment systematically**: Try multiple approaches and compare results",
    "**Document everything**: Keep detailed records of experiments and results",
    "**Test safety**: Evaluate for harmful or biased outputs",
    "**Plan for deployment**: Consider inference speed and resource requirements",
    "**Iterate based on evaluation**: Use evaluation results to guide improvements",
    "**Consider efficiency**: Parameter-efficient methods often provide great ROI",
    "**Align with human preferences**: Use RLHF for user-facing applications"
]

for practice in best_practices:
    st.markdown(f"• {practice}")
