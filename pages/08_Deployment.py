import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Model Deployment", page_icon="🚀", layout="wide")

st.title("🚀 Model Deployment")
st.markdown("### From Trained Model to Production System")

# Overview
st.header("🎯 Overview")
st.markdown("""
Model deployment transforms your trained LLM into a production-ready system that can serve real users. 
This involves choosing deployment strategies, optimizing for performance and cost, implementing safety measures, 
and setting up monitoring and maintenance systems.
""")

# Deployment strategies
st.header("🏗️ Deployment Strategies")

strategy_tabs = st.tabs([
    "☁️ Cloud Deployment",
    "🏠 On-Premises", 
    "📱 Edge Deployment",
    "🔄 Hybrid Approaches"
])

with strategy_tabs[0]:
    st.subheader("☁️ Cloud Deployment Options")
    
    cloud_options = [
        {
            "option": "Managed AI Services",
            "description": "Use cloud provider's managed LLM services",
            "examples": [
                "OpenAI API (GPT-3.5, GPT-4)",
                "Azure OpenAI Service",
                "Google Vertex AI",
                "AWS Bedrock",
                "Anthropic Claude API"
            ],
            "advantages": [
                "No infrastructure management",
                "Automatic scaling",
                "High availability",
                "Pay-per-use pricing",
                "Latest model updates"
            ],
            "disadvantages": [
                "Limited customization",
                "Data privacy concerns",
                "Vendor lock-in",
                "Ongoing costs",
                "Less control over performance"
            ],
            "best_for": "Rapid prototyping, MVP development, standard use cases",
            "cost_model": "Pay-per-token or API call"
        },
        {
            "option": "Self-Hosted on Cloud VMs",
            "description": "Deploy your own models on cloud virtual machines",
            "examples": [
                "AWS EC2 with GPU instances",
                "Google Cloud Compute Engine",
                "Azure Virtual Machines",
                "DigitalOcean Droplets",
                "Linode GPU instances"
            ],
            "advantages": [
                "Full control over model",
                "Custom optimizations",
                "Data stays in your infrastructure",
                "Predictable costs",
                "Can use custom models"
            ],
            "disadvantages": [
                "Infrastructure management required",
                "Scaling complexity",
                "Security responsibility",
                "Higher upfront costs",
                "Maintenance overhead"
            ],
            "best_for": "Custom models, data privacy requirements, cost optimization",
            "cost_model": "Fixed instance costs + storage"
        },
        {
            "option": "Container Orchestration",
            "description": "Deploy using Kubernetes, Docker, or container services",
            "examples": [
                "Amazon EKS",
                "Google GKE", 
                "Azure AKS",
                "Docker Swarm",
                "AWS Fargate"
            ],
            "advantages": [
                "Excellent scalability",
                "Resource efficiency",
                "Microservices architecture",
                "Easy CI/CD integration",
                "Environment consistency"
            ],
            "disadvantages": [
                "Container orchestration complexity",
                "Learning curve",
                "Debugging challenges",
                "Resource overhead",
                "Networking complexity"
            ],
            "best_for": "Microservices architecture, auto-scaling, DevOps teams",
            "cost_model": "Container runtime costs + orchestration"
        },
        {
            "option": "Serverless Deployment",
            "description": "Deploy using serverless computing platforms",
            "examples": [
                "AWS Lambda",
                "Google Cloud Functions",
                "Azure Functions",
                "Vercel Functions",
                "Cloudflare Workers"
            ],
            "advantages": [
                "Zero server management",
                "Automatic scaling to zero",
                "Pay only for execution time",
                "Built-in fault tolerance",
                "Fast deployment"
            ],
            "disadvantages": [
                "Cold start latency",
                "Execution time limits",
                "Memory constraints",
                "Limited to smaller models",
                "Vendor-specific APIs"
            ],
            "best_for": "Small models, intermittent usage, cost-sensitive applications",
            "cost_model": "Pay-per-execution time"
        }
    ]
    
    for option in cloud_options:
        with st.expander(f"☁️ {option['option']}"):
            st.markdown(option['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📋 Examples:**")
                for example in option['examples']:
                    st.markdown(f"• {example}")
                
                st.markdown("**✅ Advantages:**")
                for advantage in option['advantages']:
                    st.markdown(f"• {advantage}")
            
            with col2:
                st.markdown("**❌ Disadvantages:**")
                for disadvantage in option['disadvantages']:
                    st.markdown(f"• {disadvantage}")
            
            st.info(f"**🎯 Best for:** {option['best_for']}")
            st.info(f"**💰 Cost Model:** {option['cost_model']}")
    
    # Cloud deployment code example
    st.markdown("### 💻 Cloud Deployment Example")
    
    deployment_example = st.selectbox(
        "Select deployment example:",
        ["AWS SageMaker", "Google Cloud Run", "Azure Container Instances"]
    )
    
    if deployment_example == "AWS SageMaker":
        st.code("""
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Create Hugging Face Model
huggingface_model = HuggingFaceModel(
    model_data="s3://your-bucket/model.tar.gz",
    role=role,
    transformers_version="4.21",
    pytorch_version="1.12",
    py_version="py39",
    env={
        'HF_MODEL_ID': 'your-model-name',
        'HF_TASK': 'text-generation'
    }
)

# Deploy model
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge",
    endpoint_name="llm-endpoint"
)

# Make predictions
result = predictor.predict({
    'inputs': 'The future of AI is',
    'parameters': {
        'max_new_tokens': 100,
        'temperature': 0.7
    }
})
""", language='python')
    
    elif deployment_example == "Google Cloud Run":
        st.code("""
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app

# main.py
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
generator = pipeline('text-generation', model='your-model-name')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    
    result = generator(prompt, max_new_tokens=100, temperature=0.7)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

# Deploy command
# gcloud run deploy llm-service --source . --platform managed --region us-central1 --allow-unauthenticated
""", language='python')
    
    elif deployment_example == "Azure Container Instances":
        st.code("""
# docker-compose.yml
version: '3.8'
services:
  llm-service:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MODEL_NAME=your-model-name
    resources:
      limits:
        memory: 16G
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]

# Azure deployment
az container create \\
  --resource-group myResourceGroup \\
  --name llm-container \\
  --image your-registry/llm-service:latest \\
  --cpu 4 \\
  --memory 16 \\
  --gpu-count 1 \\
  --gpu-sku V100 \\
  --ports 8080
""", language='yaml')

with strategy_tabs[1]:
    st.subheader("🏠 On-Premises Deployment")
    
    st.markdown("""
    On-premises deployment gives you maximum control over your infrastructure and data, 
    but requires significant hardware investment and technical expertise.
    """)
    
    onprem_aspects = [
        {
            "aspect": "Hardware Requirements",
            "description": "Compute and storage infrastructure needed",
            "considerations": [
                "GPU requirements (NVIDIA A100, H100, RTX series)",
                "CPU cores and RAM (typically 64GB+ for inference)",
                "Storage (NVMe SSDs for model loading)",
                "Network bandwidth for multi-node setups",
                "Power and cooling requirements"
            ],
            "example_configs": {
                "Small Model (7B)": "1x RTX 4090, 32GB RAM, 1TB NVMe SSD",
                "Medium Model (13B)": "2x RTX 4090, 64GB RAM, 2TB NVMe SSD", 
                "Large Model (70B)": "4x A100 80GB, 256GB RAM, 4TB NVMe SSD"
            }
        },
        {
            "aspect": "Software Stack",
            "description": "Required software components and frameworks",
            "considerations": [
                "Operating system (Ubuntu 20.04/22.04 recommended)",
                "CUDA drivers and toolkit",
                "Container runtime (Docker, Podman)",
                "Orchestration (Kubernetes, Docker Swarm)",
                "Inference servers (TensorRT-LLM, vLLM, TGI)"
            ],
            "example_configs": {
                "Basic Setup": "Ubuntu 22.04 + CUDA 12.1 + Docker",
                "Production Setup": "Ubuntu 22.04 + CUDA 12.1 + Kubernetes + Prometheus",
                "High Performance": "Ubuntu 22.04 + CUDA 12.1 + TensorRT-LLM + NVIDIA Triton"
            }
        },
        {
            "aspect": "Security Considerations",
            "description": "Protecting your deployment and data",
            "considerations": [
                "Network security (firewalls, VPNs)",
                "Access control and authentication",
                "Data encryption at rest and in transit",
                "Regular security updates",
                "Audit logging and monitoring"
            ],
            "example_configs": {
                "Basic Security": "Firewall + SSH keys + HTTPS",
                "Enterprise Security": "Zero-trust network + MFA + encryption + SIEM",
                "Compliance Ready": "Full audit trail + role-based access + data governance"
            }
        }
    ]
    
    for aspect in onprem_aspects:
        with st.expander(f"🏗️ {aspect['aspect']}"):
            st.markdown(aspect['description'])
            
            st.markdown("**🔧 Key Considerations:**")
            for consideration in aspect['considerations']:
                st.markdown(f"• {consideration}")
            
            st.markdown("**📋 Example Configurations:**")
            for config_name, config_details in aspect['example_configs'].items():
                st.markdown(f"• **{config_name}**: {config_details}")
    
    # On-premises deployment script
    st.markdown("### 🛠️ On-Premises Setup Example")
    
    st.code("""
#!/bin/bash
# On-premises LLM deployment script

# Install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io nvidia-container-toolkit

# Configure NVIDIA Container Runtime
sudo systemctl restart docker

# Pull model serving image
docker pull vllm/vllm-openai:latest

# Create model directory
sudo mkdir -p /opt/models
sudo chown $USER:$USER /opt/models

# Download model (example with Hugging Face)
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='microsoft/DialoGPT-large', local_dir='/opt/models/dialogpt')
"

# Run inference server
docker run -d \\
  --name llm-server \\
  --gpus all \\
  -p 8000:8000 \\
  -v /opt/models:/models \\
  vllm/vllm-openai:latest \\
  --model /models/dialogpt \\
  --host 0.0.0.0 \\
  --port 8000

# Setup reverse proxy (nginx)
sudo apt install -y nginx
sudo tee /etc/nginx/sites-available/llm > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \\$host;
        proxy_set_header X-Real-IP \\$remote_addr;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/llm /etc/nginx/sites-enabled/
sudo systemctl restart nginx

echo "LLM deployment completed!"
echo "Access your model at http://your-domain.com"
""", language='bash')

with strategy_tabs[2]:
    st.subheader("📱 Edge Deployment")
    
    st.markdown("""
    Edge deployment brings models closer to users, reducing latency and enabling offline operation. 
    This requires model optimization and specialized hardware considerations.
    """)
    
    edge_scenarios = [
        {
            "scenario": "Mobile Devices",
            "description": "Deploy models on smartphones and tablets",
            "requirements": [
                "Model size < 1GB for mobile apps",
                "ARM processor optimization",
                "Battery efficiency considerations",
                "iOS/Android framework support"
            ],
            "techniques": [
                "Model quantization (INT8, INT4)",
                "Knowledge distillation",
                "Pruning and sparsification",
                "Mobile-optimized frameworks (Core ML, TensorFlow Lite)"
            ],
            "use_cases": [
                "Offline translation",
                "Voice assistants",
                "Text completion",
                "Image captioning"
            ]
        },
        {
            "scenario": "IoT Devices",
            "description": "Deploy on resource-constrained embedded systems",
            "requirements": [
                "Extremely small model size (<100MB)",
                "Low power consumption",
                "Limited RAM and storage",
                "Real-time processing constraints"
            ],
            "techniques": [
                "Extreme quantization",
                "Model compilation",
                "Custom hardware acceleration",
                "Edge TPU optimization"
            ],
            "use_cases": [
                "Smart speakers",
                "Security cameras",
                "Industrial sensors",
                "Automotive systems"
            ]
        },
        {
            "scenario": "Edge Servers",
            "description": "Deploy on local servers at network edge",
            "requirements": [
                "Regional data compliance",
                "Low latency requirements",
                "Moderate compute resources",
                "Remote management capabilities"
            ],
            "techniques": [
                "Model caching strategies",
                "Load balancing",
                "Federated deployment",
                "Edge orchestration"
            ],
            "use_cases": [
                "Content delivery networks",
                "Regional AI services",
                "Gaming applications",
                "Financial trading"
            ]
        }
    ]
    
    for scenario in edge_scenarios:
        with st.expander(f"📱 {scenario['scenario']}"):
            st.markdown(scenario['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**⚙️ Requirements:**")
                for req in scenario['requirements']:
                    st.markdown(f"• {req}")
                
                st.markdown("**🔧 Optimization Techniques:**")
                for tech in scenario['techniques']:
                    st.markdown(f"• {tech}")
            
            with col2:
                st.markdown("**🎯 Use Cases:**")
                for use_case in scenario['use_cases']:
                    st.markdown(f"• {use_case}")
    
    # Edge optimization example
    st.markdown("### 🔧 Model Optimization for Edge")
    
    st.code("""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.quantization as quantization

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Set model to evaluation mode
model.eval()

# 1. Dynamic Quantization (easiest approach)
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

# 2. Static Quantization (better performance)
# Prepare model for quantization
model.qconfig = quantization.get_default_qconfig('fbgemm')
prepared_model = quantization.prepare(model, inplace=False)

# Calibrate with sample data
def calibrate(model, tokenizer, sample_texts):
    with torch.no_grad():
        for text in sample_texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            model(**inputs)

# Run calibration
sample_texts = ["The future of AI is", "Machine learning enables", "Deep learning models"]
calibrate(prepared_model, tokenizer, sample_texts)

# Convert to quantized model
static_quantized_model = quantization.convert(prepared_model, inplace=False)

# 3. Export to mobile format (PyTorch Mobile)
# First convert to TorchScript
traced_model = torch.jit.trace(quantized_model, example_inputs)

# Optimize for mobile
from torch.utils.mobile_optimizer import optimize_for_mobile
mobile_model = optimize_for_mobile(traced_model)

# Save mobile model
mobile_model._save_for_lite_interpreter("model_mobile.ptl")

# Size comparison
original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024 / 1024

print(f"Original model size: {original_size:.1f} MB")
print(f"Quantized model size: {quantized_size:.1f} MB")
print(f"Compression ratio: {original_size/quantized_size:.1f}x")
""", language='python')

with strategy_tabs[3]:
    st.subheader("🔄 Hybrid Deployment Strategies")
    
    st.markdown("""
    Hybrid approaches combine multiple deployment strategies to optimize for performance, 
    cost, and specific requirements across different use cases.
    """)
    
    hybrid_patterns = [
        {
            "pattern": "Cloud-Edge Federation",
            "description": "Distribute computation between cloud and edge based on requirements",
            "architecture": [
                "Heavy models in cloud for complex queries",
                "Light models on edge for fast responses",
                "Intelligent routing based on query complexity",
                "Fallback mechanisms for reliability"
            ],
            "benefits": [
                "Optimal latency for different query types",
                "Cost optimization through tiered serving",
                "Improved reliability and availability",
                "Better user experience"
            ],
            "example": "Mobile app with local text completion + cloud reasoning"
        },
        {
            "pattern": "Multi-Cloud Deployment",
            "description": "Deploy across multiple cloud providers for redundancy and optimization",
            "architecture": [
                "Primary deployment on preferred cloud",
                "Secondary deployment for failover",
                "Geographic distribution for latency",
                "Load balancing across providers"
            ],
            "benefits": [
                "Reduced vendor lock-in",
                "Improved global performance",
                "Higher availability through redundancy",
                "Cost optimization through competition"
            ],
            "example": "Global service with AWS (US), GCP (Europe), Azure (Asia)"
        },
        {
            "pattern": "Tiered Model Serving",
            "description": "Use different model sizes/capabilities for different complexity levels",
            "architecture": [
                "Small model for simple queries (fast response)",
                "Medium model for moderate complexity",
                "Large model for complex reasoning",
                "Query classification for routing"
            ],
            "benefits": [
                "Cost-effective resource utilization",
                "Optimized response times",
                "Scalable architecture",
                "Better quality-cost trade-offs"
            ],
            "example": "FAQ bot (small) + customer support (medium) + research assistant (large)"
        }
    ]
    
    for pattern in hybrid_patterns:
        with st.expander(f"🔄 {pattern['pattern']}"):
            st.markdown(pattern['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🏗️ Architecture Components:**")
                for component in pattern['architecture']:
                    st.markdown(f"• {component}")
            
            with col2:
                st.markdown("**✅ Benefits:**")
                for benefit in pattern['benefits']:
                    st.markdown(f"• {benefit}")
            
            st.info(f"**💡 Example:** {pattern['example']}")

# Performance optimization
st.header("⚡ Performance Optimization")

perf_tabs = st.tabs([
    "🚀 Inference Acceleration",
    "💾 Memory Optimization", 
    "📊 Scaling Strategies",
    "🔧 Hardware Optimization"
])

with perf_tabs[0]:
    st.subheader("🚀 Inference Acceleration Techniques")
    
    acceleration_techniques = [
        {
            "technique": "Model Quantization",
            "description": "Reduce precision to speed up inference",
            "approaches": [
                "Post-training quantization (PTQ)",
                "Quantization-aware training (QAT)", 
                "Dynamic quantization",
                "Static quantization"
            ],
            "precision_options": [
                "FP32 → FP16 (2x speedup)",
                "FP16 → INT8 (2-4x speedup)",
                "INT8 → INT4 (2x additional speedup)",
                "Custom precision formats"
            ],
            "trade_offs": "Speed gain vs. potential quality loss",
            "implementation": """
# Example: INT8 quantization with optimum
from optimum.intel import INCQuantizer

quantizer = INCQuantizer.from_pretrained(model_name)
quantized_model = quantizer.quantize(
    calibration_dataset=calibration_data,
    save_directory="quantized_model"
)
"""
        },
        {
            "technique": "Model Compilation",
            "description": "Optimize computation graphs for target hardware",
            "approaches": [
                "TensorRT (NVIDIA GPUs)",
                "OpenVINO (Intel hardware)",
                "TVM (cross-platform)",
                "torch.compile() (PyTorch 2.0+)"
            ],
            "optimizations": [
                "Operator fusion",
                "Memory layout optimization",
                "Kernel auto-tuning",
                "Graph-level optimizations"
            ],
            "trade_offs": "Compilation time vs. inference speedup",
            "implementation": """
# Example: TensorRT optimization
import tensorrt as trt
from torch2trt import torch2trt

# Convert PyTorch model to TensorRT
model_trt = torch2trt(
    model, 
    [example_input], 
    fp16_mode=True,
    max_workspace_size=1<<30
)

# Use optimized model
output = model_trt(input_tensor)
"""
        },
        {
            "technique": "Speculative Decoding",
            "description": "Use smaller model to propose tokens, larger model to verify",
            "approaches": [
                "Draft-then-verify pipeline",
                "Parallel speculation",
                "Tree-based speculation",
                "Adaptive speculation depth"
            ],
            "benefits": [
                "Faster generation for long sequences",
                "No quality degradation",
                "Adaptable to different scenarios",
                "Memory efficient"
            ],
            "trade_offs": "Complexity vs. generation speedup",
            "implementation": """
# Example: Simple speculative decoding
def speculative_decode(draft_model, target_model, prompt, k=5):
    draft_tokens = draft_model.generate(prompt, max_new_tokens=k)
    acceptance_probs = target_model.score_sequence(draft_tokens)
    
    # Accept tokens based on probability threshold
    accepted_tokens = []
    for token, prob in zip(draft_tokens, acceptance_probs):
        if prob > threshold:
            accepted_tokens.append(token)
        else:
            break
    
    return accepted_tokens
"""
        }
    ]
    
    for technique in acceleration_techniques:
        with st.expander(f"🚀 {technique['technique']}"):
            st.markdown(technique['description'])
            
            if 'approaches' in technique:
                st.markdown("**🔧 Approaches:**")
                for approach in technique['approaches']:
                    st.markdown(f"• {approach}")
            
            if 'precision_options' in technique:
                st.markdown("**📊 Precision Options:**")
                for option in technique['precision_options']:
                    st.markdown(f"• {option}")
            
            if 'benefits' in technique:
                st.markdown("**✅ Benefits:**")
                for benefit in technique['benefits']:
                    st.markdown(f"• {benefit}")
            
            st.info(f"**⚖️ Trade-offs:** {technique['trade_offs']}")
            
            st.markdown("**💻 Implementation:**")
            st.code(technique['implementation'], language='python')

with perf_tabs[1]:
    st.subheader("💾 Memory Optimization")
    
    memory_techniques = [
        "**KV Cache Optimization**: Reduce memory usage of attention key-value caches",
        "**Gradient Checkpointing**: Trade computation for memory during inference",
        "**Model Sharding**: Split large models across multiple devices",
        "**Offloading**: Move inactive parameters to CPU or disk",
        "**Memory Mapping**: Use memory-mapped files for large models",
        "**Batch Size Tuning**: Optimize batch size for memory constraints"
    ]
    
    for technique in memory_techniques:
        st.markdown(f"• {technique}")
    
    # Memory optimization code example
    st.markdown("### 🧠 Memory Optimization Example")
    
    st.code("""
import torch
from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# 1. Load model with memory optimization
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,  # Use half precision
        low_cpu_mem_usage=True      # Reduce CPU memory during loading
    )

# 2. Model sharding across devices
device_map = "auto"  # Automatically distribute across available GPUs
model = load_checkpoint_and_dispatch(
    model, 
    checkpoint_path, 
    device_map=device_map,
    no_split_module_classes=["TransformerBlock"]
)

# 3. KV cache optimization
def generate_with_kv_cache_optimization(model, input_ids, max_length=100):
    with torch.no_grad():
        past_key_values = None
        generated_ids = input_ids.clone()
        
        for _ in range(max_length):
            outputs = model(
                generated_ids[:, -1:] if past_key_values else generated_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # Update cache efficiently
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Clear unnecessary gradients
            if hasattr(outputs, 'logits'):
                del outputs.logits
            torch.cuda.empty_cache()
        
        return generated_ids

# 4. Memory monitoring
def monitor_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    import psutil
    cpu_memory = psutil.virtual_memory().used / 1024**3
    print(f"CPU Memory Used: {cpu_memory:.2f}GB")

# Monitor before and after optimization
monitor_memory_usage()
""", language='python')

with perf_tabs[2]:
    st.subheader("📊 Scaling Strategies")
    
    scaling_strategies = [
        {
            "strategy": "Horizontal Scaling",
            "description": "Add more instances to handle increased load",
            "implementation": [
                "Load balancer distributes requests",
                "Multiple model replicas",
                "Auto-scaling groups",
                "Request queuing systems"
            ],
            "advantages": ["Linear scalability", "Fault tolerance", "Cost-effective"],
            "challenges": ["Load balancing complexity", "State management", "Network overhead"]
        },
        {
            "strategy": "Vertical Scaling", 
            "description": "Increase resources of existing instances",
            "implementation": [
                "Larger GPU memory",
                "More CPU cores",
                "Faster storage",
                "Increased network bandwidth"
            ],
            "advantages": ["Simpler architecture", "Lower latency", "Better resource utilization"],
            "challenges": ["Hardware limits", "Single point of failure", "Cost scaling"]
        },
        {
            "strategy": "Model Parallelism",
            "description": "Split model across multiple devices",
            "implementation": [
                "Pipeline parallelism",
                "Tensor parallelism",
                "Expert parallelism (MoE)",
                "Sequence parallelism"
            ],
            "advantages": ["Handle very large models", "Memory distribution", "Compute parallelization"],
            "challenges": ["Communication overhead", "Load balancing", "Debugging complexity"]
        }
    ]
    
    for strategy in scaling_strategies:
        with st.expander(f"📊 {strategy['strategy']}"):
            st.markdown(strategy['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔧 Implementation:**")
                for impl in strategy['implementation']:
                    st.markdown(f"• {impl}")
                
                st.markdown("**✅ Advantages:**")
                for adv in strategy['advantages']:
                    st.markdown(f"• {adv}")
            
            with col2:
                st.markdown("**⚠️ Challenges:**")
                for challenge in strategy['challenges']:
                    st.markdown(f"• {challenge}")

with perf_tabs[3]:
    st.subheader("🔧 Hardware Optimization")
    
    # Hardware comparison chart
    hardware_data = {
        'Hardware': ['CPU (Intel Xeon)', 'GPU (RTX 4090)', 'GPU (A100)', 'TPU v4', 'Apple M2 Ultra'],
        'Inference Speed (tokens/s)': [50, 200, 400, 300, 150],
        'Memory (GB)': [256, 24, 80, 32, 192],
        'Cost ($/hour)': [0.5, 1.5, 4.0, 2.5, 0.8],
        'Power Efficiency': [60, 85, 75, 90, 95]
    }
    
    hardware_df = pd.DataFrame(hardware_data)
    
    # Performance vs cost visualization
    fig = px.scatter(
        hardware_df,
        x='Cost ($/hour)',
        y='Inference Speed (tokens/s)',
        size='Memory (GB)',
        color='Power Efficiency',
        hover_name='Hardware',
        title="Hardware Performance vs Cost Trade-off",
        labels={'size': 'Memory (GB)', 'color': 'Power Efficiency (%)'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Hardware optimization tips
    hardware_tips = [
        "**GPU Selection**: Choose based on model size and batch requirements",
        "**Memory Bandwidth**: Critical for large model inference performance", 
        "**Tensor Cores**: Utilize specialized units for mixed-precision operations",
        "**NVLink/Interconnects**: Important for multi-GPU deployments",
        "**Storage Speed**: Fast SSDs reduce model loading time",
        "**Cooling**: Maintain optimal temperatures for sustained performance"
    ]
    
    for tip in hardware_tips:
        st.markdown(f"• {tip}")

# Monitoring and maintenance
st.header("📊 Monitoring & Maintenance")

monitoring_tabs = st.tabs([
    "📈 Performance Monitoring",
    "🔍 Health Checks", 
    "🚨 Alerting",
    "🔄 Model Updates"
])

with monitoring_tabs[0]:
    st.subheader("📈 Performance Monitoring")
    
    # Sample monitoring dashboard
    st.markdown("### 📊 Key Performance Metrics")
    
    # Simulate real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Requests/min", "1,247", "+12%")
        st.metric("Avg Latency", "245ms", "-8ms")
    
    with col2:
        st.metric("GPU Utilization", "78%", "+5%")
        st.metric("Memory Usage", "12.3GB", "+0.8GB")
    
    with col3:
        st.metric("Error Rate", "0.2%", "-0.1%")
        st.metric("Availability", "99.9%", "0%")
    
    with col4:
        st.metric("Cost/hour", "$4.20", "-$0.30")
        st.metric("Tokens/sec", "342", "+18")
    
    # Performance timeline
    st.markdown("### 📈 Performance Timeline")
    
    # Generate sample data
    hours = list(range(24))
    latency = [200 + 50 * np.sin(h * np.pi / 12) + np.random.normal(0, 10) for h in hours]
    throughput = [300 + 100 * np.cos(h * np.pi / 12) + np.random.normal(0, 20) for h in hours]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours, y=latency,
        mode='lines+markers',
        name='Latency (ms)',
        yaxis='y',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=hours, y=throughput,
        mode='lines+markers', 
        name='Throughput (tokens/s)',
        yaxis='y2',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title="24-Hour Performance Monitoring",
        xaxis_title="Hour of Day",
        yaxis=dict(title="Latency (ms)", side="left"),
        yaxis2=dict(title="Throughput (tokens/s)", side="right", overlaying="y"),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monitoring tools
    st.markdown("### 🛠️ Monitoring Tools")
    
    monitoring_tools = [
        ("Prometheus + Grafana", "Time-series metrics collection and visualization"),
        ("ELK Stack", "Elasticsearch, Logstash, Kibana for log analysis"),
        ("DataDog", "Cloud-native monitoring and analytics platform"),
        ("New Relic", "Application performance monitoring"),
        ("NVIDIA GPU Monitoring", "nvidia-smi, DCGM for GPU metrics"),
        ("Custom Dashboards", "Application-specific metrics and KPIs")
    ]
    
    for tool, description in monitoring_tools:
        st.markdown(f"**{tool}**: {description}")

with monitoring_tabs[1]:
    st.subheader("🔍 Health Checks")
    
    health_check_types = [
        {
            "type": "Liveness Checks",
            "purpose": "Verify service is running and responsive",
            "checks": [
                "HTTP endpoint response",
                "Process status verification",
                "Basic inference test",
                "Memory usage validation"
            ],
            "frequency": "Every 30 seconds",
            "example": """
@app.route('/health/live')
def liveness_check():
    try:
        # Quick inference test
        result = model.generate("test", max_length=5)
        return {"status": "healthy", "timestamp": time.time()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 500
"""
        },
        {
            "type": "Readiness Checks",
            "purpose": "Verify service is ready to handle requests",
            "checks": [
                "Model loaded successfully",
                "Dependencies available",
                "Resource allocation confirmed",
                "Configuration validated"
            ],
            "frequency": "Every 60 seconds",
            "example": """
@app.route('/health/ready')
def readiness_check():
    checks = {
        'model_loaded': model is not None,
        'gpu_available': torch.cuda.is_available(),
        'memory_ok': get_memory_usage() < 0.9
    }
    
    if all(checks.values()):
        return {"status": "ready", "checks": checks}
    else:
        return {"status": "not_ready", "checks": checks}, 503
"""
        },
        {
            "type": "Performance Checks",
            "purpose": "Monitor performance degradation",
            "checks": [
                "Response time thresholds",
                "Quality metrics validation",
                "Resource utilization limits",
                "Error rate monitoring"
            ],
            "frequency": "Every 5 minutes",
            "example": """
def performance_check():
    recent_latencies = get_recent_latencies(minutes=5)
    avg_latency = np.mean(recent_latencies)
    
    if avg_latency > LATENCY_THRESHOLD:
        alert_manager.send_alert("High latency detected", 
                               {"avg_latency": avg_latency})
    
    return {"avg_latency": avg_latency, "status": "ok"}
"""
        }
    ]
    
    for check_type in health_check_types:
        with st.expander(f"🔍 {check_type['type']}"):
            st.markdown(f"**Purpose:** {check_type['purpose']}")
            st.markdown(f"**Frequency:** {check_type['frequency']}")
            
            st.markdown("**Checks Performed:**")
            for check in check_type['checks']:
                st.markdown(f"• {check}")
            
            st.markdown("**Example Implementation:**")
            st.code(check_type['example'], language='python')

with monitoring_tabs[2]:
    st.subheader("🚨 Alerting Systems")
    
    alert_categories = [
        {
            "category": "Performance Alerts",
            "triggers": [
                "Latency > 500ms for 5 minutes",
                "Throughput < 100 tokens/s for 3 minutes", 
                "Error rate > 1% for 2 minutes",
                "Queue depth > 100 requests"
            ],
            "severity": "Warning/Critical",
            "actions": ["Auto-scaling", "Load balancing", "Investigation"]
        },
        {
            "category": "Resource Alerts",
            "triggers": [
                "GPU utilization > 95% for 10 minutes",
                "Memory usage > 90% for 5 minutes",
                "Disk space < 10GB available",
                "CPU temperature > 80°C"
            ],
            "severity": "Warning/Critical", 
            "actions": ["Resource scaling", "Cooling check", "Maintenance"]
        },
        {
            "category": "Quality Alerts",
            "triggers": [
                "Model output quality degradation",
                "Increased toxicity detection",
                "Factual accuracy drops",
                "User satisfaction scores decline"
            ],
            "severity": "Warning/Critical",
            "actions": ["Model validation", "Rollback", "Investigation"]
        }
    ]
    
    for category in alert_categories:
        with st.expander(f"🚨 {category['category']}"):
            st.markdown("**Alert Triggers:**")
            for trigger in category['triggers']:
                st.markdown(f"• {trigger}")
            
            st.markdown(f"**Severity Levels:** {category['severity']}")
            
            st.markdown("**Automated Actions:**")
            for action in category['actions']:
                st.markdown(f"• {action}")
    
    # Alert configuration example
    st.markdown("### ⚙️ Alert Configuration Example")
    
    st.code("""
# Alert configuration (Prometheus AlertManager style)
groups:
  - name: llm_performance
    rules:
      - alert: HighLatency
        expr: avg_latency > 500
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency detected"
          description: "Average latency {{ $value }}ms exceeds threshold"
      
      - alert: LowThroughput  
        expr: throughput < 100
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "Low throughput detected"
          description: "Throughput {{ $value }} tokens/s below minimum"
      
      - alert: GPUMemoryHigh
        expr: gpu_memory_usage > 0.90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High GPU memory usage"
          description: "GPU memory usage at {{ $value }}%"

# Alert handling code
class AlertManager:
    def __init__(self):
        self.alert_channels = {
            'slack': SlackNotifier(),
            'email': EmailNotifier(),
            'pagerduty': PagerDutyNotifier()
        }
    
    def send_alert(self, alert_type, metrics, severity='warning'):
        message = f"Alert: {alert_type}\nMetrics: {metrics}\nSeverity: {severity}"
        
        for channel in self.alert_channels.values():
            channel.send(message)
        
        # Log alert
        logger.warning(f"Alert sent: {alert_type}")
""", language='yaml')

with monitoring_tabs[3]:
    st.subheader("🔄 Model Updates & Rollbacks")
    
    st.markdown("""
    Maintaining deployed models requires strategies for updates, rollbacks, and continuous improvement.
    """)
    
    update_strategies = [
        {
            "strategy": "Blue-Green Deployment",
            "description": "Maintain two identical environments, switch traffic between them",
            "steps": [
                "Deploy new model to green environment",
                "Test green environment thoroughly", 
                "Switch traffic from blue to green",
                "Keep blue as backup for quick rollback"
            ],
            "advantages": ["Zero downtime", "Easy rollback", "Full testing"],
            "disadvantages": ["2x resource requirement", "Complex setup"]
        },
        {
            "strategy": "Canary Deployment",
            "description": "Gradually roll out new model to subset of traffic",
            "steps": [
                "Deploy new model alongside current",
                "Route small percentage to new model",
                "Monitor performance and quality",
                "Gradually increase traffic percentage"
            ],
            "advantages": ["Risk mitigation", "Gradual validation", "Easy monitoring"],
            "disadvantages": ["Slower rollout", "Complex routing logic"]
        },
        {
            "strategy": "A/B Testing",
            "description": "Compare new model against current with real traffic",
            "steps": [
                "Deploy both models in parallel",
                "Split traffic based on user segments",
                "Collect performance metrics",
                "Choose winner based on results"
            ],
            "advantages": ["Data-driven decisions", "User feedback", "Risk mitigation"],
            "disadvantages": ["Resource intensive", "Complex analysis"]
        }
    ]
    
    for strategy in update_strategies:
        with st.expander(f"🔄 {strategy['strategy']}"):
            st.markdown(strategy['description'])
            
            st.markdown("**📋 Deployment Steps:**")
            for step in strategy['steps']:
                st.markdown(f"1. {step}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**✅ Advantages:**")
                for advantage in strategy['advantages']:
                    st.markdown(f"• {advantage}")
            
            with col2:
                st.markdown("**❌ Disadvantages:**")
                for disadvantage in strategy['disadvantages']:
                    st.markdown(f"• {disadvantage}")

# Cost optimization
st.header("💰 Cost Optimization")

cost_tabs = st.tabs(["📊 Cost Analysis", "⚡ Efficiency Strategies", "📈 Cost Monitoring"])

with cost_tabs[0]:
    st.subheader("📊 Cost Breakdown Analysis")
    
    # Cost breakdown pie chart
    cost_categories = ['Compute (GPU)', 'Storage', 'Network', 'Management', 'Monitoring']
    cost_values = [65, 15, 10, 7, 3]
    
    fig = px.pie(
        values=cost_values,
        names=cost_categories,
        title="Typical LLM Deployment Cost Breakdown"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost factors
    cost_factors = [
        {
            "factor": "Model Size",
            "impact": "Larger models require more expensive hardware and memory",
            "optimization": "Use model compression, quantization, or smaller models where appropriate"
        },
        {
            "factor": "Traffic Patterns",
            "impact": "Peak traffic requires overprovisioning, idle time wastes resources",
            "optimization": "Auto-scaling, spot instances, traffic shaping"
        },
        {
            "factor": "Response Time SLA",
            "impact": "Lower latency requirements increase infrastructure costs",
            "optimization": "Tiered service levels, caching, edge deployment"
        },
        {
            "factor": "Geographic Distribution",
            "impact": "Multiple regions increase infrastructure complexity and costs",
            "optimization": "Smart region selection, traffic routing optimization"
        }
    ]
    
    for factor in cost_factors:
        with st.expander(f"💰 {factor['factor']}"):
            st.markdown(f"**Impact:** {factor['impact']}")
            st.markdown(f"**Optimization:** {factor['optimization']}")

with cost_tabs[1]:
    st.subheader("⚡ Cost Efficiency Strategies")
    
    efficiency_strategies = [
        "**Spot Instances**: Use preemptible instances for non-critical workloads (50-90% savings)",
        "**Reserved Instances**: Commit to long-term usage for significant discounts (30-70% savings)",
        "**Auto-scaling**: Scale resources based on demand to avoid overprovisioning",
        "**Model Optimization**: Reduce model size and complexity without sacrificing quality",
        "**Caching**: Cache frequent responses to reduce computation costs",
        "**Load Balancing**: Distribute traffic efficiently to minimize resource waste",
        "**Geographic Optimization**: Choose regions with lower costs for non-latency-critical workloads",
        "**Multi-tenancy**: Share resources across multiple applications or users"
    ]
    
    for strategy in efficiency_strategies:
        st.markdown(f"• {strategy}")
    
    # Cost optimization calculator
    st.markdown("### 🧮 Cost Optimization Calculator")
    
    calc_col1, calc_col2 = st.columns(2)
    
    with calc_col1:
        baseline_cost = st.number_input("Current Monthly Cost ($)", 1000, 100000, 5000)
        spot_savings = st.slider("Spot Instance Usage (%)", 0, 100, 30)
        reserved_savings = st.slider("Reserved Instance Usage (%)", 0, 100, 50)
        optimization_savings = st.slider("Model Optimization Savings (%)", 0, 50, 20)
    
    with calc_col2:
        # Calculate savings
        spot_discount = 0.7  # 70% savings
        reserved_discount = 0.4  # 40% savings
        
        total_savings = (
            (spot_savings / 100) * spot_discount +
            (reserved_savings / 100) * reserved_discount +
            (optimization_savings / 100)
        ) * baseline_cost
        
        optimized_cost = baseline_cost - total_savings
        savings_percentage = (total_savings / baseline_cost) * 100
        
        st.metric("Optimized Monthly Cost", f"${optimized_cost:,.0f}")
        st.metric("Monthly Savings", f"${total_savings:,.0f}")
        st.metric("Savings Percentage", f"{savings_percentage:.1f}%")

with cost_tabs[2]:
    st.subheader("📈 Cost Monitoring & Budgeting")
    
    # Sample cost tracking
    st.markdown("### 📊 Cost Tracking Dashboard")
    
    # Generate sample cost data
    days = list(range(1, 31))
    daily_costs = [100 + 20 * np.sin(d * np.pi / 15) + np.random.normal(0, 5) for d in days]
    cumulative_costs = np.cumsum(daily_costs)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=days, y=daily_costs,
        mode='lines+markers',
        name='Daily Cost',
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=days, y=cumulative_costs,
        mode='lines',
        name='Cumulative Cost',
        yaxis='y2'
    ))
    
    # Add budget line
    budget_line = [3000] * len(days)
    fig.add_trace(go.Scatter(
        x=days, y=budget_line,
        mode='lines',
        name='Monthly Budget',
        line=dict(dash='dash', color='red'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Monthly Cost Tracking",
        xaxis_title="Day of Month",
        yaxis=dict(title="Daily Cost ($)", side="left"),
        yaxis2=dict(title="Cumulative Cost ($)", side="right", overlaying="y"),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost alert thresholds
    st.markdown("### 🚨 Cost Alert Configuration")
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        daily_threshold = st.number_input("Daily Cost Alert ($)", 50, 1000, 150)
        monthly_threshold = st.number_input("Monthly Budget ($)", 1000, 20000, 3000)
    
    with alert_col2:
        spike_threshold = st.slider("Cost Spike Alert (%)", 10, 100, 25)
        forecast_days = st.number_input("Forecast Period (days)", 1, 30, 7)
    
    if st.button("Set Cost Alerts"):
        st.success("✅ Cost monitoring alerts configured successfully!")
        st.info(f"You will be alerted if:")
        st.info(f"• Daily costs exceed ${daily_threshold}")
        st.info(f"• Monthly costs approach ${monthly_threshold}")
        st.info(f"• Cost increase > {spike_threshold}% from previous period")

# Best practices
st.header("🎯 Deployment Best Practices")

best_practices = [
    "**Start Simple**: Begin with basic deployment, optimize incrementally based on real usage",
    "**Plan for Scale**: Design architecture that can handle 10x growth from day one",
    "**Monitor Everything**: Comprehensive monitoring of performance, costs, and quality",
    "**Automate Operations**: Use CI/CD pipelines for model updates and infrastructure changes",
    "**Security First**: Implement proper authentication, encryption, and access controls",
    "**Test Thoroughly**: Extensive testing before production deployment",
    "**Document Procedures**: Clear documentation for deployment, monitoring, and troubleshooting",
    "**Plan for Failures**: Implement redundancy, failover, and disaster recovery",
    "**Optimize Continuously**: Regular performance tuning and cost optimization",
    "**Stay Updated**: Keep up with new deployment technologies and best practices"
]

for practice in best_practices:
    st.markdown(f"• {practice}")

# Deployment checklist
st.header("✅ Deployment Checklist")

checklist_sections = [
    {
        "section": "Pre-Deployment",
        "items": [
            "Model evaluation completed and documented",
            "Deployment strategy selected and planned",
            "Infrastructure requirements identified",
            "Security requirements defined",
            "Monitoring and alerting configured",
            "Backup and rollback procedures established"
        ]
    },
    {
        "section": "Deployment",
        "items": [
            "Infrastructure provisioned and tested",
            "Model deployed to staging environment",
            "Load testing completed successfully",
            "Security scanning passed",
            "Monitoring dashboards operational",
            "Production deployment executed"
        ]
    },
    {
        "section": "Post-Deployment",
        "items": [
            "Performance metrics within expected ranges",
            "Error rates below acceptable thresholds",
            "User acceptance testing completed",
            "Documentation updated",
            "Team trained on operations procedures",
            "Incident response procedures tested"
        ]
    }
]

for section in checklist_sections:
    st.markdown(f"### {section['section']}")
    for item in section['items']:
        st.checkbox(item)
