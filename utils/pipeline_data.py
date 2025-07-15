"""
Data structures and content for the LLM training pipeline stages.
Contains stage definitions, alternatives, tools, and resources.
"""

def get_pipeline_stages():
    """Return list of all pipeline stages with their details."""
    
    stages = [
        {
            "name": "Data Collection",
            "description": "Gathering and curating large-scale text datasets for training",
            "key_concepts": [
                {"name": "Scale", "description": "Modern LLMs require hundreds of billions to trillions of tokens"},
                {"name": "Quality", "description": "High-quality, well-formatted text improves training efficiency"},
                {"name": "Diversity", "description": "Varied sources prevent overfitting to specific domains"},
                {"name": "Licensing", "description": "Ensuring legal compliance for commercial use"},
                {"name": "Deduplication", "description": "Removing duplicates to improve generalization"}
            ],
            "steps": [
                {
                    "title": "Source Identification",
                    "description": "Identify and evaluate potential data sources for your training corpus",
                    "code_example": """
# Common data sources for LLM training
sources = {
    'common_crawl': 'Web crawl data - massive scale but variable quality',
    'wikipedia': 'High-quality encyclopedic content',
    'books': 'Literature and non-fiction books',
    'academic_papers': 'Scientific and research publications',
    'news': 'News articles and journalism',
    'forums': 'Discussion forums and Q&A sites'
}

# Example: Download Common Crawl data
import requests
def download_commoncrawl_segment(segment_id):
    url = f"https://data.commoncrawl.org/crawl-data/CC-MAIN-{segment_id}/segments/"
    response = requests.get(url)
    return response.content
"""
                },
                {
                    "title": "Legal Review",
                    "description": "Ensure compliance with copyright, privacy, and licensing requirements",
                    "code_example": """
# Legal compliance checklist
legal_requirements = {
    'copyright': 'Check for copyrighted content and fair use',
    'privacy': 'Remove or anonymize personal information',
    'licensing': 'Verify data usage rights and attribution',
    'jurisdiction': 'Comply with relevant data protection laws',
    'commercial_use': 'Ensure rights for commercial applications'
}

# Example: Basic PII detection
import re
def detect_pii(text):
    patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}-\d{3}-\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b'
    }
    
    detected_pii = {}
    for pii_type, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            detected_pii[pii_type] = matches
    
    return detected_pii
"""
                },
                {
                    "title": "Data Acquisition",
                    "description": "Download and store the selected datasets",
                    "code_example": """
# Example: Hugging Face datasets integration
from datasets import load_dataset
import os

def download_dataset(dataset_name, cache_dir="/data/datasets"):
    try:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        print(f"Downloaded {dataset_name}: {dataset}")
        return dataset
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")
        return None

# Download multiple datasets
datasets_to_download = [
    "wikipedia",
    "bookcorpus", 
    "openwebtext",
    "c4"
]

for dataset_name in datasets_to_download:
    download_dataset(dataset_name)
"""
                }
            ],
            "best_practices": [
                "Start with high-quality, curated datasets before adding web-scraped content",
                "Implement robust deduplication at multiple levels (exact, near-duplicate, semantic)",
                "Maintain detailed provenance tracking for all data sources",
                "Regular quality audits throughout the collection process",
                "Legal review of all data sources and usage rights"
            ],
            "challenges": [
                {
                    "issue": "Scale Requirements",
                    "solution": "Use distributed processing and cloud storage solutions"
                },
                {
                    "issue": "Quality Control",
                    "solution": "Implement multi-stage filtering and quality scoring"
                },
                {
                    "issue": "Legal Compliance",
                    "solution": "Work with legal experts and use well-established datasets"
                }
            ],
            "alternatives": [
                {
                    "name": "Public Datasets",
                    "pros": ["Pre-cleaned", "Legal clarity", "Reproducible"],
                    "cons": ["Limited customization", "Potential overlap with other models"],
                    "complexity": "Low",
                    "cost": "Free to Low",
                    "use_cases": ["Research", "Prototyping", "Baseline models"]
                },
                {
                    "name": "Web Scraping",
                    "pros": ["Fresh content", "Large scale", "Customizable"],
                    "cons": ["Legal risks", "Quality varies", "Technical complexity"],
                    "complexity": "High",
                    "cost": "Medium",
                    "use_cases": ["Custom domains", "Real-time data", "Specialized applications"]
                },
                {
                    "name": "Licensed Content",
                    "pros": ["High quality", "Legal clarity", "Professional content"],
                    "cons": ["Expensive", "Limited availability", "Licensing restrictions"],
                    "complexity": "Medium", 
                    "cost": "High",
                    "use_cases": ["Commercial products", "Domain-specific models", "Premium applications"]
                }
            ],
            "tools": [
                {
                    "name": "Hugging Face Datasets",
                    "category": "Data Management",
                    "description": "Library for accessing and processing datasets",
                    "link": "https://huggingface.co/docs/datasets/"
                },
                {
                    "name": "Common Crawl",
                    "category": "Data Source",
                    "description": "Open repository of web crawl data",
                    "link": "https://commoncrawl.org/"
                },
                {
                    "name": "Apache Spark",
                    "category": "Processing",
                    "description": "Distributed data processing framework",
                    "link": "https://spark.apache.org/"
                }
            ],
            "resources": [
                {
                    "title": "The Pile: An 800GB Dataset of Diverse Text",
                    "type": "Paper",
                    "description": "Comprehensive overview of a large-scale training dataset",
                    "link": "https://arxiv.org/abs/2101.00027",
                    "difficulty": "Intermediate"
                },
                {
                    "title": "Data Management for Large Scale Language Models",
                    "type": "Blog Post",
                    "description": "Practical guide to handling massive datasets",
                    "difficulty": "Beginner"
                }
            ],
            "deliverables": [
                "Raw dataset inventory and source documentation",
                "Legal compliance verification",
                "Data quality assessment report",
                "Deduplication and filtering statistics"
            ],
            "metrics": [
                {"name": "Dataset Size", "value": "500B tokens", "delta": "+50B"},
                {"name": "Quality Score", "value": "8.5/10", "delta": "+0.3"},
                {"name": "Dedup Rate", "value": "15%", "delta": "-2%"}
            ],
            "duration": "2-4 weeks",
            "difficulty": "Intermediate"
        },
        
        {
            "name": "Data Preprocessing", 
            "description": "Cleaning, filtering, and formatting data for training",
            "key_concepts": [
                {"name": "Tokenization", "description": "Converting text into tokens that models can process"},
                {"name": "Normalization", "description": "Standardizing text format and encoding"},
                {"name": "Quality Filtering", "description": "Removing low-quality or harmful content"},
                {"name": "Sequence Formatting", "description": "Structuring data for model consumption"},
                {"name": "Parallel Processing", "description": "Scaling preprocessing across multiple cores/machines"}
            ],
            "steps": [
                {
                    "title": "Text Cleaning",
                    "description": "Remove unwanted characters, normalize encoding, fix formatting issues",
                    "code_example": """
import re
import unicodedata

def clean_text(text):
    # Normalize unicode
    text = unicodedata.normalize('NFD', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Fix whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    
    return text

# Example usage
dirty_text = "<p>This is    HTML  text&nbsp;with   issues!</p>"
clean_text_result = clean_text(dirty_text)
print(clean_text_result)  # "This is HTML text with issues!"
"""
                },
                {
                    "title": "Quality Filtering",
                    "description": "Filter out low-quality documents based on heuristics and ML models",
                    "code_example": """
def quality_filter(text, min_length=100, max_length=100000):
    # Length filtering
    if len(text) < min_length or len(text) > max_length:
        return False
    
    # Language detection
    try:
        from langdetect import detect
        if detect(text) != 'en':
            return False
    except:
        return False
    
    # Quality heuristics
    words = text.split()
    if len(words) == 0:
        return False
        
    # Check vocabulary diversity
    unique_words = len(set(words))
    diversity = unique_words / len(words)
    if diversity < 0.3:  # Too repetitive
        return False
    
    # Check for excessive special characters
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_chars / len(text) > 0.3:
        return False
    
    return True
"""
                },
                {
                    "title": "Tokenization",
                    "description": "Convert text into tokens using appropriate tokenizer",
                    "code_example": """
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_text(text, max_length=1024):
    # Tokenize with truncation and padding
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors="pt"
    )
    
    return encoded

# For training data preparation
def prepare_training_data(texts, tokenizer, max_length=1024):
    tokenized_data = []
    
    for text in texts:
        tokens = tokenizer.encode(text, max_length=max_length, truncation=True)
        if len(tokens) > 50:  # Minimum length threshold
            tokenized_data.append(tokens)
    
    return tokenized_data
"""
                }
            ],
            "best_practices": [
                "Implement parallel processing for large-scale data",
                "Validate preprocessing steps on small samples first",
                "Maintain statistics throughout the pipeline",
                "Use consistent text normalization across all data",
                "Preserve original data alongside processed versions"
            ],
            "challenges": [
                {
                    "issue": "Memory Overflow",
                    "solution": "Use streaming processing and batch operations"
                },
                {
                    "issue": "Processing Speed",
                    "solution": "Implement multiprocessing and optimize bottlenecks"
                },
                {
                    "issue": "Quality Inconsistency",
                    "solution": "Standardize filtering criteria and validate outputs"
                }
            ],
            "alternatives": [
                {
                    "name": "Rule-based Filtering",
                    "pros": ["Fast", "Interpretable", "Consistent"],
                    "cons": ["Limited accuracy", "Hard to tune", "Misses edge cases"],
                    "complexity": "Low",
                    "cost": "Low"
                },
                {
                    "name": "ML-based Quality Scoring",
                    "pros": ["Higher accuracy", "Adaptable", "Catches complex patterns"],
                    "cons": ["Slower", "Requires training data", "Less interpretable"],
                    "complexity": "High",
                    "cost": "Medium"
                }
            ],
            "duration": "1-3 weeks",
            "difficulty": "Intermediate"
        },
        
        {
            "name": "Model Architecture",
            "description": "Designing the neural network structure and configuration",
            "key_concepts": [
                {"name": "Transformer Architecture", "description": "Attention-based architecture that forms the backbone of modern LLMs"},
                {"name": "Scaling Laws", "description": "Relationships between model size, data size, and performance"},
                {"name": "Parameter Efficiency", "description": "Balancing model capacity with computational requirements"},
                {"name": "Architectural Variants", "description": "Different transformer configurations for different use cases"},
                {"name": "Hardware Considerations", "description": "Designing architectures that efficiently use available hardware"}
            ],
            "duration": "1-2 weeks",
            "difficulty": "Advanced"
        },
        
        {
            "name": "Training",
            "description": "The core optimization process to learn from data",
            "key_concepts": [
                {"name": "Gradient Descent", "description": "Optimization algorithm that updates model parameters"},
                {"name": "Learning Rate Scheduling", "description": "Strategies for adjusting learning rate during training"},
                {"name": "Batch Size Optimization", "description": "Balancing memory usage with training stability"},
                {"name": "Regularization", "description": "Techniques to prevent overfitting and improve generalization"},
                {"name": "Distributed Training", "description": "Scaling training across multiple GPUs and machines"}
            ],
            "duration": "2-8 weeks",
            "difficulty": "Advanced"
        },
        
        {
            "name": "Fine-Tuning",
            "description": "Adapting pre-trained models for specific tasks or domains",
            "key_concepts": [
                {"name": "Transfer Learning", "description": "Leveraging knowledge from pre-trained models"},
                {"name": "Parameter-Efficient Methods", "description": "Updating only a subset of parameters (LoRA, adapters)"},
                {"name": "Instruction Tuning", "description": "Teaching models to follow natural language instructions"},
                {"name": "RLHF", "description": "Using human feedback to align model behavior with preferences"},
                {"name": "Domain Adaptation", "description": "Specializing models for specific domains or use cases"}
            ],
            "duration": "1-4 weeks", 
            "difficulty": "Intermediate"
        },
        
        {
            "name": "Evaluation",
            "description": "Measuring model performance across multiple dimensions",
            "key_concepts": [
                {"name": "Benchmark Evaluation", "description": "Testing on standardized tasks and datasets"},
                {"name": "Human Evaluation", "description": "Getting human judgments on model outputs"},
                {"name": "Safety Assessment", "description": "Testing for harmful, biased, or inappropriate outputs"},
                {"name": "Robustness Testing", "description": "Evaluating performance under adversarial conditions"},
                {"name": "Efficiency Metrics", "description": "Measuring computational and memory requirements"}
            ],
            "duration": "1-3 weeks",
            "difficulty": "Intermediate"
        },
        
        {
            "name": "Deployment",
            "description": "Serving models in production environments",
            "key_concepts": [
                {"name": "Inference Optimization", "description": "Techniques to speed up model inference"},
                {"name": "Scaling Infrastructure", "description": "Handling varying loads and traffic patterns"},
                {"name": "Model Serving", "description": "Frameworks and platforms for serving ML models"},
                {"name": "Monitoring & Maintenance", "description": "Ongoing monitoring and model updates"},
                {"name": "Cost Optimization", "description": "Balancing performance with operational costs"}
            ],
            "duration": "2-6 weeks",
            "difficulty": "Advanced"
        }
    ]
    
    return stages

def get_stage_details(stage_index):
    """Get detailed information for a specific stage."""
    stages = get_pipeline_stages()
    if 0 <= stage_index < len(stages):
        return stages[stage_index]
    return None

def get_common_tools():
    """Return common tools used across multiple stages."""
    return {
        "data_processing": [
            "Apache Spark",
            "Dask", 
            "Ray",
            "Pandas",
            "Hugging Face Datasets"
        ],
        "model_training": [
            "PyTorch",
            "TensorFlow",
            "JAX/Flax",
            "Transformers",
            "DeepSpeed"
        ],
        "infrastructure": [
            "Docker",
            "Kubernetes", 
            "MLflow",
            "Weights & Biases",
            "TensorBoard"
        ],
        "deployment": [
            "TorchServe",
            "TensorFlow Serving",
            "NVIDIA Triton",
            "FastAPI",
            "Streamlit"
        ]
    }

def get_resource_estimates():
    """Return resource estimates for different model sizes."""
    return {
        "small_model": {
            "parameters": "125M-350M",
            "training_data": "50-100B tokens",
            "training_time": "1-7 days",
            "hardware": "1-4 GPUs",
            "estimated_cost": "$1K-$5K"
        },
        "medium_model": {
            "parameters": "1B-7B", 
            "training_data": "100-500B tokens",
            "training_time": "1-4 weeks",
            "hardware": "8-32 GPUs",
            "estimated_cost": "$10K-$100K"
        },
        "large_model": {
            "parameters": "13B-70B",
            "training_data": "500B-2T tokens", 
            "training_time": "4-16 weeks",
            "hardware": "64-512 GPUs",
            "estimated_cost": "$100K-$1M+"
        }
    }

def get_learning_path_recommendations():
    """Return recommended learning paths for different experience levels."""
    return {
        "beginner": {
            "focus": "Understanding fundamentals and using existing models",
            "recommended_stages": [
                "Data Collection (overview)",
                "Fine-Tuning", 
                "Evaluation",
                "Deployment (simple)"
            ],
            "tools_to_learn": ["Hugging Face Transformers", "Python", "Basic ML concepts"],
            "timeline": "2-3 months"
        },
        "intermediate": {
            "focus": "Building and training custom models",
            "recommended_stages": [
                "Data Collection",
                "Data Preprocessing", 
                "Model Architecture",
                "Training",
                "Fine-Tuning",
                "Evaluation",
                "Deployment"
            ],
            "tools_to_learn": ["PyTorch/TensorFlow", "Distributed training", "Cloud platforms"],
            "timeline": "6-12 months"
        },
        "advanced": {
            "focus": "Research and large-scale production systems",
            "recommended_stages": "All stages with deep understanding",
            "tools_to_learn": ["Advanced optimization", "Research techniques", "Infrastructure management"],
            "timeline": "12+ months"
        }
    }
