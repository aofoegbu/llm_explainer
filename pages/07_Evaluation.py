import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Model Evaluation", page_icon="📊", layout="wide")

st.title("📊 Model Evaluation")
st.markdown("### Measuring and Validating Language Model Performance")

# Overview
st.header("🎯 Overview")
st.markdown("""
Model evaluation is crucial for understanding your LLM's capabilities, limitations, and readiness for deployment. 
This stage involves comprehensive testing across multiple dimensions including accuracy, safety, efficiency, 
and alignment with intended use cases.
""")

# Evaluation framework
st.header("🔬 Evaluation Framework")

framework_tabs = st.tabs([
    "📏 Evaluation Types",
    "📊 Metrics & Benchmarks", 
    "🛠️ Evaluation Tools",
    "📈 Analysis Methods"
])

with framework_tabs[0]:
    st.subheader("📏 Types of Evaluation")
    
    eval_types = [
        {
            "type": "Intrinsic Evaluation",
            "description": "Measures model performance on language modeling objectives",
            "metrics": [
                "Perplexity on held-out data",
                "Cross-entropy loss",
                "Token-level accuracy",
                "Bits per character"
            ],
            "advantages": [
                "Direct measure of language modeling quality",
                "Easy to compute and compare",
                "Language-agnostic",
                "Reflects training objective"
            ],
            "limitations": [
                "May not reflect real-world performance",
                "Doesn't capture task-specific abilities",
                "Can be misleading for practical applications"
            ],
            "when_to_use": "During training monitoring and model comparison"
        },
        {
            "type": "Extrinsic Evaluation", 
            "description": "Measures performance on downstream tasks and applications",
            "metrics": [
                "Task-specific accuracy",
                "F1 scores",
                "BLEU/ROUGE scores",
                "Human evaluation ratings"
            ],
            "advantages": [
                "Reflects real-world utility",
                "Task-relevant measurements",
                "Directly relates to use cases",
                "Validates practical value"
            ],
            "limitations": [
                "Task-specific setup required",
                "May not generalize across domains",
                "Can be expensive to compute"
            ],
            "when_to_use": "Before deployment and for task-specific validation"
        },
        {
            "type": "Behavioral Evaluation",
            "description": "Tests model behavior patterns and edge cases",
            "metrics": [
                "Consistency across prompts",
                "Robustness to input variations",
                "Safety and bias measures",
                "Calibration scores"
            ],
            "advantages": [
                "Reveals unexpected behaviors",
                "Tests safety and reliability",
                "Validates alignment with values",
                "Identifies failure modes"
            ],
            "limitations": [
                "Subjective interpretation",
                "Difficult to standardize",
                "Requires domain expertise"
            ],
            "when_to_use": "For safety validation and alignment assessment"
        }
    ]
    
    for eval_type in eval_types:
        with st.expander(f"📋 {eval_type['type']}"):
            st.markdown(eval_type['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Key Metrics:**")
                for metric in eval_type['metrics']:
                    st.markdown(f"• {metric}")
                
                st.markdown("**✅ Advantages:**")
                for advantage in eval_type['advantages']:
                    st.markdown(f"• {advantage}")
            
            with col2:
                st.markdown("**❌ Limitations:**")
                for limitation in eval_type['limitations']:
                    st.markdown(f"• {limitation}")
                
                st.info(f"**🎯 When to use:** {eval_type['when_to_use']}")

with framework_tabs[1]:
    st.subheader("📊 Metrics & Benchmarks")
    
    benchmark_tabs = st.tabs([
        "🏆 Popular Benchmarks",
        "📏 Core Metrics", 
        "🎯 Task-Specific",
        "🛡️ Safety & Bias"
    ])
    
    with benchmark_tabs[0]:
        st.markdown("### 🏆 Popular LLM Benchmarks")
        
        benchmarks = [
            {
                "name": "GLUE/SuperGLUE",
                "description": "General Language Understanding Evaluation benchmark suite",
                "tasks": [
                    "Sentiment analysis (SST-2)",
                    "Natural language inference (RTE, MNLI)",
                    "Question answering (QNLI)",
                    "Textual similarity (STS-B)"
                ],
                "focus": "General language understanding",
                "difficulty": "Medium",
                "use_case": "Baseline language understanding assessment"
            },
            {
                "name": "MMLU",
                "description": "Massive Multitask Language Understanding across 57 subjects",
                "tasks": [
                    "Elementary mathematics",
                    "US history", 
                    "Computer science",
                    "Law and ethics"
                ],
                "focus": "Knowledge and reasoning across domains",
                "difficulty": "Hard",
                "use_case": "Comprehensive knowledge evaluation"
            },
            {
                "name": "HellaSwag",
                "description": "Commonsense reasoning about everyday situations",
                "tasks": [
                    "Scenario completion",
                    "Commonsense inference",
                    "Contextual reasoning"
                ],
                "focus": "Commonsense reasoning",
                "difficulty": "Medium-Hard",
                "use_case": "Testing practical reasoning abilities"
            },
            {
                "name": "HumanEval",
                "description": "Programming problems for code generation evaluation",
                "tasks": [
                    "Function implementation",
                    "Algorithm design",
                    "Code completion"
                ],
                "focus": "Code generation and programming",
                "difficulty": "Hard",
                "use_case": "Code generation model assessment"
            },
            {
                "name": "TruthfulQA",
                "description": "Tests whether models give truthful answers to questions",
                "tasks": [
                    "Factual question answering",
                    "Misinformation detection",
                    "Truth vs. falsehood"
                ],
                "focus": "Truthfulness and factual accuracy",
                "difficulty": "Hard",
                "use_case": "Safety and truthfulness validation"
            }
        ]
        
        for benchmark in benchmarks:
            with st.expander(f"🏆 {benchmark['name']}"):
                st.markdown(benchmark['description'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📋 Example Tasks:**")
                    for task in benchmark['tasks']:
                        st.markdown(f"• {task}")
                
                with col2:
                    st.markdown(f"**🎯 Focus:** {benchmark['focus']}")
                    st.markdown(f"**📈 Difficulty:** {benchmark['difficulty']}")
                    st.markdown(f"**🔧 Use Case:** {benchmark['use_case']}")
    
    with benchmark_tabs[1]:
        st.markdown("### 📏 Core Evaluation Metrics")
        
        core_metrics = [
            {
                "category": "Language Modeling",
                "metrics": [
                    ("Perplexity", "Exponential of cross-entropy loss", "Lower is better", "2^(loss)"),
                    ("Bits per Character", "Information content per character", "Lower is better", "log2(perplexity)/chars"),
                    ("Log Likelihood", "Probability of generating text", "Higher is better", "log P(text|model)"),
                    ("Cross-Entropy", "Prediction error measure", "Lower is better", "-log P(target|prediction)")
                ]
            },
            {
                "category": "Classification",
                "metrics": [
                    ("Accuracy", "Percentage of correct predictions", "Higher is better", "correct/total"),
                    ("F1 Score", "Harmonic mean of precision and recall", "Higher is better", "2*P*R/(P+R)"),
                    ("Precision", "True positives / (TP + FP)", "Higher is better", "TP/(TP+FP)"),
                    ("Recall", "True positives / (TP + FN)", "Higher is better", "TP/(TP+FN)")
                ]
            },
            {
                "category": "Generation",
                "metrics": [
                    ("BLEU", "N-gram overlap with reference", "Higher is better", "Modified n-gram precision"),
                    ("ROUGE", "Recall-oriented overlap scoring", "Higher is better", "Longest common subsequence"),
                    ("BERTScore", "Semantic similarity using BERT", "Higher is better", "Cosine similarity"),
                    ("METEOR", "Alignment-based metric", "Higher is better", "Unigram matching + penalties")
                ]
            }
        ]
        
        for category in core_metrics:
            with st.expander(f"📊 {category['category']} Metrics"):
                for name, description, direction, formula in category['metrics']:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col1:
                        st.markdown(f"**{name}**")
                    with col2:
                        st.markdown(description)
                    with col3:
                        st.markdown(f"*{direction}*")
                    st.code(formula, language='text')
                    st.markdown("---")
    
    with benchmark_tabs[2]:
        st.markdown("### 🎯 Task-Specific Evaluation")
        
        task_evals = [
            {
                "task": "Question Answering",
                "metrics": ["Exact Match (EM)", "F1 Score", "Answer Accuracy"],
                "datasets": ["SQuAD", "Natural Questions", "MS MARCO"],
                "challenges": ["Context understanding", "Multi-hop reasoning", "Factual accuracy"],
                "evaluation_code": """
def evaluate_qa(predictions, references):
    exact_matches = []
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        # Exact match
        exact_matches.append(normalize_text(pred) == normalize_text(ref))
        
        # F1 score based on token overlap
        pred_tokens = normalize_text(pred).split()
        ref_tokens = normalize_text(ref).split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            f1_scores.append(0.0)
        else:
            common_tokens = set(pred_tokens) & set(ref_tokens)
            precision = len(common_tokens) / len(pred_tokens)
            recall = len(common_tokens) / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
    
    return {
        'exact_match': np.mean(exact_matches),
        'f1_score': np.mean(f1_scores)
    }
"""
            },
            {
                "task": "Text Summarization",
                "metrics": ["ROUGE-1/2/L", "BERTScore", "Human evaluation"],
                "datasets": ["CNN/DailyMail", "XSum", "Multi-News"],
                "challenges": ["Factual consistency", "Abstractive vs extractive", "Length control"],
                "evaluation_code": """
from rouge_score import rouge_scorer

def evaluate_summarization(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for metric in rouge_scores:
            rouge_scores[metric].append(scores[metric].fmeasure)
    
    return {metric: np.mean(scores) for metric, scores in rouge_scores.items()}
"""
            },
            {
                "task": "Machine Translation",
                "metrics": ["BLEU", "chrF", "TER", "COMET"],
                "datasets": ["WMT datasets", "OPUS", "MultiUN"],
                "challenges": ["Linguistic diversity", "Cultural adaptation", "Domain specificity"],
                "evaluation_code": """
import sacrebleu

def evaluate_translation(predictions, references):
    # BLEU score
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    
    # chrF score (character-level)
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    
    return {
        'bleu': bleu.score,
        'chrf': chrf.score
    }
"""
            }
        ]
        
        for task_eval in task_evals:
            with st.expander(f"🎯 {task_eval['task']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Key Metrics:**")
                    for metric in task_eval['metrics']:
                        st.markdown(f"• {metric}")
                    
                    st.markdown("**📚 Common Datasets:**")
                    for dataset in task_eval['datasets']:
                        st.markdown(f"• {dataset}")
                
                with col2:
                    st.markdown("**⚠️ Key Challenges:**")
                    for challenge in task_eval['challenges']:
                        st.markdown(f"• {challenge}")
                
                st.markdown("**💻 Evaluation Code:**")
                st.code(task_eval['evaluation_code'], language='python')
    
    with benchmark_tabs[3]:
        st.markdown("### 🛡️ Safety & Bias Evaluation")
        
        safety_aspects = [
            {
                "aspect": "Bias Detection",
                "description": "Identify and measure various forms of bias in model outputs",
                "methods": [
                    "Demographic parity testing",
                    "Counterfactual evaluation", 
                    "Stereotype assessment",
                    "Intersectional bias analysis"
                ],
                "tools": ["Fairness Indicators", "AI Fairness 360", "What-If Tool"],
                "metrics": ["Equalized odds", "Demographic parity", "Individual fairness"]
            },
            {
                "aspect": "Toxicity Detection",
                "description": "Measure harmful, offensive, or inappropriate content generation",
                "methods": [
                    "Perspective API scoring",
                    "Adversarial prompting",
                    "Red team evaluation",
                    "Content classification"
                ],
                "tools": ["Perspective API", "Detoxify", "HatEval"],
                "metrics": ["Toxicity probability", "Severe toxicity rate", "Attack success rate"]
            },
            {
                "aspect": "Truthfulness Assessment",
                "description": "Evaluate factual accuracy and misinformation susceptibility",
                "methods": [
                    "Fact-checking alignment",
                    "Knowledge consistency",
                    "Citation accuracy",
                    "Hallucination detection"
                ],
                "tools": ["TruthfulQA", "FEVER", "FactCC"],
                "metrics": ["Factual accuracy", "Consistency score", "Citation precision"]
            },
            {
                "aspect": "Privacy Compliance",
                "description": "Ensure protection of sensitive information and privacy",
                "methods": [
                    "PII extraction testing",
                    "Membership inference attacks",
                    "Data reconstruction attempts",
                    "Differential privacy auditing"
                ],
                "tools": ["Privacy Meter", "TensorFlow Privacy", "Opacus"],
                "metrics": ["PII leakage rate", "Privacy loss", "Reconstruction error"]
            }
        ]
        
        for aspect in safety_aspects:
            with st.expander(f"🛡️ {aspect['aspect']}"):
                st.markdown(aspect['description'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔍 Evaluation Methods:**")
                    for method in aspect['methods']:
                        st.markdown(f"• {method}")
                    
                    st.markdown("**🛠️ Available Tools:**")
                    for tool in aspect['tools']:
                        st.markdown(f"• {tool}")
                
                with col2:
                    st.markdown("**📊 Key Metrics:**")
                    for metric in aspect['metrics']:
                        st.markdown(f"• {metric}")

with framework_tabs[2]:
    st.subheader("🛠️ Evaluation Tools & Frameworks")
    
    tool_categories = [
        {
            "category": "Comprehensive Frameworks",
            "tools": [
                {
                    "name": "lm-evaluation-harness",
                    "description": "Unified framework for LLM evaluation across many tasks",
                    "features": ["200+ tasks", "Standardized metrics", "Easy extensibility"],
                    "best_for": "Large-scale benchmark evaluation",
                    "example": """
# Install and use lm-evaluation-harness
pip install lm-eval

# Evaluate model on multiple tasks
lm_eval --model hf-causal-experimental \\
        --model_args pretrained=gpt2 \\
        --tasks hellaswag,arc_easy,arc_challenge \\
        --device cuda:0 \\
        --batch_size 16
"""
                },
                {
                    "name": "OpenAI Evals",
                    "description": "Framework for creating and running evaluations",
                    "features": ["Custom eval creation", "OpenAI API integration", "Community evals"],
                    "best_for": "Custom evaluation development",
                    "example": """
# Create custom evaluation
import evals

# Register custom eval
@evals.record_and_check_match
def custom_eval(sampler, **kwargs):
    # Your evaluation logic here
    pass

# Run evaluation
evals run custom_eval --model gpt-3.5-turbo
"""
                }
            ]
        },
        {
            "category": "Specialized Tools",
            "tools": [
                {
                    "name": "BIG-bench",
                    "description": "Beyond the Imitation Game collaborative benchmark",
                    "features": ["200+ tasks", "Diverse evaluation", "Community-driven"],
                    "best_for": "Comprehensive capability assessment",
                    "example": """
# Use BIG-bench tasks
from bigbench import api

# Load specific task
task = api.get_task('logical_deduction')

# Evaluate model
results = task.evaluate_model(your_model)
"""
                },
                {
                    "name": "HELM",
                    "description": "Holistic Evaluation of Language Models",
                    "features": ["Standardized scenarios", "Comprehensive metrics", "Reproducible"],
                    "best_for": "Holistic model comparison",
                    "example": """
# HELM evaluation
helm-run --suite v1 --model openai/gpt-3.5-turbo \\
         --max-eval-instances 100
"""
                }
            ]
        }
    ]
    
    for category in tool_categories:
        st.markdown(f"### {category['category']}")
        
        for tool in category['tools']:
            with st.expander(f"🛠️ {tool['name']}"):
                st.markdown(tool['description'])
                
                st.markdown("**✨ Key Features:**")
                for feature in tool['features']:
                    st.markdown(f"• {feature}")
                
                st.info(f"**🎯 Best for:** {tool['best_for']}")
                
                st.markdown("**💻 Example Usage:**")
                st.code(tool['example'], language='bash' if 'pip install' in tool['example'] or 'helm-run' in tool['example'] else 'python')

with framework_tabs[3]:
    st.subheader("📈 Analysis Methods")
    
    analysis_tabs = st.tabs(["📊 Statistical Analysis", "🔍 Error Analysis", "📈 Performance Trends"])
    
    with analysis_tabs[0]:
        st.markdown("### 📊 Statistical Analysis Techniques")
        
        statistical_methods = [
            {
                "method": "Confidence Intervals",
                "purpose": "Quantify uncertainty in performance estimates",
                "when_to_use": "When reporting benchmark results",
                "implementation": """
import numpy as np
from scipy import stats

def compute_confidence_interval(scores, confidence=0.95):
    n = len(scores)
    mean = np.mean(scores)
    sem = stats.sem(scores)  # Standard error of mean
    
    # Compute confidence interval
    ci = stats.t.interval(confidence, n-1, loc=mean, scale=sem)
    
    return mean, ci

# Example usage
scores = [0.85, 0.87, 0.83, 0.86, 0.84]
mean, (lower, upper) = compute_confidence_interval(scores)
print(f"Mean: {mean:.3f}, 95% CI: [{lower:.3f}, {upper:.3f}]")
"""
            },
            {
                "method": "Statistical Significance Testing",
                "purpose": "Determine if performance differences are statistically significant",
                "when_to_use": "When comparing multiple models",
                "implementation": """
from scipy.stats import ttest_rel, wilcoxon

def compare_models(scores_a, scores_b, test='paired_t'):
    if test == 'paired_t':
        statistic, p_value = ttest_rel(scores_a, scores_b)
    elif test == 'wilcoxon':
        statistic, p_value = wilcoxon(scores_a, scores_b)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

# Example comparison
model_a_scores = [0.85, 0.87, 0.83, 0.86, 0.84]
model_b_scores = [0.82, 0.84, 0.80, 0.83, 0.81]

result = compare_models(model_a_scores, model_b_scores)
print(f"Significant difference: {result['significant']} (p={result['p_value']:.3f})")
"""
            },
            {
                "method": "Effect Size Calculation",
                "purpose": "Measure practical significance of differences",
                "when_to_use": "To understand magnitude of improvements",
                "implementation": """
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return d

# Interpret effect size
effect_size = cohens_d(model_a_scores, model_b_scores)
interpretation = "small" if abs(effect_size) < 0.5 else "medium" if abs(effect_size) < 0.8 else "large"
print(f"Effect size: {effect_size:.3f} ({interpretation})")
"""
            }
        ]
        
        for method in statistical_methods:
            with st.expander(f"📊 {method['method']}"):
                st.markdown(f"**Purpose:** {method['purpose']}")
                st.markdown(f"**When to use:** {method['when_to_use']}")
                st.code(method['implementation'], language='python')
    
    with analysis_tabs[1]:
        st.markdown("### 🔍 Error Analysis Techniques")
        
        error_analysis = [
            "**Failure Case Analysis**: Systematically categorize and analyze incorrect predictions",
            "**Confusion Matrix Analysis**: Understand patterns in classification errors",
            "**Error Distribution**: Analyze how errors vary across different data subsets",
            "**Qualitative Analysis**: Manual inspection of failure cases for insights",
            "**Adversarial Testing**: Evaluate robustness to input perturbations",
            "**Ablation Studies**: Remove components to understand their contribution"
        ]
        
        for analysis in error_analysis:
            st.markdown(f"• {analysis}")
    
    with analysis_tabs[2]:
        st.markdown("### 📈 Performance Tracking")
        
        # Simulated performance tracking chart
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        performance_data = {
            'Date': dates,
            'MMLU Score': 65 + 10 * np.random.cumsum(np.random.randn(30)) * 0.01,
            'HumanEval Score': 45 + 8 * np.random.cumsum(np.random.randn(30)) * 0.01,
            'Safety Score': 85 + 5 * np.random.cumsum(np.random.randn(30)) * 0.01
        }
        
        perf_df = pd.DataFrame(performance_data)
        
        fig = go.Figure()
        
        for metric in ['MMLU Score', 'HumanEval Score', 'Safety Score']:
            fig.add_trace(go.Scatter(
                x=perf_df['Date'],
                y=perf_df[metric],
                mode='lines+markers',
                name=metric,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Model Performance Tracking Over Time",
            xaxis_title="Date",
            yaxis_title="Score (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Evaluation planning
st.header("📋 Evaluation Planning")

planning_tabs = st.tabs(["🎯 Evaluation Strategy", "📊 Metric Selection", "⏱️ Evaluation Timeline"])

with planning_tabs[0]:
    st.subheader("🎯 Define Evaluation Strategy")
    
    # Strategy planning form
    use_case = st.selectbox(
        "Primary Use Case:",
        ["General chat/assistant", "Domain-specific QA", "Code generation", "Content creation", "Research/analysis"]
    )
    
    target_audience = st.multiselect(
        "Target Audience:",
        ["General users", "Domain experts", "Developers", "Researchers", "Enterprise users"]
    )
    
    critical_requirements = st.multiselect(
        "Critical Requirements:",
        ["High accuracy", "Safety/non-toxicity", "Factual correctness", "Consistency", "Efficiency", "Privacy"]
    )
    
    # Generate recommendation
    if st.button("Generate Evaluation Plan"):
        st.markdown("### 📋 Recommended Evaluation Plan")
        
        # Basic recommendations based on selections
        if use_case == "General chat/assistant":
            st.markdown("**🎯 Focus Areas:**")
            st.markdown("• Instruction following (HellaSwag, MMLU)")
            st.markdown("• Safety and alignment (TruthfulQA)")
            st.markdown("• Conversational ability (human evaluation)")
        
        elif use_case == "Code generation":
            st.markdown("**🎯 Focus Areas:**")
            st.markdown("• Code correctness (HumanEval, MBPP)")
            st.markdown("• Security vulnerability detection")
            st.markdown("• Documentation quality")
        
        if "Safety/non-toxicity" in critical_requirements:
            st.markdown("**🛡️ Safety Evaluation Required:**")
            st.markdown("• Toxicity detection (Perspective API)")
            st.markdown("• Bias assessment (demographic parity)")
            st.markdown("• Red team evaluation")

with planning_tabs[1]:
    st.subheader("📊 Metric Selection Guide")
    
    # Interactive metric selector
    st.markdown("### 🎛️ Metric Selection Tool")
    
    task_type = st.selectbox(
        "Task Type:",
        ["Classification", "Generation", "Question Answering", "Summarization", "Translation"]
    )
    
    data_availability = st.selectbox(
        "Reference Data:",
        ["Human-annotated references available", "No references (need automated metrics)", "Mixed availability"]
    )
    
    evaluation_budget = st.selectbox(
        "Evaluation Budget:",
        ["Low (automated only)", "Medium (some human eval)", "High (extensive human eval)"]
    )
    
    # Metric recommendations
    metric_recommendations = {
        "Classification": {
            "primary": ["Accuracy", "F1 Score", "Precision/Recall"],
            "secondary": ["AUC-ROC", "Confusion Matrix Analysis"],
            "human": ["Error Case Review"]
        },
        "Generation": {
            "primary": ["BLEU", "ROUGE", "BERTScore"],
            "secondary": ["Perplexity", "Diversity Metrics"],
            "human": ["Fluency", "Relevance", "Creativity"]
        },
        "Question Answering": {
            "primary": ["Exact Match", "F1 Score"],
            "secondary": ["Answer Similarity", "Context Relevance"],
            "human": ["Answer Quality", "Completeness"]
        }
    }
    
    if task_type in metric_recommendations:
        recs = metric_recommendations[task_type]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🎯 Primary Metrics:**")
            for metric in recs["primary"]:
                st.markdown(f"• {metric}")
        
        with col2:
            st.markdown("**📊 Secondary Metrics:**")
            for metric in recs["secondary"]:
                st.markdown(f"• {metric}")
        
        with col3:
            st.markdown("**👥 Human Evaluation:**")
            for metric in recs["human"]:
                st.markdown(f"• {metric}")

with planning_tabs[2]:
    st.subheader("⏱️ Evaluation Timeline")
    
    # Timeline planning
    evaluation_phases = [
        ("Setup & Preparation", "Configure evaluation tools and datasets", 2),
        ("Automated Evaluation", "Run benchmark suites and automated metrics", 1),
        ("Human Evaluation", "Conduct human assessment studies", 7),
        ("Analysis & Reporting", "Analyze results and prepare reports", 3),
        ("Validation & Review", "Validate findings and peer review", 2)
    ]
    
    total_days = sum(days for _, _, days in evaluation_phases)
    
    st.markdown(f"**Estimated Total Time:** {total_days} days")
    
    # Timeline chart
    timeline_fig = go.Figure()
    
    colors = ['lightblue', 'lightgreen', 'orange', 'lightcoral', 'lightgray']
    
    for i, (phase, description, days) in enumerate(evaluation_phases):
        timeline_fig.add_trace(go.Bar(
            x=[days],
            y=[phase],
            orientation='h',
            name=phase,
            marker_color=colors[i],
            text=f"{days} days",
            textposition="middle center",
            hovertext=description
        ))
    
    timeline_fig.update_layout(
        title="Evaluation Timeline",
        xaxis_title="Days",
        showlegend=False,
        height=350
    )
    
    st.plotly_chart(timeline_fig, use_container_width=True)

# Interactive evaluation tools
st.header("🛠️ Interactive Evaluation Tools")

tool_tabs = st.tabs(["📊 Performance Calculator", "📈 A/B Test Analyzer", "🎯 Metric Comparator"])

with tool_tabs[0]:
    st.subheader("📊 Performance Calculator")
    
    # Performance metrics calculator
    st.markdown("Calculate common evaluation metrics from your results:")
    
    calc_type = st.selectbox("Metric Type:", ["Classification", "Information Retrieval", "Generation"])
    
    if calc_type == "Classification":
        col1, col2 = st.columns(2)
        
        with col1:
            true_positives = st.number_input("True Positives", min_value=0, value=85)
            false_positives = st.number_input("False Positives", min_value=0, value=10)
        
        with col2:
            true_negatives = st.number_input("True Negatives", min_value=0, value=90)
            false_negatives = st.number_input("False Negatives", min_value=0, value=15)
        
        if st.button("Calculate Metrics"):
            # Calculate metrics
            accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.3f}")
            with col2:
                st.metric("Precision", f"{precision:.3f}")
            with col3:
                st.metric("Recall", f"{recall:.3f}")
            with col4:
                st.metric("F1 Score", f"{f1:.3f}")

with tool_tabs[1]:
    st.subheader("📈 A/B Test Analyzer")
    
    st.markdown("Compare performance between two model versions:")
    
    # A/B test input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model A Results:**")
        model_a_scores = st.text_area("Scores (one per line)", value="0.85\n0.87\n0.83\n0.86\n0.84")
        
    with col2:
        st.markdown("**Model B Results:**")
        model_b_scores = st.text_area("Scores (one per line)", value="0.82\n0.84\n0.80\n0.83\n0.81", key="model_b")
    
    if st.button("Analyze A/B Test"):
        try:
            scores_a = [float(x.strip()) for x in model_a_scores.split('\n') if x.strip()]
            scores_b = [float(x.strip()) for x in model_b_scores.split('\n') if x.strip()]
            
            if len(scores_a) > 0 and len(scores_b) > 0:
                mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
                std_a, std_b = np.std(scores_a), np.std(scores_b)
                
                # Statistical test
                from scipy.stats import ttest_ind
                statistic, p_value = ttest_ind(scores_a, scores_b)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Model A Mean", f"{mean_a:.3f}")
                    st.metric("Model A Std", f"{std_a:.3f}")
                
                with col2:
                    st.metric("Model B Mean", f"{mean_b:.3f}")
                    st.metric("Model B Std", f"{std_b:.3f}")
                
                with col3:
                    improvement = ((mean_a - mean_b) / mean_b) * 100 if mean_b != 0 else 0
                    st.metric("Improvement", f"{improvement:.1f}%")
                    
                    significant = "Yes" if p_value < 0.05 else "No"
                    st.metric("Statistically Significant", significant)
                
                st.info(f"P-value: {p_value:.4f}")
                
        except ValueError:
            st.error("Please enter valid numeric scores.")

with tool_tabs[2]:
    st.subheader("🎯 Metric Comparator")
    
    st.markdown("Compare different evaluation metrics for the same model:")
    
    # Sample data for comparison
    metrics_data = {
        'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC-ROC'],
        'Value': [0.85, 0.82, 0.88, 0.77, 0.91],
        'Benchmark': [0.80, 0.78, 0.82, 0.75, 0.87]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Allow user to edit values
    edited_df = st.data_editor(metrics_df, use_container_width=True)
    
    # Visualization
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Your Model',
        x=edited_df['Metric'],
        y=edited_df['Value'],
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Benchmark',
        x=edited_df['Metric'],
        y=edited_df['Benchmark'],
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title="Model Performance vs. Benchmark",
        xaxis_title="Metrics",
        yaxis_title="Score",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Best practices
st.header("🎯 Evaluation Best Practices")

best_practices = [
    "**Use multiple metrics**: No single metric captures all aspects of performance",
    "**Establish baselines**: Compare against simple baselines and previous work",
    "**Report confidence intervals**: Quantify uncertainty in your results",
    "**Test on diverse data**: Evaluate across different domains and demographics",
    "**Include human evaluation**: Automated metrics don't capture all quality aspects",
    "**Document evaluation setup**: Ensure reproducibility and transparency",
    "**Consider computational costs**: Balance evaluation thoroughness with resources",
    "**Test edge cases**: Evaluate robustness to unusual or adversarial inputs",
    "**Regular re-evaluation**: Performance can change as models and data evolve",
    "**Cross-validate findings**: Use multiple evaluation approaches when possible"
]

for practice in best_practices:
    st.markdown(f"• {practice}")

# Evaluation checklist
st.header("✅ Evaluation Checklist")

checklist_categories = [
    {
        "category": "Planning",
        "items": [
            "Evaluation objectives clearly defined",
            "Target metrics identified",
            "Evaluation datasets selected",
            "Baseline models chosen",
            "Human evaluation protocol designed"
        ]
    },
    {
        "category": "Execution", 
        "items": [
            "Evaluation infrastructure set up",
            "Automated benchmarks executed",
            "Human evaluation conducted",
            "Statistical significance tested",
            "Error analysis performed"
        ]
    },
    {
        "category": "Analysis",
        "items": [
            "Results documented thoroughly",
            "Confidence intervals calculated",
            "Failure cases analyzed",
            "Comparison with baselines completed",
            "Findings validated independently"
        ]
    }
]

for category in checklist_categories:
    st.markdown(f"### {category['category']}")
    for item in category['items']:
        st.checkbox(item)
