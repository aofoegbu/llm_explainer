import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="In-Context Learning", page_icon="🧩", layout="wide")

st.title("🧩 In-Context Learning")
st.markdown("### Learning New Tasks from Examples Within the Context Window")

# Overview
st.header("🎯 Overview")
st.markdown("""
In-context learning (ICL) is the remarkable ability of large language models to learn new tasks 
from just a few examples provided in their input context, without any parameter updates. 
This emergent capability has revolutionized how we interact with and deploy language models.
""")

# Core concepts
st.header("🧠 Core Concepts")

concept_tabs = st.tabs([
    "🔍 What is ICL?",
    "🧬 Mechanisms", 
    "📊 Capabilities",
    "⚖️ Advantages & Limitations"
])

with concept_tabs[0]:
    st.subheader("🔍 Understanding In-Context Learning")
    
    st.markdown("""
    In-context learning refers to the ability of language models to perform new tasks 
    by conditioning on a few input-output examples, without requiring parameter updates 
    or explicit training on those tasks.
    """)
    
    icl_components = [
        {
            "component": "Task Specification",
            "description": "Define the task through natural language or examples",
            "example": "Task: Translate English to French"
        },
        {
            "component": "Demonstration Examples",
            "description": "Provide input-output pairs that illustrate the task",
            "example": "English: 'Hello' → French: 'Bonjour'"
        },
        {
            "component": "Query Input",
            "description": "The new input you want the model to process",
            "example": "English: 'Goodbye'"
        },
        {
            "component": "Expected Output",
            "description": "Model generates output following the pattern",
            "example": "French: 'Au revoir'"
        }
    ]
    
    for component in icl_components:
        with st.expander(f"🔧 {component['component']}"):
            st.markdown(component['description'])
            st.code(component['example'])
    
    # ICL vs Traditional Learning comparison
    st.markdown("### 🔄 ICL vs. Traditional Fine-Tuning")
    
    comparison_data = {
        'Aspect': ['Parameter Updates', 'Training Time', 'Data Requirements', 'Deployment', 'Task Switching'],
        'In-Context Learning': ['None', 'Instant', 'Few examples', 'Immediate', 'Very fast'],
        'Traditional Fine-Tuning': ['All/Subset', 'Hours to days', 'Thousands of examples', 'After training', 'Requires retraining']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

with concept_tabs[1]:
    st.subheader("🧬 Mechanisms Behind ICL")
    
    mechanism_tabs = st.tabs(["🧠 Cognitive Models", "🔬 Research Findings", "📊 Empirical Observations"])
    
    with mechanism_tabs[0]:
        st.markdown("### 🧠 Theoretical Mechanisms")
        
        mechanisms = [
            {
                "mechanism": "Pattern Matching",
                "description": "Model recognizes patterns in examples and applies them to new inputs",
                "evidence": [
                    "Performance improves with more relevant examples",
                    "Similar examples lead to better performance",
                    "Order of examples can affect results"
                ],
                "implications": [
                    "Example selection is crucial",
                    "Diversity in examples helps generalization",
                    "Context length limits the number of examples"
                ]
            },
            {
                "mechanism": "Induction Head Circuits",
                "description": "Specific attention patterns that implement copying and pattern completion",
                "evidence": [
                    "Attention heads focus on repeated patterns",
                    "Circuit analysis shows pattern copying behavior",
                    "Emerges at specific model scales"
                ],
                "implications": [
                    "ICL emerges from attention mechanisms",
                    "Scaling improves ICL capabilities",
                    "Architecture design affects ICL performance"
                ]
            },
            {
                "mechanism": "Gradient Descent Simulation",
                "description": "ICL implements a form of implicit gradient descent in the forward pass",
                "evidence": [
                    "ICL performance correlates with gradient-based learning",
                    "Similar convergence properties to actual gradient descent",
                    "Learning dynamics follow predictable patterns"
                ],
                "implications": [
                    "ICL is a form of meta-learning",
                    "Can predict ICL performance from training dynamics",
                    "Optimization insights apply to ICL"
                ]
            }
        ]
        
        for mechanism in mechanisms:
            with st.expander(f"🔬 {mechanism['mechanism']}"):
                st.markdown(mechanism['description'])
                
                st.markdown("**Evidence:**")
                for evidence in mechanism['evidence']:
                    st.markdown(f"• {evidence}")
                
                st.markdown("**Implications:**")
                for implication in mechanism['implications']:
                    st.markdown(f"• {implication}")
    
    with mechanism_tabs[1]:
        st.markdown("### 🔬 Key Research Findings")
        
        findings = [
            {
                "finding": "Scaling Laws for ICL",
                "description": "ICL performance scales predictably with model size",
                "details": [
                    "Larger models show better ICL capabilities",
                    "Performance follows power law scaling",
                    "Different tasks have different scaling exponents",
                    "Emergence threshold around 1B parameters"
                ]
            },
            {
                "finding": "Example Selection Effects",
                "description": "The choice and order of examples significantly impacts performance",
                "details": [
                    "Relevant examples outperform random selection",
                    "Example order can change results by 30%+",
                    "Diversity in examples improves generalization",
                    "Label bias can affect performance"
                ]
            },
            {
                "finding": "Task Complexity Limits",
                "description": "ICL works well for some tasks but struggles with others",
                "details": [
                    "Excellent for pattern recognition tasks",
                    "Good for classification and simple reasoning",
                    "Struggles with complex multi-step reasoning",
                    "Performance degrades with task complexity"
                ]
            }
        ]
        
        for finding in findings:
            with st.expander(f"📊 {finding['finding']}"):
                st.markdown(finding['description'])
                for detail in finding['details']:
                    st.markdown(f"• {detail}")
    
    with mechanism_tabs[2]:
        st.markdown("### 📊 Empirical Observations")
        
        # Simulate ICL performance scaling
        model_sizes = ['125M', '350M', '1.3B', '6.7B', '13B', '30B', '175B']
        icl_performance = [25, 35, 52, 68, 75, 82, 87]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=model_sizes,
            y=icl_performance,
            mode='lines+markers',
            name='ICL Performance',
            line=dict(width=3, color='blue'),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="In-Context Learning Performance vs Model Size",
            xaxis_title="Model Size",
            yaxis_title="ICL Performance (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Example order effects
        st.markdown("### 📈 Example Order Effects")
        
        order_effects = pd.DataFrame({
            'Example Order': ['Random', 'Increasing Difficulty', 'Decreasing Difficulty', 'Semantic Similarity', 'Optimal'],
            'Performance': [65, 72, 68, 78, 82],
            'Variance': [8, 6, 7, 4, 3]
        })
        
        fig2 = px.bar(
            order_effects, 
            x='Example Order', 
            y='Performance',
            error_y='Variance',
            title="Effect of Example Ordering on ICL Performance"
        )
        
        st.plotly_chart(fig2, use_container_width=True)

with concept_tabs[2]:
    st.subheader("📊 ICL Capabilities")
    
    capability_tabs = st.tabs(["🎯 Task Types", "📈 Performance Analysis", "🔬 Emergent Abilities"])
    
    with capability_tabs[0]:
        st.markdown("### 🎯 Task Types and ICL Performance")
        
        task_categories = [
            {
                "category": "Classification Tasks",
                "performance": "Excellent",
                "examples": [
                    "Sentiment analysis",
                    "Topic classification", 
                    "Intent recognition",
                    "Named entity recognition"
                ],
                "characteristics": [
                    "Clear input-output mappings",
                    "Finite label spaces",
                    "Pattern recognition focused",
                    "Benefits from diverse examples"
                ]
            },
            {
                "category": "Language Tasks",
                "performance": "Very Good",
                "examples": [
                    "Translation",
                    "Summarization",
                    "Paraphrasing",
                    "Style transfer"
                ],
                "characteristics": [
                    "Leverages pre-training knowledge",
                    "Benefits from few good examples",
                    "Quality depends on target language",
                    "Works well for common language pairs"
                ]
            },
            {
                "category": "Reasoning Tasks",
                "performance": "Good",
                "examples": [
                    "Arithmetic",
                    "Logical reasoning",
                    "Commonsense reasoning",
                    "Simple problem solving"
                ],
                "characteristics": [
                    "Performance varies by complexity",
                    "Benefits from step-by-step examples",
                    "Struggles with novel reasoning patterns",
                    "Improves with chain-of-thought prompting"
                ]
            },
            {
                "category": "Creative Tasks",
                "performance": "Variable",
                "examples": [
                    "Story generation",
                    "Poetry writing",
                    "Creative problem solving",
                    "Brainstorming"
                ],
                "characteristics": [
                    "Highly dependent on examples",
                    "Benefits from stylistic examples",
                    "Quality is subjective",
                    "Originality can be limited"
                ]
            }
        ]
        
        for category in task_categories:
            with st.expander(f"📋 {category['category']} - {category['performance']}"):
                st.markdown("**Example Tasks:**")
                for example in category['examples']:
                    st.markdown(f"• {example}")
                
                st.markdown("**Key Characteristics:**")
                for char in category['characteristics']:
                    st.markdown(f"• {char}")
    
    with capability_tabs[1]:
        st.markdown("### 📈 Performance Analysis")
        
        # Performance by number of examples
        num_examples = list(range(0, 11))
        performance_curves = {
            'Classification': [45, 62, 72, 78, 82, 85, 87, 88, 89, 89, 90],
            'Translation': [35, 55, 68, 75, 80, 83, 85, 86, 87, 87, 88],
            'Reasoning': [25, 42, 55, 63, 68, 72, 75, 77, 78, 79, 80],
            'Creative': [30, 45, 58, 65, 69, 71, 72, 73, 73, 74, 74]
        }
        
        fig = go.Figure()
        colors = ['blue', 'green', 'red', 'purple']
        
        for i, (task, scores) in enumerate(performance_curves.items()):
            fig.add_trace(go.Scatter(
                x=num_examples,
                y=scores,
                mode='lines+markers',
                name=task,
                line=dict(width=3, color=colors[i]),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title="ICL Performance vs Number of Examples",
            xaxis_title="Number of Examples",
            yaxis_title="Performance (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance factors
        st.markdown("### 🎯 Key Performance Factors")
        
        factors = [
            {
                "factor": "Example Quality",
                "impact": "High",
                "description": "Clear, correct, and representative examples dramatically improve performance",
                "tips": ["Use diverse examples", "Ensure correctness", "Match target distribution"]
            },
            {
                "factor": "Example Relevance",
                "impact": "High", 
                "description": "Examples similar to the target task perform much better",
                "tips": ["Use domain-specific examples", "Match input style", "Consider semantic similarity"]
            },
            {
                "factor": "Number of Examples",
                "impact": "Medium",
                "description": "More examples help, but with diminishing returns",
                "tips": ["Optimal range: 3-8 examples", "Quality over quantity", "Monitor context length"]
            },
            {
                "factor": "Example Order",
                "impact": "Medium",
                "description": "Order can significantly affect performance",
                "tips": ["Put best examples last", "Consider difficulty progression", "Test different orders"]
            }
        ]
        
        for factor in factors:
            with st.expander(f"🎯 {factor['factor']} - {factor['impact']} Impact"):
                st.markdown(factor['description'])
                st.markdown("**Optimization Tips:**")
                for tip in factor['tips']:
                    st.markdown(f"• {tip}")
    
    with capability_tabs[2]:
        st.markdown("### 🔬 Emergent Abilities in ICL")
        
        emergent_abilities = [
            {
                "ability": "Meta-Learning",
                "description": "Learning to learn from the examples themselves",
                "emergence": "Appears around 1-10B parameters",
                "examples": [
                    "Adapting to new task formats",
                    "Inferring task rules from examples",
                    "Generalizing beyond example patterns"
                ]
            },
            {
                "ability": "Compositional Reasoning",
                "description": "Combining multiple skills demonstrated in different examples",
                "emergence": "Appears around 10B+ parameters",
                "examples": [
                    "Combining classification with explanation",
                    "Multi-step reasoning from examples",
                    "Integrating multiple task aspects"
                ]
            },
            {
                "ability": "Instruction Following",
                "description": "Following complex instructions with minimal examples",
                "emergence": "Appears around 100B+ parameters",
                "examples": [
                    "Following multi-part instructions",
                    "Adapting to constraint changes",
                    "Understanding implicit requirements"
                ]
            }
        ]
        
        for ability in emergent_abilities:
            with st.expander(f"✨ {ability['ability']}"):
                st.markdown(ability['description'])
                st.markdown(f"**Emergence:** {ability['emergence']}")
                st.markdown("**Examples:**")
                for example in ability['examples']:
                    st.markdown(f"• {example}")

with concept_tabs[3]:
    st.subheader("⚖️ Advantages & Limitations")
    
    comparison_tabs = st.tabs(["✅ Advantages", "❌ Limitations", "🔄 Trade-offs"])
    
    with comparison_tabs[0]:
        st.markdown("### ✅ Advantages of In-Context Learning")
        
        advantages = [
            {
                "advantage": "Rapid Deployment",
                "description": "No training time required - immediate task adaptation",
                "benefits": [
                    "Instant task switching",
                    "No infrastructure for training",
                    "Real-time adaptation possible",
                    "Minimal computational overhead"
                ]
            },
            {
                "advantage": "Few Examples Required",
                "description": "Effective with just a handful of examples",
                "benefits": [
                    "Useful for rare tasks",
                    "Reduces data collection burden",
                    "Enables rapid prototyping",
                    "Good for personalization"
                ]
            },
            {
                "advantage": "No Parameter Updates",
                "description": "Preserves the original model capabilities",
                "benefits": [
                    "No catastrophic forgetting",
                    "Maintains general capabilities",
                    "Safe to experiment",
                    "Reversible adaptations"
                ]
            },
            {
                "advantage": "Interpretability",
                "description": "Easy to understand what the model is learning from",
                "benefits": [
                    "Transparent example influence",
                    "Easy to debug issues",
                    "Clear input-output relationships",
                    "Auditable decisions"
                ]
            }
        ]
        
        for advantage in advantages:
            with st.expander(f"✅ {advantage['advantage']}"):
                st.markdown(advantage['description'])
                st.markdown("**Key Benefits:**")
                for benefit in advantage['benefits']:
                    st.markdown(f"• {benefit}")
    
    with comparison_tabs[1]:
        st.markdown("### ❌ Limitations of In-Context Learning")
        
        limitations = [
            {
                "limitation": "Context Length Constraints",
                "description": "Limited by model's maximum context window",
                "issues": [
                    "Can't use many examples",
                    "Long examples reduce capacity",
                    "Complex tasks need more context",
                    "Cost increases with context length"
                ]
            },
            {
                "limitation": "Performance Ceiling",
                "description": "Generally lower performance than fine-tuning",
                "issues": [
                    "Typically 80-90% of fine-tuning performance",
                    "Struggles with complex tasks",
                    "Limited by pre-training knowledge",
                    "May not capture task nuances"
                ]
            },
            {
                "limitation": "Example Sensitivity",
                "description": "Very sensitive to example quality and selection",
                "issues": [
                    "Poor examples degrade performance",
                    "Example order matters significantly",
                    "Biased examples introduce bias",
                    "Hard to predict optimal examples"
                ]
            },
            {
                "limitation": "Task Complexity Limits",
                "description": "Struggles with highly complex or novel reasoning",
                "issues": [
                    "Limited multi-step reasoning",
                    "Difficulty with novel patterns",
                    "Struggles with abstract concepts",
                    "May not generalize well"
                ]
            }
        ]
        
        for limitation in limitations:
            with st.expander(f"❌ {limitation['limitation']}"):
                st.markdown(limitation['description'])
                st.markdown("**Key Issues:**")
                for issue in limitation['issues']:
                    st.markdown(f"• {issue}")
    
    with comparison_tabs[2]:
        st.markdown("### 🔄 ICL vs Fine-Tuning Trade-offs")
        
        # Trade-off visualization
        tradeoff_data = {
            'Approach': ['In-Context Learning', 'Fine-Tuning'],
            'Development Speed': [95, 25],
            'Peak Performance': [75, 95],
            'Data Requirements': [95, 30],
            'Computational Cost': [85, 40],
            'Flexibility': [90, 50],
            'Interpretability': [85, 60]
        }
        
        tradeoff_df = pd.DataFrame(tradeoff_data)
        
        # Radar chart
        categories = ['Development Speed', 'Peak Performance', 'Data Requirements', 
                     'Computational Cost', 'Flexibility', 'Interpretability']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=tradeoff_df.iloc[0, 1:].values,
            theta=categories,
            fill='toself',
            name='In-Context Learning',
            line_color='blue'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=tradeoff_df.iloc[1, 1:].values,
            theta=categories,
            fill='toself',
            name='Fine-Tuning',
            line_color='red'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="ICL vs Fine-Tuning Comparison",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Practical implementation
st.header("🛠️ Practical Implementation")

implementation_tabs = st.tabs([
    "📝 Example Design",
    "🎯 Selection Strategies", 
    "🔧 Optimization Techniques",
    "📊 Evaluation Methods"
])

with implementation_tabs[0]:
    st.subheader("📝 Example Design Guidelines")
    
    design_principles = [
        {
            "principle": "Clarity and Correctness",
            "description": "Examples must be unambiguous and correct",
            "guidelines": [
                "Use clear, simple language",
                "Ensure all examples are correct",
                "Avoid ambiguous cases",
                "Consistent formatting across examples"
            ],
            "example": {
                "good": "Input: 'I love this movie!' → Sentiment: Positive",
                "bad": "Input: 'movie good' → happy"
            }
        },
        {
            "principle": "Representativeness",
            "description": "Examples should cover the task distribution",
            "guidelines": [
                "Include diverse input types",
                "Cover different difficulty levels", 
                "Represent all output classes",
                "Include edge cases when possible"
            ],
            "example": {
                "good": "Mix of short/long texts, different topics, all sentiment classes",
                "bad": "Only short positive movie reviews"
            }
        },
        {
            "principle": "Relevance",
            "description": "Examples should be relevant to the target task",
            "guidelines": [
                "Match the domain and style",
                "Use similar input formats",
                "Align with target complexity",
                "Consider target use cases"
            ],
            "example": {
                "good": "Medical text examples for medical classification",
                "bad": "Movie reviews for medical sentiment analysis"
            }
        }
    ]
    
    for principle in design_principles:
        with st.expander(f"📐 {principle['principle']}"):
            st.markdown(principle['description'])
            
            st.markdown("**Guidelines:**")
            for guideline in principle['guidelines']:
                st.markdown(f"• {guideline}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Good Example:**")
                st.code(principle['example']['good'])
            with col2:
                st.markdown("**❌ Poor Example:**")
                st.code(principle['example']['bad'])

with implementation_tabs[1]:
    st.subheader("🎯 Example Selection Strategies")
    
    selection_strategies = [
        {
            "strategy": "Random Selection",
            "description": "Randomly sample from available examples",
            "pros": ["Simple to implement", "Unbiased selection", "Good baseline"],
            "cons": ["May miss important patterns", "Inconsistent performance", "No optimization"],
            "when_to_use": "Quick prototyping, baseline comparison",
            "implementation": """
import random

def random_selection(examples, k=5):
    return random.sample(examples, min(k, len(examples)))
"""
        },
        {
            "strategy": "Similarity-Based Selection",
            "description": "Select examples most similar to the target input",
            "pros": ["Relevant examples", "Better performance", "Adaptive to input"],
            "cons": ["Requires similarity computation", "May lack diversity", "Computationally expensive"],
            "when_to_use": "When you have large example pools",
            "implementation": """
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def similarity_selection(target, examples, embeddings, k=5):
    target_embedding = get_embedding(target)
    similarities = cosine_similarity([target_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[-k:]
    return [examples[i] for i in top_indices]
"""
        },
        {
            "strategy": "Diverse Selection",
            "description": "Select examples that maximize diversity",
            "pros": ["Covers different patterns", "Robust to various inputs", "Better generalization"],
            "cons": ["Complex optimization", "May include irrelevant examples", "Harder to implement"],
            "when_to_use": "Complex tasks requiring broad coverage",
            "implementation": """
def diverse_selection(examples, embeddings, k=5):
    selected = []
    remaining = list(range(len(examples)))
    
    # Start with random example
    first_idx = random.choice(remaining)
    selected.append(first_idx)
    remaining.remove(first_idx)
    
    # Greedily select most diverse examples
    for _ in range(k-1):
        if not remaining:
            break
            
        best_idx = None
        best_min_sim = -1
        
        for idx in remaining:
            min_sim = min([cosine_similarity([embeddings[idx]], [embeddings[s]])[0][0] 
                          for s in selected])
            if min_sim > best_min_sim:
                best_min_sim = min_sim
                best_idx = idx
        
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
    
    return [examples[i] for i in selected]
"""
        },
        {
            "strategy": "Active Learning",
            "description": "Iteratively select examples based on model uncertainty",
            "pros": ["Optimizes for model needs", "Efficient example use", "Improves over time"],
            "cons": ["Requires multiple iterations", "Complex implementation", "May overfit to model"],
            "when_to_use": "When you can iterate and have model feedback",
            "implementation": """
def active_learning_selection(model, examples, k=5):
    uncertainties = []
    
    for example in examples:
        # Get model prediction confidence
        prediction = model(example)
        uncertainty = calculate_uncertainty(prediction)
        uncertainties.append(uncertainty)
    
    # Select examples with highest uncertainty
    top_indices = np.argsort(uncertainties)[-k:]
    return [examples[i] for i in top_indices]
"""
        }
    ]
    
    for strategy in selection_strategies:
        with st.expander(f"🎯 {strategy['strategy']}"):
            st.markdown(strategy['description'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Pros:**")
                for pro in strategy['pros']:
                    st.markdown(f"• {pro}")
            with col2:
                st.markdown("**❌ Cons:**")
                for con in strategy['cons']:
                    st.markdown(f"• {con}")
            
            st.markdown(f"**When to use:** {strategy['when_to_use']}")
            st.markdown("**Implementation:**")
            st.code(strategy['implementation'], language='python')

with implementation_tabs[2]:
    st.subheader("🔧 ICL Optimization Techniques")
    
    optimization_techniques = [
        {
            "technique": "Example Ordering",
            "description": "Optimize the order of examples for better performance",
            "approaches": [
                "Increasing difficulty progression",
                "Most relevant examples last",
                "Random permutation testing",
                "Semantic clustering"
            ],
            "implementation": """
# Test different orderings
def optimize_example_order(examples, test_cases, model):
    best_order = None
    best_performance = 0
    
    for _ in range(10):  # Try 10 random orders
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        performance = evaluate_icl(model, shuffled, test_cases)
        if performance > best_performance:
            best_performance = performance
            best_order = shuffled
    
    return best_order, best_performance
"""
        },
        {
            "technique": "Template Optimization",
            "description": "Find the best prompt template format",
            "approaches": [
                "A/B test different formats",
                "Vary instruction phrasing",
                "Test different separators",
                "Optimize output formatting"
            ],
            "implementation": """
templates = [
    "Task: {task}\\nExample: {input} -> {output}\\nInput: {query}\\nOutput:",
    "{task}\\n\\n{input}: {output}\\n\\n{query}:",
    "Given: {input}\\nAnswer: {output}\\n\\nGiven: {query}\\nAnswer:",
]

def find_best_template(templates, examples, test_cases):
    best_template = None
    best_score = 0
    
    for template in templates:
        score = evaluate_template(template, examples, test_cases)
        if score > best_score:
            best_score = score
            best_template = template
    
    return best_template
"""
        },
        {
            "technique": "Context Length Management",
            "description": "Optimize usage of available context window",
            "approaches": [
                "Prioritize recent/relevant examples",
                "Compress example representations",
                "Dynamic example selection",
                "Hierarchical example organization"
            ],
            "implementation": """
def manage_context_length(examples, max_tokens, tokenizer):
    selected_examples = []
    current_tokens = 0
    
    # Sort by relevance (implement your relevance metric)
    examples = sorted(examples, key=lambda x: relevance_score(x), reverse=True)
    
    for example in examples:
        example_tokens = len(tokenizer.encode(format_example(example)))
        if current_tokens + example_tokens <= max_tokens:
            selected_examples.append(example)
            current_tokens += example_tokens
        else:
            break
    
    return selected_examples
"""
        },
        {
            "technique": "Multi-Shot Ensembling",
            "description": "Combine predictions from multiple example sets",
            "approaches": [
                "Vote across different example sets",
                "Weight by confidence scores",
                "Bootstrap sampling",
                "Cross-validation style selection"
            ],
            "implementation": """
def ensemble_icl_predictions(model, example_sets, query):
    predictions = []
    confidences = []
    
    for examples in example_sets:
        pred, conf = model.predict_with_confidence(examples, query)
        predictions.append(pred)
        confidences.append(conf)
    
    # Weighted voting
    weights = np.array(confidences) / sum(confidences)
    final_prediction = weighted_vote(predictions, weights)
    
    return final_prediction
"""
        }
    ]
    
    for technique in optimization_techniques:
        with st.expander(f"🔧 {technique['technique']}"):
            st.markdown(technique['description'])
            
            st.markdown("**Approaches:**")
            for approach in technique['approaches']:
                st.markdown(f"• {approach}")
            
            st.markdown("**Implementation Example:**")
            st.code(technique['implementation'], language='python')

with implementation_tabs[3]:
    st.subheader("📊 ICL Evaluation Methods")
    
    evaluation_methods = [
        {
            "method": "Hold-out Testing",
            "description": "Test ICL performance on unseen examples",
            "steps": [
                "Split data into examples and test sets",
                "Use examples for ICL demonstrations",
                "Evaluate on held-out test set",
                "Compare different example selections"
            ],
            "metrics": ["Accuracy", "F1-score", "Precision/Recall", "Task-specific metrics"],
            "considerations": [
                "Ensure test set represents real distribution",
                "Use stratified sampling for balanced evaluation",
                "Consider multiple random splits",
                "Account for example selection variance"
            ]
        },
        {
            "method": "Cross-Validation",
            "description": "Robust evaluation across different example sets",
            "steps": [
                "Divide examples into k folds",
                "Use k-1 folds for demonstration",
                "Test on remaining fold",
                "Average performance across folds"
            ],
            "metrics": ["Mean performance", "Standard deviation", "Confidence intervals"],
            "considerations": [
                "Choose appropriate k value",
                "Ensure sufficient examples per fold",
                "Stratify by important variables",
                "Report variance in addition to mean"
            ]
        },
        {
            "method": "Ablation Studies",
            "description": "Understand impact of different ICL components",
            "steps": [
                "Test with different numbers of examples",
                "Compare random vs optimized selection",
                "Evaluate different prompt formats",
                "Analyze example ordering effects"
            ],
            "metrics": ["Performance delta", "Effect size", "Statistical significance"],
            "considerations": [
                "Test one component at a time",
                "Use consistent evaluation setup",
                "Report confidence intervals",
                "Consider interaction effects"
            ]
        }
    ]
    
    for method in evaluation_methods:
        with st.expander(f"📊 {method['method']}"):
            st.markdown(method['description'])
            
            st.markdown("**Evaluation Steps:**")
            for step in method['steps']:
                st.markdown(f"• {step}")
            
            st.markdown("**Key Metrics:**")
            for metric in method['metrics']:
                st.markdown(f"• {metric}")
            
            st.markdown("**Important Considerations:**")
            for consideration in method['considerations']:
                st.markdown(f"• {consideration}")

# Interactive ICL explorer
st.header("🎮 Interactive ICL Explorer")

explorer_tabs = st.tabs(["🔧 ICL Builder", "📊 Performance Simulator", "🎯 Example Analyzer"])

with explorer_tabs[0]:
    st.subheader("🔧 Build Your ICL Prompt")
    
    # Task configuration
    task_type = st.selectbox("Select Task Type", [
        "Text Classification", "Sentiment Analysis", "Question Answering",
        "Translation", "Summarization", "Named Entity Recognition"
    ])
    
    # Example management
    st.markdown("### 📚 Manage Examples")
    num_examples = st.slider("Number of Examples", 1, 8, 3)
    
    examples = []
    for i in range(num_examples):
        col1, col2 = st.columns(2)
        with col1:
            input_text = st.text_input(f"Example {i+1} Input", key=f"input_{i}")
        with col2:
            output_text = st.text_input(f"Example {i+1} Output", key=f"output_{i}")
        
        if input_text and output_text:
            examples.append({"input": input_text, "output": output_text})
    
    # Query input
    query_input = st.text_area("Query Input", placeholder="Enter the text you want to process...")
    
    # Generate ICL prompt
    if st.button("🚀 Generate ICL Prompt") and examples and query_input:
        prompt_parts = []
        prompt_parts.append(f"Task: {task_type}")
        prompt_parts.append("")
        
        for i, example in enumerate(examples):
            prompt_parts.append(f"Example {i+1}:")
            prompt_parts.append(f"Input: {example['input']}")
            prompt_parts.append(f"Output: {example['output']}")
            prompt_parts.append("")
        
        prompt_parts.append("Query:")
        prompt_parts.append(f"Input: {query_input}")
        prompt_parts.append("Output:")
        
        generated_prompt = "\n".join(prompt_parts)
        
        st.markdown("### 📝 Generated ICL Prompt")
        st.code(generated_prompt)
        
        # Analyze prompt
        prompt_length = len(generated_prompt)
        estimated_tokens = prompt_length // 4
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prompt Length", f"{prompt_length} chars")
        with col2:
            st.metric("Estimated Tokens", estimated_tokens)
        with col3:
            st.metric("Examples Used", len(examples))

with explorer_tabs[1]:
    st.subheader("📊 ICL Performance Simulator")
    
    st.markdown("Simulate how different factors affect ICL performance:")
    
    # Simulation parameters
    col1, col2 = st.columns(2)
    
    with col1:
        model_size = st.selectbox("Model Size", ["1B", "7B", "13B", "70B", "175B"])
        task_complexity = st.slider("Task Complexity", 1, 10, 5)
        example_quality = st.slider("Example Quality", 1, 10, 8)
    
    with col2:
        num_examples_sim = st.slider("Number of Examples", 0, 10, 5)
        example_relevance = st.slider("Example Relevance", 1, 10, 7)
        context_efficiency = st.slider("Context Efficiency", 1, 10, 6)
    
    if st.button("🎲 Run Simulation"):
        # Simulate performance based on parameters
        base_performance = {
            "1B": 40, "7B": 60, "13B": 70, "70B": 80, "175B": 85
        }[model_size]
        
        # Apply modifiers
        complexity_modifier = (11 - task_complexity) * 0.05
        quality_modifier = example_quality * 0.03
        relevance_modifier = example_relevance * 0.02
        examples_modifier = min(num_examples_sim * 0.04, 0.2)
        efficiency_modifier = context_efficiency * 0.01
        
        final_performance = base_performance + (
            complexity_modifier + quality_modifier + relevance_modifier + 
            examples_modifier + efficiency_modifier
        ) * base_performance
        
        final_performance = max(0, min(100, final_performance))
        
        # Display results
        st.markdown("### 🎯 Simulation Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Performance", f"{final_performance:.1f}%")
        with col2:
            st.metric("Confidence Level", f"{85 + complexity_modifier*10:.0f}%")
        with col3:
            performance_tier = "Excellent" if final_performance > 80 else "Good" if final_performance > 60 else "Fair"
            st.metric("Performance Tier", performance_tier)
        
        # Recommendations
        st.markdown("### 💡 Optimization Recommendations")
        
        recommendations = []
        if example_quality < 7:
            recommendations.append("📝 Improve example quality - ensure correctness and clarity")
        if example_relevance < 7:
            recommendations.append("🎯 Select more relevant examples for your specific use case")
        if num_examples_sim < 3:
            recommendations.append("📚 Add more examples (optimal range: 3-8)")
        if task_complexity > 7:
            recommendations.append("🧩 Consider breaking down complex tasks into simpler steps")
        if context_efficiency < 7:
            recommendations.append("⚡ Optimize context usage - remove unnecessary information")
        
        if not recommendations:
            recommendations.append("✅ Your configuration looks well-optimized!")
        
        for rec in recommendations:
            st.markdown(f"• {rec}")

with explorer_tabs[2]:
    st.subheader("🎯 Example Quality Analyzer")
    
    st.markdown("Analyze the quality of your ICL examples:")
    
    # Example input
    analysis_input = st.text_area("Paste your example (format: Input → Output)", 
                                 placeholder="Input: 'Great movie!' → Output: Positive")
    
    if analysis_input and st.button("🔍 Analyze Example"):
        # Simulate example analysis
        factors = {
            "Clarity": np.random.uniform(7, 9),
            "Correctness": np.random.uniform(8, 10),
            "Relevance": np.random.uniform(6, 9),
            "Completeness": np.random.uniform(7, 9),
            "Consistency": np.random.uniform(7, 9)
        }
        
        overall_score = np.mean(list(factors.values()))
        
        st.markdown("### 📊 Quality Analysis Results")
        
        # Quality metrics
        col1, col2 = st.columns(2)
        
        with col1:
            for factor, score in factors.items():
                st.metric(factor, f"{score:.1f}/10")
        
        with col2:
            st.metric("Overall Quality", f"{overall_score:.1f}/10")
            
            quality_level = "Excellent" if overall_score > 8.5 else "Good" if overall_score > 7 else "Needs Improvement"
            color = "green" if overall_score > 8.5 else "orange" if overall_score > 7 else "red"
            st.markdown(f"**Quality Level:** :{color}[{quality_level}]")
        
        # Improvement suggestions
        st.markdown("### 💡 Improvement Suggestions")
        
        suggestions = []
        if factors["Clarity"] < 8:
            suggestions.append("📝 Make input and output more explicit and unambiguous")
        if factors["Correctness"] < 9:
            suggestions.append("✅ Double-check that the output is completely correct")
        if factors["Relevance"] < 8:
            suggestions.append("🎯 Ensure example matches your target task domain")
        if factors["Completeness"] < 8:
            suggestions.append("📋 Provide more complete context or explanation")
        if factors["Consistency"] < 8:
            suggestions.append("🔄 Maintain consistent format with other examples")
        
        if not suggestions:
            suggestions.append("🌟 This example looks high quality!")
        
        for suggestion in suggestions:
            st.markdown(f"• {suggestion}")

# Best practices and tips
st.header("💡 ICL Best Practices")

best_practices = [
    "**Start Simple**: Begin with clear, straightforward examples before adding complexity",
    "**Quality over Quantity**: 3-5 high-quality examples often outperform 10 mediocre ones",
    "**Test Example Order**: The order of examples can significantly impact performance",
    "**Match Target Domain**: Use examples from the same domain as your target use case",
    "**Include Edge Cases**: Add examples that cover boundary conditions and corner cases",
    "**Validate Thoroughly**: Test your ICL setup on diverse inputs before deployment",
    "**Monitor Performance**: Track performance over time as input distributions may shift",
    "**Document Examples**: Keep track of which examples work best for which tasks",
    "**Consider Context Limits**: Balance example richness with context window constraints",
    "**Iterate Based on Results**: Use performance feedback to refine your example selection"
]

for practice in best_practices:
    st.markdown(f"• {practice}")

# Additional resources
st.header("📚 Learning Resources")

resources = [
    {
        "title": "In-Context Learning and Induction Heads",
        "type": "Research Paper",
        "description": "Foundational paper on the mechanisms behind ICL",
        "difficulty": "Advanced"
    },
    {
        "title": "What Makes Good In-Context Examples for GPT-3?",
        "type": "Research Paper", 
        "description": "Empirical study on example selection strategies",
        "difficulty": "Intermediate"
    },
    {
        "title": "A Survey of In-Context Learning",
        "type": "Survey Paper",
        "description": "Comprehensive overview of ICL research and applications",
        "difficulty": "Intermediate"
    },
    {
        "title": "Practical Guide to In-Context Learning",
        "type": "Tutorial",
        "description": "Hands-on guide with examples and code",
        "difficulty": "Beginner"
    }
]

for resource in resources:
    with st.expander(f"📖 {resource['title']}"):
        st.markdown(f"**Type:** {resource['type']}")
        st.markdown(f"**Description:** {resource['description']}")
        st.markdown(f"**Difficulty:** {resource['difficulty']}")