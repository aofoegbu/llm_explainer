import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Prompt Engineering", page_icon="🎨", layout="wide")

st.title("🎨 Prompt Engineering")
st.markdown("### Crafting Effective Prompts for Optimal LLM Performance")

# Overview
st.header("🎯 Overview")
st.markdown("""
Prompt engineering is the art and science of crafting inputs that elicit the best responses from language models. 
Effective prompts can dramatically improve model performance across tasks without requiring fine-tuning, 
making it a crucial skill for LLM practitioners.
""")

# Core concepts
st.header("🧠 Core Concepts")

concept_tabs = st.tabs([
    "📝 Prompt Components",
    "🎯 Design Principles", 
    "🔧 Techniques",
    "📊 Evaluation"
])

with concept_tabs[0]:
    st.subheader("📝 Prompt Components")
    
    components = [
        {
            "component": "Context",
            "description": "Background information that helps the model understand the task",
            "examples": [
                "You are an expert data scientist...",
                "Given the following customer reviews...",
                "In the context of medical diagnosis..."
            ],
            "best_practices": [
                "Provide relevant domain knowledge",
                "Set appropriate tone and style",
                "Include necessary constraints",
                "Specify the target audience"
            ]
        },
        {
            "component": "Task Description",
            "description": "Clear specification of what you want the model to do",
            "examples": [
                "Classify the sentiment of the following text",
                "Summarize the key points in 3 sentences",
                "Generate a creative story based on these elements"
            ],
            "best_practices": [
                "Be specific and unambiguous",
                "Use action verbs",
                "Specify output format",
                "Include success criteria"
            ]
        },
        {
            "component": "Examples",
            "description": "Sample inputs and desired outputs to guide the model",
            "examples": [
                "Input: 'Great product!' → Output: Positive",
                "Text: '...' Summary: '...'",
                "Question: '...' Answer: '...'"
            ],
            "best_practices": [
                "Use diverse, representative examples",
                "Show edge cases and corner cases",
                "Maintain consistent formatting",
                "Include both positive and negative examples"
            ]
        },
        {
            "component": "Input Data",
            "description": "The actual data you want the model to process",
            "examples": [
                "Text to analyze: 'This movie was amazing!'",
                "Document: [actual document content]",
                "Question: 'What is the capital of France?'"
            ],
            "best_practices": [
                "Clearly delineate from other components",
                "Ensure proper formatting",
                "Validate input quality",
                "Handle special characters appropriately"
            ]
        }
    ]
    
    for component in components:
        with st.expander(f"🔍 {component['component']}"):
            st.markdown(component['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📋 Examples:**")
                for example in component['examples']:
                    st.code(example)
            
            with col2:
                st.markdown("**✨ Best Practices:**")
                for practice in component['best_practices']:
                    st.markdown(f"• {practice}")

with concept_tabs[1]:
    st.subheader("🎯 Design Principles")
    
    principles = [
        {
            "principle": "Clarity and Specificity",
            "description": "Make your intentions crystal clear",
            "guidelines": [
                "Use precise language",
                "Avoid ambiguous terms",
                "Specify desired output format",
                "Include relevant constraints"
            ],
            "example": {
                "bad": "Tell me about dogs",
                "good": "Write a 200-word informative paragraph about Golden Retrievers, focusing on their temperament and suitability as family pets."
            }
        },
        {
            "principle": "Context Relevance",
            "description": "Provide sufficient and relevant background",
            "guidelines": [
                "Include domain-specific knowledge",
                "Set appropriate context length",
                "Remove irrelevant information",
                "Consider the model's knowledge cutoff"
            ],
            "example": {
                "bad": "Solve this problem: 2x + 5 = 11",
                "good": "As a math tutor helping a 10th-grade student, solve this linear equation step-by-step, explaining each step: 2x + 5 = 11"
            }
        },
        {
            "principle": "Progressive Complexity",
            "description": "Start simple and add complexity gradually",
            "guidelines": [
                "Begin with basic requirements",
                "Add constraints incrementally",
                "Test simple cases first",
                "Build on working examples"
            ],
            "example": {
                "bad": "Create a comprehensive business plan with financial projections, market analysis, and risk assessment for a tech startup",
                "good": "First create a simple business concept. Then we'll add market analysis, then financial projections, then risk assessment."
            }
        },
        {
            "principle": "Consistency",
            "description": "Maintain consistent format and terminology",
            "guidelines": [
                "Use consistent formatting",
                "Maintain same terminology",
                "Apply uniform structure",
                "Keep style consistent"
            ],
            "example": {
                "bad": "Classify: Happy. Analyze sentiment: Sad. Determine emotion: Angry.",
                "good": "Classify sentiment: Happy → Positive. Classify sentiment: Sad → Negative. Classify sentiment: Angry → Negative."
            }
        }
    ]
    
    for principle in principles:
        with st.expander(f"🎯 {principle['principle']}"):
            st.markdown(principle['description'])
            
            st.markdown("**📋 Guidelines:**")
            for guideline in principle['guidelines']:
                st.markdown(f"• {guideline}")
            
            st.markdown("**📝 Example:**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**❌ Poor:**")
                st.code(principle['example']['bad'])
            with col2:
                st.markdown("**✅ Better:**")
                st.code(principle['example']['good'])

with concept_tabs[2]:
    st.subheader("🔧 Prompt Engineering Techniques")
    
    technique_tabs = st.tabs([
        "🔄 Few-Shot Learning",
        "🤔 Chain-of-Thought", 
        "🎭 Role Playing",
        "🔧 Template Engineering"
    ])
    
    with technique_tabs[0]:
        st.markdown("### 🔄 Few-Shot Learning")
        
        st.markdown("""
        Few-shot learning uses examples to teach the model the desired behavior pattern.
        The quality and diversity of examples significantly impact performance.
        """)
        
        shot_variations = [
            {
                "type": "Zero-Shot",
                "description": "No examples provided, rely on task description alone",
                "when_to_use": "Simple, well-defined tasks or when examples are unavailable",
                "example": """
Task: Classify the sentiment of customer reviews as Positive, Negative, or Neutral.

Review: "The product arrived quickly and works perfectly!"
Sentiment:"""
            },
            {
                "type": "One-Shot",
                "description": "Single example to demonstrate the pattern",
                "when_to_use": "When task is relatively clear but needs format guidance",
                "example": """
Task: Classify sentiment as Positive, Negative, or Neutral.

Example:
Review: "Great quality and fast shipping!"
Sentiment: Positive

Review: "The product arrived quickly and works perfectly!"
Sentiment:"""
            },
            {
                "type": "Few-Shot (3-5 examples)",
                "description": "Multiple examples covering different cases",
                "when_to_use": "Complex tasks requiring pattern recognition",
                "example": """
Task: Classify sentiment as Positive, Negative, or Neutral.

Examples:
Review: "Amazing product, highly recommend!"
Sentiment: Positive

Review: "Poor quality, broke after one day."
Sentiment: Negative

Review: "It's okay, nothing special."
Sentiment: Neutral

Review: "Fast delivery but average quality."
Sentiment: Neutral

Review: "The product arrived quickly and works perfectly!"
Sentiment:"""
            }
        ]
        
        for variation in shot_variations:
            with st.expander(f"📊 {variation['type']}"):
                st.markdown(variation['description'])
                st.markdown(f"**When to use:** {variation['when_to_use']}")
                st.markdown("**Example:**")
                st.code(variation['example'])
    
    with technique_tabs[1]:
        st.markdown("### 🤔 Chain-of-Thought Prompting")
        
        st.markdown("""
        Chain-of-thought prompting encourages the model to break down complex problems 
        into step-by-step reasoning, leading to more accurate and explainable results.
        """)
        
        cot_examples = [
            {
                "task": "Mathematical Problem",
                "without_cot": """
Problem: Sarah has 3 boxes of apples. Each box contains 12 apples. She gives away 8 apples. How many apples does she have left?

Answer: 28 apples""",
                "with_cot": """
Problem: Sarah has 3 boxes of apples. Each box contains 12 apples. She gives away 8 apples. How many apples does she have left?

Let me think through this step by step:
1. First, I need to find the total number of apples Sarah has initially
2. She has 3 boxes with 12 apples each: 3 × 12 = 36 apples
3. She gives away 8 apples: 36 - 8 = 28 apples
4. Therefore, Sarah has 28 apples left.

Answer: 28 apples"""
            },
            {
                "task": "Logical Reasoning",
                "without_cot": """
All birds can fly. Penguins are birds. Can penguins fly?

Answer: No""",
                "with_cot": """
All birds can fly. Penguins are birds. Can penguins fly?

Let me reason through this:
1. The premise states "All birds can fly"
2. The premise states "Penguins are birds"
3. Following the logic: If all birds can fly, and penguins are birds, then penguins should be able to fly
4. However, I know from real-world knowledge that penguins cannot fly
5. This reveals that the initial premise "All birds can fly" is incorrect

Answer: No, penguins cannot fly, which shows the premise is flawed."""
            }
        ]
        
        for example in cot_examples:
            with st.expander(f"🧮 {example['task']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**❌ Without Chain-of-Thought:**")
                    st.code(example['without_cot'])
                
                with col2:
                    st.markdown("**✅ With Chain-of-Thought:**")
                    st.code(example['with_cot'])
    
    with technique_tabs[2]:
        st.markdown("### 🎭 Role Playing")
        
        st.markdown("""
        Role playing assigns the model a specific persona or expertise, 
        which can improve performance on domain-specific tasks.
        """)
        
        role_examples = [
            {
                "role": "Expert Professional",
                "example": """
You are a senior software engineer with 10 years of experience in Python development. 
Review this code for potential issues and suggest improvements:

```python
def calculate_average(numbers):
    return sum(numbers) / len(numbers)
```

Please provide detailed feedback on:
1. Potential errors or edge cases
2. Code quality and best practices
3. Suggested improvements""",
                "benefits": ["Domain expertise", "Professional perspective", "Detailed analysis"]
            },
            {
                "role": "Specific Audience Perspective",
                "example": """
You are explaining cloud computing concepts to a 12-year-old who loves video games. 
Explain what "serverless computing" means using analogies they would understand.

Make sure to:
- Use gaming analogies
- Keep language simple
- Make it engaging and fun""",
                "benefits": ["Appropriate language level", "Relevant analogies", "Engaging explanations"]
            },
            {
                "role": "Creative Character",
                "example": """
You are Shakespeare, the famous playwright. Write a short soliloquy (4-6 lines) 
about the challenges of modern social media, but in your characteristic 
Elizabethan style and iambic pentameter.

Focus on themes of:
- Truth vs. appearance
- Public vs. private self
- The speed of modern communication""",
                "benefits": ["Consistent style", "Creative constraints", "Unique perspective"]
            }
        ]
        
        for role in role_examples:
            with st.expander(f"🎭 {role['role']}"):
                st.code(role['example'])
                st.markdown("**Benefits:**")
                for benefit in role['benefits']:
                    st.markdown(f"• {benefit}")
    
    with technique_tabs[3]:
        st.markdown("### 🔧 Template Engineering")
        
        st.markdown("""
        Template engineering creates reusable prompt structures that can be 
        adapted for similar tasks, ensuring consistency and efficiency.
        """)
        
        templates = [
            {
                "name": "Classification Template",
                "structure": """
{CONTEXT}

Task: Classify the following {INPUT_TYPE} into one of these categories: {CATEGORIES}

{EXAMPLES}

{INPUT_LABEL}: {INPUT_TEXT}
{OUTPUT_LABEL}:""",
                "variables": {
                    "CONTEXT": "Background information about the classification task",
                    "INPUT_TYPE": "Type of data being classified (text, image, etc.)",
                    "CATEGORIES": "List of possible classification labels",
                    "EXAMPLES": "Few-shot examples (optional)",
                    "INPUT_LABEL": "Label for the input data",
                    "INPUT_TEXT": "Actual data to classify",
                    "OUTPUT_LABEL": "Label for expected output"
                }
            },
            {
                "name": "Generation Template",
                "structure": """
{ROLE_CONTEXT}

{TASK_DESCRIPTION}

Requirements:
{REQUIREMENTS}

{EXAMPLES}

Input: {INPUT}
Output:""",
                "variables": {
                    "ROLE_CONTEXT": "Role or expertise assignment",
                    "TASK_DESCRIPTION": "What needs to be generated",
                    "REQUIREMENTS": "Constraints and specifications",
                    "EXAMPLES": "Sample inputs and outputs",
                    "INPUT": "Actual input data"
                }
            },
            {
                "name": "Analysis Template",
                "structure": """
{EXPERT_ROLE}

Analyze the following {ANALYSIS_TARGET} focusing on:
{ANALYSIS_DIMENSIONS}

Provide your analysis in this format:
{OUTPUT_FORMAT}

{ANALYSIS_TARGET}: {INPUT_DATA}

Analysis:""",
                "variables": {
                    "EXPERT_ROLE": "Subject matter expert assignment",
                    "ANALYSIS_TARGET": "What is being analyzed",
                    "ANALYSIS_DIMENSIONS": "Specific aspects to focus on",
                    "OUTPUT_FORMAT": "Structure for the output",
                    "INPUT_DATA": "Data to be analyzed"
                }
            }
        ]
        
        for template in templates:
            with st.expander(f"📋 {template['name']}"):
                st.markdown("**Template Structure:**")
                st.code(template['structure'])
                
                st.markdown("**Template Variables:**")
                for var, description in template['variables'].items():
                    st.markdown(f"• **{var}**: {description}")

with concept_tabs[3]:
    st.subheader("📊 Prompt Evaluation")
    
    eval_tabs = st.tabs(["📏 Evaluation Metrics", "🧪 Testing Strategies", "🔄 Iteration Process"])
    
    with eval_tabs[0]:
        st.markdown("### 📏 Evaluation Metrics")
        
        metrics = [
            {
                "metric": "Accuracy",
                "description": "Percentage of correct responses",
                "when_to_use": "Classification and factual tasks",
                "calculation": "Correct responses / Total responses",
                "considerations": ["Define what constitutes 'correct'", "Account for partial correctness", "Consider inter-annotator agreement"]
            },
            {
                "metric": "Relevance",
                "description": "How well responses address the prompt",
                "when_to_use": "Open-ended generation tasks",
                "calculation": "Typically rated on a scale (1-5)",
                "considerations": ["Define relevance criteria clearly", "Use multiple evaluators", "Consider context appropriateness"]
            },
            {
                "metric": "Consistency",
                "description": "Stability of responses across similar inputs",
                "when_to_use": "When reliability is important",
                "calculation": "Variance in responses to similar prompts",
                "considerations": ["Test with paraphrased prompts", "Check for format consistency", "Measure semantic consistency"]
            },
            {
                "metric": "Efficiency",
                "description": "Token usage and response time",
                "when_to_use": "Production deployments",
                "calculation": "Tokens per task / Time per response",
                "considerations": ["Balance quality vs. cost", "Consider prompt length impact", "Monitor response latency"]
            }
        ]
        
        for metric in metrics:
            with st.expander(f"📊 {metric['metric']}"):
                st.markdown(metric['description'])
                st.markdown(f"**When to use:** {metric['when_to_use']}")
                st.markdown(f"**Calculation:** {metric['calculation']}")
                st.markdown("**Considerations:**")
                for consideration in metric['considerations']:
                    st.markdown(f"• {consideration}")
    
    with eval_tabs[1]:
        st.markdown("### 🧪 Testing Strategies")
        
        strategies = [
            {
                "strategy": "A/B Testing",
                "description": "Compare different prompt versions on the same task",
                "implementation": [
                    "Create 2-3 prompt variants",
                    "Test on same dataset split",
                    "Use statistical significance testing",
                    "Consider both automated and human evaluation"
                ],
                "example": "Test different instruction phrasings: 'Classify sentiment' vs 'Determine emotional tone'"
            },
            {
                "strategy": "Edge Case Testing",
                "description": "Test prompts on challenging or unusual inputs",
                "implementation": [
                    "Create edge case test sets",
                    "Include ambiguous examples",
                    "Test with out-of-domain data",
                    "Evaluate failure modes"
                ],
                "example": "Test sentiment classifier on sarcastic comments, mixed emotions, or very short texts"
            },
            {
                "strategy": "Ablation Studies",
                "description": "Remove prompt components to understand their impact",
                "implementation": [
                    "Remove one component at a time",
                    "Test with minimal vs. maximal context",
                    "Vary number of examples",
                    "Compare different example selections"
                ],
                "example": "Test impact of examples: 0-shot vs 1-shot vs 3-shot vs 5-shot"
            }
        ]
        
        for strategy in strategies:
            with st.expander(f"🔬 {strategy['strategy']}"):
                st.markdown(strategy['description'])
                st.markdown("**Implementation Steps:**")
                for step in strategy['implementation']:
                    st.markdown(f"• {step}")
                st.markdown(f"**Example:** {strategy['example']}")
    
    with eval_tabs[2]:
        st.markdown("### 🔄 Iteration Process")
        
        iteration_steps = [
            {
                "step": "1. Baseline Establishment",
                "actions": [
                    "Create initial prompt version",
                    "Test on representative dataset",
                    "Measure baseline performance",
                    "Document results and observations"
                ]
            },
            {
                "step": "2. Error Analysis",
                "actions": [
                    "Identify common failure patterns",
                    "Categorize types of errors",
                    "Analyze edge cases",
                    "Look for systematic biases"
                ]
            },
            {
                "step": "3. Hypothesis Formation",
                "actions": [
                    "Propose specific improvements",
                    "Predict expected impact",
                    "Prioritize changes by potential impact",
                    "Consider implementation complexity"
                ]
            },
            {
                "step": "4. Implementation & Testing",
                "actions": [
                    "Implement one change at a time",
                    "Test on validation set",
                    "Compare against baseline",
                    "Document what worked and what didn't"
                ]
            },
            {
                "step": "5. Validation & Deployment",
                "actions": [
                    "Test on held-out test set",
                    "Validate with human evaluation",
                    "Check for unintended side effects",
                    "Deploy with monitoring"
                ]
            }
        ]
        
        for step_info in iteration_steps:
            with st.expander(f"📋 {step_info['step']}"):
                for action in step_info['actions']:
                    st.markdown(f"• {action}")

# Interactive prompt builder
st.header("🛠️ Interactive Prompt Builder")

builder_tabs = st.tabs(["📝 Component Builder", "🎮 Live Testing", "📊 Performance Tracker"])

with builder_tabs[0]:
    st.subheader("📝 Build Your Prompt")
    
    # Prompt builder form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Task Configuration")
        task_type = st.selectbox("Task Type", [
            "Text Classification", "Sentiment Analysis", "Summarization", 
            "Question Answering", "Creative Writing", "Code Generation",
            "Translation", "Analysis", "Custom"
        ])
        
        role = st.text_area("Role/Context (optional)", 
                           placeholder="You are an expert data scientist...")
        
        task_description = st.text_area("Task Description", 
                                       placeholder="Classify the sentiment of the following text...")
        
        output_format = st.text_area("Output Format (optional)",
                                    placeholder="Format: [Category] - [Confidence]")
    
    with col2:
        st.markdown("### 📚 Examples")
        num_examples = st.slider("Number of Examples", 0, 5, 2)
        
        examples = []
        for i in range(num_examples):
            example_input = st.text_input(f"Example {i+1} Input", key=f"ex_input_{i}")
            example_output = st.text_input(f"Example {i+1} Output", key=f"ex_output_{i}")
            if example_input and example_output:
                examples.append({"input": example_input, "output": example_output})
        
        st.markdown("### 📝 Test Input")
        test_input = st.text_area("Input to test with your prompt")
    
    # Generate prompt
    if st.button("🔨 Generate Prompt"):
        prompt_parts = []
        
        if role:
            prompt_parts.append(role)
            prompt_parts.append("")
        
        if task_description:
            prompt_parts.append(task_description)
            prompt_parts.append("")
        
        if output_format:
            prompt_parts.append(f"Output format: {output_format}")
            prompt_parts.append("")
        
        if examples:
            prompt_parts.append("Examples:")
            for ex in examples:
                prompt_parts.append(f"Input: {ex['input']}")
                prompt_parts.append(f"Output: {ex['output']}")
                prompt_parts.append("")
        
        if test_input:
            prompt_parts.append(f"Input: {test_input}")
            prompt_parts.append("Output:")
        
        generated_prompt = "\n".join(prompt_parts)
        
        st.markdown("### 📋 Generated Prompt")
        st.code(generated_prompt)
        
        # Calculate prompt statistics
        prompt_length = len(generated_prompt)
        estimated_tokens = prompt_length // 4  # Rough estimation
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Characters", prompt_length)
        with col2:
            st.metric("Est. Tokens", estimated_tokens)
        with col3:
            st.metric("Examples Used", len(examples))

with builder_tabs[1]:
    st.subheader("🎮 Live Prompt Testing")
    
    st.markdown("""
    **Note:** This is a simulation interface. In a real implementation, 
    this would connect to an actual language model API for testing.
    """)
    
    test_prompt = st.text_area("Enter your prompt to test", height=200)
    
    if test_prompt and st.button("🚀 Test Prompt"):
        # Simulate response (in real implementation, this would call an LLM)
        st.markdown("### 🤖 Simulated Response")
        st.info("This would show the actual model response in a real implementation.")
        
        # Simulate metrics
        st.markdown("### 📊 Response Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Response Time", "1.2s")
        with col2:
            st.metric("Token Usage", "150")
        with col3:
            st.metric("Estimated Cost", "$0.002")
        with col4:
            st.metric("Quality Score", "8.5/10")

with builder_tabs[2]:
    st.subheader("📊 Performance Tracking")
    
    st.markdown("Track and compare different prompt versions:")
    
    # Simulated performance data
    prompt_versions = pd.DataFrame({
        'Version': ['v1.0', 'v1.1', 'v1.2', 'v2.0', 'v2.1'],
        'Accuracy': [0.72, 0.78, 0.81, 0.85, 0.87],
        'Avg_Tokens': [120, 135, 150, 140, 145],
        'Cost_Per_1K': [0.15, 0.18, 0.20, 0.19, 0.20],
        'Human_Rating': [7.2, 7.8, 8.1, 8.5, 8.7]
    })
    
    # Performance charts
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=prompt_versions['Version'],
        y=prompt_versions['Accuracy'],
        mode='lines+markers',
        name='Accuracy',
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=prompt_versions['Version'],
        y=prompt_versions['Human_Rating']/10,  # Normalize to 0-1
        mode='lines+markers',
        name='Human Rating (scaled)',
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=prompt_versions['Version'],
        y=prompt_versions['Avg_Tokens'],
        mode='lines+markers',
        name='Avg Tokens',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Prompt Performance Evolution",
        xaxis_title="Prompt Version",
        yaxis=dict(title="Performance Metrics", side="left"),
        yaxis2=dict(title="Token Usage", side="right", overlaying="y"),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance table
    st.markdown("### 📋 Detailed Performance Comparison")
    st.dataframe(prompt_versions, use_container_width=True)

# Best practices
st.header("✨ Prompt Engineering Best Practices")

best_practices_tabs = st.tabs(["🎯 General Guidelines", "⚠️ Common Pitfalls", "🔧 Advanced Tips"])

with best_practices_tabs[0]:
    st.subheader("🎯 General Guidelines")
    
    guidelines = [
        "**Be Specific**: Vague prompts lead to inconsistent results",
        "**Provide Context**: Give the model necessary background information",
        "**Use Examples**: Show don't just tell - demonstrate the desired pattern",
        "**Iterate Systematically**: Make one change at a time and measure impact",
        "**Test Edge Cases**: Ensure robustness across different input types",
        "**Consider Token Efficiency**: Balance detail with conciseness",
        "**Document Everything**: Keep track of what works and what doesn't",
        "**Validate with Humans**: Automated metrics don't capture everything",
        "**Think About Bias**: Consider how prompts might introduce unwanted bias",
        "**Plan for Scale**: Design prompts that work consistently at scale"
    ]
    
    for guideline in guidelines:
        st.markdown(f"• {guideline}")

with best_practices_tabs[1]:
    st.subheader("⚠️ Common Pitfalls")
    
    pitfalls = [
        {
            "pitfall": "Overloading with Information",
            "description": "Including too much irrelevant context",
            "solution": "Focus on essential information only",
            "example": "Don't include entire documentation when only specific function is needed"
        },
        {
            "pitfall": "Inconsistent Examples",
            "description": "Examples that contradict each other or the task",
            "solution": "Carefully curate examples to be consistent and representative",
            "example": "All sentiment examples should follow the same labeling scheme"
        },
        {
            "pitfall": "Prompt Injection Vulnerability",
            "description": "Prompts that can be manipulated by user input",
            "solution": "Validate and sanitize inputs, use clear delimiters",
            "example": "Use triple quotes or special tokens to separate user input"
        },
        {
            "pitfall": "Assuming Model Knowledge",
            "description": "Relying on knowledge the model might not have",
            "solution": "Provide necessary context or verify model capabilities",
            "example": "Don't assume model knows recent events or proprietary information"
        },
        {
            "pitfall": "Ignoring Output Variability",
            "description": "Not accounting for non-deterministic outputs",
            "solution": "Test multiple times, use temperature settings appropriately",
            "example": "Creative tasks may need higher temperature, classification tasks lower"
        }
    ]
    
    for pitfall in pitfalls:
        with st.expander(f"⚠️ {pitfall['pitfall']}"):
            st.markdown(pitfall['description'])
            st.markdown(f"**Solution:** {pitfall['solution']}")
            st.markdown(f"**Example:** {pitfall['example']}")

with best_practices_tabs[2]:
    st.subheader("🔧 Advanced Tips")
    
    advanced_tips = [
        {
            "tip": "Prompt Chaining",
            "description": "Break complex tasks into a series of simpler prompts",
            "benefit": "Better control, easier debugging, more reliable results",
            "example": "1) Extract key info → 2) Analyze extracted info → 3) Generate final output"
        },
        {
            "tip": "Dynamic Examples",
            "description": "Select examples based on input similarity",
            "benefit": "More relevant examples improve performance",
            "example": "For text classification, choose examples most similar to input text"
        },
        {
            "tip": "Meta-Prompting",
            "description": "Have the model help design better prompts",
            "benefit": "Leverage model's understanding of effective prompts",
            "example": "Ask model to suggest improvements to your prompt"
        },
        {
            "tip": "Constrained Generation",
            "description": "Use format constraints to guide output structure",
            "benefit": "More consistent, parseable outputs",
            "example": "Require JSON format, specific word counts, or structured templates"
        },
        {
            "tip": "Prompt Ensembling",
            "description": "Use multiple prompt variations and combine results",
            "benefit": "More robust and reliable outputs",
            "example": "Generate responses with 3 different prompts and select best one"
        }
    ]
    
    for tip in advanced_tips:
        with st.expander(f"🚀 {tip['tip']}"):
            st.markdown(tip['description'])
            st.markdown(f"**Benefit:** {tip['benefit']}")
            st.markdown(f"**Example:** {tip['example']}")

# Resources
st.header("📚 Learning Resources")

resources = [
    {
        "title": "Prompt Engineering Guide",
        "type": "Documentation",
        "description": "Comprehensive guide covering all major techniques",
        "difficulty": "Beginner to Advanced"
    },
    {
        "title": "OpenAI Prompt Engineering Best Practices",
        "type": "Official Guide",
        "description": "Best practices from OpenAI team",
        "difficulty": "Intermediate"
    },
    {
        "title": "Chain-of-Thought Prompting Paper",
        "type": "Research Paper",
        "description": "Original research on chain-of-thought techniques",
        "difficulty": "Advanced"
    },
    {
        "title": "Prompt Engineering Course",
        "type": "Online Course",
        "description": "Hands-on course with practical exercises",
        "difficulty": "Beginner"
    }
]

for resource in resources:
    with st.expander(f"📖 {resource['title']}"):
        st.markdown(f"**Type:** {resource['type']}")
        st.markdown(f"**Description:** {resource['description']}")
        st.markdown(f"**Difficulty:** {resource['difficulty']}")