import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Data Preprocessing", page_icon="🧹", layout="wide")

st.title("🧹 Data Preprocessing for LLM Training")
st.markdown("### Transforming Raw Data into Training-Ready Format")

# Overview
st.header("🎯 Overview")
st.markdown("""
Data preprocessing is where raw collected data is transformed into a clean, consistent format suitable for training. 
This critical stage involves cleaning, filtering, tokenization, and formatting to ensure optimal model performance 
and training efficiency.
""")

# Preprocessing pipeline visualization
st.header("🔄 Preprocessing Pipeline")

# Create pipeline flowchart
pipeline_steps = [
    "Raw Data", "Language Detection", "Quality Filtering", "Deduplication", 
    "Text Normalization", "Tokenization", "Sequence Formatting", "Final Dataset"
]

fig = go.Figure()

# Add boxes for each step
for i, step in enumerate(pipeline_steps):
    fig.add_shape(
        type="rect",
        x0=i*1.2, y0=0, x1=i*1.2+1, y1=0.8,
        fillcolor="lightblue" if i % 2 == 0 else "lightgreen",
        line=dict(color="black", width=1)
    )
    
    fig.add_annotation(
        x=i*1.2+0.5, y=0.4,
        text=step,
        showarrow=False,
        font=dict(size=10)
    )
    
    # Add arrows
    if i < len(pipeline_steps) - 1:
        fig.add_annotation(
            x=i*1.2+1.1, y=0.4,
            text="→",
            showarrow=False,
            font=dict(size=16)
        )

fig.update_layout(
    title="Data Preprocessing Pipeline Flow",
    showlegend=False,
    xaxis=dict(showgrid=False, showticklabels=False, range=[-0.5, len(pipeline_steps)*1.2]),
    yaxis=dict(showgrid=False, showticklabels=False, range=[-0.2, 1]),
    height=200
)

st.plotly_chart(fig, use_container_width=True)

# Main preprocessing steps
st.header("📋 Key Preprocessing Steps")

tabs = st.tabs([
    "🔍 Quality Filtering", 
    "🧹 Text Cleaning", 
    "🔤 Tokenization", 
    "📏 Sequence Formatting",
    "⚡ Performance Optimization"
])

with tabs[0]:
    st.subheader("🔍 Quality Filtering")
    st.markdown("Remove low-quality content that could harm model training.")
    
    filtering_methods = [
        {
            "name": "Length-based Filtering",
            "description": "Remove documents that are too short or too long",
            "example": "Min: 50 tokens, Max: 100,000 tokens",
            "code": """
def filter_by_length(text, min_tokens=50, max_tokens=100000):
    tokens = text.split()
    return min_tokens <= len(tokens) <= max_tokens

# Apply filtering
filtered_docs = [doc for doc in documents if filter_by_length(doc)]
"""
        },
        {
            "name": "Language Purity",
            "description": "Ensure documents are primarily in the target language",
            "example": "Minimum 90% target language confidence",
            "code": """
from langdetect import detect, LangDetectError

def filter_by_language(text, target_lang='en', confidence=0.9):
    try:
        detected_lang = detect(text)
        return detected_lang == target_lang
    except LangDetectError:
        return False

# Filter non-English documents
english_docs = [doc for doc in documents if filter_by_language(doc)]
"""
        },
        {
            "name": "Content Quality Scoring",
            "description": "Score documents based on linguistic quality metrics",
            "example": "Grammar, vocabulary diversity, coherence",
            "code": """
import re
from collections import Counter

def quality_score(text):
    # Basic quality metrics
    words = text.split()
    unique_words = len(set(words))
    total_words = len(words)
    
    # Vocabulary diversity
    diversity = unique_words / total_words if total_words > 0 else 0
    
    # Grammar indicators (simplified)
    sentences = len(re.split(r'[.!?]+', text))
    avg_sentence_length = total_words / sentences if sentences > 0 else 0
    
    # Combine metrics
    score = (diversity * 0.4) + (min(avg_sentence_length/20, 1) * 0.6)
    return score

# Filter by quality threshold
quality_docs = [doc for doc in documents if quality_score(doc) > 0.3]
"""
        }
    ]
    
    for method in filtering_methods:
        with st.expander(f"🎯 {method['name']}"):
            st.markdown(method['description'])
            st.markdown(f"**Example:** {method['example']}")
            st.code(method['code'], language='python')

with tabs[1]:
    st.subheader("🧹 Text Cleaning")
    st.markdown("Standardize and clean text content for consistent processing.")
    
    cleaning_operations = [
        ("Unicode Normalization", "Standardize character encoding", "NFD normalization"),
        ("HTML/XML Removal", "Strip markup and formatting tags", "BeautifulSoup parsing"),
        ("Special Character Handling", "Normalize punctuation and symbols", "Regex replacement"),
        ("Whitespace Normalization", "Standardize spacing and line breaks", "Regex normalization"),
        ("Case Normalization", "Optional lowercase conversion", "Context-dependent"),
        ("Encoding Fixes", "Handle character encoding issues", "UTF-8 enforcement")
    ]
    
    for operation, description, method in cleaning_operations:
        st.markdown(f"**{operation}**: {description} _{method}_")
    
    # Interactive cleaning example
    st.markdown("### 🧪 Interactive Text Cleaning Demo")
    
    sample_dirty_text = st.text_area(
        "Input text to clean:",
        value="<p>This is    SOME text with   &nbsp; HTML   tags!</p>\n\n  Extra   whitespace..."
    )
    
    if st.button("Clean Text"):
        import re
        
        # Simulate cleaning steps
        cleaned = sample_dirty_text
        
        # Remove HTML tags
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        
        # Handle HTML entities
        cleaned = cleaned.replace('&nbsp;', ' ').replace('&amp;', '&')
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        # Show before/after
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before:**")
            st.code(sample_dirty_text)
        with col2:
            st.markdown("**After:**")
            st.code(cleaned)

with tabs[2]:
    st.subheader("🔤 Tokenization")
    st.markdown("Convert text into tokens that the model can understand.")
    
    tokenization_tabs = st.tabs(["📚 Tokenization Methods", "🔧 Implementation", "📊 Comparison"])
    
    with tokenization_tabs[0]:
        tokenizers = [
            {
                "name": "Byte-Pair Encoding (BPE)",
                "description": "Merges frequently occurring byte pairs",
                "pros": ["Handles OOV words", "Compact vocabulary", "Language agnostic"],
                "cons": ["May split common words", "Training required"],
                "use_case": "GPT family, many modern LLMs"
            },
            {
                "name": "WordPiece",
                "description": "Merges characters/subwords based on likelihood",
                "pros": ["Good for morphologically rich languages", "Handles rare words"],
                "cons": ["More complex training", "Language-specific tuning"],
                "use_case": "BERT, T5"
            },
            {
                "name": "SentencePiece",
                "description": "Unigram language model-based segmentation", 
                "pros": ["No pre-tokenization needed", "Handles any text", "Unicode aware"],
                "cons": ["Computationally intensive", "Large model size"],
                "use_case": "T5, mT5, XLNet"
            }
        ]
        
        for tokenizer in tokenizers:
            with st.expander(f"🔤 {tokenizer['name']}"):
                st.markdown(tokenizer['description'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Advantages:**")
                    for pro in tokenizer['pros']:
                        st.markdown(f"• {pro}")
                with col2:
                    st.markdown("**❌ Challenges:**")
                    for con in tokenizer['cons']:
                        st.markdown(f"• {con}")
                
                st.markdown(f"**🎯 Common in:** {tokenizer['use_case']}")
    
    with tokenization_tabs[1]:
        st.markdown("### 🛠️ Tokenizer Training Example")
        
        st.code("""
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# Initialize BPE tokenizer
tokenizer = Tokenizer(models.BPE())

# Set pre-tokenizer (splits on whitespace and punctuation)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

# Set decoder
tokenizer.decoder = decoders.ByteLevel()

# Initialize trainer
trainer = trainers.BpeTrainer(
    vocab_size=50000,
    min_frequency=2,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
)

# Train on your corpus
files = ["corpus.txt"]
tokenizer.train(files, trainer)

# Save tokenizer
tokenizer.save("tokenizer.json")

# Use tokenizer
encoded = tokenizer.encode("Hello, world!")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
""", language='python')
    
    with tokenization_tabs[2]:
        st.markdown("### 📊 Tokenizer Comparison")
        
        # Create comparison chart
        comparison_data = {
            'Method': ['BPE', 'WordPiece', 'SentencePiece', 'Word-level'],
            'Vocabulary Efficiency': [8, 7, 9, 5],
            'OOV Handling': [9, 8, 9, 3],
            'Training Speed': [8, 6, 5, 10],
            'Inference Speed': [9, 8, 7, 10]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create radar chart using Scatterpolar
        fig = go.Figure()
        
        methods = df_comparison['Method'].tolist()
        metrics = ['Vocabulary Efficiency', 'OOV Handling', 'Training Speed', 'Inference Speed']
        
        for metric in metrics:
            fig.add_trace(go.Scatterpolar(
                r=df_comparison[metric].tolist(),
                theta=methods,
                fill='toself',
                name=metric,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            title="Tokenizer Performance Comparison",
            showlegend=True,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("📏 Sequence Formatting")
    st.markdown("Format tokenized text into training sequences with proper structure.")
    
    formatting_aspects = [
        {
            "aspect": "Sequence Length",
            "description": "Determine optimal sequence length for training",
            "considerations": [
                "Model architecture constraints (e.g., 2048 for GPT-2)",
                "Memory limitations during training",
                "Task-specific requirements",
                "Computational efficiency trade-offs"
            ],
            "example": "Most models use 512, 1024, 2048, or 4096 tokens"
        },
        {
            "aspect": "Special Tokens",
            "description": "Add tokens for sequence structure and control",
            "considerations": [
                "Beginning of sequence: <BOS> or <s>",
                "End of sequence: <EOS> or </s>",
                "Padding: <PAD>",
                "Unknown words: <UNK>",
                "Task-specific tokens: <MASK>, <SEP>"
            ],
            "example": "<s> This is a sentence. </s> <PAD> <PAD>"
        },
        {
            "aspect": "Attention Masks",
            "description": "Indicate which tokens should be attended to",
            "considerations": [
                "Mask padding tokens (value: 0)",
                "Attend to content tokens (value: 1)",
                "Causal masking for autoregressive models",
                "Custom masking for specific tasks"
            ],
            "example": "[1, 1, 1, 1, 1, 0, 0] for 5 content + 2 padding tokens"
        }
    ]
    
    for aspect in formatting_aspects:
        with st.expander(f"📐 {aspect['aspect']}"):
            st.markdown(aspect['description'])
            st.markdown("**Key Considerations:**")
            for consideration in aspect['considerations']:
                st.markdown(f"• {consideration}")
            st.markdown(f"**Example:** {aspect['example']}")
    
    # Sequence formatting code example
    st.markdown("### 💻 Sequence Formatting Implementation")
    
    st.code("""
def format_sequences(texts, tokenizer, max_length=512):
    formatted_sequences = []
    
    for text in texts:
        # Tokenize
        tokens = tokenizer.encode(text)
        
        # Add special tokens
        tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
        
        # Truncate or pad
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens.extend([tokenizer.pad_token_id] * (max_length - len(tokens)))
        
        # Create attention mask
        attention_mask = [1 if token != tokenizer.pad_token_id else 0 for token in tokens]
        
        formatted_sequences.append({
            'input_ids': tokens,
            'attention_mask': attention_mask
        })
    
    return formatted_sequences

# Example usage
sequences = format_sequences(corpus_texts, tokenizer, max_length=1024)
""", language='python')

with tabs[4]:
    st.subheader("⚡ Performance Optimization")
    st.markdown("Optimize preprocessing for large-scale data processing.")
    
    optimization_strategies = [
        {
            "strategy": "Parallel Processing",
            "description": "Use multiple CPU cores for preprocessing",
            "implementation": "multiprocessing.Pool or concurrent.futures",
            "benefit": "2-8x speedup depending on CPU cores"
        },
        {
            "strategy": "Batch Processing",
            "description": "Process data in batches rather than one-by-one",
            "implementation": "Process chunks of 1000-10000 documents",
            "benefit": "Reduced I/O overhead and better memory usage"
        },
        {
            "strategy": "Streaming Processing",
            "description": "Process data without loading everything into memory",
            "implementation": "Generator functions and itertools",
            "benefit": "Handle datasets larger than available RAM"
        },
        {
            "strategy": "Caching",
            "description": "Cache intermediate results to avoid recomputation",
            "implementation": "Redis, filesystem cache, or memory cache",
            "benefit": "Faster iteration during development"
        }
    ]
    
    for strategy in optimization_strategies:
        with st.expander(f"🚀 {strategy['strategy']}"):
            st.markdown(strategy['description'])
            st.markdown(f"**Implementation:** {strategy['implementation']}")
            st.markdown(f"**Benefit:** {strategy['benefit']}")
    
    # Performance optimization example
    st.markdown("### 🔧 Parallel Processing Example")
    
    st.code("""
import multiprocessing as mp
from functools import partial

def process_batch(batch, tokenizer, max_length):
    \"\"\"Process a batch of texts\"\"\"
    results = []
    for text in batch:
        # Apply all preprocessing steps
        cleaned = clean_text(text)
        if quality_score(cleaned) > threshold:
            tokens = tokenizer.encode(cleaned)
            formatted = format_sequence(tokens, max_length)
            results.append(formatted)
    return results

def parallel_preprocess(texts, tokenizer, max_length=512, batch_size=1000, n_workers=4):
    \"\"\"Preprocess texts in parallel\"\"\"
    
    # Split into batches
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    # Create partial function with fixed parameters
    process_func = partial(process_batch, tokenizer=tokenizer, max_length=max_length)
    
    # Process in parallel
    with mp.Pool(n_workers) as pool:
        results = pool.map(process_func, batches)
    
    # Flatten results
    return [item for batch_result in results for item in batch_result]

# Usage
processed_data = parallel_preprocess(raw_texts, tokenizer, n_workers=8)
""", language='python')

# Quality assurance
st.header("🎯 Quality Assurance")

qa_tabs = st.tabs(["📊 Metrics", "🔍 Validation", "🐛 Common Issues"])

with qa_tabs[0]:
    st.subheader("📊 Key Quality Metrics")
    
    # Sample metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Processing Rate", "50K docs/hour", "+5K")
        st.metric("Quality Score", "0.85", "+0.02")
    
    with col2:
        st.metric("Dedup Rate", "12%", "-2%")
        st.metric("Token Efficiency", "94%", "+1%")
    
    with col3:
        st.metric("Language Purity", "99.2%", "+0.3%")
        st.metric("Avg Sequence Length", "487 tokens", "+12")
    
    with col4:
        st.metric("Memory Usage", "8.2 GB", "-0.5 GB")
        st.metric("Error Rate", "0.1%", "-0.05%")

with qa_tabs[1]:
    st.subheader("🔍 Validation Checks")
    
    validation_checks = [
        "✅ Token distribution analysis",
        "✅ Vocabulary coverage verification", 
        "✅ Sequence length distribution check",
        "✅ Special token usage validation",
        "✅ Encoding/decoding round-trip test",
        "✅ Sample quality manual review",
        "✅ Performance benchmark comparison",
        "✅ Memory usage profiling"
    ]
    
    for check in validation_checks:
        st.markdown(check)

with qa_tabs[2]:
    st.subheader("🐛 Common Issues and Solutions")
    
    issues = [
        {
            "issue": "Memory Overflow",
            "symptoms": "Process killed, out of memory errors",
            "solutions": [
                "Reduce batch size",
                "Use streaming processing",
                "Add memory monitoring",
                "Process in smaller chunks"
            ]
        },
        {
            "issue": "Slow Processing Speed",
            "symptoms": "Low throughput, long processing times",
            "solutions": [
                "Enable parallel processing",
                "Optimize regex patterns",
                "Use compiled tokenizers",
                "Profile and optimize bottlenecks"
            ]
        },
        {
            "issue": "Inconsistent Quality",
            "symptoms": "Variable output quality, unexpected tokens",
            "solutions": [
                "Standardize cleaning pipeline",
                "Add validation checks",
                "Review filtering thresholds",
                "Implement quality scoring"
            ]
        }
    ]
    
    for issue in issues:
        st.error(f"**{issue['issue']}**: {issue['symptoms']}")
        st.info("**Solutions:**")
        for solution in issue['solutions']:
            st.markdown(f"• {solution}")

# Interactive tools
st.header("🛠️ Interactive Tools")

tool_tabs = st.tabs(["🧮 Tokenizer Tester", "📏 Sequence Analyzer", "🎯 Quality Scorer"])

with tool_tabs[0]:
    st.subheader("🧮 Tokenizer Testing Tool")
    
    test_text = st.text_area("Enter text to tokenize:", 
                            value="Hello, how are you doing today?")
    
    vocab_size = st.selectbox("Simulated Vocab Size", [1000, 5000, 10000, 30000, 50000])
    
    if st.button("Analyze Tokenization"):
        # Simulate tokenization (simplified)
        words = test_text.split()
        estimated_tokens = len(test_text) // 4  # Rough estimate
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Words", len(words))
        with col2:
            st.metric("Estimated Tokens", estimated_tokens)
        with col3:
            st.metric("Compression Ratio", f"{len(words)/estimated_tokens:.2f}")

with tool_tabs[1]:
    st.subheader("📏 Sequence Length Analyzer")
    
    max_length = st.slider("Max Sequence Length", 128, 4096, 512)
    sample_lengths = st.multiselect(
        "Sample document lengths (tokens)",
        [50, 100, 200, 500, 800, 1200, 2000, 3000],
        default=[200, 500, 800]
    )
    
    if sample_lengths:
        # Analyze truncation/padding
        truncated = sum(1 for length in sample_lengths if length > max_length)
        padded = sum(1 for length in sample_lengths if length < max_length)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", len(sample_lengths))
        with col2:
            st.metric("Truncated", truncated)
        with col3:
            st.metric("Padded", padded)

with tool_tabs[2]:
    st.subheader("🎯 Text Quality Scorer")
    
    quality_text = st.text_area("Enter text to score:", 
                               placeholder="Paste text to analyze quality...")
    
    if quality_text and st.button("Score Quality"):
        # Simple quality metrics
        words = quality_text.split()
        unique_words = len(set(words))
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Calculate scores (simplified)
        diversity_score = min(unique_words / len(words) if words else 0, 1.0) * 100
        readability_score = min(avg_word_length / 8, 1.0) * 100
        overall_score = (diversity_score + readability_score) / 2
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Diversity Score", f"{diversity_score:.1f}%")
        with col2:
            st.metric("Readability Score", f"{readability_score:.1f}%")
        with col3:
            st.metric("Overall Quality", f"{overall_score:.1f}%")
