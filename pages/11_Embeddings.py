import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Embeddings", page_icon="🔗", layout="wide")

st.title("🔗 Embeddings")
st.markdown("### Vector Representations of Text for Semantic Understanding")

# Overview
st.header("🎯 Overview")
st.markdown("""
Text embeddings are dense vector representations that capture the semantic meaning of words, 
sentences, or documents. They enable machines to understand and process text by converting 
it into numerical form while preserving meaningful relationships between different pieces of text.
""")

# Core concepts
st.header("🧠 Core Concepts")

concept_tabs = st.tabs([
    "📊 What are Embeddings?",
    "🏗️ Types & Architectures", 
    "🔍 Properties",
    "📈 Applications"
])

with concept_tabs[0]:
    st.subheader("📊 Understanding Embeddings")
    
    st.markdown("""
    Text embeddings transform discrete text into continuous vector spaces where semantically 
    similar content is positioned closer together. This enables mathematical operations on text 
    and efficient similarity computations.
    """)
    
    # Embedding visualization simulation
    st.markdown("### 🎯 Embedding Space Visualization")
    
    # Generate sample embedding data
    np.random.seed(42)
    
    # Create clusters for different semantic categories
    animals = np.random.multivariate_normal([2, 3], [[0.5, 0.1], [0.1, 0.5]], 8)
    food = np.random.multivariate_normal([-2, 1], [[0.6, 0.2], [0.2, 0.4]], 8)
    colors = np.random.multivariate_normal([1, -2], [[0.4, 0.1], [0.1, 0.6]], 8)
    
    animal_words = ['dog', 'cat', 'bird', 'fish', 'lion', 'elephant', 'rabbit', 'horse']
    food_words = ['apple', 'pizza', 'bread', 'rice', 'cake', 'soup', 'salad', 'pasta']
    color_words = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=animals[:, 0], y=animals[:, 1],
        mode='markers+text',
        text=animal_words,
        textposition="top center",
        name='Animals',
        marker=dict(size=10, color='blue'),
        textfont=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=food[:, 0], y=food[:, 1],
        mode='markers+text',
        text=food_words,
        textposition="top center", 
        name='Food',
        marker=dict(size=10, color='green'),
        textfont=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=colors[:, 0], y=colors[:, 1],
        mode='markers+text',
        text=color_words,
        textposition="top center",
        name='Colors',
        marker=dict(size=10, color='red'),
        textfont=dict(size=10)
    ))
    
    fig.update_layout(
        title="2D Visualization of Text Embeddings by Semantic Category",
        xaxis_title="Embedding Dimension 1",
        yaxis_title="Embedding Dimension 2",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Key Observations:**
    - Words with similar meanings cluster together in the embedding space
    - Distance between points represents semantic similarity
    - Embeddings enable mathematical operations like finding nearest neighbors
    - Different semantic categories occupy distinct regions
    """)
    
    # Properties of embeddings
    st.markdown("### 🔍 Key Properties")
    
    properties = [
        {
            "property": "Semantic Similarity",
            "description": "Similar words have similar embeddings",
            "example": "distance('cat', 'dog') < distance('cat', 'airplane')",
            "math": "cosine_similarity(embed('king'), embed('queen')) > 0.8"
        },
        {
            "property": "Compositionality", 
            "description": "Word embeddings can be combined arithmetically",
            "example": "king - man + woman ≈ queen",
            "math": "embed('king') - embed('man') + embed('woman') ≈ embed('queen')"
        },
        {
            "property": "Dimensionality",
            "description": "Embeddings are typically 50-4096 dimensional vectors",
            "example": "BERT: 768 dimensions, GPT-3: 12,288 dimensions",
            "math": "embed('word') ∈ ℝᵈ where d = embedding_dimension"
        },
        {
            "property": "Contextuality",
            "description": "Modern embeddings consider surrounding context",
            "example": "'bank' in 'river bank' vs 'money bank' have different embeddings",
            "math": "embed('bank'|context₁) ≠ embed('bank'|context₂)"
        }
    ]
    
    for prop in properties:
        with st.expander(f"🔍 {prop['property']}"):
            st.markdown(prop['description'])
            st.markdown(f"**Example:** {prop['example']}")
            st.code(prop['math'])

with concept_tabs[1]:
    st.subheader("🏗️ Types & Architectures")
    
    embedding_types = st.tabs([
        "📝 Word Embeddings",
        "📄 Sentence Embeddings", 
        "📚 Document Embeddings",
        "🎨 Multimodal Embeddings"
    ])
    
    with embedding_types[0]:
        st.markdown("### 📝 Word-Level Embeddings")
        
        word_models = [
            {
                "model": "Word2Vec",
                "year": "2013",
                "approach": "Skip-gram and CBOW neural networks",
                "strengths": [
                    "Fast training and inference",
                    "Good semantic relationships", 
                    "Widely supported",
                    "Interpretable vector arithmetic"
                ],
                "limitations": [
                    "Static embeddings (no context)",
                    "Out-of-vocabulary words",
                    "Ignores word order",
                    "Limited to single words"
                ],
                "dimensions": "50-300",
                "best_for": "Basic word similarity, lightweight applications"
            },
            {
                "model": "GloVe",
                "year": "2014", 
                "approach": "Global word-word co-occurrence statistics",
                "strengths": [
                    "Incorporates global statistics",
                    "Good performance on analogy tasks",
                    "Faster training than Word2Vec",
                    "Linear structure in vector space"
                ],
                "limitations": [
                    "Static embeddings",
                    "Requires large corpus",
                    "Memory intensive",
                    "No context awareness"
                ],
                "dimensions": "50-300",
                "best_for": "Word analogies, semantic similarity tasks"
            },
            {
                "model": "FastText",
                "year": "2016",
                "approach": "Subword information with character n-grams",
                "strengths": [
                    "Handles out-of-vocabulary words",
                    "Good for morphologically rich languages",
                    "Subword information",
                    "Fast training"
                ],
                "limitations": [
                    "Larger model size",
                    "May include noise from subwords",
                    "Still static embeddings",
                    "Sensitive to text preprocessing"
                ],
                "dimensions": "100-300",
                "best_for": "Multilingual tasks, rare words, morphology"
            }
        ]
        
        for model in word_models:
            with st.expander(f"📝 {model['model']} ({model['year']})"):
                st.markdown(f"**Approach:** {model['approach']}")
                st.markdown(f"**Dimensions:** {model['dimensions']}")
                st.markdown(f"**Best for:** {model['best_for']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Strengths:**")
                    for strength in model['strengths']:
                        st.markdown(f"• {strength}")
                with col2:
                    st.markdown("**❌ Limitations:**")
                    for limitation in model['limitations']:
                        st.markdown(f"• {limitation}")
    
    with embedding_types[1]:
        st.markdown("### 📄 Sentence-Level Embeddings")
        
        sentence_models = [
            {
                "model": "Universal Sentence Encoder",
                "year": "2018",
                "approach": "Transformer and Deep Averaging Network",
                "strengths": [
                    "High-quality sentence representations",
                    "Pre-trained on diverse data",
                    "Good transfer learning",
                    "Multiple architecture options"
                ],
                "limitations": [
                    "Fixed context length",
                    "Computational overhead",
                    "Limited domain adaptation",
                    "Static per-sentence representation"
                ],
                "dimensions": "512",
                "best_for": "Sentence similarity, text classification"
            },
            {
                "model": "Sentence-BERT (SBERT)",
                "year": "2019",
                "approach": "Siamese BERT networks with pooling",
                "strengths": [
                    "BERT-quality sentence embeddings",
                    "Efficient similarity computation",
                    "Good clustering performance",
                    "Fine-tunable"
                ],
                "limitations": [
                    "Requires pre-trained BERT",
                    "Large model size",
                    "May lose some context",
                    "Training complexity"
                ],
                "dimensions": "384-1024",
                "best_for": "Semantic search, clustering, retrieval"
            },
            {
                "model": "InstructOR",
                "year": "2022",
                "approach": "Instruction-tuned text embeddings",
                "strengths": [
                    "Task-specific embeddings",
                    "Follows natural language instructions",
                    "Versatile across domains",
                    "High performance"
                ],
                "limitations": [
                    "Requires instruction design",
                    "Computational overhead",
                    "Limited to instruction format",
                    "Newer with less adoption"
                ],
                "dimensions": "768",
                "best_for": "Multi-task embeddings, instruction-following"
            }
        ]
        
        for model in sentence_models:
            with st.expander(f"📄 {model['model']} ({model['year']})"):
                st.markdown(f"**Approach:** {model['approach']}")
                st.markdown(f"**Dimensions:** {model['dimensions']}")
                st.markdown(f"**Best for:** {model['best_for']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Strengths:**")
                    for strength in model['strengths']:
                        st.markdown(f"• {strength}")
                with col2:
                    st.markdown("**❌ Limitations:**")
                    for limitation in model['limitations']:
                        st.markdown(f"• {limitation}")
    
    with embedding_types[2]:
        st.markdown("### 📚 Document-Level Embeddings")
        
        document_models = [
            {
                "model": "Doc2Vec",
                "year": "2014",
                "approach": "Extension of Word2Vec with document vectors",
                "use_cases": ["Document classification", "Document similarity", "Content recommendation"],
                "pros": ["Captures document-level semantics", "Variable document length", "Unsupervised learning"],
                "cons": ["Requires retraining for new documents", "Limited context window", "Parameter tuning sensitive"]
            },
            {
                "model": "BERT CLS Token",
                "year": "2018",
                "approach": "Use BERT's [CLS] token as document representation",
                "use_cases": ["Document classification", "Sentiment analysis", "Text similarity"],
                "pros": ["Pre-trained representations", "Contextual understanding", "Transfer learning"],
                "cons": ["Fixed sequence length", "Computational intensive", "May not capture full document"]
            },
            {
                "model": "Longformer",
                "year": "2020",
                "approach": "Attention mechanism for long documents",
                "use_cases": ["Long document processing", "Scientific papers", "Legal documents"],
                "pros": ["Handles long sequences", "Efficient attention", "Captures long-range dependencies"],
                "cons": ["Still memory intensive", "Complex implementation", "Limited pre-trained models"]
            }
        ]
        
        for model in document_models:
            with st.expander(f"📚 {model['model']} ({model['year']})"):
                st.markdown(f"**Approach:** {model['approach']}")
                
                st.markdown("**Use Cases:**")
                for use_case in model['use_cases']:
                    st.markdown(f"• {use_case}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Pros:**")
                    for pro in model['pros']:
                        st.markdown(f"• {pro}")
                with col2:
                    st.markdown("**❌ Cons:**")
                    for con in model['cons']:
                        st.markdown(f"• {con}")
    
    with embedding_types[3]:
        st.markdown("### 🎨 Multimodal Embeddings")
        
        multimodal_models = [
            {
                "model": "CLIP",
                "year": "2021",
                "modalities": ["Text", "Images"],
                "approach": "Contrastive learning between text and image pairs",
                "applications": [
                    "Image search with text queries",
                    "Zero-shot image classification",
                    "Content moderation",
                    "Creative applications"
                ],
                "capabilities": [
                    "Joint text-image understanding",
                    "Cross-modal retrieval",
                    "Zero-shot transfer",
                    "Semantic alignment"
                ]
            },
            {
                "model": "DALL-E 2",
                "year": "2022",
                "modalities": ["Text", "Images"],
                "approach": "Diffusion models with text conditioning",
                "applications": [
                    "Text-to-image generation",
                    "Image editing with text",
                    "Style transfer",
                    "Creative content creation"
                ],
                "capabilities": [
                    "High-quality image generation",
                    "Text-guided editing",
                    "Style manipulation",
                    "Concept composition"
                ]
            },
            {
                "model": "AudioCLIP",
                "year": "2021",
                "modalities": ["Text", "Audio", "Images"],
                "approach": "Extends CLIP to include audio modality",
                "applications": [
                    "Audio-visual understanding",
                    "Cross-modal retrieval",
                    "Sound classification",
                    "Multimodal search"
                ],
                "capabilities": [
                    "Audio-text alignment",
                    "Cross-modal search",
                    "Sound understanding",
                    "Multimodal reasoning"
                ]
            }
        ]
        
        for model in multimodal_models:
            with st.expander(f"🎨 {model['model']} ({model['year']})"):
                st.markdown(f"**Modalities:** {', '.join(model['modalities'])}")
                st.markdown(f"**Approach:** {model['approach']}")
                
                st.markdown("**Applications:**")
                for app in model['applications']:
                    st.markdown(f"• {app}")
                
                st.markdown("**Key Capabilities:**")
                for cap in model['capabilities']:
                    st.markdown(f"• {cap}")

with concept_tabs[2]:
    st.subheader("🔍 Embedding Properties")
    
    property_tabs = st.tabs(["📐 Mathematical Properties", "🎯 Quality Metrics", "⚡ Computational Aspects"])
    
    with property_tabs[0]:
        st.markdown("### 📐 Mathematical Properties")
        
        math_properties = [
            {
                "property": "Vector Operations",
                "description": "Embeddings support mathematical operations that preserve semantic meaning",
                "operations": [
                    "Addition: embed('king') + embed('crown') ≈ embed('royal')",
                    "Subtraction: embed('king') - embed('man') + embed('woman') ≈ embed('queen')",
                    "Dot Product: similarity(A, B) = A · B / (|A| × |B|)",
                    "Distance: euclidean_distance(A, B) = √Σ(Aᵢ - Bᵢ)²"
                ]
            },
            {
                "property": "Similarity Measures",
                "description": "Different ways to compute similarity between embeddings",
                "operations": [
                    "Cosine Similarity: cos(θ) = A·B / (|A||B|) ∈ [-1, 1]",
                    "Euclidean Distance: ||A - B||₂ = √Σ(Aᵢ - Bᵢ)²",
                    "Manhattan Distance: ||A - B||₁ = Σ|Aᵢ - Bᵢ|",
                    "Dot Product: A·B = ΣAᵢBᵢ (for normalized vectors)"
                ]
            },
            {
                "property": "Dimensionality",
                "description": "The number of dimensions affects capacity and computational cost",
                "operations": [
                    "Higher dimensions: More expressive, captures finer distinctions",
                    "Lower dimensions: More efficient, easier to visualize",
                    "Typical ranges: 50-300 (Word2Vec), 768 (BERT), 1536 (OpenAI)",
                    "Trade-off: Expressiveness vs. computational efficiency"
                ]
            }
        ]
        
        for prop in math_properties:
            with st.expander(f"📐 {prop['property']}"):
                st.markdown(prop['description'])
                st.markdown("**Examples:**")
                for op in prop['operations']:
                    st.code(op)
        
        # Similarity comparison visualization
        st.markdown("### 📊 Similarity Measure Comparison")
        
        # Generate sample data for comparison
        A = np.array([1, 2, 3])
        B = np.array([2, 3, 4])
        C = np.array([-1, -2, -3])
        
        vectors = {'A': A, 'B': B, 'C': C}
        
        similarity_data = []
        for name1, vec1 in vectors.items():
            for name2, vec2 in vectors.items():
                if name1 != name2:
                    cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    euclidean_dist = np.linalg.norm(vec1 - vec2)
                    dot_product = np.dot(vec1, vec2)
                    
                    similarity_data.append({
                        'Vector Pair': f"{name1}-{name2}",
                        'Cosine Similarity': cosine_sim,
                        'Euclidean Distance': euclidean_dist,
                        'Dot Product': dot_product
                    })
        
        sim_df = pd.DataFrame(similarity_data)
        st.dataframe(sim_df, use_container_width=True)
    
    with property_tabs[1]:
        st.markdown("### 🎯 Quality Metrics")
        
        quality_metrics = [
            {
                "metric": "Intrinsic Evaluation",
                "description": "Evaluate embeddings on semantic tasks",
                "methods": [
                    "Word similarity: Compare with human similarity ratings",
                    "Analogy tasks: king - man + woman = queen",
                    "Clustering: Group semantically similar words",
                    "Odd-one-out: Identify the semantically different word"
                ],
                "pros": ["Direct evaluation", "Interpretable results", "Standard benchmarks"],
                "cons": ["May not reflect downstream performance", "Limited scope", "Human annotation bias"]
            },
            {
                "metric": "Extrinsic Evaluation", 
                "description": "Evaluate embeddings on downstream tasks",
                "methods": [
                    "Text classification: Use embeddings as features",
                    "Information retrieval: Search and ranking tasks",
                    "Named entity recognition: Sequence labeling",
                    "Machine translation: Translation quality"
                ],
                "pros": ["Task-relevant evaluation", "Practical significance", "End-to-end assessment"],
                "cons": ["Confounded by other factors", "Task-specific", "Expensive evaluation"]
            },
            {
                "metric": "Geometric Properties",
                "description": "Analyze the structure of embedding space",
                "methods": [
                    "Uniformity: How evenly distributed are embeddings",
                    "Alignment: How well aligned are similar concepts",
                    "Isotropy: Are all directions equally informative",
                    "Hubness: Do some points appear as neighbors too often"
                ],
                "pros": ["Theory-grounded", "Model-agnostic", "Reveals biases"],
                "cons": ["May not correlate with performance", "Complex interpretation", "Limited practical guidance"]
            }
        ]
        
        for metric in quality_metrics:
            with st.expander(f"🎯 {metric['metric']}"):
                st.markdown(metric['description'])
                
                st.markdown("**Methods:**")
                for method in metric['methods']:
                    st.markdown(f"• {method}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Pros:**")
                    for pro in metric['pros']:
                        st.markdown(f"• {pro}")
                with col2:
                    st.markdown("**❌ Cons:**")
                    for con in metric['cons']:
                        st.markdown(f"• {con}")
    
    with property_tabs[2]:
        st.markdown("### ⚡ Computational Aspects")
        
        # Performance comparison chart
        model_performance = pd.DataFrame({
            'Model': ['Word2Vec', 'GloVe', 'FastText', 'BERT', 'Sentence-BERT', 'OpenAI Ada'],
            'Training Time (hours)': [2, 4, 3, 48, 12, np.nan],
            'Inference Speed (texts/sec)': [10000, 8000, 9000, 100, 500, 1000],
            'Memory Usage (GB)': [0.5, 1.0, 1.5, 4.0, 2.0, np.nan],
            'Embedding Dim': [300, 300, 300, 768, 384, 1536]
        })
        
        # Training time chart
        fig1 = px.bar(
            model_performance.dropna(subset=['Training Time (hours)']),
            x='Model',
            y='Training Time (hours)',
            title="Training Time Comparison"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Inference speed chart  
        fig2 = px.bar(
            model_performance,
            x='Model',
            y='Inference Speed (texts/sec)',
            title="Inference Speed Comparison (log scale)",
            log_y=True
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("### 💡 Performance Optimization Tips")
        
        optimization_tips = [
            "**Dimension Reduction**: Use PCA or t-SNE for visualization and compression",
            "**Quantization**: Reduce precision to save memory (float32 → int8)",
            "**Caching**: Store frequently used embeddings in memory or fast storage",
            "**Batch Processing**: Process multiple texts together for efficiency",
            "**Model Selection**: Choose the right model for your speed/quality requirements",
            "**Hardware Acceleration**: Use GPUs for large-scale embedding computation",
            "**Approximate Search**: Use techniques like LSH for fast similarity search",
            "**Sparse Embeddings**: Use sparse representations when applicable"
        ]
        
        for tip in optimization_tips:
            st.markdown(f"• {tip}")

with concept_tabs[3]:
    st.subheader("📈 Applications")
    
    application_tabs = st.tabs([
        "🔍 Search & Retrieval",
        "📊 Classification & Clustering",
        "🤖 Recommendation Systems",
        "🧠 Advanced Applications"
    ])
    
    with application_tabs[0]:
        st.markdown("### 🔍 Search & Retrieval Applications")
        
        search_applications = [
            {
                "application": "Semantic Search",
                "description": "Find relevant documents based on meaning, not just keywords",
                "how_it_works": [
                    "Convert queries and documents to embeddings",
                    "Compute similarity between query and document embeddings",
                    "Rank documents by semantic similarity",
                    "Return most relevant results"
                ],
                "advantages": [
                    "Finds semantically related content",
                    "Handles synonyms and paraphrases",
                    "Works across languages",
                    "No need for exact keyword matches"
                ],
                "example": "Query: 'car problems' finds documents about 'vehicle issues', 'automotive troubles'"
            },
            {
                "application": "Question Answering",
                "description": "Retrieve relevant passages to answer user questions",
                "how_it_works": [
                    "Embed questions and candidate passages",
                    "Find passages most similar to question",
                    "Extract or generate answers from passages",
                    "Rank answers by relevance and confidence"
                ],
                "advantages": [
                    "Handles complex, natural language questions",
                    "Finds relevant context efficiently",
                    "Supports open-domain QA",
                    "Scales to large knowledge bases"
                ],
                "example": "Q: 'How do photosynthesis work?' → finds biology textbook passages about photosynthesis"
            },
            {
                "application": "Cross-Modal Retrieval",
                "description": "Search across different modalities (text, image, audio)",
                "how_it_works": [
                    "Use multimodal embeddings (e.g., CLIP)",
                    "Embed queries and content in shared space",
                    "Compute cross-modal similarities",
                    "Return relevant content regardless of modality"
                ],
                "advantages": [
                    "Unified search across media types",
                    "Rich query capabilities",
                    "Enables creative applications",
                    "Breaks down modality silos"
                ],
                "example": "Text query 'sunset over mountains' finds relevant images and videos"
            }
        ]
        
        for app in search_applications:
            with st.expander(f"🔍 {app['application']}"):
                st.markdown(app['description'])
                
                st.markdown("**How it works:**")
                for step in app['how_it_works']:
                    st.markdown(f"• {step}")
                
                st.markdown("**Advantages:**")
                for advantage in app['advantages']:
                    st.markdown(f"• {advantage}")
                
                st.info(f"**Example:** {app['example']}")
    
    with application_tabs[1]:
        st.markdown("### 📊 Classification & Clustering")
        
        # Classification performance simulation
        st.markdown("#### 🎯 Text Classification Performance")
        
        classification_data = pd.DataFrame({
            'Embedding Method': ['TF-IDF', 'Word2Vec', 'GloVe', 'BERT', 'Sentence-BERT'],
            'Accuracy (%)': [78, 82, 83, 89, 87],
            'Training Time (min)': [5, 30, 25, 120, 60],
            'Inference Speed (docs/sec)': [1000, 500, 600, 50, 200]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Accuracy (%)',
            x=classification_data['Embedding Method'],
            y=classification_data['Accuracy (%)'],
            yaxis='y'
        ))
        
        fig.update_layout(
            title='Text Classification Performance by Embedding Method',
            xaxis_title='Embedding Method',
            yaxis_title='Accuracy (%)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Clustering applications
        st.markdown("#### 🔗 Clustering Applications")
        
        clustering_apps = [
            {
                "application": "Customer Segmentation",
                "description": "Group customers by behavior and preferences",
                "embedding_input": "Customer reviews, support tickets, purchase history",
                "clustering_method": "K-means, hierarchical clustering",
                "output": "Customer segments for targeted marketing"
            },
            {
                "application": "Content Organization",
                "description": "Automatically organize documents and articles",
                "embedding_input": "Article text, document content, metadata",
                "clustering_method": "DBSCAN, spectral clustering",
                "output": "Topic-based content categories"
            },
            {
                "application": "Research Paper Clustering",
                "description": "Group academic papers by research area",
                "embedding_input": "Paper abstracts, titles, keywords",
                "clustering_method": "Hierarchical clustering, topic modeling",
                "output": "Research area clusters and hierarchies"
            }
        ]
        
        for app in clustering_apps:
            with st.expander(f"🔗 {app['application']}"):
                st.markdown(app['description'])
                st.markdown(f"**Input:** {app['embedding_input']}")
                st.markdown(f"**Method:** {app['clustering_method']}")
                st.markdown(f"**Output:** {app['output']}")
    
    with application_tabs[2]:
        st.markdown("### 🤖 Recommendation Systems")
        
        recommendation_types = [
            {
                "type": "Content-Based Filtering",
                "description": "Recommend items similar to user's past preferences",
                "embedding_use": [
                    "Embed item descriptions and user profiles",
                    "Find items similar to user's liked items",
                    "Rank recommendations by similarity",
                    "Update recommendations based on feedback"
                ],
                "pros": ["No cold start problem", "Explainable recommendations", "Works with sparse data"],
                "cons": ["Limited diversity", "Requires rich content", "May over-specialize"],
                "example": "Movie recommendation based on genre, actors, plot similarity"
            },
            {
                "type": "Collaborative Filtering",
                "description": "Recommend based on similar users' preferences",
                "embedding_use": [
                    "Embed users and items in shared space",
                    "Learn embeddings from interaction data",
                    "Find similar users or items",
                    "Predict preferences for new items"
                ],
                "pros": ["Discovers hidden patterns", "High accuracy", "Handles complex preferences"],
                "cons": ["Cold start problem", "Sparsity issues", "Less explainable"],
                "example": "Netflix recommendations based on viewing patterns of similar users"
            },
            {
                "type": "Hybrid Approaches",
                "description": "Combine multiple recommendation strategies",
                "embedding_use": [
                    "Use both content and collaborative embeddings",
                    "Ensemble different embedding models",
                    "Multi-task learning with shared embeddings",
                    "Context-aware embedding selection"
                ],
                "pros": ["Best of both worlds", "Robust performance", "Flexible adaptation"],
                "cons": ["Increased complexity", "More parameters", "Harder to tune"],
                "example": "Amazon recommendations using product features + user behavior + social signals"
            }
        ]
        
        for rec_type in recommendation_types:
            with st.expander(f"🤖 {rec_type['type']}"):
                st.markdown(rec_type['description'])
                
                st.markdown("**How embeddings are used:**")
                for use in rec_type['embedding_use']:
                    st.markdown(f"• {use}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Pros:**")
                    for pro in rec_type['pros']:
                        st.markdown(f"• {pro}")
                with col2:
                    st.markdown("**❌ Cons:**")
                    for con in rec_type['cons']:
                        st.markdown(f"• {con}")
                
                st.info(f"**Example:** {rec_type['example']}")
    
    with application_tabs[3]:
        st.markdown("### 🧠 Advanced Applications")
        
        advanced_apps = [
            {
                "application": "Zero-Shot Learning",
                "description": "Classify into categories not seen during training",
                "technique": "Use semantic embeddings to bridge between seen and unseen classes",
                "benefits": ["No training data needed for new classes", "Rapid deployment", "Scalable classification"],
                "challenges": ["Performance gap vs supervised", "Relies on semantic descriptions", "Domain transfer issues"]
            },
            {
                "application": "Few-Shot Learning",
                "description": "Learn new tasks with minimal examples",
                "technique": "Use pre-trained embeddings with similarity-based matching",
                "benefits": ["Fast adaptation", "Efficient use of data", "Transfer learning"],
                "challenges": ["Example selection matters", "May not generalize well", "Limited complexity"]
            },
            {
                "application": "Anomaly Detection",
                "description": "Identify unusual or suspicious content",
                "technique": "Detect outliers in embedding space using distance or density",
                "benefits": ["Unsupervised detection", "Adapts to normal patterns", "Scalable monitoring"],
                "challenges": ["Defining normal vs anomalous", "High-dimensional spaces", "False positive management"]
            },
            {
                "application": "Knowledge Graph Embeddings",
                "description": "Represent entities and relations in vector space",
                "technique": "Learn embeddings that preserve graph structure and semantics",
                "benefits": ["Efficient knowledge representation", "Enables reasoning", "Link prediction"],
                "challenges": ["Complex optimization", "Scalability issues", "Incomplete knowledge"]
            },
            {
                "application": "Style Transfer",
                "description": "Transfer style while preserving content",
                "technique": "Separate content and style embeddings for independent manipulation",
                "benefits": ["Creative applications", "Controllable generation", "Cross-domain transfer"],
                "challenges": ["Content-style disentanglement", "Quality control", "Evaluation metrics"]
            }
        ]
        
        for app in advanced_apps:
            with st.expander(f"🧠 {app['application']}"):
                st.markdown(app['description'])
                st.markdown(f"**Technique:** {app['technique']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Benefits:**")
                    for benefit in app['benefits']:
                        st.markdown(f"• {benefit}")
                with col2:
                    st.markdown("**⚠️ Challenges:**")
                    for challenge in app['challenges']:
                        st.markdown(f"• {challenge}")

# Implementation guide
st.header("🛠️ Implementation Guide")

implementation_tabs = st.tabs([
    "🚀 Quick Start",
    "🔧 Advanced Techniques", 
    "📊 Evaluation & Optimization",
    "🏭 Production Deployment"
])

with implementation_tabs[0]:
    st.subheader("🚀 Quick Start Guide")
    
    quick_start_steps = [
        {
            "step": "1. Choose Embedding Model",
            "description": "Select appropriate embedding model for your use case",
            "code": """
# Option 1: Use pre-trained embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Option 2: Use OpenAI embeddings
import openai
openai.api_key = 'your-api-key'

# Option 3: Use local word embeddings
import gensim
model = gensim.models.Word2Vec.load('word2vec.model')
""",
            "considerations": [
                "Task requirements (word/sentence/document level)",
                "Performance requirements (speed vs quality)",
                "Resource constraints (memory, compute)",
                "Domain specificity (general vs specialized)"
            ]
        },
        {
            "step": "2. Generate Embeddings",
            "description": "Convert your text data into vector representations",
            "code": """
# Sentence-level embeddings
texts = ["Hello world", "How are you?", "Nice to meet you"]
embeddings = model.encode(texts)
print(f"Shape: {embeddings.shape}")  # (3, 384)

# OpenAI embeddings
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']

embedding = get_embedding("Your text here")
""",
            "considerations": [
                "Batch processing for efficiency",
                "Handle text preprocessing (tokenization, cleaning)",
                "Memory management for large datasets",
                "Error handling and retries"
            ]
        },
        {
            "step": "3. Compute Similarities",
            "description": "Find similar texts using vector operations",
            "code": """
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Compute pairwise similarities
similarities = cosine_similarity(embeddings)
print(similarities)

# Find most similar text to query
def find_similar(query, texts, embeddings, top_k=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top-k most similar
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'text': texts[idx],
            'similarity': similarities[idx]
        })
    return results

results = find_similar("greeting", texts, embeddings)
""",
            "considerations": [
                "Choose appropriate similarity metric",
                "Normalize embeddings if needed",
                "Consider computational efficiency",
                "Handle edge cases (empty results, ties)"
            ]
        },
        {
            "step": "4. Store and Index",
            "description": "Efficiently store and search large embedding collections",
            "code": """
# Option 1: Simple storage with pickle
import pickle
with open('embeddings.pkl', 'wb') as f:
    pickle.dump({'texts': texts, 'embeddings': embeddings}, f)

# Option 2: Use vector database (Pinecone example)
import pinecone
pinecone.init(api_key="your-key", environment="us-west1-gcp")

index = pinecone.Index("my-index")
vectors = [(f"id_{i}", embedding.tolist()) for i, embedding in enumerate(embeddings)]
index.upsert(vectors)

# Option 3: Use FAISS for fast similarity search
import faiss
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product
index.add(embeddings.astype('float32'))

# Search
scores, indices = index.search(query_embedding.astype('float32'), k=5)
""",
            "considerations": [
                "Scalability requirements",
                "Query performance needs",
                "Data persistence requirements",
                "Cost vs performance trade-offs"
            ]
        }
    ]
    
    for step_info in quick_start_steps:
        with st.expander(f"📋 {step_info['step']}"):
            st.markdown(step_info['description'])
            st.code(step_info['code'], language='python')
            
            st.markdown("**Key Considerations:**")
            for consideration in step_info['considerations']:
                st.markdown(f"• {consideration}")

with implementation_tabs[1]:
    st.subheader("🔧 Advanced Techniques")
    
    advanced_techniques = [
        {
            "technique": "Fine-tuning Embeddings",
            "description": "Adapt pre-trained embeddings to your specific domain",
            "approaches": [
                "Contrastive learning with domain-specific pairs",
                "Metric learning with triplet loss",
                "Multi-task learning with auxiliary tasks",
                "Domain adaptation techniques"
            ],
            "code": """
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create training examples (positive pairs)
train_examples = [
    InputExample(texts=['The weather is nice', 'Beautiful sunny day'], label=1.0),
    InputExample(texts=['I love pizza', 'Pizza is delicious'], label=1.0),
    InputExample(texts=['Car accident', 'Beautiful flower'], label=0.0)
]

# Create data loader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define loss function
train_loss = losses.CosineSimilarityLoss(model)

# Fine-tune
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100
)
"""
        },
        {
            "technique": "Embedding Fusion",
            "description": "Combine multiple embedding sources for better representations",
            "approaches": [
                "Concatenation of different embeddings",
                "Weighted averaging with learned weights",
                "Attention-based fusion mechanisms",
                "Hierarchical combination strategies"
            ],
            "code": """
import numpy as np
from sklearn.decomposition import PCA

def fuse_embeddings(embeddings_list, method='concat'):
    if method == 'concat':
        return np.concatenate(embeddings_list, axis=1)
    
    elif method == 'average':
        return np.mean(embeddings_list, axis=0)
    
    elif method == 'weighted_average':
        weights = [0.6, 0.3, 0.1]  # Example weights
        weighted = [w * emb for w, emb in zip(weights, embeddings_list)]
        return np.sum(weighted, axis=0)
    
    elif method == 'pca_fusion':
        # Reduce dimensionality after concatenation
        fused = np.concatenate(embeddings_list, axis=1)
        pca = PCA(n_components=384)
        return pca.fit_transform(fused)

# Example usage
bert_embeddings = model1.encode(texts)
sentence_bert_embeddings = model2.encode(texts)
word2vec_embeddings = get_word2vec_embeddings(texts)

fused = fuse_embeddings([
    bert_embeddings, 
    sentence_bert_embeddings, 
    word2vec_embeddings
], method='weighted_average')
"""
        },
        {
            "technique": "Hierarchical Embeddings",
            "description": "Create multi-level representations for complex documents",
            "approaches": [
                "Word → Sentence → Document hierarchy",
                "Attention pooling at each level",
                "Graph-based document representation",
                "Multi-scale embedding learning"
            ],
            "code": """
def create_hierarchical_embeddings(document, word_model, sentence_model):
    sentences = document.split('.')
    
    # Word-level embeddings
    word_embeddings = []
    for sentence in sentences:
        words = sentence.split()
        word_vecs = [word_model[word] for word in words if word in word_model]
        if word_vecs:
            # Average word embeddings for sentence
            sentence_from_words = np.mean(word_vecs, axis=0)
            word_embeddings.append(sentence_from_words)
    
    # Sentence-level embeddings
    sentence_embeddings = sentence_model.encode(sentences)
    
    # Document-level embedding (attention-weighted average)
    def attention_pooling(embeddings, attention_weights=None):
        if attention_weights is None:
            # Simple average
            return np.mean(embeddings, axis=0)
        else:
            # Weighted average
            return np.average(embeddings, axis=0, weights=attention_weights)
    
    # Combine hierarchical information
    doc_embedding = {
        'word_level': attention_pooling(word_embeddings) if word_embeddings else None,
        'sentence_level': attention_pooling(sentence_embeddings),
        'combined': np.concatenate([
            attention_pooling(word_embeddings) if word_embeddings else np.zeros(300),
            attention_pooling(sentence_embeddings)
        ])
    }
    
    return doc_embedding
"""
        },
        {
            "technique": "Dynamic Embeddings",
            "description": "Adapt embeddings based on context or user behavior",
            "approaches": [
                "Context-dependent embedding selection",
                "User-personalized embeddings",
                "Time-aware embedding updates",
                "Adaptive embedding dimensions"
            ],
            "code": """
class DynamicEmbedder:
    def __init__(self, models_dict):
        self.models = models_dict
        self.usage_stats = {}
    
    def select_model(self, text, context=None, user_id=None):
        # Context-based selection
        if context == 'technical':
            return self.models['scientific']
        elif context == 'casual':
            return self.models['general']
        
        # User-based selection
        if user_id in self.usage_stats:
            # Use model that worked best for this user
            best_model = max(self.usage_stats[user_id], 
                           key=self.usage_stats[user_id].get)
            return self.models[best_model]
        
        # Default fallback
        return self.models['general']
    
    def embed_with_context(self, text, context=None, user_id=None):
        model = self.select_model(text, context, user_id)
        embedding = model.encode([text])[0]
        
        # Add context-specific adjustments
        if context:
            context_vector = self.get_context_vector(context)
            embedding = embedding + 0.1 * context_vector
        
        return embedding
    
    def update_performance(self, user_id, model_name, performance):
        if user_id not in self.usage_stats:
            self.usage_stats[user_id] = {}
        self.usage_stats[user_id][model_name] = performance

# Usage
embedder = DynamicEmbedder({
    'general': SentenceTransformer('all-MiniLM-L6-v2'),
    'scientific': SentenceTransformer('allenai-specter')
})

embedding = embedder.embed_with_context(
    "Machine learning models", 
    context="technical", 
    user_id="user123"
)
"""
        }
    ]
    
    for technique in advanced_techniques:
        with st.expander(f"🔧 {technique['technique']}"):
            st.markdown(technique['description'])
            
            st.markdown("**Approaches:**")
            for approach in technique['approaches']:
                st.markdown(f"• {approach}")
            
            st.markdown("**Implementation:**")
            st.code(technique['code'], language='python')

with implementation_tabs[2]:
    st.subheader("📊 Evaluation & Optimization")
    
    evaluation_methods = [
        {
            "method": "Similarity Evaluation",
            "description": "Evaluate how well embeddings capture semantic similarity",
            "metrics": ["Spearman correlation with human judgments", "Pearson correlation", "Ranking accuracy"],
            "code": """
from scipy.stats import spearmanr, pearsonr
import numpy as np

def evaluate_similarity(model, similarity_dataset):
    \"\"\"
    Evaluate embeddings on word/sentence similarity dataset
    Dataset format: [(word1, word2, human_score), ...]
    \"\"\"
    model_scores = []
    human_scores = []
    
    for word1, word2, human_score in similarity_dataset:
        # Get embeddings
        emb1 = model.encode([word1])[0]
        emb2 = model.encode([word2])[0]
        
        # Compute cosine similarity
        model_score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        model_scores.append(model_score)
        human_scores.append(human_score)
    
    # Compute correlations
    spearman_corr, _ = spearmanr(model_scores, human_scores)
    pearson_corr, _ = pearsonr(model_scores, human_scores)
    
    return {
        'spearman': spearman_corr,
        'pearson': pearson_corr,
        'model_scores': model_scores,
        'human_scores': human_scores
    }

# Example evaluation
results = evaluate_similarity(model, similarity_dataset)
print(f"Spearman correlation: {results['spearman']:.3f}")
"""
        },
        {
            "method": "Downstream Task Evaluation",
            "description": "Evaluate embeddings on actual use case tasks",
            "metrics": ["Classification accuracy", "Retrieval precision/recall", "Clustering silhouette score"],
            "code": """
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

def evaluate_classification(embeddings, labels, cv_folds=5):
    \"\"\"Evaluate embeddings on classification task\"\"\"
    classifier = LogisticRegression(random_state=42)
    
    # Cross-validation
    cv_scores = cross_val_score(classifier, embeddings, labels, 
                               cv=cv_folds, scoring='accuracy')
    
    # Full training for detailed metrics
    classifier.fit(embeddings, labels)
    predictions = classifier.predict(embeddings)
    
    return {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'classification_report': classification_report(labels, predictions)
    }

def evaluate_retrieval(query_embeddings, doc_embeddings, relevance_labels, k=10):
    \"\"\"Evaluate embeddings on retrieval task\"\"\"
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Compute similarities
    similarities = cosine_similarity(query_embeddings, doc_embeddings)
    
    # Get top-k documents for each query
    top_k_indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]
    
    precision_scores = []
    recall_scores = []
    
    for i, (query_top_k, query_relevance) in enumerate(zip(top_k_indices, relevance_labels)):
        relevant_retrieved = sum(query_relevance[idx] for idx in query_top_k)
        total_relevant = sum(query_relevance)
        
        precision = relevant_retrieved / k
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    return {
        'precision_at_k': np.mean(precision_scores),
        'recall_at_k': np.mean(recall_scores),
        'f1_at_k': 2 * np.mean(precision_scores) * np.mean(recall_scores) / 
                   (np.mean(precision_scores) + np.mean(recall_scores))
    }
"""
        },
        {
            "method": "Embedding Quality Analysis",
            "description": "Analyze intrinsic properties of embedding space",
            "metrics": ["Isotropy", "Uniformity", "Alignment", "Hubness"],
            "code": """
import numpy as np
from sklearn.neighbors import NearestNeighbors

def analyze_embedding_quality(embeddings):
    \"\"\"Comprehensive embedding quality analysis\"\"\"
    
    # 1. Isotropy: How uniform are directions in embedding space
    def compute_isotropy(embeddings):
        # Compute principal components
        from sklearn.decomposition import PCA
        pca = PCA()
        pca.fit(embeddings)
        
        # Isotropy = how evenly distributed the explained variance is
        explained_var = pca.explained_variance_ratio_
        isotropy = 1 - np.var(explained_var) / np.mean(explained_var)
        return isotropy
    
    # 2. Uniformity: How evenly distributed are embeddings
    def compute_uniformity(embeddings, t=2):
        n = len(embeddings)
        distances = []
        for i in range(min(1000, n)):  # Sample for efficiency
            for j in range(i+1, min(1000, n)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(np.exp(-t * dist**2))
        return np.log(np.mean(distances))
    
    # 3. Alignment: How well aligned are similar concepts
    def compute_alignment(embeddings, similarity_pairs):
        alignments = []
        for (idx1, idx2) in similarity_pairs:
            cosine_sim = np.dot(embeddings[idx1], embeddings[idx2]) / \\
                        (np.linalg.norm(embeddings[idx1]) * np.linalg.norm(embeddings[idx2]))
            alignments.append(cosine_sim)
        return np.mean(alignments)
    
    # 4. Hubness: Do some points appear as neighbors too often?
    def compute_hubness(embeddings, k=10):
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(embeddings)
        _, indices = nbrs.kneighbors(embeddings)
        
        # Count how often each point appears as a neighbor
        neighbor_counts = np.zeros(len(embeddings))
        for neighbors in indices:
            for neighbor in neighbors[1:]:  # Exclude self
                neighbor_counts[neighbor] += 1
        
        # Skewness of neighbor counts indicates hubness
        return np.std(neighbor_counts) / np.mean(neighbor_counts)
    
    results = {
        'isotropy': compute_isotropy(embeddings),
        'uniformity': compute_uniformity(embeddings),
        'hubness': compute_hubness(embeddings)
    }
    
    return results

# Usage
quality_metrics = analyze_embedding_quality(embeddings)
print(f"Isotropy: {quality_metrics['isotropy']:.3f}")
print(f"Uniformity: {quality_metrics['uniformity']:.3f}")
print(f"Hubness: {quality_metrics['hubness']:.3f}")
"""
        }
    ]
    
    for method in evaluation_methods:
        with st.expander(f"📊 {method['method']}"):
            st.markdown(method['description'])
            
            st.markdown("**Key Metrics:**")
            for metric in method['metrics']:
                st.markdown(f"• {metric}")
            
            st.markdown("**Implementation:**")
            st.code(method['code'], language='python')

with implementation_tabs[3]:
    st.subheader("🏭 Production Deployment")
    
    deployment_considerations = [
        {
            "aspect": "Scalability",
            "description": "Handle large-scale embedding operations efficiently",
            "strategies": [
                "Batch processing for embedding generation",
                "Distributed computing with frameworks like Ray or Dask",
                "GPU acceleration for model inference",
                "Caching frequently accessed embeddings"
            ],
            "code": """
import ray
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Option 1: Ray for distributed processing
@ray.remote
def embed_batch(texts, model_path):
    model = SentenceTransformer(model_path)
    return model.encode(texts)

def distributed_embedding(all_texts, model_path, batch_size=1000):
    ray.init()
    
    # Split into batches
    batches = [all_texts[i:i+batch_size] 
              for i in range(0, len(all_texts), batch_size)]
    
    # Process batches in parallel
    futures = [embed_batch.remote(batch, model_path) for batch in batches]
    results = ray.get(futures)
    
    # Combine results
    return np.vstack(results)

# Option 2: Thread-based parallelization
class EmbeddingService:
    def __init__(self, model, max_workers=4):
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache = {}
    
    def embed_with_cache(self, text):
        if text in self.cache:
            return self.cache[text]
        
        embedding = self.model.encode([text])[0]
        self.cache[text] = embedding
        return embedding
    
    def embed_batch_async(self, texts):
        futures = [self.executor.submit(self.embed_with_cache, text) 
                  for text in texts]
        return [future.result() for future in futures]
"""
        },
        {
            "aspect": "Storage & Indexing",
            "description": "Efficiently store and search embedding vectors",
            "strategies": [
                "Vector databases (Pinecone, Weaviate, Qdrant)",
                "Approximate nearest neighbor search (FAISS, Annoy)",
                "Compression techniques (quantization, PCA)",
                "Distributed storage solutions"
            ],
            "code": """
# Vector database integration
import pinecone
import faiss
import numpy as np

class VectorStore:
    def __init__(self, storage_type='faiss'):
        self.storage_type = storage_type
        
        if storage_type == 'pinecone':
            pinecone.init(api_key="your-key", environment="us-west1-gcp")
            self.index = pinecone.Index("embeddings")
        
        elif storage_type == 'faiss':
            self.dimension = None
            self.index = None
            self.id_map = {}
    
    def add_embeddings(self, embeddings, ids=None, metadata=None):
        if self.storage_type == 'pinecone':
            vectors = [(str(i), emb.tolist(), meta) 
                      for i, (emb, meta) in enumerate(zip(embeddings, metadata or [{}]*len(embeddings)))]
            self.index.upsert(vectors)
        
        elif self.storage_type == 'faiss':
            if self.index is None:
                self.dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings.astype('float32'))
            self.index.add(embeddings.astype('float32'))
            
            # Store ID mapping
            if ids:
                start_idx = len(self.id_map)
                for i, id_val in enumerate(ids):
                    self.id_map[start_idx + i] = id_val
    
    def search(self, query_embedding, top_k=10):
        if self.storage_type == 'pinecone':
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            return results['matches']
        
        elif self.storage_type == 'faiss':
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                result = {
                    'score': float(score),
                    'id': self.id_map.get(idx, idx)
                }
                results.append(result)
            return results

# Usage
vector_store = VectorStore('faiss')
vector_store.add_embeddings(embeddings, ids=text_ids)
results = vector_store.search(query_embedding, top_k=5)
"""
        },
        {
            "aspect": "Monitoring & Maintenance",
            "description": "Monitor embedding quality and performance in production",
            "strategies": [
                "Performance metrics tracking",
                "Embedding drift detection",
                "A/B testing different embedding models",
                "Automated retraining pipelines"
            ],
            "code": """
import logging
from datetime import datetime
import json

class EmbeddingMonitor:
    def __init__(self, log_file='embedding_metrics.log'):
        self.logger = logging.getLogger('embedding_monitor')
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.baseline_metrics = None
        self.alert_thresholds = {
            'similarity_drop': 0.1,
            'latency_increase': 2.0,
            'error_rate': 0.05
        }
    
    def log_embedding_request(self, text, embedding, latency, success=True):
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'embedding_norm': float(np.linalg.norm(embedding)),
            'latency_ms': latency * 1000,
            'success': success
        }
        self.logger.info(json.dumps(log_data))
    
    def compute_embedding_drift(self, current_embeddings, reference_embeddings):
        \"\"\"Detect if embeddings have drifted from baseline\"\"\"
        from scipy.spatial.distance import cdist
        
        # Compute average distance between current and reference
        distances = cdist(current_embeddings, reference_embeddings, metric='cosine')
        avg_distance = np.mean(np.min(distances, axis=1))
        
        return avg_distance
    
    def check_performance_degradation(self, current_metrics):
        \"\"\"Check if performance has degraded significantly\"\"\"
        if self.baseline_metrics is None:
            self.baseline_metrics = current_metrics
            return False
        
        alerts = []
        
        # Check similarity drop
        if 'avg_similarity' in current_metrics:
            similarity_drop = self.baseline_metrics['avg_similarity'] - current_metrics['avg_similarity']
            if similarity_drop > self.alert_thresholds['similarity_drop']:
                alerts.append(f"Similarity dropped by {similarity_drop:.3f}")
        
        # Check latency increase
        if 'avg_latency' in current_metrics:
            latency_ratio = current_metrics['avg_latency'] / self.baseline_metrics['avg_latency']
            if latency_ratio > self.alert_thresholds['latency_increase']:
                alerts.append(f"Latency increased by {latency_ratio:.1f}x")
        
        return alerts
    
    def trigger_retraining_check(self, performance_drop_threshold=0.15):
        \"\"\"Check if model should be retrained\"\"\"
        # Implement logic to check if retraining is needed
        # This could be based on performance metrics, data drift, etc.
        pass

# Usage
monitor = EmbeddingMonitor()

# Log each embedding request
embedding = model.encode(["example text"])[0]
monitor.log_embedding_request("example text", embedding, latency=0.1)

# Check for performance issues
current_metrics = {'avg_similarity': 0.75, 'avg_latency': 0.15}
alerts = monitor.check_performance_degradation(current_metrics)
if alerts:
    print("Performance alerts:", alerts)
"""
        }
    ]
    
    for consideration in deployment_considerations:
        with st.expander(f"🏭 {consideration['aspect']}"):
            st.markdown(consideration['description'])
            
            st.markdown("**Key Strategies:**")
            for strategy in consideration['strategies']:
                st.markdown(f"• {strategy}")
            
            st.markdown("**Implementation Example:**")
            st.code(consideration['code'], language='python')

# Best practices
st.header("💡 Best Practices")

best_practices = [
    "**Choose the Right Model**: Select embeddings based on your specific use case and constraints",
    "**Normalize Embeddings**: Use unit vectors for cosine similarity comparisons",
    "**Handle Out-of-Vocabulary**: Plan for words/phrases not seen during training",
    "**Monitor Performance**: Track embedding quality and similarity metrics in production",
    "**Cache Strategically**: Store frequently used embeddings to reduce computation",
    "**Batch Processing**: Process multiple texts together for efficiency",
    "**Evaluate Thoroughly**: Test embeddings on your specific downstream tasks",
    "**Consider Context**: Use contextual embeddings when word meaning depends on context",
    "**Plan for Scale**: Design your embedding pipeline to handle growth in data volume",
    "**Version Control**: Track embedding model versions and their performance"
]

for practice in best_practices:
    st.markdown(f"• {practice}")

# Resources
st.header("📚 Learning Resources")

resources = [
    {
        "title": "Attention Is All You Need",
        "type": "Research Paper",
        "description": "Foundational paper on transformer architecture",
        "difficulty": "Advanced"
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "type": "Research Paper",
        "description": "Influential paper on contextual embeddings",
        "difficulty": "Advanced"
    },
    {
        "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
        "type": "Research Paper",
        "description": "Efficient sentence-level embeddings",
        "difficulty": "Intermediate"
    },
    {
        "title": "Embedding Techniques Handbook",
        "type": "Tutorial",
        "description": "Comprehensive guide to text embeddings",
        "difficulty": "Beginner"
    }
]

for resource in resources:
    with st.expander(f"📖 {resource['title']}"):
        st.markdown(f"**Type:** {resource['type']}")
        st.markdown(f"**Description:** {resource['description']}")
        st.markdown(f"**Difficulty:** {resource['difficulty']}")