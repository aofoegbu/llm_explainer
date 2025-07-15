import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Retrieval Augmented Generation", page_icon="🔍", layout="wide")

st.title("🔍 Retrieval Augmented Generation (RAG)")
st.markdown("### Enhancing LLMs with External Knowledge Retrieval")

# Overview
st.header("🎯 Overview")
st.markdown("""
Retrieval Augmented Generation (RAG) combines the power of large language models with external knowledge bases 
to provide more accurate, up-to-date, and factually grounded responses. RAG addresses key limitations of LLMs 
such as outdated knowledge, hallucination, and lack of domain-specific information.
""")

# Core concepts
st.header("🧠 Core Concepts")

concept_tabs = st.tabs([
    "🔍 What is RAG?",
    "🏗️ Architecture", 
    "⚙️ Components",
    "🔄 Workflow"
])

with concept_tabs[0]:
    st.subheader("🔍 Understanding RAG")
    
    st.markdown("""
    RAG is a framework that retrieves relevant information from external knowledge sources 
    and uses this information to augment the context provided to a language model for generation.
    """)
    
    # RAG vs Traditional LLM comparison
    st.markdown("### 🔄 RAG vs. Traditional LLM")
    
    comparison_data = {
        'Aspect': [
            'Knowledge Source',
            'Information Freshness', 
            'Factual Accuracy',
            'Domain Specificity',
            'Hallucination Risk',
            'Cost of Updates',
            'Response Latency',
            'Computational Requirements'
        ],
        'Traditional LLM': [
            'Training data only',
            'Static (training cutoff)',
            'Moderate (prone to hallucination)',
            'General purpose',
            'High',
            'Expensive (retraining)',
            'Low',
            'Model inference only'
        ],
        'RAG System': [
            'Training data + external knowledge',
            'Dynamic (real-time updates)',
            'High (grounded in retrieved facts)',
            'Customizable per domain',
            'Low (fact-checked)',
            'Low (update knowledge base)',
            'Higher (retrieval + generation)',
            'Retrieval system + model inference'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Benefits and challenges
    st.markdown("### ✅ Key Benefits")
    benefits = [
        "**Up-to-date Information**: Access to current knowledge beyond training cutoff",
        "**Reduced Hallucination**: Grounded responses based on retrieved facts",
        "**Domain Expertise**: Incorporate specialized knowledge bases",
        "**Transparency**: Show sources and evidence for claims",
        "**Cost-Effective Updates**: Update knowledge without retraining models",
        "**Customizable Knowledge**: Tailor information sources to specific use cases"
    ]
    
    for benefit in benefits:
        st.markdown(f"• {benefit}")
    
    st.markdown("### ⚠️ Key Challenges")
    challenges = [
        "**Retrieval Quality**: Effectiveness depends on search accuracy",
        "**Latency Overhead**: Additional time for retrieval operations",
        "**Knowledge Base Maintenance**: Requires ongoing curation and updates",
        "**Context Integration**: Effectively combining retrieved information with queries",
        "**Scalability**: Managing large-scale retrieval across extensive knowledge bases",
        "**Relevance Filtering**: Avoiding noise from irrelevant retrieved content"
    ]
    
    for challenge in challenges:
        st.markdown(f"• {challenge}")

with concept_tabs[1]:
    st.subheader("🏗️ RAG Architecture")
    
    st.markdown("""
    RAG systems consist of several interconnected components that work together to retrieve 
    relevant information and generate augmented responses.
    """)
    
    # Architecture diagram (simplified text representation)
    st.markdown("### 🔧 System Architecture")
    
    architecture_components = [
        {
            "component": "Knowledge Base",
            "description": "Repository of documents, articles, or structured data",
            "examples": ["Document databases", "Vector stores", "APIs", "Web crawls"],
            "role": "Source of external knowledge to augment model responses"
        },
        {
            "component": "Retrieval System", 
            "description": "Finds relevant information based on user queries",
            "examples": ["Dense retrieval", "Sparse retrieval", "Hybrid search", "Semantic search"],
            "role": "Bridge between user queries and knowledge base content"
        },
        {
            "component": "Context Augmentation",
            "description": "Combines retrieved information with original query",
            "examples": ["Prompt engineering", "Context ranking", "Information synthesis"],
            "role": "Prepares enriched context for the language model"
        },
        {
            "component": "Language Model",
            "description": "Generates responses using augmented context",
            "examples": ["GPT models", "Claude", "Local LLMs", "Fine-tuned models"],
            "role": "Produces final responses based on query and retrieved information"
        }
    ]
    
    for component in architecture_components:
        with st.expander(f"🔧 {component['component']}"):
            st.markdown(component['description'])
            st.markdown(f"**Role:** {component['role']}")
            st.markdown("**Examples:**")
            for example in component['examples']:
                st.markdown(f"• {example}")
    
    # Architecture patterns
    st.markdown("### 📐 Common Architecture Patterns")
    
    patterns = [
        {
            "pattern": "Sequential RAG",
            "description": "Retrieve first, then generate",
            "flow": "Query → Retrieve → Augment Context → Generate",
            "pros": ["Simple implementation", "Clear separation of concerns", "Easy to debug"],
            "cons": ["Fixed retrieval scope", "No iterative refinement", "May miss relevant info"],
            "best_for": "Simple Q&A, fact-checking, basic knowledge lookup"
        },
        {
            "pattern": "Iterative RAG",
            "description": "Multiple retrieval-generation cycles",
            "flow": "Query → Retrieve → Generate → Evaluate → Retrieve More → Refine",
            "pros": ["Adaptive information gathering", "Better coverage", "Self-correcting"],
            "cons": ["Higher latency", "Complex implementation", "More API calls"],
            "best_for": "Research tasks, complex questions, multi-step reasoning"
        },
        {
            "pattern": "Conditional RAG",
            "description": "Retrieve only when necessary",
            "flow": "Query → Assess Need → (Conditionally) Retrieve → Generate",
            "pros": ["Efficient resource usage", "Lower latency for simple queries", "Cost optimization"],
            "cons": ["Complex decision logic", "Risk of missing needed info", "Hard to tune thresholds"],
            "best_for": "Mixed workloads, cost-sensitive applications, general assistants"
        },
        {
            "pattern": "Agentic RAG",
            "description": "Agent decides retrieval strategy",
            "flow": "Query → Plan → Execute Retrieval Strategy → Synthesize → Generate",
            "pros": ["Intelligent retrieval planning", "Handles complex queries", "Tool integration"],
            "cons": ["Very complex", "Hard to predict behavior", "High computational cost"],
            "best_for": "Research assistants, complex analysis, multi-source integration"
        }
    ]
    
    for pattern in patterns:
        with st.expander(f"📐 {pattern['pattern']}"):
            st.markdown(pattern['description'])
            st.markdown(f"**Flow:** {pattern['flow']}")
            st.markdown(f"**Best for:** {pattern['best_for']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Pros:**")
                for pro in pattern['pros']:
                    st.markdown(f"• {pro}")
            with col2:
                st.markdown("**❌ Cons:**")
                for con in pattern['cons']:
                    st.markdown(f"• {con}")

with concept_tabs[2]:
    st.subheader("⚙️ RAG Components")
    
    component_tabs = st.tabs([
        "📚 Knowledge Sources",
        "🔍 Retrieval Methods",
        "🧠 Context Integration",
        "⚡ Optimization Techniques"
    ])
    
    with component_tabs[0]:
        st.markdown("### 📚 Knowledge Sources")
        
        knowledge_sources = [
            {
                "source": "Document Collections",
                "description": "Structured and unstructured text documents",
                "examples": [
                    "Corporate documentation and policies",
                    "Research papers and publications", 
                    "Product manuals and guides",
                    "Historical records and archives"
                ],
                "preprocessing": [
                    "Text extraction and cleaning",
                    "Chunking into manageable segments", 
                    "Metadata extraction and tagging",
                    "Deduplication and quality filtering"
                ],
                "challenges": [
                    "Handling different document formats",
                    "Maintaining document freshness",
                    "Managing access permissions",
                    "Dealing with large file sizes"
                ]
            },
            {
                "source": "Structured Databases",
                "description": "Relational and NoSQL databases with structured information",
                "examples": [
                    "Customer relationship management systems",
                    "Product catalogs and inventories",
                    "Financial and transactional data",
                    "Scientific datasets and measurements"
                ],
                "preprocessing": [
                    "Schema analysis and mapping",
                    "Data normalization and cleaning",
                    "Join optimization for retrieval",
                    "Index creation for fast access"
                ],
                "challenges": [
                    "Converting structured data to text",
                    "Handling complex relationships",
                    "Managing data privacy and security",
                    "Optimizing query performance"
                ]
            },
            {
                "source": "Web Content",
                "description": "Dynamic web pages and online resources",
                "examples": [
                    "News articles and blog posts",
                    "Wikipedia and reference sites",
                    "Product reviews and forums",
                    "Social media and community content"
                ],
                "preprocessing": [
                    "Web scraping and crawling",
                    "HTML parsing and content extraction",
                    "Duplicate content detection",
                    "Content freshness tracking"
                ],
                "challenges": [
                    "Handling dynamic content",
                    "Respecting robots.txt and rate limits",
                    "Managing content quality and reliability",
                    "Dealing with anti-scraping measures"
                ]
            },
            {
                "source": "APIs and Live Data",
                "description": "Real-time data from external services",
                "examples": [
                    "Weather and traffic information",
                    "Stock prices and financial data",
                    "Social media feeds and trends",
                    "IoT sensor data and telemetry"
                ],
                "preprocessing": [
                    "API response parsing and normalization",
                    "Rate limiting and caching strategies",
                    "Data validation and error handling",
                    "Real-time processing pipelines"
                ],
                "challenges": [
                    "Managing API rate limits and costs",
                    "Handling service unavailability",
                    "Processing real-time data streams",
                    "Maintaining data consistency"
                ]
            }
        ]
        
        for source in knowledge_sources:
            with st.expander(f"📚 {source['source']}"):
                st.markdown(source['description'])
                
                st.markdown("**Examples:**")
                for example in source['examples']:
                    st.markdown(f"• {example}")
                
                st.markdown("**Preprocessing Steps:**")
                for step in source['preprocessing']:
                    st.markdown(f"• {step}")
                
                st.markdown("**Key Challenges:**")
                for challenge in source['challenges']:
                    st.markdown(f"• {challenge}")
    
    with component_tabs[1]:
        st.markdown("### 🔍 Retrieval Methods")
        
        retrieval_methods = [
            {
                "method": "Dense Retrieval",
                "description": "Uses neural embeddings to find semantically similar content",
                "how_it_works": [
                    "Encode queries and documents into dense vectors",
                    "Compute similarity using cosine distance or dot product",
                    "Return top-k most similar documents",
                    "Leverage pre-trained or fine-tuned embedding models"
                ],
                "advantages": [
                    "Captures semantic similarity beyond exact keywords",
                    "Works well for natural language queries",
                    "Handles synonyms and paraphrases effectively",
                    "Can be fine-tuned for specific domains"
                ],
                "disadvantages": [
                    "Computationally expensive for large corpora",
                    "May miss exact keyword matches",
                    "Requires good embedding model quality",
                    "Black box - hard to interpret results"
                ],
                "best_for": "Semantic search, Q&A systems, research assistance"
            },
            {
                "method": "Sparse Retrieval (BM25)",
                "description": "Traditional keyword-based search using term frequency",
                "how_it_works": [
                    "Index documents using inverted indexes",
                    "Calculate relevance scores based on term frequencies",
                    "Apply document length normalization",
                    "Rank results by BM25 relevance score"
                ],
                "advantages": [
                    "Fast and efficient for large collections",
                    "Excellent for exact keyword matches",
                    "Interpretable scoring mechanism",
                    "Well-established and battle-tested"
                ],
                "disadvantages": [
                    "Misses semantic relationships",
                    "Struggles with synonyms and paraphrases",
                    "Vocabulary mismatch problems",
                    "Limited understanding of context"
                ],
                "best_for": "Keyword search, document retrieval, legal/compliance queries"
            },
            {
                "method": "Hybrid Retrieval",
                "description": "Combines dense and sparse retrieval for better coverage",
                "how_it_works": [
                    "Run both dense and sparse retrieval in parallel",
                    "Normalize and combine scores from both methods",
                    "Re-rank results using learned combination weights",
                    "Apply fusion techniques like reciprocal rank fusion"
                ],
                "advantages": [
                    "Best of both worlds - semantic and exact matching",
                    "More robust across different query types",
                    "Better recall and precision balance",
                    "Adaptable to different use cases"
                ],
                "disadvantages": [
                    "More complex implementation and tuning",
                    "Higher computational overhead",
                    "Requires careful score normalization",
                    "More components to maintain and optimize"
                ],
                "best_for": "General-purpose search, enterprise applications, complex domains"
            },
            {
                "method": "Re-ranking",
                "description": "Refines initial retrieval results using sophisticated models",
                "how_it_works": [
                    "Retrieve initial candidate set using fast method",
                    "Apply more sophisticated ranking model to candidates",
                    "Use cross-encoders or other deep models for ranking",
                    "Return re-ordered results with better relevance"
                ],
                "advantages": [
                    "Improves precision of final results",
                    "Can use more compute on smaller candidate set",
                    "Incorporates complex query-document interactions",
                    "Better handling of nuanced relevance judgments"
                ],
                "disadvantages": [
                    "Additional latency from two-stage process",
                    "Limited by quality of initial retrieval",
                    "More complex pipeline to manage",
                    "Higher computational cost overall"
                ],
                "best_for": "High-precision applications, competitive search, critical decisions"
            }
        ]
        
        for method in retrieval_methods:
            with st.expander(f"🔍 {method['method']}"):
                st.markdown(method['description'])
                st.markdown(f"**Best for:** {method['best_for']}")
                
                st.markdown("**How it works:**")
                for step in method['how_it_works']:
                    st.markdown(f"• {step}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Advantages:**")
                    for advantage in method['advantages']:
                        st.markdown(f"• {advantage}")
                with col2:
                    st.markdown("**❌ Disadvantages:**")
                    for disadvantage in method['disadvantages']:
                        st.markdown(f"• {disadvantage}")
        
        # Performance comparison
        st.markdown("### 📊 Retrieval Method Performance Comparison")
        
        performance_data = pd.DataFrame({
            'Method': ['BM25', 'Dense Retrieval', 'Hybrid', 'Re-ranking'],
            'Precision': [0.72, 0.78, 0.83, 0.87],
            'Recall': [0.68, 0.74, 0.79, 0.81],
            'Latency (ms)': [15, 45, 60, 85],
            'Semantic Understanding': [2, 8, 7, 9],
            'Exact Match': [9, 6, 8, 8]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=performance_data['Precision'],
            y=performance_data['Recall'],
            mode='markers+text',
            text=performance_data['Method'],
            textposition="top center",
            marker=dict(
                size=performance_data['Latency (ms)'],
                sizemode='diameter',
                sizeref=2,
                color=performance_data['Semantic Understanding'],
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Semantic Understanding")
            ),
            name='Retrieval Methods'
        ))
        
        fig.update_layout(
            title="Retrieval Method Comparison (Bubble size = Latency)",
            xaxis_title="Precision",
            yaxis_title="Recall",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with component_tabs[2]:
        st.markdown("### 🧠 Context Integration")
        
        integration_strategies = [
            {
                "strategy": "Simple Concatenation",
                "description": "Directly append retrieved content to the query",
                "implementation": """
def simple_concatenation(query, retrieved_docs):
    context = "\\n\\n".join([doc['content'] for doc in retrieved_docs])
    prompt = f"Context: {context}\\n\\nQuestion: {query}\\n\\nAnswer:"
    return prompt
""",
                "pros": ["Simple to implement", "Transparent process", "Works with any LLM"],
                "cons": ["May exceed context limits", "No relevance weighting", "Poor handling of contradictions"]
            },
            {
                "strategy": "Ranked Integration",
                "description": "Include retrieved content based on relevance ranking",
                "implementation": """
def ranked_integration(query, retrieved_docs, max_tokens=2000):
    # Sort by relevance score
    sorted_docs = sorted(retrieved_docs, key=lambda x: x['score'], reverse=True)
    
    context_parts = []
    total_tokens = 0
    
    for doc in sorted_docs:
        doc_tokens = estimate_tokens(doc['content'])
        if total_tokens + doc_tokens <= max_tokens:
            context_parts.append(f"Source: {doc['title']}\\n{doc['content']}")
            total_tokens += doc_tokens
        else:
            break
    
    context = "\\n\\n---\\n\\n".join(context_parts)
    prompt = f"Use the following sources to answer the question:\\n\\n{context}\\n\\nQuestion: {query}\\n\\nAnswer:"
    return prompt
""",
                "pros": ["Respects context limits", "Prioritizes relevant content", "Includes source attribution"],
                "cons": ["May miss important info in lower-ranked docs", "Still simple concatenation", "No semantic integration"]
            },
            {
                "strategy": "Summarization-Based",
                "description": "Summarize retrieved content before including in context",
                "implementation": """
def summarization_integration(query, retrieved_docs, summarizer_model):
    # Summarize each document
    summaries = []
    for doc in retrieved_docs:
        summary_prompt = f"Summarize the following text in 2-3 sentences:\\n{doc['content']}"
        summary = summarizer_model.generate(summary_prompt)
        summaries.append(f"Source {doc['title']}: {summary}")
    
    # Combine summaries
    context = "\\n\\n".join(summaries)
    prompt = f"Based on the following information:\\n\\n{context}\\n\\nQuestion: {query}\\n\\nAnswer:"
    return prompt
""",
                "pros": ["Condenses information efficiently", "Reduces token usage", "Highlights key points"],
                "cons": ["May lose important details", "Requires additional model calls", "Summarization quality dependent"]
            },
            {
                "strategy": "Question-Focused Extraction",
                "description": "Extract only information relevant to the specific question",
                "implementation": """
def question_focused_extraction(query, retrieved_docs, extraction_model):
    relevant_excerpts = []
    
    for doc in retrieved_docs:
        extraction_prompt = f'''
        Document: {doc['content']}
        Question: {query}
        
        Extract only the parts of the document that are directly relevant to answering the question.
        If nothing is relevant, respond with "No relevant information."
        
        Relevant excerpt:'''
        
        excerpt = extraction_model.generate(extraction_prompt)
        if excerpt != "No relevant information.":
            relevant_excerpts.append(f"From {doc['title']}: {excerpt}")
    
    context = "\\n\\n".join(relevant_excerpts)
    prompt = f"Relevant information:\\n\\n{context}\\n\\nQuestion: {query}\\n\\nAnswer:"
    return prompt
""",
                "pros": ["Highly focused on query", "Reduces noise", "Better token efficiency"],
                "cons": ["Complex extraction process", "May miss subtle connections", "Requires sophisticated extraction model"]
            }
        ]
        
        for strategy in integration_strategies:
            with st.expander(f"🧠 {strategy['strategy']}"):
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
                
                st.markdown("**Implementation:**")
                st.code(strategy['implementation'], language='python')
    
    with component_tabs[3]:
        st.markdown("### ⚡ Optimization Techniques")
        
        optimization_techniques = [
            {
                "technique": "Query Optimization",
                "description": "Improve query formulation for better retrieval",
                "approaches": [
                    "Query expansion with synonyms and related terms",
                    "Query rewriting for better matching",
                    "Multi-query generation for comprehensive coverage", 
                    "Query classification for method selection"
                ],
                "example": """
# Query expansion example
def expand_query(original_query, expansion_model):
    expansion_prompt = f'''
    Original query: {original_query}
    
    Generate 3 alternative phrasings of this query that might find relevant information:
    1.
    2. 
    3.
    '''
    expanded_queries = expansion_model.generate(expansion_prompt)
    return parse_expanded_queries(expanded_queries)

# Use all queries for retrieval
all_results = []
for query in [original_query] + expanded_queries:
    results = retrieval_system.search(query)
    all_results.extend(results)

# Deduplicate and rank
final_results = deduplicate_and_rank(all_results)
"""
            },
            {
                "technique": "Caching Strategies",
                "description": "Cache retrieval results to reduce latency and costs",
                "approaches": [
                    "Semantic caching based on query similarity",
                    "Result caching with TTL for freshness",
                    "Hierarchical caching at multiple levels",
                    "Precomputed results for common queries"
                ],
                "example": """
import hashlib
from datetime import datetime, timedelta

class SemanticCache:
    def __init__(self, similarity_threshold=0.95, ttl_hours=24):
        self.cache = {}
        self.query_embeddings = {}
        self.similarity_threshold = similarity_threshold
        self.ttl = timedelta(hours=ttl_hours)
    
    def get_cache_key(self, query):
        return hashlib.md5(query.encode()).hexdigest()
    
    def is_expired(self, timestamp):
        return datetime.now() - timestamp > self.ttl
    
    def get(self, query, query_embedding):
        # Check for exact match first
        cache_key = self.get_cache_key(query)
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if not self.is_expired(timestamp):
                return result
        
        # Check for semantic similarity
        for cached_query, cached_embedding in self.query_embeddings.items():
            similarity = cosine_similarity(query_embedding, cached_embedding)
            if similarity > self.similarity_threshold:
                cached_key = self.get_cache_key(cached_query)
                if cached_key in self.cache:
                    result, timestamp = self.cache[cached_key]
                    if not self.is_expired(timestamp):
                        return result
        
        return None
    
    def set(self, query, query_embedding, result):
        cache_key = self.get_cache_key(query)
        self.cache[cache_key] = (result, datetime.now())
        self.query_embeddings[query] = query_embedding
"""
            },
            {
                "technique": "Chunk Optimization",
                "description": "Optimize document chunking for better retrieval",
                "approaches": [
                    "Semantic chunking based on content structure",
                    "Overlapping chunks to preserve context",
                    "Adaptive chunk sizes based on content type",
                    "Hierarchical chunking with multiple granularities"
                ],
                "example": """
def semantic_chunking(text, max_chunk_size=500, overlap=50):
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        
        # If adding this sentence would exceed max size, finalize current chunk
        if current_length + sentence_length > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Start new chunk with overlap
            if overlap > 0:
                overlap_sentences = current_chunk[-overlap:]
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk = [sentence]
                current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def hierarchical_chunking(document):
    # Create chunks at multiple levels
    return {
        'paragraphs': chunk_by_paragraphs(document),
        'sections': chunk_by_sections(document), 
        'pages': chunk_by_pages(document),
        'semantic': semantic_chunking(document)
    }
"""
            },
            {
                "technique": "Relevance Filtering", 
                "description": "Filter out irrelevant retrieved content",
                "approaches": [
                    "Threshold-based filtering using similarity scores",
                    "ML-based relevance classification",
                    "Query-document entailment checking",
                    "Diversity filtering to avoid redundancy"
                ],
                "example": """
def relevance_filter(query, retrieved_docs, min_score=0.7, max_results=10):
    # Score-based filtering
    relevant_docs = [doc for doc in retrieved_docs if doc['score'] >= min_score]
    
    # Diversity filtering
    diverse_docs = []
    used_content = set()
    
    for doc in relevant_docs:
        # Simple content hash for deduplication
        content_hash = hash(doc['content'][:200])  # First 200 chars
        
        if content_hash not in used_content:
            diverse_docs.append(doc)
            used_content.add(content_hash)
            
            if len(diverse_docs) >= max_results:
                break
    
    return diverse_docs

def entailment_filter(query, retrieved_docs, entailment_model):
    filtered_docs = []
    
    for doc in retrieved_docs:
        # Check if document content entails relevance to query
        entailment_score = entailment_model.predict(
            premise=doc['content'][:500],  # Truncate for efficiency
            hypothesis=f"This text is relevant to: {query}"
        )
        
        if entailment_score > 0.8:  # High confidence threshold
            filtered_docs.append(doc)
    
    return filtered_docs
"""
            }
        ]
        
        for technique in optimization_techniques:
            with st.expander(f"⚡ {technique['technique']}"):
                st.markdown(technique['description'])
                
                st.markdown("**Key Approaches:**")
                for approach in technique['approaches']:
                    st.markdown(f"• {approach}")
                
                st.markdown("**Implementation Example:**")
                st.code(technique['example'], language='python')

with concept_tabs[3]:
    st.subheader("🔄 RAG Workflow")
    
    st.markdown("""
    The RAG workflow describes the step-by-step process of how a query flows through 
    the system to produce an augmented response.
    """)
    
    # Workflow visualization
    workflow_steps = [
        {
            "step": "1. Query Processing",
            "description": "Analyze and prepare the user query",
            "details": [
                "Query understanding and intent detection",
                "Query preprocessing and normalization", 
                "Query expansion and rewriting if needed",
                "Embedding generation for dense retrieval"
            ],
            "inputs": ["User query", "Query context", "User preferences"],
            "outputs": ["Processed query", "Query embedding", "Retrieval strategy"]
        },
        {
            "step": "2. Retrieval Execution", 
            "description": "Search for relevant information",
            "details": [
                "Execute search across knowledge sources",
                "Apply retrieval methods (dense, sparse, hybrid)",
                "Score and rank retrieved candidates",
                "Filter results by relevance thresholds"
            ],
            "inputs": ["Processed query", "Knowledge base index", "Retrieval parameters"],
            "outputs": ["Ranked document list", "Relevance scores", "Source metadata"]
        },
        {
            "step": "3. Context Preparation",
            "description": "Prepare augmented context for generation",
            "details": [
                "Select top-k most relevant documents",
                "Apply context integration strategy",
                "Manage context length constraints",
                "Format context for the language model"
            ],
            "inputs": ["Retrieved documents", "Original query", "Context limits"],
            "outputs": ["Augmented prompt", "Source references", "Context metadata"]
        },
        {
            "step": "4. Response Generation",
            "description": "Generate response using augmented context",
            "details": [
                "Send augmented prompt to language model",
                "Generate response based on retrieved context",
                "Apply response quality filters",
                "Extract generated answer and reasoning"
            ],
            "inputs": ["Augmented prompt", "Generation parameters", "Quality filters"],
            "outputs": ["Generated response", "Confidence score", "Reasoning trace"]
        },
        {
            "step": "5. Post-processing",
            "description": "Refine and validate the final response",
            "details": [
                "Fact-checking against retrieved sources",
                "Citation and source attribution",
                "Response formatting and presentation",
                "Quality assessment and confidence scoring"
            ],
            "inputs": ["Generated response", "Source documents", "Quality metrics"],
            "outputs": ["Final response", "Source citations", "Confidence assessment"]
        }
    ]
    
    for step_info in workflow_steps:
        with st.expander(f"🔄 {step_info['step']}: {step_info['description']}"):
            st.markdown("**Key Operations:**")
            for detail in step_info['details']:
                st.markdown(f"• {detail}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Inputs:**")
                for inp in step_info['inputs']:
                    st.markdown(f"• {inp}")
            with col2:
                st.markdown("**Processing:**")
                st.markdown("Transforms inputs through the operations listed above")
            with col3:
                st.markdown("**Outputs:**")
                for outp in step_info['outputs']:
                    st.markdown(f"• {outp}")
    
    # Flow diagram
    st.markdown("### 📊 Workflow Performance Metrics")
    
    # Simulate workflow performance data
    workflow_metrics = pd.DataFrame({
        'Step': ['Query Processing', 'Retrieval', 'Context Prep', 'Generation', 'Post-processing'],
        'Avg Latency (ms)': [50, 200, 100, 800, 150],
        'Success Rate (%)': [99.5, 94.2, 98.1, 92.8, 96.7],
        'Resource Usage': [1, 4, 2, 8, 3]
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Latency (ms)',
        x=workflow_metrics['Step'],
        y=workflow_metrics['Avg Latency (ms)'],
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        name='Success Rate (%)',
        x=workflow_metrics['Step'], 
        y=workflow_metrics['Success Rate (%)'],
        yaxis='y2',
        mode='lines+markers',
        line=dict(color='red'),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="RAG Workflow Performance by Stage",
        xaxis_title="Workflow Step",
        yaxis=dict(title="Latency (ms)", side="left"),
        yaxis2=dict(title="Success Rate (%)", side="right", overlaying="y"),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Implementation guide
st.header("🛠️ Implementation Guide")

implementation_tabs = st.tabs([
    "🚀 Quick Start",
    "🏗️ System Design",
    "🔧 Advanced Patterns",
    "📊 Evaluation & Monitoring"
])

with implementation_tabs[0]:
    st.subheader("🚀 Quick Start RAG Implementation")
    
    quick_start_code = """
# Basic RAG Implementation Example
import openai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SimpleRAG:
    def __init__(self, documents, openai_api_key):
        self.documents = documents
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Create embeddings and index
        self.document_embeddings = self.encoder.encode(documents)
        self.index = faiss.IndexFlatIP(self.document_embeddings.shape[1])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(self.document_embeddings.astype('float32'))
        self.index.add(self.document_embeddings.astype('float32'))
    
    def retrieve(self, query, top_k=3):
        # Encode query
        query_embedding = self.encoder.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Return relevant documents
        retrieved_docs = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            retrieved_docs.append({
                'content': self.documents[idx],
                'score': float(score),
                'rank': i + 1
            })
        
        return retrieved_docs
    
    def generate_response(self, query, retrieved_docs):
        # Create context from retrieved documents
        context = "\\n\\n".join([f"Document {doc['rank']}: {doc['content']}" 
                                for doc in retrieved_docs])
        
        # Create prompt
        prompt = f'''Based on the following context, answer the question.
        
Context:
{context}

Question: {query}

Answer: Provide a comprehensive answer based on the context above. If the context doesn't contain enough information, say so.'''

        # Generate response
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    def query(self, question):
        # Full RAG pipeline
        retrieved_docs = self.retrieve(question)
        response = self.generate_response(question, retrieved_docs)
        
        return {
            'answer': response,
            'sources': retrieved_docs,
            'context_used': len(retrieved_docs)
        }

# Usage example
documents = [
    "The capital of France is Paris. It is known for the Eiffel Tower.",
    "Python is a programming language. It's great for data science.",
    "Machine learning involves training models on data to make predictions.",
    "The weather today is sunny with a temperature of 75°F."
]

rag_system = SimpleRAG(documents, "your-openai-api-key")
result = rag_system.query("What is the capital of France?")

print("Answer:", result['answer'])
print("Sources used:", len(result['sources']))
"""
    
    st.code(quick_start_code, language='python')
    
    st.markdown("### 🔧 Setup Steps")
    
    setup_steps = [
        "**Install Dependencies**: `pip install openai sentence-transformers faiss-cpu`",
        "**Prepare Knowledge Base**: Collect and preprocess your documents", 
        "**Configure API Keys**: Set up OpenAI or other LLM provider credentials",
        "**Create Embeddings**: Generate vector representations of your documents",
        "**Build Index**: Create searchable index for fast retrieval",
        "**Test System**: Validate with sample queries and refine as needed"
    ]
    
    for i, step in enumerate(setup_steps, 1):
        st.markdown(f"{i}. {step}")

with implementation_tabs[1]:
    st.subheader("🏗️ Production System Design")
    
    system_design_considerations = [
        {
            "component": "Knowledge Base Management",
            "description": "Scalable storage and indexing of knowledge sources",
            "design_patterns": [
                "Document ingestion pipelines with format standardization",
                "Incremental indexing for real-time updates",
                "Version control for knowledge base changes",
                "Backup and disaster recovery strategies"
            ],
            "technologies": [
                "Vector databases: Pinecone, Weaviate, Qdrant",
                "Document stores: Elasticsearch, MongoDB",
                "Data pipelines: Apache Airflow, Prefect",
                "Storage: AWS S3, Google Cloud Storage"
            ],
            "code_example": """
# Knowledge base management system
class KnowledgeBaseManager:
    def __init__(self, vector_db, document_store):
        self.vector_db = vector_db
        self.document_store = document_store
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def ingest_document(self, document, metadata):
        # Store original document
        doc_id = self.document_store.store(document, metadata)
        
        # Create chunks
        chunks = self.chunk_document(document)
        
        # Generate embeddings and store
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_model.encode([chunk])[0]
            chunk_id = f"{doc_id}_chunk_{i}"
            
            self.vector_db.upsert({
                'id': chunk_id,
                'vector': embedding.tolist(),
                'metadata': {
                    'document_id': doc_id,
                    'chunk_index': i,
                    'content': chunk,
                    **metadata
                }
            })
        
        return doc_id
    
    def update_document(self, doc_id, new_content, new_metadata):
        # Remove old chunks
        self.vector_db.delete_by_metadata({'document_id': doc_id})
        
        # Re-ingest with new content
        return self.ingest_document(new_content, new_metadata)
    
    def search(self, query, top_k=10, filters=None):
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = self.vector_db.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filters
        )
        
        return results['matches']
"""
        },
        {
            "component": "Retrieval Service Architecture",
            "description": "Scalable and efficient retrieval system design",
            "design_patterns": [
                "Microservices architecture for different retrieval methods",
                "Caching layers for frequently accessed content",
                "Load balancing across retrieval instances",
                "Circuit breakers for fault tolerance"
            ],
            "technologies": [
                "APIs: FastAPI, Flask, GraphQL",
                "Caching: Redis, Memcached",
                "Load balancing: NGINX, HAProxy",
                "Monitoring: Prometheus, Grafana"
            ],
            "code_example": """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import redis

app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict = {}

class RetrievalService:
    def __init__(self):
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
        self.hybrid_fusion = HybridFusion()
    
    async def retrieve(self, query: str, top_k: int, filters: dict):
        # Check cache first
        cache_key = f"retrieval:{hash(query)}:{top_k}"
        cached_result = redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        # Parallel retrieval
        dense_task = asyncio.create_task(
            self.dense_retriever.search(query, top_k, filters)
        )
        sparse_task = asyncio.create_task(
            self.sparse_retriever.search(query, top_k, filters)
        )
        
        dense_results, sparse_results = await asyncio.gather(
            dense_task, sparse_task
        )
        
        # Fuse results
        fused_results = self.hybrid_fusion.combine(
            dense_results, sparse_results, top_k
        )
        
        # Cache results (expire in 1 hour)
        redis_client.setex(cache_key, 3600, json.dumps(fused_results))
        
        return fused_results

retrieval_service = RetrievalService()

@app.post("/retrieve")
async def retrieve_documents(request: QueryRequest):
    try:
        results = await retrieval_service.retrieve(
            request.query, request.top_k, request.filters
        )
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""
        },
        {
            "component": "Generation Service",
            "description": "Scalable LLM integration and response generation",
            "design_patterns": [
                "Model serving with auto-scaling",
                "Request queuing and batching for efficiency",
                "Multiple model support with routing",
                "Response caching and deduplication"
            ],
            "technologies": [
                "Model serving: Ray Serve, TorchServe, TensorFlow Serving",
                "Queue systems: Apache Kafka, RabbitMQ",
                "Container orchestration: Kubernetes, Docker Swarm",
                "API gateways: Kong, Ambassador"
            ],
            "code_example": """
import asyncio
from typing import List, Dict
import tiktoken

class GenerationService:
    def __init__(self):
        self.models = {
            'gpt-3.5-turbo': OpenAIModel('gpt-3.5-turbo'),
            'gpt-4': OpenAIModel('gpt-4'),
            'claude': AnthropicModel('claude-2'),
            'local-llm': LocalLLMModel('path/to/model')
        }
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def select_model(self, query: str, context: str, requirements: Dict):
        # Model selection logic based on requirements
        total_tokens = len(self.tokenizer.encode(query + context))
        
        if requirements.get('fast_response', False):
            return 'gpt-3.5-turbo'
        elif total_tokens > 8000:
            return 'gpt-4'  # Better for long contexts
        elif requirements.get('privacy', False):
            return 'local-llm'
        else:
            return 'gpt-3.5-turbo'  # Default
    
    async def generate_response(self, query: str, context: str, 
                              model_requirements: Dict = None):
        model_name = self.select_model(query, context, model_requirements or {})
        model = self.models[model_name]
        
        prompt = self.build_prompt(query, context)
        
        # Generate with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await model.generate(prompt)
                return {
                    'answer': response,
                    'model_used': model_name,
                    'tokens_used': len(self.tokenizer.encode(prompt + response)),
                    'attempt': attempt + 1
                }
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def build_prompt(self, query: str, context: str):
        return f'''Use the following context to answer the question. Be accurate and cite sources when possible.

Context:
{context}

Question: {query}

Answer:'''

@app.post("/generate")
async def generate_answer(request: GenerationRequest):
    try:
        result = await generation_service.generate_response(
            request.query, 
            request.context,
            request.model_requirements
        )
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""
        }
    ]
    
    for component in system_design_considerations:
        with st.expander(f"🏗️ {component['component']}"):
            st.markdown(component['description'])
            
            st.markdown("**Design Patterns:**")
            for pattern in component['design_patterns']:
                st.markdown(f"• {pattern}")
            
            st.markdown("**Key Technologies:**")
            for tech in component['technologies']:
                st.markdown(f"• {tech}")
            
            st.markdown("**Implementation Example:**")
            st.code(component['code_example'], language='python')

with implementation_tabs[2]:
    st.subheader("🔧 Advanced RAG Patterns")
    
    advanced_patterns = [
        {
            "pattern": "Multi-Modal RAG",
            "description": "Integrate text, images, and other modalities in retrieval",
            "use_cases": [
                "Technical documentation with diagrams",
                "Product catalogs with images and descriptions",
                "Research papers with figures and charts",
                "Educational content with multimedia"
            ],
            "implementation": """
class MultiModalRAG:
    def __init__(self):
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.image_encoder = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.text_index = faiss.IndexFlatIP(384)  # Text embedding dimension
        self.image_index = faiss.IndexFlatIP(512)  # Image embedding dimension
        
    def index_document(self, document):
        doc_id = document['id']
        
        # Index text content
        if 'text' in document:
            text_embedding = self.text_encoder.encode([document['text']])[0]
            self.text_index.add(text_embedding.reshape(1, -1))
        
        # Index images
        if 'images' in document:
            for img_path in document['images']:
                image = Image.open(img_path)
                image_embedding = self.image_encoder.encode_image(image)
                self.image_index.add(image_embedding.reshape(1, -1))
    
    def retrieve_multimodal(self, query, query_type='text'):
        results = []
        
        if query_type == 'text':
            # Text-to-text retrieval
            query_embedding = self.text_encoder.encode([query])
            scores, indices = self.text_index.search(query_embedding, k=5)
            results.extend(self.format_text_results(scores, indices))
            
            # Text-to-image retrieval (using CLIP)
            text_features = self.image_encoder.encode_text(query)
            img_scores, img_indices = self.image_index.search(text_features, k=3)
            results.extend(self.format_image_results(img_scores, img_indices))
        
        elif query_type == 'image':
            # Image-to-image and image-to-text retrieval
            query_image = Image.open(query)
            image_features = self.image_encoder.encode_image(query_image)
            
            # Similar images
            img_scores, img_indices = self.image_index.search(image_features, k=5)
            results.extend(self.format_image_results(img_scores, img_indices))
            
            # Related text content
            text_scores, text_indices = self.text_index.search(image_features, k=3)
            results.extend(self.format_text_results(text_scores, text_indices))
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
"""
        },
        {
            "pattern": "Hierarchical RAG",
            "description": "Multi-level retrieval from abstract to specific information",
            "use_cases": [
                "Large technical documentation systems",
                "Multi-level knowledge hierarchies",
                "Academic research databases",
                "Enterprise knowledge management"
            ],
            "implementation": """
class HierarchicalRAG:
    def __init__(self):
        self.levels = {
            'category': CategoryIndex(),      # High-level topics
            'document': DocumentIndex(),      # Document-level
            'section': SectionIndex(),        # Section-level
            'paragraph': ParagraphIndex()     # Fine-grained
        }
    
    def hierarchical_retrieve(self, query, max_results_per_level=5):
        results = {}
        
        # Level 1: Category-level retrieval
        category_results = self.levels['category'].search(query, max_results_per_level)
        relevant_categories = [r['category_id'] for r in category_results]
        
        # Level 2: Document-level within relevant categories
        document_results = self.levels['document'].search(
            query, 
            filters={'category_id': relevant_categories},
            max_results=max_results_per_level
        )
        relevant_documents = [r['document_id'] for r in document_results]
        
        # Level 3: Section-level within relevant documents
        section_results = self.levels['section'].search(
            query,
            filters={'document_id': relevant_documents},
            max_results=max_results_per_level
        )
        
        # Level 4: Paragraph-level within relevant sections
        relevant_sections = [r['section_id'] for r in section_results]
        paragraph_results = self.levels['paragraph'].search(
            query,
            filters={'section_id': relevant_sections},
            max_results=max_results_per_level
        )
        
        return {
            'categories': category_results,
            'documents': document_results,
            'sections': section_results,
            'paragraphs': paragraph_results
        }
    
    def generate_hierarchical_context(self, hierarchical_results, query):
        context_parts = []
        
        # Add high-level context from categories
        if hierarchical_results['categories']:
            context_parts.append("TOPIC OVERVIEW:")
            for cat in hierarchical_results['categories'][:2]:
                context_parts.append(f"- {cat['title']}: {cat['description']}")
        
        # Add document-level context
        if hierarchical_results['documents']:
            context_parts.append("\\nRELEVANT DOCUMENTS:")
            for doc in hierarchical_results['documents'][:3]:
                context_parts.append(f"- {doc['title']}: {doc['summary']}")
        
        # Add detailed content from paragraphs
        if hierarchical_results['paragraphs']:
            context_parts.append("\\nDETAILED INFORMATION:")
            for para in hierarchical_results['paragraphs'][:5]:
                context_parts.append(f"- {para['content']}")
        
        return "\\n".join(context_parts)
"""
        },
        {
            "pattern": "Adaptive RAG",
            "description": "Dynamic retrieval strategy based on query complexity",
            "use_cases": [
                "General-purpose assistants",
                "Variable query complexity systems",
                "Resource-constrained environments",
                "Multi-tenant RAG systems"
            ],
            "implementation": """
class AdaptiveRAG:
    def __init__(self):
        self.query_classifier = QueryComplexityClassifier()
        self.retrieval_strategies = {
            'simple': SimpleRetrieval(),
            'moderate': HybridRetrieval(),
            'complex': IterativeRetrieval(),
            'expert': AgenticRetrieval()
        }
    
    def assess_query_complexity(self, query):
        # Analyze query characteristics
        features = {
            'length': len(query.split()),
            'question_words': count_question_words(query),
            'technical_terms': count_technical_terms(query),
            'multi_part': has_multiple_questions(query),
            'temporal_references': has_temporal_refs(query),
            'comparison_requests': has_comparisons(query)
        }
        
        complexity_score = self.query_classifier.predict(features)
        
        if complexity_score < 0.3:
            return 'simple'
        elif complexity_score < 0.6:
            return 'moderate'
        elif complexity_score < 0.8:
            return 'complex'
        else:
            return 'expert'
    
    def adaptive_retrieve(self, query, user_context=None):
        complexity = self.assess_query_complexity(query)
        strategy = self.retrieval_strategies[complexity]
        
        # Adjust parameters based on complexity
        if complexity == 'simple':
            # Fast, basic retrieval
            return strategy.retrieve(query, top_k=3, max_iterations=1)
        
        elif complexity == 'moderate':
            # Hybrid approach with moderate depth
            return strategy.retrieve(query, top_k=5, use_reranking=True)
        
        elif complexity == 'complex':
            # Iterative retrieval with refinement
            return strategy.retrieve(
                query, 
                top_k=10, 
                max_iterations=3,
                use_query_expansion=True
            )
        
        else:  # expert
            # Full agentic approach
            return strategy.retrieve(
                query,
                user_context=user_context,
                use_tools=True,
                max_depth=5
            )
    
    def generate_adaptive_response(self, query, user_context=None):
        complexity = self.assess_query_complexity(query)
        retrieved_docs = self.adaptive_retrieve(query, user_context)
        
        # Adjust generation parameters based on complexity
        generation_params = {
            'simple': {'max_tokens': 150, 'temperature': 0.1},
            'moderate': {'max_tokens': 300, 'temperature': 0.2},
            'complex': {'max_tokens': 500, 'temperature': 0.3},
            'expert': {'max_tokens': 800, 'temperature': 0.4}
        }
        
        params = generation_params[complexity]
        
        response = self.llm.generate(
            query=query,
            context=self.format_context(retrieved_docs),
            **params
        )
        
        return {
            'answer': response,
            'complexity': complexity,
            'sources': retrieved_docs,
            'retrieval_method': complexity
        }
"""
        }
    ]
    
    for pattern in advanced_patterns:
        with st.expander(f"🔧 {pattern['pattern']}"):
            st.markdown(pattern['description'])
            
            st.markdown("**Use Cases:**")
            for use_case in pattern['use_cases']:
                st.markdown(f"• {use_case}")
            
            st.markdown("**Implementation:**")
            st.code(pattern['implementation'], language='python')

with implementation_tabs[3]:
    st.subheader("📊 Evaluation & Monitoring")
    
    evaluation_metrics = [
        {
            "category": "Retrieval Quality",
            "metrics": [
                {
                    "metric": "Retrieval Precision@K",
                    "description": "Fraction of retrieved documents that are relevant",
                    "formula": "Relevant Retrieved / Total Retrieved",
                    "code": """
def retrieval_precision_at_k(retrieved_docs, relevance_labels, k):
    retrieved_k = retrieved_docs[:k]
    relevant_count = sum(1 for doc in retrieved_k if relevance_labels.get(doc['id'], 0) > 0)
    return relevant_count / k if k > 0 else 0

# Usage
precision_5 = retrieval_precision_at_k(results, ground_truth, k=5)
"""
                },
                {
                    "metric": "Retrieval Recall@K", 
                    "description": "Fraction of relevant documents that are retrieved",
                    "formula": "Relevant Retrieved / Total Relevant",
                    "code": """
def retrieval_recall_at_k(retrieved_docs, relevance_labels, k):
    retrieved_k = retrieved_docs[:k]
    retrieved_relevant = sum(1 for doc in retrieved_k if relevance_labels.get(doc['id'], 0) > 0)
    total_relevant = sum(1 for score in relevance_labels.values() if score > 0)
    return retrieved_relevant / total_relevant if total_relevant > 0 else 0
"""
                },
                {
                    "metric": "Mean Reciprocal Rank (MRR)",
                    "description": "Average of reciprocal ranks of first relevant document",
                    "formula": "1/N * Σ(1/rank_of_first_relevant)",
                    "code": """
def mean_reciprocal_rank(retrieved_docs_list, relevance_labels_list):
    reciprocal_ranks = []
    
    for retrieved_docs, relevance_labels in zip(retrieved_docs_list, relevance_labels_list):
        for i, doc in enumerate(retrieved_docs):
            if relevance_labels.get(doc['id'], 0) > 0:
                reciprocal_ranks.append(1 / (i + 1))
                break
        else:
            reciprocal_ranks.append(0)  # No relevant document found
    
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
"""
                }
            ]
        },
        {
            "category": "Generation Quality",
            "metrics": [
                {
                    "metric": "Faithfulness",
                    "description": "How well the generated answer is supported by retrieved context",
                    "formula": "Fraction of answer claims supported by context",
                    "code": """
def calculate_faithfulness(answer, context, claim_extractor, entailment_model):
    # Extract claims from the answer
    claims = claim_extractor.extract_claims(answer)
    
    supported_claims = 0
    for claim in claims:
        # Check if claim is entailed by context
        entailment_score = entailment_model.predict(
            premise=context,
            hypothesis=claim
        )
        if entailment_score > 0.8:  # High confidence threshold
            supported_claims += 1
    
    return supported_claims / len(claims) if claims else 1.0
"""
                },
                {
                    "metric": "Answer Relevance",
                    "description": "How well the answer addresses the original question",
                    "formula": "Semantic similarity between question and answer",
                    "code": """
def calculate_answer_relevance(question, answer, similarity_model):
    # Encode question and answer
    question_embedding = similarity_model.encode([question])[0]
    answer_embedding = similarity_model.encode([answer])[0]
    
    # Calculate cosine similarity
    similarity = cosine_similarity(
        question_embedding.reshape(1, -1),
        answer_embedding.reshape(1, -1)
    )[0][0]
    
    return float(similarity)
"""
                },
                {
                    "metric": "Context Relevance",
                    "description": "How relevant the retrieved context is to the question",
                    "formula": "Average relevance of context chunks to question",
                    "code": """
def calculate_context_relevance(question, context_chunks, similarity_model):
    question_embedding = similarity_model.encode([question])[0]
    
    relevance_scores = []
    for chunk in context_chunks:
        chunk_embedding = similarity_model.encode([chunk])[0]
        similarity = cosine_similarity(
            question_embedding.reshape(1, -1),
            chunk_embedding.reshape(1, -1)
        )[0][0]
        relevance_scores.append(similarity)
    
    return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
"""
                }
            ]
        },
        {
            "category": "System Performance",
            "metrics": [
                {
                    "metric": "End-to-End Latency",
                    "description": "Total time from query to response",
                    "formula": "t_response - t_query",
                    "code": """
import time
from functools import wraps

def measure_latency(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        latency = end_time - start_time
        result['latency_ms'] = latency * 1000
        return result
    return wrapper

@measure_latency
def rag_query(question):
    retrieved_docs = retrieval_system.search(question)
    response = generation_system.generate(question, retrieved_docs)
    return {'answer': response, 'sources': retrieved_docs}
"""
                },
                {
                    "metric": "Throughput",
                    "description": "Number of queries processed per unit time",
                    "formula": "Queries / Time",
                    "code": """
import asyncio
import time

async def measure_throughput(rag_system, queries, concurrent_requests=10):
    start_time = time.time()
    
    # Process queries in batches
    semaphore = asyncio.Semaphore(concurrent_requests)
    
    async def process_query(query):
        async with semaphore:
            return await rag_system.query_async(query)
    
    # Execute all queries
    tasks = [process_query(query) for query in queries]
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    throughput = len(queries) / total_time
    
    return {
        'throughput_qps': throughput,
        'total_queries': len(queries),
        'total_time_s': total_time,
        'avg_latency_ms': (total_time / len(queries)) * 1000
    }
"""
                },
                {
                    "metric": "Resource Utilization",
                    "description": "CPU, memory, and GPU usage during operation",
                    "formula": "Resource usage / Total available",
                    "code": """
import psutil
import GPUtil
import threading
import time

class ResourceMonitor:
    def __init__(self):
        self.monitoring = False
        self.metrics = []
    
    def start_monitoring(self, interval=1):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        self.monitor_thread.join()
        return self.get_summary()
    
    def _monitor_loop(self, interval):
        while self.monitoring:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # GPU (if available)
            gpu_info = GPUtil.getGPUs()
            gpu_usage = gpu_info[0].load * 100 if gpu_info else 0
            gpu_memory = gpu_info[0].memoryUtil * 100 if gpu_info else 0
            
            self.metrics.append({
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'gpu_percent': gpu_usage,
                'gpu_memory_percent': gpu_memory
            })
            
            time.sleep(interval)
    
    def get_summary(self):
        if not self.metrics:
            return {}
        
        return {
            'avg_cpu': sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics),
            'avg_memory': sum(m['memory_percent'] for m in self.metrics) / len(self.metrics),
            'avg_gpu': sum(m['gpu_percent'] for m in self.metrics) / len(self.metrics),
            'avg_gpu_memory': sum(m['gpu_memory_percent'] for m in self.metrics) / len(self.metrics),
            'peak_memory': max(m['memory_percent'] for m in self.metrics),
            'samples': len(self.metrics)
        }

# Usage
monitor = ResourceMonitor()
monitor.start_monitoring()

# Run your RAG workload
results = run_rag_evaluation(test_queries)

resource_usage = monitor.stop_monitoring()
"""
                }
            ]
        }
    ]
    
    for category in evaluation_metrics:
        with st.expander(f"📊 {category['category']}"):
            for metric_info in category['metrics']:
                st.markdown(f"### {metric_info['metric']}")
                st.markdown(metric_info['description'])
                st.markdown(f"**Formula:** {metric_info['formula']}")
                st.code(metric_info['code'], language='python')
                st.markdown("---")
    
    # Monitoring dashboard concept
    st.markdown("### 📈 Monitoring Dashboard")
    
    # Simulate monitoring data
    monitoring_data = pd.DataFrame({
        'Timestamp': pd.date_range('2024-01-01', periods=24, freq='H'),
        'Queries_per_Hour': np.random.normal(1000, 200, 24),
        'Avg_Latency_ms': np.random.normal(800, 150, 24),
        'Success_Rate': np.random.normal(0.95, 0.02, 24),
        'Retrieval_Precision': np.random.normal(0.82, 0.05, 24)
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monitoring_data['Timestamp'],
        y=monitoring_data['Queries_per_Hour'],
        mode='lines',
        name='Queries/Hour',
        yaxis='y'
    ))
    
    fig.add_trace(go.Scatter(
        x=monitoring_data['Timestamp'],
        y=monitoring_data['Avg_Latency_ms'],
        mode='lines',
        name='Avg Latency (ms)',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="RAG System Monitoring Dashboard",
        xaxis_title="Time",
        yaxis=dict(title="Queries/Hour", side="left"),
        yaxis2=dict(title="Latency (ms)", side="right", overlaying="y"),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Applications and use cases
st.header("🎯 Applications & Use Cases")

use_case_tabs = st.tabs([
    "🏢 Enterprise",
    "🔬 Research & Education",
    "🛒 E-commerce",
    "⚕️ Healthcare"
])

with use_case_tabs[0]:
    st.subheader("🏢 Enterprise Applications")
    
    enterprise_cases = [
        {
            "use_case": "Customer Support Automation",
            "description": "Automated customer service with access to knowledge base",
            "knowledge_sources": [
                "Product documentation and manuals",
                "FAQ databases and help articles", 
                "Support ticket history and resolutions",
                "Company policies and procedures"
            ],
            "benefits": [
                "24/7 availability for customer support",
                "Consistent and accurate responses",
                "Reduced load on human support agents",
                "Faster resolution of common issues"
            ],
            "implementation_considerations": [
                "Integration with existing ticketing systems",
                "Escalation to human agents for complex issues",
                "Regular updates to knowledge base content",
                "Multi-language support for global customers"
            ],
            "success_metrics": [
                "First-contact resolution rate",
                "Average response time",
                "Customer satisfaction scores",
                "Agent workload reduction"
            ]
        },
        {
            "use_case": "Internal Knowledge Management",
            "description": "Employee access to organizational knowledge and expertise",
            "knowledge_sources": [
                "Internal wikis and documentation",
                "Meeting notes and project reports",
                "Expert knowledge and best practices",
                "Training materials and procedures"
            ],
            "benefits": [
                "Faster access to institutional knowledge",
                "Reduced knowledge silos between teams",
                "Onboarding acceleration for new employees",
                "Preservation of expert knowledge"
            ],
            "implementation_considerations": [
                "Access control and permissions management",
                "Integration with collaboration tools",
                "Content governance and quality control",
                "Regular knowledge base updates"
            ],
            "success_metrics": [
                "Time to find information",
                "Employee productivity metrics",
                "Knowledge sharing frequency",
                "New employee onboarding time"
            ]
        },
        {
            "use_case": "Compliance and Legal Research",
            "description": "Automated research and compliance checking",
            "knowledge_sources": [
                "Legal documents and regulations",
                "Compliance policies and procedures",
                "Case law and legal precedents",
                "Industry standards and guidelines"
            ],
            "benefits": [
                "Faster compliance research and verification",
                "Reduced risk of regulatory violations",
                "Consistent interpretation of regulations",
                "Cost reduction in legal research"
            ],
            "implementation_considerations": [
                "Accuracy and reliability requirements",
                "Regular updates for regulatory changes",
                "Audit trails for compliance verification",
                "Integration with legal case management"
            ],
            "success_metrics": [
                "Research time reduction",
                "Compliance violation reduction",
                "Legal research accuracy",
                "Cost per legal query"
            ]
        }
    ]
    
    for case in enterprise_cases:
        with st.expander(f"🏢 {case['use_case']}"):
            st.markdown(case['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Knowledge Sources:**")
                for source in case['knowledge_sources']:
                    st.markdown(f"• {source}")
                
                st.markdown("**Benefits:**")
                for benefit in case['benefits']:
                    st.markdown(f"• {benefit}")
            
            with col2:
                st.markdown("**Implementation Considerations:**")
                for consideration in case['implementation_considerations']:
                    st.markdown(f"• {consideration}")
                
                st.markdown("**Success Metrics:**")
                for metric in case['success_metrics']:
                    st.markdown(f"• {metric}")

with use_case_tabs[1]:
    st.subheader("🔬 Research & Education")
    
    research_cases = [
        {
            "application": "Academic Research Assistant",
            "description": "Support researchers with literature review and knowledge discovery",
            "features": [
                "Multi-database paper search and retrieval",
                "Citation network analysis and exploration",
                "Methodology comparison across studies",
                "Research gap identification"
            ],
            "example_workflow": """
1. Researcher queries: "Recent advances in graph neural networks for drug discovery"
2. System searches across:
   - PubMed for biomedical papers
   - arXiv for machine learning preprints
   - Google Scholar for broader academic coverage
3. Retrieves relevant papers and extracts key information
4. Synthesizes findings and identifies research trends
5. Suggests related work and potential research directions
"""
        },
        {
            "application": "Educational Content Creation",
            "description": "Generate personalized learning materials from knowledge bases",
            "features": [
                "Curriculum-aligned content generation",
                "Difficulty level adaptation",
                "Multi-modal content integration",
                "Assessment question generation"
            ],
            "example_workflow": """
1. Educator requests: "Create a lesson on photosynthesis for 8th grade"
2. System retrieves from:
   - Textbook content and curriculum standards
   - Scientific articles and research papers
   - Educational videos and interactive content
3. Adapts content for age-appropriate language
4. Generates lesson plan with activities and assessments
5. Provides additional resources for different learning styles
"""
        },
        {
            "application": "Scientific Literature Analysis",
            "description": "Large-scale analysis of scientific publications and trends",
            "features": [
                "Citation network analysis",
                "Research trend identification",
                "Cross-disciplinary connection discovery",
                "Hypothesis generation support"
            ],
            "example_workflow": """
1. Scientist asks: "What are the emerging connections between AI and climate science?"
2. System analyzes:
   - Recent publications in both fields
   - Citation patterns and collaborations
   - Methodological overlaps and innovations
3. Identifies emerging research areas and collaborations
4. Suggests potential interdisciplinary opportunities
5. Provides evidence-based insights for research planning
"""
        }
    ]
    
    for case in research_cases:
        with st.expander(f"🔬 {case['application']}"):
            st.markdown(case['description'])
            
            st.markdown("**Key Features:**")
            for feature in case['features']:
                st.markdown(f"• {feature}")
            
            st.markdown("**Example Workflow:**")
            st.code(case['example_workflow'])

with use_case_tabs[2]:
    st.subheader("🛒 E-commerce Applications")
    
    ecommerce_cases = [
        {
            "use_case": "Product Discovery & Recommendations",
            "description": "Enhanced product search and personalized recommendations",
            "rag_integration": [
                "Product catalogs with detailed specifications",
                "Customer reviews and ratings analysis",
                "Purchase history and behavioral data",
                "Seasonal trends and market analysis"
            ],
            "customer_benefits": [
                "Natural language product search",
                "Contextual product recommendations",
                "Comparison assistance across products",
                "Personalized shopping experience"
            ],
            "business_benefits": [
                "Increased conversion rates",
                "Higher average order value",
                "Reduced product return rates",
                "Improved customer lifetime value"
            ]
        },
        {
            "use_case": "Shopping Assistant Chatbot",
            "description": "AI-powered shopping guidance and support",
            "rag_integration": [
                "Real-time inventory and pricing data",
                "Product compatibility and bundling info",
                "Shipping and return policy information",
                "Promotional offers and discount codes"
            ],
            "customer_benefits": [
                "Instant answers to product questions",
                "Size and fit recommendations",
                "Price comparison and deal alerts",
                "Order tracking and support"
            ],
            "business_benefits": [
                "Reduced customer service costs",
                "24/7 customer support availability",
                "Increased customer engagement",
                "Data-driven product insights"
            ]
        },
        {
            "use_case": "Market Intelligence & Pricing",
            "description": "Competitive analysis and dynamic pricing optimization",
            "rag_integration": [
                "Competitor pricing and product data",
                "Market trends and consumer sentiment",
                "Supply chain and cost information",
                "Economic indicators and forecasts"
            ],
            "customer_benefits": [
                "Competitive pricing and better value",
                "Early access to trending products",
                "Informed purchase decisions",
                "Price alerts and notifications"
            ],
            "business_benefits": [
                "Optimized pricing strategies",
                "Competitive advantage maintenance",
                "Improved profit margins",
                "Market opportunity identification"
            ]
        }
    ]
    
    for case in ecommerce_cases:
        with st.expander(f"🛒 {case['use_case']}"):
            st.markdown(case['description'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**RAG Integration:**")
                for integration in case['rag_integration']:
                    st.markdown(f"• {integration}")
            
            with col2:
                st.markdown("**Customer Benefits:**")
                for benefit in case['customer_benefits']:
                    st.markdown(f"• {benefit}")
            
            with col3:
                st.markdown("**Business Benefits:**")
                for benefit in case['business_benefits']:
                    st.markdown(f"• {benefit}")

with use_case_tabs[3]:
    st.subheader("⚕️ Healthcare Applications")
    
    healthcare_cases = [
        {
            "application": "Clinical Decision Support",
            "description": "Evidence-based medical recommendations and diagnostic assistance",
            "knowledge_sources": [
                "Medical literature and research papers",
                "Clinical guidelines and protocols",
                "Drug interaction databases",
                "Patient history and case studies"
            ],
            "capabilities": [
                "Differential diagnosis suggestions",
                "Treatment protocol recommendations",
                "Drug dosage and interaction checking",
                "Evidence-based practice support"
            ],
            "safety_considerations": [
                "Human physician oversight required",
                "Clear uncertainty communication",
                "Audit trails for all recommendations",
                "Regular validation against outcomes"
            ]
        },
        {
            "application": "Medical Research & Literature Review",
            "description": "Accelerated medical research and evidence synthesis",
            "knowledge_sources": [
                "PubMed and medical databases",
                "Clinical trial registries",
                "Medical device and drug approvals",
                "Healthcare policy documents"
            ],
            "capabilities": [
                "Systematic literature reviews",
                "Meta-analysis support",
                "Research gap identification",
                "Clinical trial matching"
            ],
            "safety_considerations": [
                "Peer review and validation processes",
                "Transparent methodology reporting",
                "Bias detection and mitigation",
                "Ethical review compliance"
            ]
        },
        {
            "application": "Patient Education & Communication",
            "description": "Personalized patient information and education",
            "knowledge_sources": [
                "Patient education materials",
                "Medical condition databases",
                "Treatment option information",
                "Healthcare provider networks"
            ],
            "capabilities": [
                "Condition-specific education",
                "Treatment option explanations",
                "Medication instructions",
                "Lifestyle recommendation"
            ],
            "safety_considerations": [
                "Medical accuracy verification",
                "Clear disclaimers about medical advice",
                "Privacy and data protection",
                "Healthcare provider involvement"
            ]
        }
    ]
    
    for case in healthcare_cases:
        with st.expander(f"⚕️ {case['application']}"):
            st.markdown(case['description'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Knowledge Sources:**")
                for source in case['knowledge_sources']:
                    st.markdown(f"• {source}")
            
            with col2:
                st.markdown("**Capabilities:**")
                for capability in case['capabilities']:
                    st.markdown(f"• {capability}")
            
            with col3:
                st.markdown("**Safety Considerations:**")
                for consideration in case['safety_considerations']:
                    st.markdown(f"• {consideration}")

# Best practices
st.header("💡 RAG Best Practices")

best_practices = [
    "**Data Quality First**: Ensure high-quality, accurate, and up-to-date knowledge sources",
    "**Chunk Strategically**: Optimize document chunking for both retrieval and context quality",
    "**Monitor Performance**: Continuously track retrieval and generation quality metrics",
    "**Handle Edge Cases**: Plan for scenarios with no relevant results or conflicting information",
    "**Maintain Freshness**: Implement strategies for keeping knowledge bases current",
    "**Optimize for Speed**: Balance retrieval quality with response time requirements",
    "**Source Attribution**: Always provide clear citations and source references",
    "**Failure Gracefully**: Design fallback mechanisms for system component failures",
    "**Security & Privacy**: Implement proper access controls and data protection measures",
    "**Iterative Improvement**: Use user feedback and performance data to continuously enhance the system"
]

for practice in best_practices:
    st.markdown(f"• {practice}")

# Resources
st.header("📚 Learning Resources")

resources = [
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "type": "Research Paper",
        "description": "Original RAG paper by Facebook AI Research",
        "difficulty": "Advanced"
    },
    {
        "title": "RAG vs Fine-tuning: Which is the best tool to boost your LLM application?",
        "type": "Blog Post",
        "description": "Comparative analysis of RAG and fine-tuning approaches",
        "difficulty": "Intermediate"
    },
    {
        "title": "Building RAG Applications with LangChain",
        "type": "Tutorial",
        "description": "Practical guide to implementing RAG systems",
        "difficulty": "Beginner"
    },
    {
        "title": "Advanced RAG Techniques",
        "type": "Technical Guide",
        "description": "Deep dive into optimization and advanced patterns",
        "difficulty": "Advanced"
    }
]

for resource in resources:
    with st.expander(f"📖 {resource['title']}"):
        st.markdown(f"**Type:** {resource['type']}")
        st.markdown(f"**Description:** {resource['description']}")
        st.markdown(f"**Difficulty:** {resource['difficulty']}")