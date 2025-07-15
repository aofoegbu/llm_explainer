import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="LLM Explainability & Interpretability", page_icon="🔍", layout="wide")

st.title("🔍 LLM Explainability & Interpretability")
st.markdown("### Understanding How Large Language Models Make Decisions")

# Overview
st.header("🎯 Overview")
st.markdown("""
Explainability and interpretability in LLMs are crucial for building trust, debugging models, 
ensuring fairness, and meeting regulatory requirements. This involves understanding how models 
process information, make decisions, and generate outputs.
""")

# Core concepts
st.header("🧠 Core Concepts")

concept_tabs = st.tabs([
    "🔬 Interpretability Methods",
    "📊 Attention Analysis", 
    "🧪 Probing Techniques",
    "🎭 Behavioral Analysis"
])

with concept_tabs[0]:
    st.subheader("🔬 Interpretability Methods")
    
    st.markdown("""
    Various methods exist to understand and explain LLM behavior, each providing 
    different insights into model functioning.
    """)
    
    interpretability_methods = [
        {
            "method": "Gradient-based Explanations",
            "description": "Use gradients to understand input importance",
            "techniques": [
                "Saliency Maps",
                "Integrated Gradients", 
                "GradCAM for Transformers",
                "SmoothGrad"
            ],
            "strengths": [
                "Model-agnostic approach",
                "Computationally efficient",
                "Fine-grained attribution",
                "Well-established theory"
            ],
            "limitations": [
                "Can be noisy",
                "May not capture interactions",
                "Sensitive to model architecture",
                "Requires model access"
            ],
            "implementation": """
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

class GradientExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def saliency_map(self, text, target_token_id=None):
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs['input_ids']
        
        # Enable gradients for input embeddings
        embeddings = self.model.get_input_embeddings()
        input_embeds = embeddings(input_ids)
        input_embeds.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(inputs_embeds=input_embeds)
        logits = outputs.logits
        
        # Target: next token prediction or specific token
        if target_token_id is None:
            target_token_id = input_ids[0, -1]  # Last token
        
        # Get probability for target token
        target_logit = logits[0, -1, target_token_id]
        
        # Backward pass
        target_logit.backward()
        
        # Get gradients
        gradients = input_embeds.grad
        
        # Compute saliency (L2 norm of gradients)
        saliency = torch.norm(gradients, dim=-1).squeeze()
        
        return {
            'tokens': self.tokenizer.convert_ids_to_tokens(input_ids[0]),
            'saliency_scores': saliency.detach().numpy(),
            'input_ids': input_ids[0].tolist()
        }
    
    def integrated_gradients(self, text, target_token_id=None, steps=50):
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs['input_ids']
        
        embeddings = self.model.get_input_embeddings()
        input_embeds = embeddings(input_ids)
        
        # Baseline: zero embeddings
        baseline = torch.zeros_like(input_embeds)
        
        # Interpolation path
        alphas = torch.linspace(0, 1, steps).view(-1, 1, 1, 1)
        
        gradients = []
        for alpha in alphas:
            # Interpolated input
            interpolated = baseline + alpha * (input_embeds - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(inputs_embeds=interpolated)
            logits = outputs.logits
            
            if target_token_id is None:
                target_token_id = input_ids[0, -1]
            
            target_logit = logits[0, -1, target_token_id]
            
            # Backward pass
            grad = torch.autograd.grad(target_logit, interpolated)[0]
            gradients.append(grad)
        
        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)
        
        # Integrated gradients
        integrated_grads = (input_embeds - baseline) * avg_gradients
        attribution = torch.norm(integrated_grads, dim=-1).squeeze()
        
        return {
            'tokens': self.tokenizer.convert_ids_to_tokens(input_ids[0]),
            'attribution_scores': attribution.detach().numpy(),
            'input_ids': input_ids[0].tolist()
        }

# Usage example
explainer = GradientExplainer(model, tokenizer)
result = explainer.integrated_gradients("The capital of France is Paris")

# Visualize results
def visualize_attribution(tokens, scores):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(scores)))
    
    bars = plt.bar(range(len(tokens)), scores, color=colors)
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.ylabel('Attribution Score')
    plt.title('Token Attribution Scores')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, 
                              norm=plt.Normalize(vmin=min(scores), vmax=max(scores)))
    plt.colorbar(sm)
    
    plt.tight_layout()
    plt.show()
"""
        },
        {
            "method": "Feature Attribution",
            "description": "Identify which input features contribute to outputs",
            "techniques": [
                "LIME (Local Interpretable Model-agnostic Explanations)",
                "SHAP (SHapley Additive exPlanations)",
                "Occlusion Analysis",
                "Perturbation-based Methods"
            ],
            "strengths": [
                "Model-agnostic",
                "Intuitive explanations",
                "Well-validated approaches",
                "Good for local explanations"
            ],
            "limitations": [
                "Computationally expensive",
                "May not capture global behavior",
                "Sensitive to perturbation strategy",
                "Assumes feature independence"
            ],
            "implementation": """
import numpy as np
from sklearn.linear_model import LinearRegression
import itertools

class LIMEExplainer:
    def __init__(self, model, tokenizer, num_samples=1000):
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
    
    def explain_prediction(self, text, target_class=None):
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        
        # Generate perturbed samples
        samples, labels = self.generate_samples(text, tokens)
        
        # Train local linear model
        local_model = LinearRegression()
        local_model.fit(samples, labels)
        
        # Get feature importance
        importance = local_model.coef_
        
        return {
            'tokens': tokens,
            'importance': importance,
            'local_score': local_model.score(samples, labels)
        }
    
    def generate_samples(self, original_text, tokens):
        samples = []
        labels = []
        
        # Get original prediction
        original_pred = self.predict_text(original_text)
        
        for _ in range(self.num_samples):
            # Randomly mask tokens
            mask = np.random.binomial(1, 0.5, len(tokens))
            
            # Create perturbed text
            perturbed_tokens = [token if mask[i] else '[MASK]' 
                              for i, token in enumerate(tokens)]
            perturbed_text = self.tokenizer.convert_tokens_to_string(perturbed_tokens)
            
            # Get prediction for perturbed text
            pred = self.predict_text(perturbed_text)
            
            samples.append(mask)
            labels.append(pred)
        
        return np.array(samples), np.array(labels)
    
    def predict_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Return probability of positive class (simplified)
            probs = F.softmax(logits, dim=-1)
            return probs[0, -1, :].max().item()

class SHAPExplainer:
    def __init__(self, model, tokenizer, background_texts=None):
        self.model = model
        self.tokenizer = tokenizer
        self.background_texts = background_texts or []
    
    def shapley_values(self, text, max_features=10):
        tokens = self.tokenizer.tokenize(text)
        n_features = min(len(tokens), max_features)
        
        # Get original prediction
        original_pred = self.predict_text(text)
        
        # Calculate Shapley values
        shapley_values = {}
        
        for i in range(n_features):
            marginal_contributions = []
            
            # Generate all possible coalitions
            for r in range(n_features):
                for coalition in itertools.combinations(range(n_features), r):
                    if i not in coalition:
                        # Coalition without feature i
                        without_i = self.create_coalition_text(text, tokens, coalition)
                        pred_without = self.predict_text(without_i)
                        
                        # Coalition with feature i
                        with_i = self.create_coalition_text(text, tokens, coalition + (i,))
                        pred_with = self.predict_text(with_i)
                        
                        # Marginal contribution
                        marginal = pred_with - pred_without
                        marginal_contributions.append(marginal)
            
            # Average marginal contribution is Shapley value
            shapley_values[i] = np.mean(marginal_contributions)
        
        return {
            'tokens': tokens[:n_features],
            'shapley_values': list(shapley_values.values())
        }
    
    def create_coalition_text(self, original_text, tokens, coalition):
        # Create text with only coalition features
        coalition_tokens = ['[MASK]'] * len(tokens)
        for idx in coalition:
            coalition_tokens[idx] = tokens[idx]
        
        return self.tokenizer.convert_tokens_to_string(coalition_tokens)
    
    def predict_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            return probs[0, -1, :].max().item()
"""
        },
        {
            "method": "Concept-based Explanations",
            "description": "Explain predictions in terms of human-interpretable concepts",
            "techniques": [
                "TCAV (Testing with Concept Activation Vectors)",
                "Concept Bottleneck Models",
                "Network Dissection",
                "Compositional Explanations"
            ],
            "strengths": [
                "Human-interpretable concepts",
                "Global model understanding",
                "Causal reasoning support",
                "Domain knowledge integration"
            ],
            "limitations": [
                "Requires concept annotation",
                "May miss important concepts",
                "Computational overhead",
                "Concept definition challenges"
            ],
            "implementation": """
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

class TCAVExplainer:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.activations = {}
        
        # Register forward hook
        self.register_hooks()
    
    def register_hooks(self):
        def hook_fn(module, input, output):
            self.activations[self.layer_name] = output.detach()
        
        # Find the layer by name
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook_fn)
                break
    
    def get_concept_vector(self, concept_examples, random_examples):
        # Get activations for concept examples
        concept_activations = []
        for example in concept_examples:
            self.forward_pass(example)
            activation = self.activations[self.layer_name]
            concept_activations.append(activation.mean(dim=1).flatten())
        
        # Get activations for random examples
        random_activations = []
        for example in random_examples:
            self.forward_pass(example)
            activation = self.activations[self.layer_name]
            random_activations.append(activation.mean(dim=1).flatten())
        
        # Create labels
        X = torch.cat(concept_activations + random_activations)
        y = [1] * len(concept_activations) + [0] * len(random_activations)
        
        # Train linear classifier
        classifier = LogisticRegression()
        classifier.fit(X.numpy(), y)
        
        # Concept vector is the decision boundary normal
        concept_vector = torch.tensor(classifier.coef_[0])
        return concept_vector / torch.norm(concept_vector)
    
    def tcav_score(self, test_examples, concept_vector):
        gradients = []
        
        for example in test_examples:
            # Forward pass with gradient computation
            self.model.zero_grad()
            output = self.forward_pass(example)
            
            # Get target logit (e.g., prediction for specific class)
            target_logit = output.logits.max()
            
            # Backward pass
            target_logit.backward()
            
            # Get gradients for the layer
            for name, param in self.model.named_parameters():
                if self.layer_name in name:
                    gradient = param.grad.flatten()
                    gradients.append(gradient)
                    break
        
        # Calculate TCAV score
        directional_derivatives = []
        for grad in gradients:
            dd = torch.dot(grad, concept_vector)
            directional_derivatives.append(dd)
        
        # TCAV score: fraction of positive directional derivatives
        positive_count = sum(1 for dd in directional_derivatives if dd > 0)
        tcav_score = positive_count / len(directional_derivatives)
        
        return tcav_score
    
    def forward_pass(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        return self.model(**inputs)

# Usage example
tcav = TCAVExplainer(model, layer_name="transformer.h.11")

# Define concept examples (e.g., positive sentiment)
positive_examples = [
    "I love this movie!",
    "This is amazing!",
    "Great job!",
    "Fantastic work!"
]

random_examples = [
    "The weather is nice.",
    "I went to the store.",
    "The book is on the table.",
    "Time flies quickly."
]

# Get concept vector for "positivity"
concept_vector = tcav.get_concept_vector(positive_examples, random_examples)

# Test TCAV score on movie reviews
test_examples = [
    "This movie was terrible.",
    "I enjoyed the film.",
    "The plot was confusing.",
    "Outstanding performance!"
]

tcav_score = tcav.tcav_score(test_examples, concept_vector)
print(f"TCAV Score for positivity concept: {tcav_score:.3f}")
"""
        }
    ]
    
    for method in interpretability_methods:
        with st.expander(f"🔬 {method['method']}"):
            st.markdown(method['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Techniques:**")
                for technique in method['techniques']:
                    st.markdown(f"• {technique}")
                
                st.markdown("**Strengths:**")
                for strength in method['strengths']:
                    st.markdown(f"• {strength}")
            
            with col2:
                st.markdown("**Limitations:**")
                for limitation in method['limitations']:
                    st.markdown(f"• {limitation}")
            
            st.markdown("**Implementation Example:**")
            st.code(method['implementation'], language='python')

with concept_tabs[1]:
    st.subheader("📊 Attention Analysis")
    
    st.markdown("""
    Attention mechanisms provide direct insights into what the model focuses on 
    when making predictions, offering interpretable explanations.
    """)
    
    attention_techniques = [
        {
            "technique": "Attention Visualization",
            "description": "Visualize attention weights between tokens",
            "applications": [
                "Understanding token relationships",
                "Identifying important context",
                "Debugging model behavior",
                "Validating domain knowledge"
            ],
            "code": """
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer

class AttentionVisualizer:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
    
    def get_attention_weights(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            attentions = outputs.attentions  # Tuple of attention weights
        
        return {
            'tokens': self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
            'attentions': [attn.squeeze().numpy() for attn in attentions],
            'input_ids': inputs['input_ids'][0]
        }
    
    def plot_attention_heatmap(self, text, layer=0, head=0):
        result = self.get_attention_weights(text)
        tokens = result['tokens']
        attention = result['attentions'][layer][head]  # [seq_len, seq_len]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='Blues',
            annot=False,
            fmt='.2f'
        )
        plt.title(f'Attention Weights - Layer {layer}, Head {head}')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def plot_attention_patterns(self, text, max_layers=4):
        result = self.get_attention_weights(text)
        tokens = result['tokens']
        attentions = result['attentions']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i in range(min(max_layers, len(attentions))):
            # Average across attention heads
            avg_attention = attentions[i].mean(axis=0)
            
            sns.heatmap(
                avg_attention,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='Blues',
                ax=axes[i],
                cbar=True
            )
            axes[i].set_title(f'Layer {i} - Average Attention')
            axes[i].set_xlabel('Key Tokens')
            axes[i].set_ylabel('Query Tokens')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_attention_distribution(self, text):
        result = self.get_attention_weights(text)
        tokens = result['tokens']
        attentions = result['attentions']
        
        # Calculate attention statistics
        stats = {}
        
        for layer_idx, layer_attention in enumerate(attentions):
            layer_stats = {
                'max_attention': layer_attention.max(),
                'min_attention': layer_attention.min(),
                'mean_attention': layer_attention.mean(),
                'std_attention': layer_attention.std(),
                'sparsity': (layer_attention < 0.01).sum() / layer_attention.numel()
            }
            stats[f'layer_{layer_idx}'] = layer_stats
        
        return stats

# Usage
visualizer = AttentionVisualizer("bert-base-uncased")
text = "The quick brown fox jumps over the lazy dog"

# Plot attention heatmap
visualizer.plot_attention_heatmap(text, layer=2, head=0)

# Analyze attention patterns
visualizer.plot_attention_patterns(text)

# Get attention statistics
stats = visualizer.analyze_attention_distribution(text)
print("Attention Statistics:", stats)
"""
        },
        {
            "technique": "Attention Flow Analysis",
            "description": "Track information flow through attention layers",
            "applications": [
                "Understanding information propagation",
                "Identifying bottlenecks",
                "Analyzing compositional reasoning",
                "Debugging attention collapse"
            ],
            "code": """
class AttentionFlowAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def compute_attention_flow(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            attentions = outputs.attentions
        
        # Compute attention flow matrix
        seq_len = attentions[0].size(-1)
        num_layers = len(attentions)
        
        # Initialize flow matrix: [layer, from_token, to_token]
        flow_matrix = torch.zeros(num_layers, seq_len, seq_len)
        
        for layer_idx, attention in enumerate(attentions):
            # Average across heads and batch
            avg_attention = attention.mean(dim=(0, 1))  # [seq_len, seq_len]
            flow_matrix[layer_idx] = avg_attention
        
        return flow_matrix, self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    def analyze_information_flow(self, text, source_token_idx, target_token_idx):
        flow_matrix, tokens = self.compute_attention_flow(text)
        
        # Track flow from source to target across layers
        flow_path = []
        
        for layer in range(flow_matrix.size(0)):
            attention_weight = flow_matrix[layer, target_token_idx, source_token_idx]
            flow_path.append(attention_weight.item())
        
        return {
            'source_token': tokens[source_token_idx],
            'target_token': tokens[target_token_idx],
            'flow_path': flow_path,
            'total_flow': sum(flow_path),
            'max_flow_layer': np.argmax(flow_path)
        }
    
    def detect_attention_patterns(self, text):
        flow_matrix, tokens = self.compute_attention_flow(text)
        patterns = {}
        
        # Detect common patterns
        for layer_idx, attention in enumerate(flow_matrix):
            # Self-attention pattern
            self_attention = torch.diag(attention).mean()
            patterns[f'layer_{layer_idx}_self_attention'] = self_attention.item()
            
            # Local attention pattern (neighboring tokens)
            local_attention = 0
            for i in range(len(tokens) - 1):
                local_attention += attention[i, i+1] + attention[i+1, i]
            patterns[f'layer_{layer_idx}_local_attention'] = local_attention.item() / (2 * (len(tokens) - 1))
            
            # Global attention pattern (distant tokens)
            global_attention = 0
            count = 0
            for i in range(len(tokens)):
                for j in range(len(tokens)):
                    if abs(i - j) > 3:  # Distant tokens
                        global_attention += attention[i, j]
                        count += 1
            patterns[f'layer_{layer_idx}_global_attention'] = global_attention.item() / count if count > 0 else 0
        
        return patterns
    
    def visualize_flow_graph(self, text, min_weight=0.1):
        flow_matrix, tokens = self.compute_attention_flow(text)
        
        # Create networkx graph
        import networkx as nx
        
        G = nx.DiGraph()
        
        # Add nodes
        for i, token in enumerate(tokens):
            G.add_node(i, label=token)
        
        # Add edges for each layer
        for layer_idx, attention in enumerate(flow_matrix):
            for i in range(len(tokens)):
                for j in range(len(tokens)):
                    if attention[i, j] > min_weight and i != j:
                        G.add_edge(j, i, weight=attention[i, j].item(), layer=layer_idx)
        
        return G
"""
        }
    ]
    
    for technique in attention_techniques:
        with st.expander(f"📊 {technique['technique']}"):
            st.markdown(technique['description'])
            
            st.markdown("**Applications:**")
            for app in technique['applications']:
                st.markdown(f"• {app}")
            
            st.markdown("**Implementation:**")
            st.code(technique['code'], language='python')

# Attention visualization example
st.markdown("### 📊 Interactive Attention Visualization")

# Sample attention data for visualization
sample_tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
sample_attention = np.random.rand(len(sample_tokens), len(sample_tokens))
sample_attention = sample_attention / sample_attention.sum(axis=1, keepdims=True)

fig = go.Figure(data=go.Heatmap(
    z=sample_attention,
    x=sample_tokens,
    y=sample_tokens,
    colorscale='Blues',
    hoverongaps=False
))

fig.update_layout(
    title="Sample Attention Heatmap",
    xaxis_title="Key Tokens",
    yaxis_title="Query Tokens",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

with concept_tabs[2]:
    st.subheader("🧪 Probing Techniques")
    
    st.markdown("""
    Probing techniques use auxiliary tasks to understand what linguistic and 
    conceptual knowledge is encoded in different layers of LLMs.
    """)
    
    probing_methods = [
        {
            "method": "Linear Probing",
            "description": "Train linear classifiers on frozen model representations",
            "purpose": "Test if specific information is linearly accessible",
            "implementation": """
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class LinearProbe:
    def __init__(self, model, tokenizer, layer_name):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_name = layer_name
        self.representations = {}
        
        # Register forward hook
        self.register_hook()
    
    def register_hook(self):
        def hook_fn(module, input, output):
            self.representations[self.layer_name] = output.detach()
        
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook_fn)
                break
    
    def extract_representations(self, texts, token_positions=None):
        representations = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            
            with torch.no_grad():
                _ = self.model(**inputs)
                
                # Get representation from the specified layer
                layer_output = self.representations[self.layer_name]
                
                if token_positions is None:
                    # Use average pooling
                    representation = layer_output.mean(dim=1).squeeze()
                else:
                    # Use specific token positions
                    representation = layer_output[0, token_positions, :].mean(dim=0)
                
                representations.append(representation.numpy())
        
        return np.array(representations)
    
    def probe_syntax(self, sentences, pos_tags):
        # Probe for part-of-speech information
        X = self.extract_representations(sentences)
        
        # Flatten POS tags if needed
        y = []
        for tags in pos_tags:
            if isinstance(tags, list):
                y.extend(tags)
            else:
                y.append(tags)
        
        # Train linear classifier
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X, y)
        
        # Evaluate
        predictions = probe.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        return {
            'accuracy': accuracy,
            'probe_model': probe,
            'feature_importance': probe.coef_
        }
    
    def probe_semantics(self, texts, semantic_labels):
        # Probe for semantic information (e.g., sentiment, topics)
        X = self.extract_representations(texts)
        y = semantic_labels
        
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X, y)
        
        predictions = probe.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        return {
            'accuracy': accuracy,
            'probe_model': probe,
            'predictions': predictions
        }
    
    def probe_world_knowledge(self, factual_statements, truth_labels):
        # Probe for world knowledge
        X = self.extract_representations(factual_statements)
        y = truth_labels  # 1 for true statements, 0 for false
        
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X, y)
        
        predictions = probe.predict(X)
        accuracy = accuracy_score(y, predictions)
        
        return {
            'accuracy': accuracy,
            'probe_model': probe,
            'confidence': probe.predict_proba(X)
        }

# Usage example
probe = LinearProbe(model, tokenizer, "transformer.h.6")

# Probe for POS tagging
sentences = [
    "The cat sits on the mat",
    "Dogs run quickly through parks",
    "She writes beautiful poetry daily"
]
pos_tags = [
    ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"],
    ["NOUN", "VERB", "ADV", "ADP", "NOUN"],
    ["PRON", "VERB", "ADJ", "NOUN", "ADV"]
]

syntax_results = probe.probe_syntax(sentences, pos_tags)
print(f"POS Tagging Accuracy: {syntax_results['accuracy']:.3f}")

# Probe for sentiment
texts = ["I love this movie", "This is terrible", "It's okay I guess"]
sentiments = ["positive", "negative", "neutral"]

semantic_results = probe.probe_semantics(texts, sentiments)
print(f"Sentiment Accuracy: {semantic_results['accuracy']:.3f}")
"""
        },
        {
            "method": "Diagnostic Classification",
            "description": "Use diagnostic tasks to test specific capabilities",
            "purpose": "Evaluate model understanding of linguistic phenomena",
            "implementation": """
class DiagnosticProbe:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def test_syntactic_agreement(self, test_cases):
        # Test subject-verb agreement
        results = []
        
        for case in test_cases:
            correct_sentence = case['correct']
            incorrect_sentence = case['incorrect']
            
            # Get model perplexity for both sentences
            correct_ppl = self.compute_perplexity(correct_sentence)
            incorrect_ppl = self.compute_perplexity(incorrect_sentence)
            
            # Model should assign lower perplexity to correct sentence
            correct_prediction = correct_ppl < incorrect_ppl
            
            results.append({
                'correct_sentence': correct_sentence,
                'incorrect_sentence': incorrect_sentence,
                'correct_perplexity': correct_ppl,
                'incorrect_perplexity': incorrect_ppl,
                'model_correct': correct_prediction
            })
        
        accuracy = sum(1 for r in results if r['model_correct']) / len(results)
        return {'accuracy': accuracy, 'detailed_results': results}
    
    def test_semantic_understanding(self, analogy_pairs):
        # Test semantic analogies: A is to B as C is to ?
        results = []
        
        for pair in analogy_pairs:
            a, b, c, expected_d = pair['A'], pair['B'], pair['C'], pair['D']
            
            # Create prompt
            prompt = f"{a} is to {b} as {c} is to"
            
            # Generate completion
            generated = self.generate_completion(prompt, max_tokens=5)
            
            # Check if generated completion matches expected
            correct = expected_d.lower() in generated.lower()
            
            results.append({
                'analogy': f"{a}:{b} :: {c}:{expected_d}",
                'generated': generated,
                'correct': correct
            })
        
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        return {'accuracy': accuracy, 'detailed_results': results}
    
    def test_commonsense_reasoning(self, reasoning_questions):
        # Test commonsense reasoning capabilities
        results = []
        
        for question in reasoning_questions:
            prompt = question['question']
            correct_answer = question['correct_answer']
            
            # Generate answer
            generated_answer = self.generate_completion(prompt, max_tokens=20)
            
            # Simple matching (in practice, use more sophisticated evaluation)
            correct = any(ans.lower() in generated_answer.lower() 
                         for ans in correct_answer if isinstance(correct_answer, list))
            
            if not isinstance(correct_answer, list):
                correct = correct_answer.lower() in generated_answer.lower()
            
            results.append({
                'question': prompt,
                'correct_answer': correct_answer,
                'generated_answer': generated_answer,
                'correct': correct
            })
        
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        return {'accuracy': accuracy, 'detailed_results': results}
    
    def compute_perplexity(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss)
        
        return perplexity.item()
    
    def generate_completion(self, prompt, max_tokens=50):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):].strip()
        
        return completion

# Usage example
diagnostic = DiagnosticProbe(model, tokenizer)

# Test syntactic agreement
agreement_cases = [
    {
        'correct': "The cat sits on the mat",
        'incorrect': "The cat sit on the mat"
    },
    {
        'correct': "The dogs are running",
        'incorrect': "The dogs is running"
    }
]

syntax_results = diagnostic.test_syntactic_agreement(agreement_cases)
print(f"Syntactic Agreement Accuracy: {syntax_results['accuracy']:.3f}")

# Test semantic analogies
analogies = [
    {'A': 'man', 'B': 'king', 'C': 'woman', 'D': 'queen'},
    {'A': 'cat', 'B': 'kitten', 'C': 'dog', 'D': 'puppy'}
]

semantic_results = diagnostic.test_semantic_understanding(analogies)
print(f"Semantic Analogy Accuracy: {semantic_results['accuracy']:.3f}")
"""
        }
    ]
    
    for method in probing_methods:
        with st.expander(f"🧪 {method['method']}"):
            st.markdown(method['description'])
            st.markdown(f"**Purpose:** {method['purpose']}")
            
            st.markdown("**Implementation:**")
            st.code(method['implementation'], language='python')

# Probing results visualization
st.markdown("### 📊 Probing Results by Layer")

# Simulate probing results across layers
layers = list(range(12))
pos_accuracy = [0.45, 0.62, 0.78, 0.85, 0.89, 0.91, 0.88, 0.84, 0.79, 0.75, 0.71, 0.68]
sentiment_accuracy = [0.52, 0.58, 0.65, 0.72, 0.79, 0.83, 0.87, 0.91, 0.93, 0.91, 0.89, 0.86]
world_knowledge = [0.48, 0.51, 0.55, 0.59, 0.64, 0.69, 0.74, 0.78, 0.82, 0.84, 0.83, 0.81]

probing_df = pd.DataFrame({
    'Layer': layers,
    'POS Tagging': pos_accuracy,
    'Sentiment': sentiment_accuracy,
    'World Knowledge': world_knowledge
})

fig = px.line(probing_df, x='Layer', y=['POS Tagging', 'Sentiment', 'World Knowledge'],
              title="Probing Accuracy Across Transformer Layers",
              labels={'value': 'Accuracy', 'variable': 'Probing Task'})

st.plotly_chart(fig, use_container_width=True)

with concept_tabs[3]:
    st.subheader("🎭 Behavioral Analysis")
    
    st.markdown("""
    Behavioral analysis examines how models respond to different inputs and 
    situations to understand their capabilities and limitations.
    """)
    
    # Add comprehensive behavioral analysis content here
    behavioral_tests = [
        {
            "test": "Consistency Testing",
            "description": "Test model consistency across paraphrased inputs",
            "metrics": ["Response consistency", "Semantic stability", "Factual consistency"],
            "example": """
def test_consistency(model, tokenizer, base_question, paraphrases):
    base_response = generate_response(model, tokenizer, base_question)
    
    consistency_scores = []
    for paraphrase in paraphrases:
        para_response = generate_response(model, tokenizer, paraphrase)
        
        # Compute semantic similarity
        similarity = compute_semantic_similarity(base_response, para_response)
        consistency_scores.append(similarity)
    
    return {
        'base_question': base_question,
        'base_response': base_response,
        'consistency_scores': consistency_scores,
        'average_consistency': np.mean(consistency_scores)
    }
"""
        },
        {
            "test": "Robustness Testing", 
            "description": "Test model robustness to adversarial inputs",
            "metrics": ["Attack success rate", "Output quality degradation", "Safety violations"],
            "example": """
def test_robustness(model, tokenizer, clean_inputs, adversarial_inputs):
    results = []
    
    for clean, adversarial in zip(clean_inputs, adversarial_inputs):
        clean_response = generate_response(model, tokenizer, clean)
        adv_response = generate_response(model, tokenizer, adversarial)
        
        # Check if adversarial input succeeded
        attack_success = check_attack_success(clean_response, adv_response)
        
        results.append({
            'clean_input': clean,
            'adversarial_input': adversarial,
            'clean_response': clean_response,
            'adversarial_response': adv_response,
            'attack_success': attack_success
        })
    
    return results
"""
        }
    ]
    
    for test in behavioral_tests:
        with st.expander(f"🎭 {test['test']}"):
            st.markdown(test['description'])
            
            st.markdown("**Key Metrics:**")
            for metric in test['metrics']:
                st.markdown(f"• {metric}")
            
            st.markdown("**Example Implementation:**")
            st.code(test['example'], language='python')

# Best practices
st.header("💡 Explainability Best Practices")

best_practices = [
    "**Multiple Methods**: Use complementary explainability techniques for comprehensive understanding",
    "**Layer-wise Analysis**: Examine different layers to understand information processing progression",
    "**Task-specific Probing**: Design probes specific to your domain and use case",
    "**Human Evaluation**: Validate explanations with domain experts and end users",
    "**Causal Analysis**: Go beyond correlation to understand causal relationships",
    "**Adversarial Testing**: Test explanation robustness with adversarial inputs",
    "**Documentation**: Maintain clear documentation of explanation methods and limitations",
    "**Iterative Improvement**: Continuously refine explanation methods based on findings",
    "**Stakeholder Communication**: Present explanations appropriate for different audiences",
    "**Ethical Considerations**: Consider fairness and bias implications of explanations"
]

for practice in best_practices:
    st.markdown(f"• {practice}")

# Resources
st.header("📚 Learning Resources")

resources = [
    {
        "title": "Attention is All You Need",
        "type": "Research Paper",
        "description": "Original transformer paper with attention mechanism details",
        "difficulty": "Advanced"
    },
    {
        "title": "Visualizing Attention in Transformer Models",
        "type": "Tutorial",
        "description": "Practical guide to attention visualization techniques",
        "difficulty": "Intermediate"
    },
    {
        "title": "Probing Tasks for Neural Networks",
        "type": "Survey Paper",
        "description": "Comprehensive overview of probing methodologies",
        "difficulty": "Advanced"
    },
    {
        "title": "Interpretable Machine Learning",
        "type": "Book",
        "description": "General principles of ML interpretability",
        "difficulty": "Beginner"
    }
]

for resource in resources:
    with st.expander(f"📖 {resource['title']}"):
        st.markdown(f"**Type:** {resource['type']}")
        st.markdown(f"**Description:** {resource['description']}")
        st.markdown(f"**Difficulty:** {resource['difficulty']}")