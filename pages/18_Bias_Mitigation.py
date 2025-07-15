import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Bias Mitigation in LLMs", page_icon="⚖️", layout="wide")

st.title("⚖️ Bias Mitigation in LLMs")
st.markdown("### Building Fair and Equitable Language Models")

# Overview
st.header("🎯 Overview")
st.markdown("""
Bias mitigation in Large Language Models is crucial for ensuring fair, equitable, and safe AI systems. 
This involves identifying, measuring, and reducing various forms of bias throughout the model lifecycle 
from data collection to deployment and monitoring.
""")

# Core concepts
st.header("🧠 Core Concepts")

concept_tabs = st.tabs([
    "🔍 Bias Types & Detection",
    "📊 Measurement Techniques", 
    "🛠️ Mitigation Strategies",
    "📈 Monitoring & Evaluation"
])

with concept_tabs[0]:
    st.subheader("🔍 Types of Bias and Detection Methods")
    
    st.markdown("""
    Understanding different types of bias is the first step toward building fairer AI systems.
    Each type requires specific detection and mitigation approaches.
    """)
    
    bias_types = [
        {
            "type": "Data Bias",
            "description": "Bias present in training data reflecting societal inequalities",
            "subtypes": [
                "Historical bias - Past discriminatory practices in data",
                "Representation bias - Underrepresentation of certain groups",
                "Measurement bias - Systematic errors in data collection",
                "Aggregation bias - Inappropriate grouping of different populations"
            ],
            "examples": [
                "Gender bias in job descriptions",
                "Racial bias in criminal justice data",
                "Geographic bias in language samples",
                "Socioeconomic bias in educational data"
            ],
            "detection_methods": """
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class DataBiasDetector:
    def __init__(self, dataset):
        self.dataset = dataset
        
    def analyze_representation_bias(self, demographic_column, target_column=None):
        # Analyze representation across demographic groups
        demographic_counts = self.dataset[demographic_column].value_counts()
        total_samples = len(self.dataset)
        
        representation_analysis = {}
        for group, count in demographic_counts.items():
            percentage = (count / total_samples) * 100
            representation_analysis[group] = {
                'count': count,
                'percentage': percentage,
                'underrepresented': percentage < 10  # Threshold for underrepresentation
            }
        
        # If target column provided, analyze conditional representation
        if target_column:
            conditional_analysis = {}
            for target_value in self.dataset[target_column].unique():
                subset = self.dataset[self.dataset[target_column] == target_value]
                group_dist = subset[demographic_column].value_counts(normalize=True) * 100
                conditional_analysis[target_value] = group_dist.to_dict()
            
            representation_analysis['conditional_distribution'] = conditional_analysis
        
        return representation_analysis
    
    def detect_historical_bias(self, text_column, bias_keywords):
        # Detect potentially biased language patterns
        bias_detection = {}
        
        for category, keywords in bias_keywords.items():
            category_bias = {
                'keyword_counts': {},
                'biased_samples': [],
                'bias_score': 0
            }
            
            for keyword in keywords:
                # Count occurrences of biased keywords
                keyword_count = self.dataset[text_column].str.contains(
                    keyword, case=False, na=False
                ).sum()
                
                category_bias['keyword_counts'][keyword] = keyword_count
                
                if keyword_count > 0:
                    # Get samples containing the keyword
                    biased_samples = self.dataset[
                        self.dataset[text_column].str.contains(keyword, case=False, na=False)
                    ][text_column].tolist()
                    category_bias['biased_samples'].extend(biased_samples[:5])  # Limit examples
            
            # Calculate bias score for this category
            total_keyword_mentions = sum(category_bias['keyword_counts'].values())
            category_bias['bias_score'] = total_keyword_mentions / len(self.dataset)
            
            bias_detection[category] = category_bias
        
        return bias_detection
    
    def analyze_intersectional_bias(self, demographic_columns, target_column):
        # Analyze bias at intersections of multiple demographic attributes
        intersectional_analysis = {}
        
        # Create intersection groups
        intersection_groups = self.dataset.groupby(demographic_columns)
        
        for group_values, group_data in intersection_groups:
            group_key = ' + '.join([f"{col}:{val}" for col, val in zip(demographic_columns, group_values)])
            
            # Analyze outcomes for this intersection group
            if target_column in group_data.columns:
                target_distribution = group_data[target_column].value_counts(normalize=True)
                
                intersectional_analysis[group_key] = {
                    'size': len(group_data),
                    'percentage_of_total': len(group_data) / len(self.dataset) * 100,
                    'target_distribution': target_distribution.to_dict(),
                    'sample_data': group_data.head(3).to_dict('records')
                }
        
        return intersectional_analysis
    
    def statistical_parity_test(self, demographic_column, target_column):
        # Test for statistical parity across demographic groups
        overall_positive_rate = (self.dataset[target_column] == 1).mean()
        
        parity_analysis = {}
        
        for group in self.dataset[demographic_column].unique():
            group_data = self.dataset[self.dataset[demographic_column] == group]
            group_positive_rate = (group_data[target_column] == 1).mean()
            
            # Calculate disparity
            disparity = group_positive_rate - overall_positive_rate
            disparity_ratio = group_positive_rate / overall_positive_rate if overall_positive_rate > 0 else float('inf')
            
            parity_analysis[group] = {
                'positive_rate': group_positive_rate,
                'disparity': disparity,
                'disparity_ratio': disparity_ratio,
                'sample_size': len(group_data),
                'statistically_significant': abs(disparity) > 0.05  # 5% threshold
            }
        
        return parity_analysis

# Usage example
bias_keywords = {
    'gender': ['he said', 'she said', 'male doctor', 'female nurse', 'guys', 'gals'],
    'race': ['urban youth', 'articulate black', 'exotic', 'primitive'],
    'age': ['young and energetic', 'old and wise', 'millennial', 'boomer'],
    'religion': ['christian values', 'muslim terrorist', 'jewish money']
}

detector = DataBiasDetector(training_data)

# Analyze representation bias
repr_bias = detector.analyze_representation_bias('gender', 'job_category')
print("Representation Analysis:", repr_bias)

# Detect historical bias patterns
hist_bias = detector.detect_historical_bias('text', bias_keywords)
print("Historical Bias Detection:", hist_bias)

# Analyze intersectional bias
intersect_bias = detector.analyze_intersectional_bias(['gender', 'race'], 'hired')
print("Intersectional Analysis:", intersect_bias)
"""
        },
        {
            "type": "Algorithmic Bias",
            "description": "Bias introduced by model architecture and training process",
            "subtypes": [
                "Selection bias - Biased sampling during training",
                "Confirmation bias - Model reinforcing existing patterns",
                "Optimization bias - Biased objective functions",
                "Evaluation bias - Biased metrics and test sets"
            ],
            "examples": [
                "Word embeddings associating 'doctor' with 'he'",
                "Sentiment analysis performing worse on AAE",
                "Language models generating stereotypical completions",
                "Named entity recognition failing on non-Western names"
            ],
            "detection_methods": """
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class AlgorithmicBiasDetector:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def embedding_bias_test(self, target_words, attribute_words_1, attribute_words_2):
        # Word Embedding Association Test (WEAT)
        
        def get_embedding(word):
            inputs = self.tokenizer(word, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use last hidden state mean as word embedding
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            return embedding.numpy()
        
        # Get embeddings for all words
        target_embeddings = [get_embedding(word) for word in target_words]
        attr1_embeddings = [get_embedding(word) for word in attribute_words_1]
        attr2_embeddings = [get_embedding(word) for word in attribute_words_2]
        
        # Calculate mean embeddings for attribute sets
        mean_attr1 = np.mean(attr1_embeddings, axis=0)
        mean_attr2 = np.mean(attr2_embeddings, axis=0)
        
        # Calculate bias scores for each target word
        bias_scores = []
        for target_emb in target_embeddings:
            # Cosine similarity to each attribute set
            sim_attr1 = cosine_similarity([target_emb], [mean_attr1])[0][0]
            sim_attr2 = cosine_similarity([target_emb], [mean_attr2])[0][0]
            
            # Bias score: difference in similarities
            bias_score = sim_attr1 - sim_attr2
            bias_scores.append(bias_score)
        
        # Overall bias: mean of individual bias scores
        overall_bias = np.mean(bias_scores)
        
        return {
            'target_words': target_words,
            'individual_bias_scores': dict(zip(target_words, bias_scores)),
            'overall_bias': overall_bias,
            'interpretation': 'Positive bias toward attribute_1, negative toward attribute_2' if overall_bias > 0 else 'Positive bias toward attribute_2, negative toward attribute_1'
        }
    
    def generation_bias_test(self, prompts, demographic_groups):
        # Test for biased text generation
        generation_analysis = {}
        
        for group in demographic_groups:
            group_results = {
                'generated_texts': [],
                'sentiment_scores': [],
                'stereotype_mentions': 0,
                'positive_traits': 0,
                'negative_traits': 0
            }
            
            for prompt_template in prompts:
                # Insert demographic group into prompt
                prompt = prompt_template.format(group=group)
                
                # Generate text
                inputs = self.tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7,
                        num_return_sequences=3
                    )
                
                for output in outputs:
                    generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                    generated_text = generated_text[len(prompt):].strip()
                    
                    group_results['generated_texts'].append(generated_text)
                    
                    # Analyze generated text for bias indicators
                    sentiment = self.analyze_sentiment(generated_text)
                    group_results['sentiment_scores'].append(sentiment)
                    
                    # Count stereotype mentions (simplified)
                    if self.contains_stereotypes(generated_text, group):
                        group_results['stereotype_mentions'] += 1
            
            # Calculate summary statistics
            group_results['avg_sentiment'] = np.mean(group_results['sentiment_scores'])
            group_results['stereotype_rate'] = group_results['stereotype_mentions'] / len(group_results['generated_texts'])
            
            generation_analysis[group] = group_results
        
        return generation_analysis
    
    def performance_disparity_test(self, test_data, demographic_column, target_column):
        # Test for performance disparities across demographic groups
        performance_analysis = {}
        
        for group in test_data[demographic_column].unique():
            group_data = test_data[test_data[demographic_column] == group]
            
            # Get model predictions
            predictions = []
            true_labels = group_data[target_column].tolist()
            
            for text in group_data['text']:
                prediction = self.predict(text)
                predictions.append(prediction)
            
            # Calculate performance metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average='weighted')
            recall = recall_score(true_labels, predictions, average='weighted')
            f1 = f1_score(true_labels, predictions, average='weighted')
            
            performance_analysis[group] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'sample_size': len(group_data),
                'error_examples': self.get_error_examples(group_data, predictions, true_labels)
            }
        
        # Calculate performance disparities
        accuracy_values = [metrics['accuracy'] for metrics in performance_analysis.values()]
        performance_analysis['disparity_metrics'] = {
            'max_accuracy_gap': max(accuracy_values) - min(accuracy_values),
            'coefficient_of_variation': np.std(accuracy_values) / np.mean(accuracy_values),
            'equalized_odds_violation': self.calculate_equalized_odds_violation(performance_analysis)
        }
        
        return performance_analysis
    
    def analyze_sentiment(self, text):
        # Simplified sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'positive', 'happy']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'negative', 'sad', 'angry']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        return positive_count - negative_count
    
    def contains_stereotypes(self, text, group):
        # Simplified stereotype detection
        stereotype_keywords = {
            'women': ['emotional', 'nurturing', 'weak', 'submissive'],
            'men': ['aggressive', 'strong', 'dominant', 'unemotional'],
            'elderly': ['slow', 'forgetful', 'outdated', 'frail'],
            'young': ['inexperienced', 'irresponsible', 'naive']
        }
        
        if group.lower() in stereotype_keywords:
            keywords = stereotype_keywords[group.lower()]
            return any(keyword in text.lower() for keyword in keywords)
        
        return False

# Usage example
detector = AlgorithmicBiasDetector(model, tokenizer)

# Test embedding bias (gender-career associations)
target_words = ['doctor', 'nurse', 'engineer', 'teacher', 'CEO', 'secretary']
male_words = ['he', 'him', 'his', 'man', 'male', 'boy']
female_words = ['she', 'her', 'hers', 'woman', 'female', 'girl']

embedding_bias = detector.embedding_bias_test(target_words, male_words, female_words)
print("Embedding Bias Results:", embedding_bias)

# Test generation bias
prompts = [
    "The {group} person is typically",
    "A {group} worker would",
    "Most {group} people are"
]
demographic_groups = ['Black', 'White', 'Asian', 'Hispanic']

generation_bias = detector.generation_bias_test(prompts, demographic_groups)
print("Generation Bias Results:", generation_bias)
"""
        },
        {
            "type": "Deployment Bias",
            "description": "Bias emerging during model deployment and real-world usage",
            "subtypes": [
                "Population bias - Differences between training and deployment populations",
                "Temporal bias - Changes in data distribution over time",
                "Usage bias - Different usage patterns across user groups",
                "Feedback bias - Biased user feedback affecting model updates"
            ],
            "examples": [
                "Model trained on formal text used in informal settings",
                "Healthcare AI tested on one demographic, deployed broadly",
                "Recommendation systems creating filter bubbles",
                "Voice assistants favoring certain accents"
            ],
            "detection_methods": """
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

class DeploymentBiasMonitor:
    def __init__(self):
        self.baseline_metrics = {}
        self.monitoring_data = []
        
    def set_baseline(self, validation_data, demographic_column, performance_metrics):
        # Establish baseline performance across demographic groups
        self.baseline_metrics = {}
        
        for group in validation_data[demographic_column].unique():
            group_data = validation_data[validation_data[demographic_column] == group]
            
            self.baseline_metrics[group] = {
                'sample_size': len(group_data),
                'metrics': performance_metrics[group] if group in performance_metrics else {}
            }
    
    def monitor_deployment_performance(self, deployment_data, demographic_column, time_column):
        # Monitor performance drift over time across demographic groups
        time_series_analysis = {}
        
        # Group by time periods (e.g., weeks, months)
        deployment_data['time_period'] = pd.to_datetime(deployment_data[time_column]).dt.to_period('W')
        
        for period in deployment_data['time_period'].unique():
            period_data = deployment_data[deployment_data['time_period'] == period]
            period_analysis = {}
            
            for group in period_data[demographic_column].unique():
                group_data = period_data[period_data[demographic_column] == group]
                
                # Calculate current metrics
                current_metrics = self.calculate_performance_metrics(group_data)
                
                # Compare with baseline
                if group in self.baseline_metrics:
                    baseline = self.baseline_metrics[group]['metrics']
                    metric_changes = {}
                    
                    for metric, current_value in current_metrics.items():
                        if metric in baseline:
                            baseline_value = baseline[metric]
                            change = current_value - baseline_value
                            percent_change = (change / baseline_value) * 100 if baseline_value != 0 else 0
                            
                            metric_changes[metric] = {
                                'current': current_value,
                                'baseline': baseline_value,
                                'change': change,
                                'percent_change': percent_change,
                                'significant_change': abs(percent_change) > 10  # 10% threshold
                            }
                    
                    period_analysis[group] = {
                        'sample_size': len(group_data),
                        'metric_changes': metric_changes,
                        'data_drift_detected': self.detect_data_drift(group_data)
                    }
            
            time_series_analysis[str(period)] = period_analysis
        
        return time_series_analysis
    
    def detect_population_shift(self, training_data, deployment_data, feature_columns):
        # Detect shifts in population characteristics
        shift_analysis = {}
        
        for feature in feature_columns:
            if feature in training_data.columns and feature in deployment_data.columns:
                # For numerical features
                if training_data[feature].dtype in ['int64', 'float64']:
                    # Use Kolmogorov-Smirnov test
                    ks_stat, p_value = stats.ks_2samp(
                        training_data[feature].dropna(),
                        deployment_data[feature].dropna()
                    )
                    
                    shift_analysis[feature] = {
                        'type': 'numerical',
                        'ks_statistic': ks_stat,
                        'p_value': p_value,
                        'significant_shift': p_value < 0.05,
                        'training_mean': training_data[feature].mean(),
                        'deployment_mean': deployment_data[feature].mean(),
                        'mean_shift': deployment_data[feature].mean() - training_data[feature].mean()
                    }
                
                # For categorical features
                else:
                    train_dist = training_data[feature].value_counts(normalize=True)
                    deploy_dist = deployment_data[feature].value_counts(normalize=True)
                    
                    # Calculate total variation distance
                    all_categories = set(train_dist.index) | set(deploy_dist.index)
                    tvd = 0
                    
                    for category in all_categories:
                        train_prob = train_dist.get(category, 0)
                        deploy_prob = deploy_dist.get(category, 0)
                        tvd += abs(train_prob - deploy_prob)
                    
                    tvd = tvd / 2  # Total variation distance
                    
                    shift_analysis[feature] = {
                        'type': 'categorical',
                        'total_variation_distance': tvd,
                        'significant_shift': tvd > 0.1,  # 10% threshold
                        'training_distribution': train_dist.to_dict(),
                        'deployment_distribution': deploy_dist.to_dict()
                    }
        
        return shift_analysis
    
    def analyze_feedback_bias(self, feedback_data, demographic_column, feedback_column):
        # Analyze potential bias in user feedback
        feedback_analysis = {}
        
        for group in feedback_data[demographic_column].unique():
            group_feedback = feedback_data[feedback_data[demographic_column] == group]
            
            # Analyze feedback patterns
            feedback_stats = {
                'sample_size': len(group_feedback),
                'feedback_rate': len(group_feedback) / len(feedback_data),
                'avg_rating': group_feedback[feedback_column].mean(),
                'rating_distribution': group_feedback[feedback_column].value_counts(normalize=True).to_dict(),
                'feedback_variance': group_feedback[feedback_column].var()
            }
            
            # Detect potential bias indicators
            overall_avg = feedback_data[feedback_column].mean()
            feedback_stats['bias_indicators'] = {
                'lower_than_average': feedback_stats['avg_rating'] < overall_avg - 0.5,
                'higher_than_average': feedback_stats['avg_rating'] > overall_avg + 0.5,
                'low_participation': feedback_stats['feedback_rate'] < 0.1,
                'high_variance': feedback_stats['feedback_variance'] > feedback_data[feedback_column].var() * 1.5
            }
            
            feedback_analysis[group] = feedback_stats
        
        return feedback_analysis
    
    def generate_bias_report(self, analysis_results):
        # Generate comprehensive bias monitoring report
        report = {
            'summary': {
                'total_groups_monitored': 0,
                'groups_with_significant_bias': 0,
                'most_biased_metric': '',
                'recommendations': []
            },
            'detailed_findings': analysis_results,
            'action_items': []
        }
        
        # Analyze results and generate recommendations
        biased_groups = 0
        metric_bias_counts = {}
        
        for period, period_data in analysis_results.items():
            for group, group_data in period_data.items():
                report['summary']['total_groups_monitored'] += 1
                
                group_has_bias = False
                for metric, metric_data in group_data.get('metric_changes', {}).items():
                    if metric_data.get('significant_change', False):
                        group_has_bias = True
                        metric_bias_counts[metric] = metric_bias_counts.get(metric, 0) + 1
                
                if group_has_bias:
                    biased_groups += 1
        
        report['summary']['groups_with_significant_bias'] = biased_groups
        
        if metric_bias_counts:
            report['summary']['most_biased_metric'] = max(metric_bias_counts.items(), key=lambda x: x[1])[0]
        
        # Generate recommendations
        if biased_groups > 0:
            report['summary']['recommendations'] = [
                "Investigate root causes of performance disparities",
                "Consider retraining with more balanced data",
                "Implement group-specific model adjustments",
                "Increase monitoring frequency for affected groups"
            ]
        
        return report

# Usage example
monitor = DeploymentBiasMonitor()

# Set baseline from validation data
baseline_performance = {
    'group_A': {'accuracy': 0.85, 'precision': 0.82},
    'group_B': {'accuracy': 0.83, 'precision': 0.80}
}
monitor.set_baseline(validation_df, 'demographic_group', baseline_performance)

# Monitor deployment performance
deployment_analysis = monitor.monitor_deployment_performance(
    deployment_df, 'demographic_group', 'timestamp'
)

# Detect population shifts
population_shift = monitor.detect_population_shift(
    training_df, deployment_df, ['age', 'education', 'income']
)

# Analyze feedback bias
feedback_bias = monitor.analyze_feedback_bias(
    feedback_df, 'demographic_group', 'user_rating'
)

# Generate comprehensive report
bias_report = monitor.generate_bias_report(deployment_analysis)
print("Bias Monitoring Report:", bias_report)
"""
        }
    ]
    
    for bias_type in bias_types:
        with st.expander(f"🔍 {bias_type['type']}"):
            st.markdown(bias_type['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Subtypes:**")
                for subtype in bias_type['subtypes']:
                    st.markdown(f"• {subtype}")
            
            with col2:
                st.markdown("**Examples:**")
                for example in bias_type['examples']:
                    st.markdown(f"• {example}")
            
            st.markdown("**Detection Methods:**")
            st.code(bias_type['detection_methods'], language='python')

# Interactive bias detection demo
st.markdown("### 🔍 Interactive Bias Detection Demo")

demo_data = pd.DataFrame({
    'text': [
        "The doctor said he would see you now",
        "The nurse was very caring and gentle", 
        "The engineer explained the technical details",
        "The teacher was patient with her students",
        "The CEO made a decisive business decision",
        "The secretary scheduled the meeting efficiently"
    ],
    'profession': ['doctor', 'nurse', 'engineer', 'teacher', 'CEO', 'secretary'],
    'gender_pronoun': ['he', 'she', 'they', 'she', 'they', 'she']
})

st.markdown("**Sample Data Analysis:**")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Gender-Profession Distribution:**")
    gender_prof_dist = demo_data.groupby(['profession', 'gender_pronoun']).size().unstack(fill_value=0)
    st.dataframe(gender_prof_dist)

with col2:
    st.markdown("**Bias Indicators:**")
    st.markdown("• 'Doctor' associated with male pronoun")
    st.markdown("• 'Nurse' associated with female pronoun") 
    st.markdown("• 'Secretary' associated with female pronoun")
    st.markdown("• Potential reinforcement of stereotypes")

with concept_tabs[1]:
    st.subheader("📊 Bias Measurement Techniques")
    
    st.markdown("""
    Quantitative measurement of bias is essential for understanding the scope of the problem 
    and tracking progress in mitigation efforts.
    """)
    
    measurement_techniques = [
        {
            "technique": "Statistical Parity",
            "description": "Measures whether outcomes are equally distributed across groups",
            "formula": "P(Y=1|A=a) = P(Y=1|A=b) for all groups a,b",
            "use_cases": [
                "Hiring decisions",
                "Loan approvals", 
                "Criminal justice",
                "Educational opportunities"
            ],
            "implementation": """
def measure_statistical_parity(predictions, sensitive_attribute):
    parity_results = {}
    overall_positive_rate = predictions.mean()
    
    for group in sensitive_attribute.unique():
        group_mask = sensitive_attribute == group
        group_positive_rate = predictions[group_mask].mean()
        
        # Calculate parity metrics
        difference = group_positive_rate - overall_positive_rate
        ratio = group_positive_rate / overall_positive_rate if overall_positive_rate > 0 else float('inf')
        
        parity_results[group] = {
            'positive_rate': group_positive_rate,
            'difference_from_overall': difference,
            'ratio_to_overall': ratio,
            'satisfies_parity': abs(difference) < 0.05  # 5% tolerance
        }
    
    return parity_results
"""
        },
        {
            "technique": "Equalized Odds", 
            "description": "Measures whether error rates are equal across groups",
            "formula": "P(Y=1|A=a,T=t) = P(Y=1|A=b,T=t) for all groups a,b and true labels t",
            "use_cases": [
                "Medical diagnosis",
                "Risk assessment",
                "Fraud detection",
                "Content moderation"
            ],
            "implementation": """
def measure_equalized_odds(y_true, y_pred, sensitive_attribute):
    odds_results = {}
    
    for group in sensitive_attribute.unique():
        group_mask = sensitive_attribute == group
        group_y_true = y_true[group_mask]
        group_y_pred = y_pred[group_mask]
        
        # True Positive Rate (Sensitivity)
        tpr = ((group_y_true == 1) & (group_y_pred == 1)).sum() / (group_y_true == 1).sum()
        
        # False Positive Rate 
        fpr = ((group_y_true == 0) & (group_y_pred == 1)).sum() / (group_y_true == 0).sum()
        
        # True Negative Rate (Specificity)
        tnr = ((group_y_true == 0) & (group_y_pred == 0)).sum() / (group_y_true == 0).sum()
        
        # False Negative Rate
        fnr = ((group_y_true == 1) & (group_y_pred == 0)).sum() / (group_y_true == 1).sum()
        
        odds_results[group] = {
            'true_positive_rate': tpr,
            'false_positive_rate': fpr,
            'true_negative_rate': tnr,
            'false_negative_rate': fnr,
            'sample_size': group_mask.sum()
        }
    
    # Calculate disparities
    groups = list(odds_results.keys())
    if len(groups) >= 2:
        group_a, group_b = groups[0], groups[1]
        
        tpr_disparity = abs(odds_results[group_a]['true_positive_rate'] - 
                           odds_results[group_b]['true_positive_rate'])
        fpr_disparity = abs(odds_results[group_a]['false_positive_rate'] - 
                           odds_results[group_b]['false_positive_rate'])
        
        odds_results['disparities'] = {
            'tpr_disparity': tpr_disparity,
            'fpr_disparity': fpr_disparity,
            'satisfies_equalized_odds': tpr_disparity < 0.05 and fpr_disparity < 0.05
        }
    
    return odds_results
"""
        },
        {
            "technique": "Calibration",
            "description": "Measures whether predicted probabilities match actual outcomes across groups",
            "formula": "P(T=1|S=s,A=a) = s for all scores s and groups a",
            "use_cases": [
                "Risk scoring",
                "Probability estimation",
                "Medical prognosis",
                "Financial modeling"
            ],
            "implementation": """
def measure_calibration(y_true, y_prob, sensitive_attribute, n_bins=10):
    calibration_results = {}
    
    for group in sensitive_attribute.unique():
        group_mask = sensitive_attribute == group
        group_y_true = y_true[group_mask]
        group_y_prob = y_prob[group_mask]
        
        # Create probability bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (group_y_prob > bin_lower) & (group_y_prob <= bin_upper)
            
            if in_bin.sum() > 0:
                # Mean predicted probability in bin
                bin_pred_prob = group_y_prob[in_bin].mean()
                
                # Actual positive rate in bin
                bin_true_prob = group_y_true[in_bin].mean()
                
                # Calibration error for this bin
                bin_calibration_error = abs(bin_pred_prob - bin_true_prob)
                
                calibration_data.append({
                    'bin_range': f'({bin_lower:.1f}, {bin_upper:.1f}]',
                    'predicted_prob': bin_pred_prob,
                    'actual_prob': bin_true_prob,
                    'calibration_error': bin_calibration_error,
                    'count': in_bin.sum()
                })
        
        # Overall calibration metrics
        if calibration_data:
            # Expected Calibration Error (ECE)
            total_samples = len(group_y_true)
            ece = sum(
                (data['count'] / total_samples) * data['calibration_error'] 
                for data in calibration_data
            )
            
            # Maximum Calibration Error (MCE)
            mce = max(data['calibration_error'] for data in calibration_data)
            
            calibration_results[group] = {
                'expected_calibration_error': ece,
                'maximum_calibration_error': mce,
                'bin_details': calibration_data,
                'well_calibrated': ece < 0.05  # 5% tolerance
            }
    
    return calibration_results
"""
        },
        {
            "technique": "Individual Fairness",
            "description": "Measures whether similar individuals receive similar outcomes",
            "formula": "d(M(x1), M(x2)) ≤ L × d(x1, x2) for similar inputs x1, x2",
            "use_cases": [
                "Personalized recommendations",
                "Individual assessments",
                "Custom pricing",
                "Personalized content"
            ],
            "implementation": """
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def measure_individual_fairness(X, predictions, sensitive_features, similarity_threshold=0.1):
    fairness_results = {
        'violations': [],
        'similarity_consistency': {},
        'lipschitz_constant': None
    }
    
    # Calculate pairwise similarities
    feature_distances = euclidean_distances(X)
    prediction_distances = euclidean_distances(predictions.reshape(-1, 1))
    
    violations = []
    
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            # Check if individuals are similar in features
            feature_distance = feature_distances[i, j]
            pred_distance = prediction_distances[i, j]
            
            # Individuals are similar if feature distance is small
            if feature_distance < similarity_threshold:
                # Check if predictions are also similar
                if pred_distance > similarity_threshold * 2:  # Allow some tolerance
                    
                    # Check if they differ mainly in sensitive attributes
                    sensitive_distance = np.linalg.norm(
                        sensitive_features.iloc[i] - sensitive_features.iloc[j]
                    )
                    
                    if sensitive_distance > feature_distance * 0.5:
                        violations.append({
                            'individual_1': i,
                            'individual_2': j,
                            'feature_distance': feature_distance,
                            'prediction_distance': pred_distance,
                            'sensitive_distance': sensitive_distance,
                            'unfairness_score': pred_distance / feature_distance
                        })
    
    fairness_results['violations'] = violations
    fairness_results['violation_rate'] = len(violations) / (len(X) * (len(X) - 1) / 2)
    
    # Calculate approximate Lipschitz constant
    if len(violations) > 0:
        lipschitz_estimates = [
            v['prediction_distance'] / v['feature_distance'] 
            for v in violations if v['feature_distance'] > 0
        ]
        fairness_results['lipschitz_constant'] = max(lipschitz_estimates) if lipschitz_estimates else None
    
    return fairness_results
"""
        }
    ]
    
    for technique in measurement_techniques:
        with st.expander(f"📊 {technique['technique']}"):
            st.markdown(technique['description'])
            st.markdown(f"**Formula:** {technique['formula']}")
            
            st.markdown("**Use Cases:**")
            for use_case in technique['use_cases']:
                st.markdown(f"• {use_case}")
            
            st.markdown("**Implementation:**")
            st.code(technique['implementation'], language='python')

# Bias metrics comparison chart
st.markdown("### 📊 Bias Metrics Comparison")

# Simulate bias metrics across different groups
groups = ['Group A', 'Group B', 'Group C', 'Group D']
metrics_data = {
    'Group': groups,
    'Statistical Parity': [0.02, 0.15, 0.08, 0.12],
    'Equalized Odds': [0.03, 0.18, 0.06, 0.14],
    'Calibration Error': [0.01, 0.09, 0.04, 0.07],
    'Individual Fairness': [0.05, 0.22, 0.11, 0.16]
}

bias_metrics_df = pd.DataFrame(metrics_data)

fig = px.bar(bias_metrics_df, x='Group', y=['Statistical Parity', 'Equalized Odds', 'Calibration Error', 'Individual Fairness'],
             title="Bias Metrics Across Different Groups",
             labels={'value': 'Bias Score (lower is better)', 'variable': 'Metric'},
             barmode='group')

st.plotly_chart(fig, use_container_width=True)

with concept_tabs[2]:
    st.subheader("🛠️ Bias Mitigation Strategies")
    
    st.markdown("""
    Effective bias mitigation requires a multi-pronged approach addressing bias 
    at different stages of the ML pipeline.
    """)
    
    mitigation_strategies = [
        {
            "strategy": "Data-level Mitigation",
            "description": "Addressing bias through data collection and preprocessing",
            "techniques": [
                "Balanced sampling",
                "Data augmentation",
                "Synthetic data generation", 
                "Bias-aware feature selection"
            ],
            "implementation": """
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

class DataBiasMitigation:
    def __init__(self):
        self.strategies = {}
    
    def balanced_sampling(self, data, sensitive_column, target_column, strategy='oversample'):
        # Balance representation across sensitive groups
        balanced_data = []
        
        # Get group sizes
        group_sizes = data.groupby([sensitive_column, target_column]).size()
        target_size = group_sizes.max()  # Target size for balancing
        
        for (sensitive_value, target_value), group_size in group_sizes.items():
            group_data = data[
                (data[sensitive_column] == sensitive_value) & 
                (data[target_column] == target_value)
            ]
            
            if strategy == 'oversample' and group_size < target_size:
                # Oversample minority groups
                oversampled = resample(
                    group_data,
                    replace=True,
                    n_samples=target_size,
                    random_state=42
                )
                balanced_data.append(oversampled)
            
            elif strategy == 'undersample' and group_size > target_size:
                # Undersample majority groups
                undersampled = resample(
                    group_data,
                    replace=False,
                    n_samples=target_size,
                    random_state=42
                )
                balanced_data.append(undersampled)
            
            else:
                balanced_data.append(group_data)
        
        return pd.concat(balanced_data, ignore_index=True)
    
    def bias_aware_augmentation(self, texts, sensitive_attributes, augmentation_factor=2):
        # Augment data while preserving sensitive attribute balance
        augmented_data = []
        
        # Group by sensitive attributes
        for attr_value in sensitive_attributes.unique():
            attr_texts = texts[sensitive_attributes == attr_value]
            
            # Apply augmentation techniques
            for text in attr_texts:
                augmented_data.append(text)  # Original
                
                # Generate variations (simplified - in practice use more sophisticated methods)
                for _ in range(augmentation_factor - 1):
                    augmented_text = self.augment_text(text)
                    augmented_data.append(augmented_text)
        
        return augmented_data
    
    def augment_text(self, text):
        # Simple text augmentation (in practice, use more sophisticated methods)
        augmentation_techniques = [
            self.synonym_replacement,
            self.sentence_reordering,
            self.random_insertion,
        ]
        
        technique = np.random.choice(augmentation_techniques)
        return technique(text)
    
    def synonym_replacement(self, text):
        # Replace words with synonyms (simplified)
        synonyms = {
            'good': ['great', 'excellent', 'fine'],
            'bad': ['poor', 'terrible', 'awful'],
            'big': ['large', 'huge', 'enormous'],
            'small': ['tiny', 'little', 'mini']
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in synonyms:
                words[i] = np.random.choice(synonyms[word.lower()])
        
        return ' '.join(words)
    
    def sentence_reordering(self, text):
        # Reorder sentences (simplified)
        sentences = text.split('. ')
        if len(sentences) > 1:
            np.random.shuffle(sentences)
            return '. '.join(sentences)
        return text
    
    def random_insertion(self, text):
        # Insert neutral words (simplified)
        neutral_words = ['actually', 'really', 'quite', 'very', 'somewhat']
        words = text.split()
        
        if len(words) > 2:
            insert_pos = np.random.randint(1, len(words))
            words.insert(insert_pos, np.random.choice(neutral_words))
        
        return ' '.join(words)
    
    def remove_biased_features(self, data, feature_columns, bias_threshold=0.8):
        # Remove features highly correlated with sensitive attributes
        bias_scores = {}
        
        for feature in feature_columns:
            if feature in data.columns:
                # Calculate correlation with sensitive attributes
                correlations = []
                for sensitive_col in ['gender', 'race', 'age']:  # Example sensitive attributes
                    if sensitive_col in data.columns:
                        if data[feature].dtype in ['int64', 'float64'] and data[sensitive_col].dtype in ['int64', 'float64']:
                            corr = abs(data[feature].corr(data[sensitive_col]))
                            correlations.append(corr)
                
                if correlations:
                    max_correlation = max(correlations)
                    bias_scores[feature] = max_correlation
        
        # Remove highly biased features
        features_to_remove = [
            feature for feature, bias_score in bias_scores.items()
            if bias_score > bias_threshold
        ]
        
        cleaned_data = data.drop(columns=features_to_remove)
        
        return cleaned_data, features_to_remove, bias_scores
    
    def synthetic_fair_data_generation(self, original_data, sensitive_column, target_column, n_samples=1000):
        # Generate synthetic data with better fairness properties
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data
        feature_columns = [col for col in original_data.columns if col not in [sensitive_column, target_column]]
        X = original_data[feature_columns]
        y = original_data[target_column]
        sensitive = original_data[sensitive_column]
        
        # Encode categorical variables
        encoders = {}
        X_encoded = X.copy()
        
        for col in X.columns:
            if X[col].dtype == 'object':
                encoders[col] = LabelEncoder()
                X_encoded[col] = encoders[col].fit_transform(X[col].astype(str))
        
        # Train model to learn data distribution
        model = RandomForestClassifier(random_state=42)
        model.fit(X_encoded, y)
        
        # Generate synthetic samples
        synthetic_data = []
        
        # Ensure balanced representation across sensitive groups
        unique_sensitive = sensitive.unique()
        samples_per_group = n_samples // len(unique_sensitive)
        
        for sensitive_value in unique_sensitive:
            group_data = original_data[original_data[sensitive_column] == sensitive_value]
            
            for _ in range(samples_per_group):
                # Sample features from group distribution
                sample_idx = np.random.randint(0, len(group_data))
                base_sample = group_data.iloc[sample_idx]
                
                # Add noise to create variation
                synthetic_sample = base_sample.copy()
                
                for col in feature_columns:
                    if X[col].dtype in ['int64', 'float64']:
                        noise = np.random.normal(0, X[col].std() * 0.1)
                        synthetic_sample[col] = base_sample[col] + noise
                
                # Predict target using fair model (simplified)
                synthetic_sample[target_column] = np.random.choice([0, 1])  # Balanced targets
                synthetic_sample[sensitive_column] = sensitive_value
                
                synthetic_data.append(synthetic_sample)
        
        return pd.DataFrame(synthetic_data)

# Usage example
mitigator = DataBiasMitigation()

# Balance sampling
balanced_data = mitigator.balanced_sampling(
    training_data, 'gender', 'hired', strategy='oversample'
)

# Remove biased features
cleaned_data, removed_features, bias_scores = mitigator.remove_biased_features(
    training_data, ['name_length', 'zip_code', 'education'], bias_threshold=0.7
)

print(f"Removed features: {removed_features}")
print(f"Bias scores: {bias_scores}")

# Generate synthetic fair data
synthetic_data = mitigator.synthetic_fair_data_generation(
    training_data, 'race', 'approved', n_samples=5000
)
"""
        },
        {
            "strategy": "Algorithm-level Mitigation",
            "description": "Modifying training algorithms to promote fairness",
            "techniques": [
                "Fairness constraints",
                "Adversarial debiasing",
                "Multi-task learning",
                "Regularization methods"
            ],
            "implementation": """
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

class FairClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super().__init__()
        
        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Adversarial network for sensitive attribute prediction
        self.adversary = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # Binary sensitive attribute
        )
        
    def forward(self, x, return_hidden=False):
        # Get hidden representation from classifier
        hidden = self.classifier[:-1](x)  # All layers except the last
        
        # Final classification
        logits = self.classifier[-1](hidden)
        
        if return_hidden:
            return logits, hidden
        return logits
    
    def adversarial_forward(self, hidden):
        return self.adversary(hidden)

class FairnessTrainer:
    def __init__(self, model, alpha=1.0, beta=1.0):
        self.model = model
        self.alpha = alpha  # Weight for classification loss
        self.beta = beta    # Weight for adversarial loss
        
        self.classifier_optimizer = optim.Adam(self.model.classifier.parameters(), lr=0.001)
        self.adversary_optimizer = optim.Adam(self.model.adversary.parameters(), lr=0.001)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def train_step(self, X_batch, y_batch, sensitive_batch):
        # Phase 1: Train classifier to maximize accuracy while fooling adversary
        self.classifier_optimizer.zero_grad()
        
        logits, hidden = self.model(X_batch, return_hidden=True)
        
        # Classification loss
        class_loss = self.criterion(logits, y_batch)
        
        # Adversarial loss (fool the adversary)
        adv_logits = self.model.adversarial_forward(hidden)
        # Use uniform distribution as target to fool adversary
        uniform_target = torch.ones_like(sensitive_batch) * 0.5
        adv_loss = -self.criterion(adv_logits, sensitive_batch)  # Negative to fool adversary
        
        # Combined loss
        total_loss = self.alpha * class_loss + self.beta * adv_loss
        total_loss.backward()
        self.classifier_optimizer.step()
        
        # Phase 2: Train adversary to predict sensitive attributes
        self.adversary_optimizer.zero_grad()
        
        with torch.no_grad():
            _, hidden = self.model(X_batch, return_hidden=True)
        
        adv_logits = self.model.adversarial_forward(hidden.detach())
        adv_loss_real = self.criterion(adv_logits, sensitive_batch)
        
        adv_loss_real.backward()
        self.adversary_optimizer.step()
        
        return {
            'classification_loss': class_loss.item(),
            'adversarial_loss': adv_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def fairness_regularization_loss(self, logits, sensitive_attributes):
        # Implement fairness regularization
        
        # Demographic parity regularization
        sensitive_groups = sensitive_attributes.unique()
        group_predictions = []
        
        for group in sensitive_groups:
            group_mask = sensitive_attributes == group
            group_logits = logits[group_mask]
            
            if len(group_logits) > 0:
                group_pred_prob = torch.softmax(group_logits, dim=1)[:, 1].mean()
                group_predictions.append(group_pred_prob)
        
        if len(group_predictions) > 1:
            # Minimize variance in prediction rates across groups
            group_predictions = torch.stack(group_predictions)
            demographic_parity_loss = torch.var(group_predictions)
        else:
            demographic_parity_loss = torch.tensor(0.0)
        
        return demographic_parity_loss

class ConstrainedOptimizer:
    def __init__(self, model, fairness_constraint='demographic_parity', tolerance=0.05):
        self.model = model
        self.fairness_constraint = fairness_constraint
        self.tolerance = tolerance
        
    def lagrangian_optimization(self, X, y, sensitive_attr, epochs=100):
        # Lagrangian method for constrained optimization
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Lagrange multiplier for fairness constraint
        lambda_fair = torch.tensor(0.0, requires_grad=True)
        lambda_optimizer = optim.SGD([lambda_fair], lr=0.01)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            lambda_optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(X)
            
            # Classification loss
            class_loss = nn.CrossEntropyLoss()(logits, y)
            
            # Fairness constraint violation
            constraint_violation = self.compute_constraint_violation(logits, sensitive_attr)
            
            # Lagrangian
            lagrangian = class_loss + lambda_fair * constraint_violation
            
            # Update model parameters
            lagrangian.backward(retain_graph=True)
            optimizer.step()
            
            # Update Lagrange multiplier
            if constraint_violation.item() > self.tolerance:
                lambda_grad = constraint_violation.detach()
                lambda_fair.grad = lambda_grad
                lambda_optimizer.step()
                lambda_fair.data = torch.clamp(lambda_fair.data, min=0)  # Ensure non-negative
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss={class_loss.item():.4f}, "
                      f"Constraint={constraint_violation.item():.4f}, "
                      f"Lambda={lambda_fair.item():.4f}")
    
    def compute_constraint_violation(self, logits, sensitive_attr):
        if self.fairness_constraint == 'demographic_parity':
            return self.demographic_parity_violation(logits, sensitive_attr)
        elif self.fairness_constraint == 'equalized_odds':
            return self.equalized_odds_violation(logits, sensitive_attr)
        else:
            raise ValueError(f"Unknown constraint: {self.fairness_constraint}")
    
    def demographic_parity_violation(self, logits, sensitive_attr):
        probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of positive class
        
        violations = []
        groups = sensitive_attr.unique()
        
        for i, group_a in enumerate(groups):
            for group_b in groups[i+1:]:
                mask_a = sensitive_attr == group_a
                mask_b = sensitive_attr == group_b
                
                if mask_a.sum() > 0 and mask_b.sum() > 0:
                    rate_a = probs[mask_a].mean()
                    rate_b = probs[mask_b].mean()
                    violation = torch.abs(rate_a - rate_b)
                    violations.append(violation)
        
        return torch.stack(violations).max() if violations else torch.tensor(0.0)

# Usage example
# Initialize fair classifier
model = FairClassifier(input_dim=feature_dim, hidden_dim=64)
trainer = FairnessTrainer(model, alpha=1.0, beta=0.5)

# Training loop
for epoch in range(100):
    for batch_idx, (X_batch, y_batch, sensitive_batch) in enumerate(dataloader):
        losses = trainer.train_step(X_batch, y_batch, sensitive_batch)
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}: {losses}")

# Constrained optimization
constrained_model = ConstrainedOptimizer(model, fairness_constraint='demographic_parity')
constrained_model.lagrangian_optimization(X_train, y_train, sensitive_train)
"""
        },
        {
            "strategy": "Post-processing Mitigation",
            "description": "Adjusting model outputs to achieve fairness",
            "techniques": [
                "Threshold optimization",
                "Calibration adjustments",
                "Output redistribution",
                "Fairness-aware ensembling"
            ],
            "implementation": """
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_curve, accuracy_score

class PostProcessingFairness:
    def __init__(self):
        self.group_thresholds = {}
        self.calibration_maps = {}
    
    def optimize_thresholds(self, y_true, y_prob, sensitive_attr, fairness_metric='equalized_odds'):
        # Optimize classification thresholds per group for fairness
        
        optimal_thresholds = {}
        
        for group in sensitive_attr.unique():
            group_mask = sensitive_attr == group
            group_y_true = y_true[group_mask]
            group_y_prob = y_prob[group_mask]
            
            if fairness_metric == 'equalized_odds':
                # Optimize for equal TPR and FPR across groups
                optimal_threshold = self.find_optimal_threshold_equalized_odds(
                    group_y_true, group_y_prob
                )
            elif fairness_metric == 'demographic_parity':
                # Optimize for equal positive prediction rates
                optimal_threshold = self.find_optimal_threshold_demographic_parity(
                    group_y_true, group_y_prob, target_rate=0.5
                )
            else:
                # Default: maximize accuracy
                optimal_threshold = self.find_optimal_threshold_accuracy(
                    group_y_true, group_y_prob
                )
            
            optimal_thresholds[group] = optimal_threshold
        
        self.group_thresholds = optimal_thresholds
        return optimal_thresholds
    
    def find_optimal_threshold_accuracy(self, y_true, y_prob):
        def accuracy_at_threshold(threshold):
            y_pred = (y_prob >= threshold).astype(int)
            return -accuracy_score(y_true, y_pred)  # Negative for minimization
        
        result = minimize_scalar(accuracy_at_threshold, bounds=(0, 1), method='bounded')
        return result.x
    
    def find_optimal_threshold_equalized_odds(self, y_true, y_prob):
        # Find threshold that balances TPR and FPR
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Find threshold where TPR - FPR is maximized (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]
    
    def find_optimal_threshold_demographic_parity(self, y_true, y_prob, target_rate):
        def rate_difference(threshold):
            y_pred = (y_prob >= threshold).astype(int)
            actual_rate = y_pred.mean()
            return abs(actual_rate - target_rate)
        
        result = minimize_scalar(rate_difference, bounds=(0, 1), method='bounded')
        return result.x
    
    def apply_group_thresholds(self, y_prob, sensitive_attr):
        # Apply group-specific thresholds
        y_pred_fair = np.zeros(len(y_prob))
        
        for group, threshold in self.group_thresholds.items():
            group_mask = sensitive_attr == group
            y_pred_fair[group_mask] = (y_prob[group_mask] >= threshold).astype(int)
        
        return y_pred_fair
    
    def calibrate_probabilities(self, y_true, y_prob, sensitive_attr, method='platt'):
        # Calibrate probabilities per group
        
        from sklearn.calibration import calibration_curve
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        
        calibration_models = {}
        
        for group in sensitive_attr.unique():
            group_mask = sensitive_attr == group
            group_y_true = y_true[group_mask]
            group_y_prob = y_prob[group_mask]
            
            if method == 'platt':
                # Platt scaling (sigmoid calibration)
                calibrator = LogisticRegression()
                calibrator.fit(group_y_prob.reshape(-1, 1), group_y_true)
                calibration_models[group] = calibrator
                
            elif method == 'isotonic':
                # Isotonic regression
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(group_y_prob, group_y_true)
                calibration_models[group] = calibrator
        
        self.calibration_models = calibration_models
        return calibration_models
    
    def apply_calibration(self, y_prob, sensitive_attr):
        # Apply group-specific calibration
        y_prob_calibrated = np.zeros(len(y_prob))
        
        for group in sensitive_attr.unique():
            group_mask = sensitive_attr == group
            group_y_prob = y_prob[group_mask]
            
            if group in self.calibration_models:
                calibrator = self.calibration_models[group]
                
                if hasattr(calibrator, 'predict_proba'):  # Platt scaling
                    y_prob_calibrated[group_mask] = calibrator.predict_proba(
                        group_y_prob.reshape(-1, 1)
                    )[:, 1]
                else:  # Isotonic regression
                    y_prob_calibrated[group_mask] = calibrator.transform(group_y_prob)
            else:
                y_prob_calibrated[group_mask] = group_y_prob
        
        return y_prob_calibrated
    
    def fairness_aware_ensemble(self, predictions_list, sensitive_attr, weights=None):
        # Ensemble multiple models with fairness considerations
        
        if weights is None:
            weights = np.ones(len(predictions_list)) / len(predictions_list)
        
        # Calculate fairness scores for each model
        fairness_scores = []
        
        for preds in predictions_list:
            # Calculate demographic parity violation
            dp_violation = self.calculate_demographic_parity_violation(preds, sensitive_attr)
            fairness_scores.append(1 / (1 + dp_violation))  # Higher score for lower violation
        
        # Adjust weights based on fairness
        fairness_weights = np.array(fairness_scores)
        fairness_weights = fairness_weights / fairness_weights.sum()
        
        # Combine original weights with fairness weights
        final_weights = weights * fairness_weights
        final_weights = final_weights / final_weights.sum()
        
        # Ensemble predictions
        ensemble_pred = np.zeros(len(predictions_list[0]))
        
        for i, preds in enumerate(predictions_list):
            ensemble_pred += final_weights[i] * preds
        
        return ensemble_pred, final_weights
    
    def calculate_demographic_parity_violation(self, predictions, sensitive_attr):
        # Calculate maximum demographic parity violation
        groups = sensitive_attr.unique()
        rates = []
        
        for group in groups:
            group_mask = sensitive_attr == group
            if group_mask.sum() > 0:
                group_rate = predictions[group_mask].mean()
                rates.append(group_rate)
        
        if len(rates) > 1:
            return max(rates) - min(rates)
        return 0.0

# Usage example
postprocessor = PostProcessingFairness()

# Optimize thresholds for fairness
optimal_thresholds = postprocessor.optimize_thresholds(
    y_true, y_prob, sensitive_attr, fairness_metric='equalized_odds'
)

print("Optimal thresholds:", optimal_thresholds)

# Apply group-specific thresholds
fair_predictions = postprocessor.apply_group_thresholds(y_prob, sensitive_attr)

# Calibrate probabilities
calibration_models = postprocessor.calibrate_probabilities(
    y_true, y_prob, sensitive_attr, method='platt'
)

calibrated_probs = postprocessor.apply_calibration(y_prob, sensitive_attr)

# Fairness-aware ensemble
ensemble_pred, ensemble_weights = postprocessor.fairness_aware_ensemble(
    [model1_preds, model2_preds, model3_preds], sensitive_attr
)

print("Fairness-aware ensemble weights:", ensemble_weights)
"""
        }
    ]
    
    for strategy in mitigation_strategies:
        with st.expander(f"🛠️ {strategy['strategy']}"):
            st.markdown(strategy['description'])
            
            st.markdown("**Key Techniques:**")
            for technique in strategy['techniques']:
                st.markdown(f"• {technique}")
            
            st.markdown("**Implementation Example:**")
            st.code(strategy['implementation'], language='python')

# Mitigation effectiveness comparison
st.markdown("### 📈 Mitigation Strategy Effectiveness")

# Simulate effectiveness data
strategies = ['Baseline', 'Data Balancing', 'Adversarial Training', 'Threshold Optimization', 'Combined Approach']
effectiveness_data = {
    'Strategy': strategies,
    'Bias Reduction (%)': [0, 25, 40, 30, 60],
    'Accuracy Loss (%)': [0, -5, -8, -2, -10],
    'Implementation Complexity': [1, 2, 4, 2, 5]
}

effectiveness_df = pd.DataFrame(effectiveness_data)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=effectiveness_df['Accuracy Loss (%)'],
    y=effectiveness_df['Bias Reduction (%)'],
    mode='markers+text',
    text=effectiveness_df['Strategy'],
    textposition="top center",
    marker=dict(
        size=effectiveness_df['Implementation Complexity'] * 10,
        color=effectiveness_df['Bias Reduction (%)'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Bias Reduction (%)")
    ),
    hovertemplate='Strategy: %{text}<br>Bias Reduction: %{y}%<br>Accuracy Loss: %{x}%<br>Complexity: %{marker.size}<extra></extra>'
))

fig.update_layout(
    title="Bias Mitigation Strategy Trade-offs",
    xaxis_title="Accuracy Loss (%)",
    yaxis_title="Bias Reduction (%)",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

with concept_tabs[3]:
    st.subheader("📈 Monitoring & Evaluation")
    
    st.markdown("""
    Continuous monitoring and evaluation are essential for maintaining fairness 
    in deployed systems and detecting emerging biases.
    """)
    
    monitoring_approaches = [
        {
            "approach": "Real-time Bias Monitoring",
            "description": "Continuous tracking of bias metrics in production",
            "components": [
                "Live metric dashboards",
                "Automated bias alerts",
                "Performance tracking by group",
                "Drift detection systems"
            ],
            "implementation": """
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class RealTimeBiasMonitor:
    def __init__(self, alert_thresholds=None):
        self.alert_thresholds = alert_thresholds or {
            'demographic_parity': 0.1,
            'equalized_odds': 0.1,
            'calibration_error': 0.05
        }
        self.monitoring_data = []
        self.alerts = []
    
    def update_metrics(self, predictions, true_labels, sensitive_attrs, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate bias metrics
        metrics = self.calculate_bias_metrics(predictions, true_labels, sensitive_attrs)
        metrics['timestamp'] = timestamp
        
        self.monitoring_data.append(metrics)
        
        # Check for alerts
        alerts = self.check_alerts(metrics)
        if alerts:
            self.alerts.extend(alerts)
        
        return metrics, alerts
    
    def calculate_bias_metrics(self, predictions, true_labels, sensitive_attrs):
        metrics = {}
        
        # Demographic Parity
        dp_violation = self.demographic_parity_violation(predictions, sensitive_attrs)
        metrics['demographic_parity_violation'] = dp_violation
        
        # Equalized Odds
        eo_violation = self.equalized_odds_violation(predictions, true_labels, sensitive_attrs)
        metrics['equalized_odds_violation'] = eo_violation
        
        # Calibration Error
        if hasattr(predictions, 'predict_proba'):
            probs = predictions.predict_proba(X)[:, 1]
            cal_error = self.calibration_error(true_labels, probs, sensitive_attrs)
            metrics['calibration_error'] = cal_error
        
        # Overall fairness score
        metrics['fairness_score'] = 1 - max(dp_violation, eo_violation)
        
        return metrics
    
    def check_alerts(self, metrics):
        alerts = []
        
        for metric, value in metrics.items():
            if metric in self.alert_thresholds:
                if value > self.alert_thresholds[metric]:
                    alert = {
                        'timestamp': metrics['timestamp'],
                        'metric': metric,
                        'value': value,
                        'threshold': self.alert_thresholds[metric],
                        'severity': 'high' if value > 2 * self.alert_thresholds[metric] else 'medium'
                    }
                    alerts.append(alert)
        
        return alerts
    
    def get_trend_analysis(self, window_hours=24):
        # Analyze trends over specified time window
        if not self.monitoring_data:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_data = [
            d for d in self.monitoring_data 
            if d['timestamp'] > cutoff_time
        ]
        
        if not recent_data:
            return {}
        
        trends = {}
        
        for metric in ['demographic_parity_violation', 'equalized_odds_violation', 'fairness_score']:
            values = [d[metric] for d in recent_data if metric in d]
            
            if len(values) > 1:
                # Calculate trend (slope)
                x = np.arange(len(values))
                slope, _ = np.polyfit(x, values, 1)
                
                trends[metric] = {
                    'slope': slope,
                    'direction': 'improving' if slope < 0 else 'degrading',
                    'current_value': values[-1],
                    'change_rate': slope * len(values)
                }
        
        return trends
    
    def generate_report(self):
        # Generate comprehensive monitoring report
        if not self.monitoring_data:
            return "No monitoring data available"
        
        latest_metrics = self.monitoring_data[-1]
        trends = self.get_trend_analysis()
        
        report = {
            'summary': {
                'last_updated': latest_metrics['timestamp'],
                'current_fairness_score': latest_metrics.get('fairness_score', 'N/A'),
                'active_alerts': len([a for a in self.alerts if 
                                    (datetime.now() - a['timestamp']).hours < 24]),
                'trend_summary': self.summarize_trends(trends)
            },
            'current_metrics': latest_metrics,
            'trends': trends,
            'recent_alerts': self.alerts[-10:],  # Last 10 alerts
            'recommendations': self.generate_recommendations(latest_metrics, trends)
        }
        
        return report
    
    def summarize_trends(self, trends):
        improving = sum(1 for t in trends.values() if t['direction'] == 'improving')
        degrading = sum(1 for t in trends.values() if t['direction'] == 'degrading')
        
        if improving > degrading:
            return "Overall improving"
        elif degrading > improving:
            return "Overall degrading"
        else:
            return "Stable"
    
    def generate_recommendations(self, current_metrics, trends):
        recommendations = []
        
        # High bias detection
        if current_metrics.get('demographic_parity_violation', 0) > 0.15:
            recommendations.append("High demographic parity violation detected - consider threshold adjustment")
        
        if current_metrics.get('equalized_odds_violation', 0) > 0.15:
            recommendations.append("High equalized odds violation - review model performance across groups")
        
        # Trend-based recommendations
        for metric, trend_data in trends.items():
            if trend_data['direction'] == 'degrading' and abs(trend_data['slope']) > 0.01:
                recommendations.append(f"Degrading trend in {metric} - investigate potential causes")
        
        # Alert-based recommendations
        high_severity_alerts = [a for a in self.alerts[-10:] if a.get('severity') == 'high']
        if high_severity_alerts:
            recommendations.append("High severity alerts detected - immediate attention required")
        
        return recommendations

# Usage example
monitor = RealTimeBiasMonitor()

# Simulate real-time monitoring
for hour in range(24):
    # Simulate predictions and ground truth
    n_samples = 1000
    predictions = np.random.binomial(1, 0.6, n_samples)
    true_labels = np.random.binomial(1, 0.5, n_samples)
    sensitive_attrs = np.random.choice(['A', 'B'], n_samples)
    
    # Add some bias over time
    if hour > 12:
        bias_factor = (hour - 12) * 0.02
        mask = sensitive_attrs == 'A'
        predictions[mask] = np.random.binomial(1, 0.6 + bias_factor, mask.sum())
    
    timestamp = datetime.now() - timedelta(hours=24-hour)
    
    metrics, alerts = monitor.update_metrics(
        predictions, true_labels, sensitive_attrs, timestamp
    )
    
    if alerts:
        print(f"Hour {hour}: Alerts detected - {alerts}")

# Generate monitoring report
report = monitor.generate_report()
print("Monitoring Report:", report['summary'])
"""
        },
        {
            "approach": "Audit and Compliance",
            "description": "Systematic evaluation of fairness and compliance requirements",
            "components": [
                "Fairness audits",
                "Regulatory compliance checks",
                "Stakeholder impact assessment",
                "Documentation and reporting"
            ],
            "implementation": """
class FairnessAudit:
    def __init__(self):
        self.audit_checklist = {
            'data_fairness': [
                'Representative sampling across demographic groups',
                'Historical bias analysis completed',
                'Data quality assessment by group',
                'Intersectional analysis conducted'
            ],
            'model_fairness': [
                'Multiple fairness metrics evaluated',
                'Performance parity analysis completed',
                'Bias mitigation techniques applied',
                'Model interpretability assessed'
            ],
            'deployment_fairness': [
                'Real-world performance monitoring',
                'Feedback loop bias analysis',
                'User experience assessment by group',
                'Continuous monitoring system active'
            ],
            'governance': [
                'Ethical review process completed',
                'Stakeholder consultation conducted',
                'Clear accountability assigned',
                'Regular audit schedule established'
            ]
        }
    
    def conduct_comprehensive_audit(self, model, test_data, sensitive_attrs):
        audit_results = {
            'overall_score': 0,
            'category_scores': {},
            'detailed_findings': {},
            'recommendations': [],
            'compliance_status': {}
        }
        
        # Technical fairness assessment
        technical_results = self.technical_fairness_assessment(
            model, test_data, sensitive_attrs
        )
        
        # Process audit
        process_results = self.process_audit()
        
        # Stakeholder impact assessment
        impact_results = self.stakeholder_impact_assessment()
        
        # Combine results
        audit_results['detailed_findings'] = {
            'technical': technical_results,
            'process': process_results,
            'impact': impact_results
        }
        
        # Calculate scores
        audit_results['category_scores'] = {
            'technical_fairness': self.score_technical_fairness(technical_results),
            'process_compliance': self.score_process_compliance(process_results),
            'stakeholder_impact': self.score_stakeholder_impact(impact_results)
        }
        
        audit_results['overall_score'] = np.mean(list(audit_results['category_scores'].values()))
        
        # Generate recommendations
        audit_results['recommendations'] = self.generate_audit_recommendations(audit_results)
        
        return audit_results
    
    def technical_fairness_assessment(self, model, test_data, sensitive_attrs):
        results = {}
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(test_data)[:, 1]
            y_pred = (y_prob > 0.5).astype(int)
        else:
            y_pred = model.predict(test_data)
            y_prob = None
        
        # Statistical parity
        results['statistical_parity'] = self.measure_statistical_parity(y_pred, sensitive_attrs)
        
        # Equalized odds (if ground truth available)
        if 'true_labels' in test_data.columns:
            results['equalized_odds'] = self.measure_equalized_odds(
                test_data['true_labels'], y_pred, sensitive_attrs
            )
        
        # Calibration (if probabilities available)
        if y_prob is not None and 'true_labels' in test_data.columns:
            results['calibration'] = self.measure_calibration(
                test_data['true_labels'], y_prob, sensitive_attrs
            )
        
        # Performance disparities
        results['performance_disparities'] = self.measure_performance_disparities(
            model, test_data, sensitive_attrs
        )
        
        return results
    
    def process_audit(self):
        # Audit development and deployment processes
        process_scores = {}
        
        for category, checklist in self.audit_checklist.items():
            # In practice, this would involve interviews, documentation review, etc.
            # For demo, we'll simulate scores
            scores = []
            for item in checklist:
                # Simulate audit findings
                score = np.random.uniform(0.6, 1.0)  # Most items partially completed
                scores.append(score)
            
            process_scores[category] = {
                'average_score': np.mean(scores),
                'item_scores': dict(zip(checklist, scores)),
                'completion_rate': sum(1 for s in scores if s > 0.8) / len(scores)
            }
        
        return process_scores
    
    def stakeholder_impact_assessment(self):
        # Assess impact on different stakeholder groups
        stakeholder_groups = [
            'end_users', 'affected_communities', 'business_users', 
            'regulators', 'developers', 'society'
        ]
        
        impact_assessment = {}
        
        for group in stakeholder_groups:
            # In practice, this would involve surveys, interviews, focus groups
            impact_assessment[group] = {
                'positive_impact_score': np.random.uniform(0.5, 0.9),
                'negative_impact_score': np.random.uniform(0.1, 0.4),
                'fairness_perception': np.random.uniform(0.6, 0.8),
                'trust_level': np.random.uniform(0.5, 0.8),
                'concerns_raised': np.random.choice([True, False], p=[0.3, 0.7])
            }
        
        return impact_assessment
    
    def score_technical_fairness(self, technical_results):
        scores = []
        
        for metric, results in technical_results.items():
            if isinstance(results, dict):
                # Extract numerical fairness scores
                if 'fairness_score' in results:
                    scores.append(results['fairness_score'])
                elif 'violation' in results:
                    scores.append(1 - results['violation'])  # Convert violation to score
        
        return np.mean(scores) if scores else 0.5
    
    def generate_audit_recommendations(self, audit_results):
        recommendations = []
        
        overall_score = audit_results['overall_score']
        
        if overall_score < 0.7:
            recommendations.append("Overall fairness score below acceptable threshold - comprehensive review needed")
        
        # Category-specific recommendations
        for category, score in audit_results['category_scores'].items():
            if score < 0.6:
                recommendations.append(f"Low score in {category} - immediate action required")
            elif score < 0.8:
                recommendations.append(f"Moderate concerns in {category} - improvement needed")
        
        # Technical recommendations
        technical_findings = audit_results['detailed_findings']['technical']
        
        for metric, results in technical_findings.items():
            if isinstance(results, dict) and 'violation' in results:
                if results['violation'] > 0.1:
                    recommendations.append(f"High {metric} violation - consider mitigation strategies")
        
        return recommendations
    
    def generate_compliance_report(self, audit_results, regulatory_framework='GDPR'):
        # Generate compliance report for specific regulatory framework
        
        compliance_mapping = {
            'GDPR': {
                'data_protection': ['data_fairness'],
                'algorithmic_accountability': ['model_fairness', 'deployment_fairness'],
                'transparency': ['governance'],
                'non_discrimination': ['technical_fairness']
            },
            'EU_AI_Act': {
                'risk_assessment': ['model_fairness'],
                'human_oversight': ['governance'],
                'accuracy_robustness': ['technical_fairness'],
                'transparency': ['governance']
            }
        }
        
        if regulatory_framework not in compliance_mapping:
            return "Regulatory framework not supported"
        
        framework_requirements = compliance_mapping[regulatory_framework]
        compliance_status = {}
        
        for requirement, categories in framework_requirements.items():
            scores = [
                audit_results['category_scores'].get(cat, 0) 
                for cat in categories
            ]
            avg_score = np.mean(scores)
            
            compliance_status[requirement] = {
                'score': avg_score,
                'status': 'compliant' if avg_score > 0.8 else 'non_compliant',
                'evidence': f"Based on assessment of {', '.join(categories)}"
            }
        
        return compliance_status

# Usage example
auditor = FairnessAudit()

# Conduct comprehensive audit
audit_results = auditor.conduct_comprehensive_audit(
    model=trained_model,
    test_data=test_dataset,
    sensitive_attrs=test_sensitive_attrs
)

print("Audit Results Summary:")
print(f"Overall Score: {audit_results['overall_score']:.3f}")
print("Category Scores:", audit_results['category_scores'])
print("Recommendations:", audit_results['recommendations'])

# Generate compliance report
compliance_report = auditor.generate_compliance_report(audit_results, 'GDPR')
print("GDPR Compliance Status:", compliance_report)
"""
        }
    ]
    
    for approach in monitoring_approaches:
        with st.expander(f"📈 {approach['approach']}"):
            st.markdown(approach['description'])
            
            st.markdown("**Key Components:**")
            for component in approach['components']:
                st.markdown(f"• {component}")
            
            st.markdown("**Implementation Example:**")
            st.code(approach['implementation'], language='python')

# Monitoring dashboard simulation
st.markdown("### 📊 Bias Monitoring Dashboard")

# Simulate monitoring metrics over time
dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
np.random.seed(42)

monitoring_data = pd.DataFrame({
    'Date': dates,
    'Demographic Parity Violation': np.random.normal(0.05, 0.02, len(dates)).clip(0, 0.2),
    'Equalized Odds Violation': np.random.normal(0.04, 0.015, len(dates)).clip(0, 0.15),
    'Calibration Error': np.random.normal(0.03, 0.01, len(dates)).clip(0, 0.1),
    'Overall Fairness Score': 1 - np.random.normal(0.08, 0.02, len(dates)).clip(0.05, 0.15)
})

fig = px.line(monitoring_data, x='Date', 
              y=['Demographic Parity Violation', 'Equalized Odds Violation', 'Calibration Error'],
              title="Bias Metrics Over Time",
              labels={'value': 'Violation Score', 'variable': 'Metric'})

# Add threshold lines
fig.add_hline(y=0.1, line_dash="dash", line_color="red", 
              annotation_text="Alert Threshold", annotation_position="top right")

st.plotly_chart(fig, use_container_width=True)

# Current status metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    current_dp = monitoring_data['Demographic Parity Violation'].iloc[-1]
    st.metric("Demographic Parity", f"{current_dp:.3f}", 
              delta=f"{current_dp - monitoring_data['Demographic Parity Violation'].iloc[-2]:.3f}")

with col2:
    current_eo = monitoring_data['Equalized Odds Violation'].iloc[-1]
    st.metric("Equalized Odds", f"{current_eo:.3f}",
              delta=f"{current_eo - monitoring_data['Equalized Odds Violation'].iloc[-2]:.3f}")

with col3:
    current_cal = monitoring_data['Calibration Error'].iloc[-1]
    st.metric("Calibration Error", f"{current_cal:.3f}",
              delta=f"{current_cal - monitoring_data['Calibration Error'].iloc[-2]:.3f}")

with col4:
    current_fair = monitoring_data['Overall Fairness Score'].iloc[-1]
    st.metric("Fairness Score", f"{current_fair:.3f}",
              delta=f"{current_fair - monitoring_data['Overall Fairness Score'].iloc[-2]:.3f}")

# Best practices
st.header("💡 Bias Mitigation Best Practices")

best_practices = [
    "**Proactive Approach**: Address bias early in the development lifecycle, not as an afterthought",
    "**Multi-metric Evaluation**: Use multiple fairness metrics as they can conflict with each other",
    "**Stakeholder Involvement**: Include affected communities in the design and evaluation process",
    "**Intersectional Analysis**: Consider intersections of multiple protected attributes",
    "**Context-Aware Solutions**: Tailor mitigation strategies to specific domain and use case",
    "**Continuous Monitoring**: Implement ongoing bias monitoring throughout the model lifecycle",
    "**Transparent Documentation**: Maintain clear documentation of bias assessment and mitigation efforts",
    "**Trade-off Awareness**: Understand and communicate trade-offs between fairness and other objectives",
    "**Regular Audits**: Conduct periodic fairness audits with external validators",
    "**Feedback Loops**: Establish mechanisms for receiving and acting on bias-related feedback"
]

for practice in best_practices:
    st.markdown(f"• {practice}")

# Resources
st.header("📚 Learning Resources")

resources = [
    {
        "title": "Fairness and Machine Learning: Limitations and Opportunities",
        "type": "Book",
        "description": "Comprehensive guide to algorithmic fairness by Barocas, Hardt, and Narayanan",
        "difficulty": "Advanced"
    },
    {
        "title": "AI Fairness 360 Toolkit",
        "type": "Tool",
        "description": "Open-source toolkit for detecting and mitigating bias in ML models",
        "difficulty": "Intermediate"
    },
    {
        "title": "Fairness Indicators",
        "type": "Tool",
        "description": "TensorFlow library for evaluating fairness metrics",
        "difficulty": "Intermediate"
    },
    {
        "title": "Algorithmic Impact Assessments",
        "type": "Framework",
        "description": "Systematic approach to evaluating algorithmic systems",
        "difficulty": "Beginner"
    }
]

for resource in resources:
    with st.expander(f"📖 {resource['title']}"):
        st.markdown(f"**Type:** {resource['type']}")
        st.markdown(f"**Description:** {resource['description']}")
        st.markdown(f"**Difficulty:** {resource['difficulty']}")