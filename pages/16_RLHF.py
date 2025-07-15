import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="RLHF - Reinforcement Learning from Human Feedback", page_icon="🎯", layout="wide")

st.title("🎯 Reinforcement Learning from Human Feedback (RLHF)")
st.markdown("### Aligning Language Models with Human Preferences")

# Overview
st.header("🎯 Overview")
st.markdown("""
Reinforcement Learning from Human Feedback (RLHF) is a technique used to train language models 
to produce outputs that align with human preferences and values. It has been crucial in creating 
helpful, harmless, and honest AI assistants like ChatGPT and Claude.
""")

# Core concepts
st.header("🧠 Core Concepts")

concept_tabs = st.tabs([
    "🔄 RLHF Process",
    "🏆 Reward Modeling", 
    "🎮 Policy Optimization",
    "📊 Evaluation Methods"
])

with concept_tabs[0]:
    st.subheader("🔄 The RLHF Process")
    
    st.markdown("""
    RLHF consists of three main stages that transform a base language model into 
    an aligned assistant that follows human preferences.
    """)
    
    # RLHF stages
    rlhf_stages = [
        {
            "stage": "Stage 1: Supervised Fine-tuning (SFT)",
            "description": "Train the model on high-quality demonstration data",
            "inputs": [
                "Pre-trained base model",
                "Human-written demonstrations",
                "High-quality instruction-response pairs"
            ],
            "process": [
                "Collect demonstrations of desired behavior",
                "Fine-tune base model on demonstrations",
                "Create initial aligned model",
                "Validate performance on held-out data"
            ],
            "outputs": [
                "SFT model with basic alignment",
                "Improved instruction following",
                "Better response quality baseline"
            ],
            "implementation": """
# Supervised Fine-tuning Implementation
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

class SFTTrainer:
    def __init__(self, model_name, learning_rate=1e-5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Add padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_data(self, demonstrations):
        # Format demonstrations as instruction-response pairs
        formatted_data = []
        
        for demo in demonstrations:
            # Create input-output pair
            input_text = f"Human: {demo['instruction']}\\n\\nAssistant: "
            full_text = input_text + demo['response'] + self.tokenizer.eos_token
            
            # Tokenize
            tokens = self.tokenizer.encode(full_text, truncation=True, max_length=2048)
            input_len = len(self.tokenizer.encode(input_text))
            
            formatted_data.append({
                'input_ids': tokens,
                'labels': [-100] * (input_len - 1) + tokens[input_len-1:],  # Mask instruction tokens
                'attention_mask': [1] * len(tokens)
            })
        
        return formatted_data
    
    def train_step(self, batch):
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, demonstrations, epochs=3, batch_size=4):
        data = self.prepare_data(demonstrations)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch in dataloader:
                # Move to device
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                loss = self.train_step(batch)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
            total_loss += avg_loss
        
        return total_loss / epochs

# Example usage
demonstrations = [
    {
        "instruction": "Explain what photosynthesis is",
        "response": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen..."
    },
    {
        "instruction": "How do I bake a chocolate cake?",
        "response": "To bake a chocolate cake, you'll need flour, sugar, cocoa powder, eggs, butter, and baking powder..."
    }
]

trainer = SFTTrainer("gpt2-medium")
avg_loss = trainer.train(demonstrations)
"""
        },
        {
            "stage": "Stage 2: Reward Model Training",
            "description": "Train a model to predict human preferences",
            "inputs": [
                "SFT model outputs",
                "Human preference comparisons",
                "Ranking data for response pairs"
            ],
            "process": [
                "Generate response pairs from SFT model",
                "Collect human preference rankings",
                "Train reward model on preference data",
                "Validate reward model accuracy"
            ],
            "outputs": [
                "Reward model that predicts human preferences",
                "Scalar reward scores for any response",
                "Preference prediction capability"
            ],
            "implementation": """
# Reward Model Training Implementation
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

class RewardModel(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        # Replace language modeling head with reward head
        hidden_size = self.base_model.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Freeze base model parameters (optional)
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, input_ids, attention_mask=None):
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use hidden state at the last token position
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        if attention_mask is not None:
            # Get last non-padded token for each sequence
            last_positions = attention_mask.sum(dim=1) - 1
            sequence_hidden = hidden_states[range(len(hidden_states)), last_positions]
        else:
            sequence_hidden = hidden_states[:, -1]  # Last token
        
        # Compute reward score
        reward = self.reward_head(sequence_hidden)
        return reward.squeeze(-1)

class RewardModelTrainer:
    def __init__(self, model_name, learning_rate=1e-5):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = RewardModel(model_name)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_comparison_data(self, comparisons):
        # Format preference comparison data
        formatted_data = []
        
        for comparison in comparisons:
            prompt = comparison['prompt']
            chosen = comparison['chosen_response']
            rejected = comparison['rejected_response']
            
            # Tokenize chosen and rejected responses
            chosen_text = f"Human: {prompt}\\n\\nAssistant: {chosen}"
            rejected_text = f"Human: {prompt}\\n\\nAssistant: {rejected}"
            
            chosen_tokens = self.tokenizer(
                chosen_text, 
                truncation=True, 
                max_length=1024, 
                padding=True, 
                return_tensors="pt"
            )
            
            rejected_tokens = self.tokenizer(
                rejected_text,
                truncation=True,
                max_length=1024,
                padding=True,
                return_tensors="pt"
            )
            
            formatted_data.append({
                'chosen_input_ids': chosen_tokens['input_ids'].squeeze(),
                'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(),
                'rejected_input_ids': rejected_tokens['input_ids'].squeeze(),
                'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze()
            })
        
        return formatted_data
    
    def compute_loss(self, chosen_rewards, rejected_rewards):
        # Bradley-Terry model loss
        # P(chosen > rejected) = sigmoid(reward_chosen - reward_rejected)
        logits = chosen_rewards - rejected_rewards
        loss = -F.logsigmoid(logits).mean()
        return loss
    
    def train_step(self, batch):
        self.optimizer.zero_grad()
        
        # Get rewards for chosen and rejected responses
        chosen_rewards = self.model(
            batch['chosen_input_ids'],
            batch['chosen_attention_mask']
        )
        
        rejected_rewards = self.model(
            batch['rejected_input_ids'],
            batch['rejected_attention_mask']
        )
        
        # Compute loss
        loss = self.compute_loss(chosen_rewards, rejected_rewards)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute accuracy (how often chosen > rejected)
        with torch.no_grad():
            correct = (chosen_rewards > rejected_rewards).float()
            accuracy = correct.mean().item()
        
        return loss.item(), accuracy
    
    def train(self, comparisons, epochs=3, batch_size=4):
        data = self.prepare_comparison_data(comparisons)
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            total_accuracy = 0
            
            for batch in dataloader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                loss, accuracy = self.train_step(batch)
                total_loss += loss
                total_accuracy += accuracy
            
            avg_loss = total_loss / len(dataloader)
            avg_accuracy = total_accuracy / len(dataloader)
            
            print(f"Epoch {epoch + 1}")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {avg_accuracy:.4f}")

# Example preference data
comparisons = [
    {
        "prompt": "Explain quantum computing",
        "chosen_response": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information...",
        "rejected_response": "Quantum computing is just really fast regular computing with quantum stuff."
    }
]

reward_trainer = RewardModelTrainer("gpt2-medium")
reward_trainer.train(comparisons)
"""
        },
        {
            "stage": "Stage 3: Reinforcement Learning",
            "description": "Optimize the policy using the reward model",
            "inputs": [
                "SFT model as initial policy",
                "Trained reward model",
                "Training prompts"
            ],
            "process": [
                "Generate responses from current policy",
                "Score responses with reward model",
                "Update policy to maximize rewards",
                "Apply KL divergence constraint"
            ],
            "outputs": [
                "Final RLHF-trained model",
                "Aligned behavior with human preferences",
                "Improved helpfulness and safety"
            ],
            "implementation": """
# PPO (Proximal Policy Optimization) for RLHF
import torch.distributions as dist

class PPOTrainer:
    def __init__(self, policy_model, reward_model, ref_model, kl_coeff=0.1):
        self.policy = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model  # Reference model for KL penalty
        self.kl_coeff = kl_coeff
        
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=1e-6)
        
    def generate_responses(self, prompts, max_length=512, num_return_sequences=1):
        responses = []
        
        for prompt in prompts:
            # Tokenize prompt
            inputs = self.policy.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.policy.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.policy.tokenizer.eos_token_id
                )
            
            # Decode response
            response_text = self.policy.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            responses.append({
                'prompt': prompt,
                'response': response_text,
                'input_ids': outputs[0]
            })
        
        return responses
    
    def compute_rewards(self, responses):
        rewards = []
        
        for response in responses:
            # Get reward from reward model
            full_text = response['prompt'] + response['response']
            inputs = self.reward_model.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            
            with torch.no_grad():
                reward = self.reward_model(**inputs)
                rewards.append(reward.item())
        
        return torch.tensor(rewards)
    
    def compute_kl_penalty(self, responses):
        kl_penalties = []
        
        for response in responses:
            input_ids = response['input_ids'].unsqueeze(0)
            
            # Get log probabilities from current policy
            with torch.no_grad():
                policy_outputs = self.policy(input_ids, labels=input_ids)
                policy_logprobs = -policy_outputs.loss
            
            # Get log probabilities from reference model
            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids, labels=input_ids)
                ref_logprobs = -ref_outputs.loss
            
            # Compute KL divergence
            kl = policy_logprobs - ref_logprobs
            kl_penalties.append(kl.item())
        
        return torch.tensor(kl_penalties)
    
    def ppo_update(self, responses, rewards, advantages, epochs=4):
        # Convert responses to training data
        input_ids = []
        attention_masks = []
        old_logprobs = []
        
        for response in responses:
            tokens = response['input_ids']
            input_ids.append(tokens)
            attention_masks.append(torch.ones_like(tokens))
            
            # Compute old log probabilities
            with torch.no_grad():
                outputs = self.policy(tokens.unsqueeze(0), labels=tokens.unsqueeze(0))
                old_logprobs.append(-outputs.loss)
        
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        old_logprobs = torch.stack(old_logprobs)
        
        # PPO updates
        for _ in range(epochs):
            # Forward pass
            outputs = self.policy(input_ids, attention_mask=attention_masks, labels=input_ids)
            new_logprobs = -outputs.loss
            
            # Compute ratio
            ratio = torch.exp(new_logprobs - old_logprobs)
            
            # Compute PPO loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        return policy_loss.item()
    
    def train_step(self, prompts):
        # Generate responses
        responses = self.generate_responses(prompts)
        
        # Compute rewards
        rewards = self.compute_rewards(responses)
        
        # Compute KL penalty
        kl_penalties = self.compute_kl_penalty(responses)
        
        # Final rewards with KL penalty
        final_rewards = rewards - self.kl_coeff * kl_penalties
        
        # Compute advantages (simplified - just use rewards)
        advantages = final_rewards - final_rewards.mean()
        advantages = advantages / (advantages.std() + 1e-8)
        
        # PPO update
        policy_loss = self.ppo_update(responses, final_rewards, advantages)
        
        return {
            'policy_loss': policy_loss,
            'avg_reward': rewards.mean().item(),
            'avg_kl': kl_penalties.mean().item()
        }

# Example RLHF training
prompts = [
    "Explain the benefits of renewable energy",
    "How can I improve my public speaking skills?",
    "What are the main causes of climate change?"
]

ppo_trainer = PPOTrainer(sft_model, reward_model, reference_model)

# Training loop
for step in range(1000):
    metrics = ppo_trainer.train_step(prompts)
    
    if step % 100 == 0:
        print(f"Step {step}")
        print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
        print(f"  Avg Reward: {metrics['avg_reward']:.4f}")
        print(f"  Avg KL: {metrics['avg_kl']:.4f}")
"""
        }
    ]
    
    for stage in rlhf_stages:
        with st.expander(f"🔄 {stage['stage']}"):
            st.markdown(stage['description'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Inputs:**")
                for inp in stage['inputs']:
                    st.markdown(f"• {inp}")
            
            with col2:
                st.markdown("**Process:**")
                for process in stage['process']:
                    st.markdown(f"• {process}")
            
            with col3:
                st.markdown("**Outputs:**")
                for output in stage['outputs']:
                    st.markdown(f"• {output}")
            
            st.markdown("**Implementation Example:**")
            st.code(stage['implementation'], language='python')

with concept_tabs[1]:
    st.subheader("🏆 Reward Modeling")
    
    st.markdown("""
    Reward modeling is the core of RLHF, translating human preferences into 
    scalar reward signals that can guide reinforcement learning.
    """)
    
    reward_topics = st.tabs([
        "📊 Preference Collection",
        "🧠 Model Architecture",
        "📈 Training Strategies",
        "⚖️ Challenges & Solutions"
    ])
    
    with reward_topics[0]:
        st.markdown("### 📊 Preference Collection Methods")
        
        collection_methods = [
            {
                "method": "Pairwise Comparisons",
                "description": "Humans compare two responses and choose the better one",
                "advantages": [
                    "Easier for humans than absolute scoring",
                    "More reliable and consistent judgments",
                    "Captures relative preferences well",
                    "Scalable with crowd workers"
                ],
                "disadvantages": [
                    "Requires many comparisons for coverage",
                    "May miss fine-grained quality differences",
                    "Intransitive preferences possible",
                    "Limited to binary choices"
                ],
                "implementation": """
# Pairwise Comparison Data Collection
class PreferenceCollector:
    def __init__(self, model, prompts):
        self.model = model
        self.prompts = prompts
        self.comparisons = []
    
    def generate_response_pairs(self, prompt, num_pairs=5):
        pairs = []
        
        for _ in range(num_pairs):
            # Generate two different responses
            response_a = self.model.generate(
                prompt, 
                temperature=0.8, 
                max_tokens=200
            )
            response_b = self.model.generate(
                prompt, 
                temperature=0.8, 
                max_tokens=200
            )
            
            pairs.append({
                'prompt': prompt,
                'response_a': response_a,
                'response_b': response_b
            })
        
        return pairs
    
    def collect_human_preferences(self, pairs):
        # In practice, this would interface with human annotators
        # Here we simulate the interface
        labeled_pairs = []
        
        for pair in pairs:
            print(f"Prompt: {pair['prompt']}")
            print(f"Response A: {pair['response_a']}")
            print(f"Response B: {pair['response_b']}")
            
            # Human annotator chooses A, B, or tie
            choice = input("Which response is better? (A/B/T for tie): ").upper()
            
            if choice == 'A':
                chosen = pair['response_a']
                rejected = pair['response_b']
            elif choice == 'B':
                chosen = pair['response_b']
                rejected = pair['response_a']
            else:  # Tie
                continue  # Skip ties for simplicity
            
            labeled_pairs.append({
                'prompt': pair['prompt'],
                'chosen': chosen,
                'rejected': rejected,
                'annotator_confidence': self.get_confidence_rating()
            })
        
        return labeled_pairs
    
    def get_confidence_rating(self):
        # Get annotator confidence in their judgment
        confidence = input("How confident are you? (1-5): ")
        return int(confidence) if confidence.isdigit() else 3
    
    def validate_consistency(self, comparisons):
        # Check for inconsistent preferences
        inconsistencies = []
        
        for i, comp1 in enumerate(comparisons):
            for j, comp2 in enumerate(comparisons[i+1:], i+1):
                if (comp1['prompt'] == comp2['prompt'] and
                    comp1['chosen'] == comp2['rejected'] and
                    comp1['rejected'] == comp2['chosen']):
                    inconsistencies.append((i, j))
        
        return inconsistencies
"""
            },
            {
                "method": "Absolute Scoring",
                "description": "Humans rate responses on absolute quality scales",
                "advantages": [
                    "Provides absolute quality measures",
                    "Can capture fine-grained differences",
                    "Efficient for single response evaluation",
                    "Allows for multi-dimensional scoring"
                ],
                "disadvantages": [
                    "Subjective scale interpretation",
                    "Annotator bias and drift",
                    "Harder to achieve consistency",
                    "Scale boundary effects"
                ],
                "implementation": """
# Absolute Scoring Data Collection
class AbsoluteScorer:
    def __init__(self, scoring_dimensions):
        self.dimensions = scoring_dimensions
        # e.g., ['helpfulness', 'accuracy', 'safety', 'clarity']
    
    def score_response(self, prompt, response):
        scores = {}
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("\\nPlease rate the response on each dimension (1-10):")
        
        for dimension in self.dimensions:
            score = input(f"{dimension.capitalize()}: ")
            scores[dimension] = int(score) if score.isdigit() else 5
        
        # Compute overall score
        overall_score = sum(scores.values()) / len(scores)
        scores['overall'] = overall_score
        
        return scores
    
    def collect_scores(self, prompt_response_pairs):
        scored_data = []
        
        for pair in prompt_response_pairs:
            scores = self.score_response(pair['prompt'], pair['response'])
            
            scored_data.append({
                'prompt': pair['prompt'],
                'response': pair['response'],
                'scores': scores,
                'timestamp': datetime.now().isoformat()
            })
        
        return scored_data
    
    def normalize_scores(self, scored_data, annotator_id):
        # Normalize scores to account for annotator bias
        all_scores = [item['scores']['overall'] for item in scored_data]
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        
        normalized_data = []
        for item in scored_data:
            normalized_overall = (item['scores']['overall'] - mean_score) / std_score
            
            normalized_item = item.copy()
            normalized_item['scores']['normalized_overall'] = normalized_overall
            normalized_item['annotator_stats'] = {
                'mean': mean_score,
                'std': std_score,
                'annotator_id': annotator_id
            }
            
            normalized_data.append(normalized_item)
        
        return normalized_data
"""
            },
            {
                "method": "Constitutional AI",
                "description": "Use AI to evaluate responses against constitutional principles",
                "advantages": [
                    "Scalable automated evaluation",
                    "Consistent application of principles",
                    "Reduces human annotation burden",
                    "Can incorporate complex criteria"
                ],
                "disadvantages": [
                    "Limited by AI evaluator capability",
                    "May perpetuate model biases",
                    "Less diverse than human judgment",
                    "Requires careful prompt engineering"
                ],
                "implementation": """
# Constitutional AI Evaluation
class ConstitutionalEvaluator:
    def __init__(self, evaluator_model, constitution):
        self.evaluator = evaluator_model
        self.constitution = constitution
        # Constitution: list of principles/criteria
    
    def evaluate_response(self, prompt, response):
        evaluations = {}
        
        for principle in self.constitution:
            eval_prompt = f'''
Evaluate the following response according to this principle: {principle}

Original Prompt: {prompt}
Response: {response}

Rate the response on how well it follows this principle (1-10):
'''
            
            evaluation = self.evaluator.generate(eval_prompt)
            try:
                score = float(evaluation.strip())
                evaluations[principle] = score
            except ValueError:
                evaluations[principle] = 5.0  # Default neutral score
        
        return evaluations
"""
            }
        ]