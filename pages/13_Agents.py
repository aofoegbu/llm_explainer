import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="AI Agents", page_icon="🤖", layout="wide")

st.title("🤖 AI Agents")
st.markdown("### Autonomous Reasoning and Decision-Making Systems")

# Overview
st.header("🎯 Overview")
st.markdown("""
AI Agents are autonomous systems that can perceive their environment, reason about it, 
make decisions, and take actions to achieve specific goals. Modern LLM-based agents 
combine language understanding with tool usage, planning, and execution capabilities.
""")

# Core concepts
st.header("🧠 Core Concepts")

concept_tabs = st.tabs([
    "🔍 What are AI Agents?",
    "🏗️ Agent Architectures", 
    "🧩 Agent Components",
    "🔄 Agent Workflows"
])

with concept_tabs[0]:
    st.subheader("🔍 Understanding AI Agents")
    
    st.markdown("""
    AI Agents are systems that exhibit autonomous behavior by perceiving, reasoning, 
    planning, and acting in their environment to achieve goals. They go beyond simple 
    input-output models to maintain state, use tools, and adapt their behavior.
    """)
    
    # Agent characteristics
    agent_characteristics = [
        {
            "characteristic": "Autonomy",
            "description": "Operates independently without constant human intervention",
            "examples": [
                "Self-directed task execution",
                "Goal-oriented decision making",
                "Error recovery and adaptation",
                "Resource management"
            ]
        },
        {
            "characteristic": "Reactivity",
            "description": "Responds to changes in the environment or context",
            "examples": [
                "Real-time adaptation to new information",
                "Context-aware behavior modification",
                "Dynamic strategy adjustment",
                "Feedback incorporation"
            ]
        },
        {
            "characteristic": "Proactivity",
            "description": "Takes initiative to achieve goals",
            "examples": [
                "Goal-driven behavior",
                "Planning and foresight",
                "Opportunity identification",
                "Preventive actions"
            ]
        },
        {
            "characteristic": "Social Ability",
            "description": "Interacts with humans and other agents",
            "examples": [
                "Natural language communication",
                "Collaboration with other agents",
                "Understanding social context",
                "Explanation and justification"
            ]
        }
    ]
    
    for char in agent_characteristics:
        with st.expander(f"🔍 {char['characteristic']}"):
            st.markdown(char['description'])
            st.markdown("**Examples:**")
            for example in char['examples']:
                st.markdown(f"• {example}")

with concept_tabs[1]:
    st.subheader("🏗️ Agent Architectures")
    
    architecture_tabs = st.tabs([
        "🔄 ReAct Agents",
        "🧠 Planning Agents",
        "🔗 Tool-Using Agents",
        "🤝 Multi-Agent Systems"
    ])
    
    with architecture_tabs[0]:
        st.markdown("### 🔄 ReAct (Reasoning + Acting) Agents")
        
        st.markdown("""
        ReAct agents interleave reasoning and acting, allowing them to think through 
        problems step-by-step while taking actions and incorporating observations.
        """)
        
        react_components = [
            {
                "component": "Thought",
                "description": "Internal reasoning about the current situation",
                "purpose": "Planning next actions and analyzing information",
                "example": "I need to find information about climate change impacts on agriculture."
            },
            {
                "component": "Action",
                "description": "External action taken in the environment",
                "purpose": "Gathering information or making changes",
                "example": "Search[climate change agriculture impacts]"
            },
            {
                "component": "Observation",
                "description": "Feedback from the environment after action",
                "purpose": "Incorporating new information into decision making",
                "example": "Found 15 research papers on climate impacts on crop yields."
            }
        ]
        
        for component in react_components:
            with st.expander(f"🔄 {component['component']}"):
                st.markdown(component['description'])
                st.markdown(f"**Purpose:** {component['purpose']}")
                st.markdown(f"**Example:** {component['example']}")
        
        st.markdown("### 💻 ReAct Implementation Example")
        st.code("""
class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.memory = []
    
    def run(self, query, max_iterations=10):
        self.memory.append(f"Question: {query}")
        
        for i in range(max_iterations):
            # Generate thought and action
            prompt = self.build_prompt()
            response = self.llm.generate(prompt)
            
            if "Thought:" in response:
                thought = self.extract_thought(response)
                self.memory.append(f"Thought: {thought}")
            
            if "Action:" in response:
                action = self.extract_action(response)
                self.memory.append(f"Action: {action}")
                
                # Execute action
                observation = self.execute_action(action)
                self.memory.append(f"Observation: {observation}")
                
                # Check if done
                if "Final Answer:" in response:
                    return self.extract_final_answer(response)
            
            elif "Final Answer:" in response:
                return self.extract_final_answer(response)
        
        return "Max iterations reached without final answer"
    
    def build_prompt(self):
        context = "\\n".join(self.memory[-10:])  # Last 10 entries
        return f'''
You are a helpful assistant that can think and take actions.

Available tools:
- Search[query]: Search for information
- Calculate[expression]: Perform calculations
- GetWeather[location]: Get weather information

Use this format:
Thought: [your reasoning]
Action: [tool_name[parameters]]
Observation: [result will be provided]
...
Final Answer: [your final response]

{context}

Thought:'''
    
    def execute_action(self, action):
        # Parse action and execute appropriate tool
        if action.startswith("Search["):
            query = action[7:-1]
            return self.tools['search'].run(query)
        elif action.startswith("Calculate["):
            expression = action[10:-1]
            return self.tools['calculator'].run(expression)
        # ... handle other tools
        else:
            return "Unknown action"
""", language='python')
    
    with architecture_tabs[1]:
        st.markdown("### 🧠 Planning Agents")
        
        st.markdown("""
        Planning agents create comprehensive plans before execution, decomposing 
        complex tasks into manageable sub-tasks and organizing them strategically.
        """)
        
        planning_types = [
            {
                "type": "Hierarchical Planning",
                "description": "Break down goals into sub-goals recursively",
                "approach": "Top-down task decomposition",
                "benefits": ["Clear goal structure", "Manageable complexity", "Parallel execution"],
                "challenges": ["Planning overhead", "Rigid structure", "Adaptation difficulty"]
            },
            {
                "type": "Sequential Planning",
                "description": "Create step-by-step execution sequences",
                "approach": "Linear task ordering with dependencies",
                "benefits": ["Simple execution", "Clear progress tracking", "Easy debugging"],
                "challenges": ["Limited flexibility", "Sequential bottlenecks", "Error propagation"]
            },
            {
                "type": "Contingency Planning",
                "description": "Plan for multiple scenarios and failure modes",
                "approach": "If-then planning with alternative paths",
                "benefits": ["Robust execution", "Error handling", "Adaptive behavior"],
                "challenges": ["Complex planning", "Resource overhead", "Decision complexity"]
            }
        ]
        
        for plan_type in planning_types:
            with st.expander(f"🧠 {plan_type['type']}"):
                st.markdown(plan_type['description'])
                st.markdown(f"**Approach:** {plan_type['approach']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Benefits:**")
                    for benefit in plan_type['benefits']:
                        st.markdown(f"• {benefit}")
                with col2:
                    st.markdown("**Challenges:**")
                    for challenge in plan_type['challenges']:
                        st.markdown(f"• {challenge}")
    
    with architecture_tabs[2]:
        st.markdown("### 🔗 Tool-Using Agents")
        
        st.markdown("""
        Tool-using agents extend LLM capabilities by integrating external tools, 
        APIs, and services to perform actions beyond text generation.
        """)
        
        tool_categories = [
            {
                "category": "Information Retrieval",
                "tools": [
                    "Web search engines",
                    "Database queries", 
                    "API calls",
                    "Document retrieval"
                ],
                "use_cases": [
                    "Real-time information lookup",
                    "Fact verification",
                    "Research assistance",
                    "Knowledge base access"
                ]
            },
            {
                "category": "Computation & Analysis",
                "tools": [
                    "Calculators and math engines",
                    "Data analysis libraries",
                    "Statistical tools",
                    "Machine learning models"
                ],
                "use_cases": [
                    "Mathematical computations",
                    "Data processing",
                    "Predictive modeling",
                    "Statistical analysis"
                ]
            },
            {
                "category": "Communication & Integration",
                "tools": [
                    "Email and messaging APIs",
                    "Calendar systems",
                    "Notification services",
                    "Workflow automation"
                ],
                "use_cases": [
                    "Automated communications",
                    "Scheduling and planning",
                    "Alert systems",
                    "Process automation"
                ]
            },
            {
                "category": "Content Generation",
                "tools": [
                    "Image generation APIs",
                    "Code execution environments",
                    "Document templates",
                    "Media processing tools"
                ],
                "use_cases": [
                    "Visual content creation",
                    "Code development",
                    "Document automation",
                    "Media manipulation"
                ]
            }
        ]
        
        for category in tool_categories:
            with st.expander(f"🔗 {category['category']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Available Tools:**")
                    for tool in category['tools']:
                        st.markdown(f"• {tool}")
                with col2:
                    st.markdown("**Use Cases:**")
                    for use_case in category['use_cases']:
                        st.markdown(f"• {use_case}")
    
    with architecture_tabs[3]:
        st.markdown("### 🤝 Multi-Agent Systems")
        
        st.markdown("""
        Multi-agent systems involve multiple AI agents working together, each with 
        specialized roles and capabilities, to solve complex problems collaboratively.
        """)
        
        multiagent_patterns = [
            {
                "pattern": "Hierarchical Organization",
                "description": "Manager agents coordinate worker agents",
                "structure": "Tree-like hierarchy with clear command structure",
                "advantages": ["Clear responsibility", "Efficient coordination", "Scalable organization"],
                "disadvantages": ["Single point of failure", "Communication bottlenecks", "Rigid structure"]
            },
            {
                "pattern": "Peer-to-Peer Collaboration",
                "description": "Agents work together as equals",
                "structure": "Flat network with direct agent-to-agent communication",
                "advantages": ["Flexible collaboration", "Fault tolerance", "Emergent behavior"],
                "disadvantages": ["Coordination complexity", "Potential conflicts", "Scalability limits"]
            },
            {
                "pattern": "Specialized Teams",
                "description": "Groups of agents with complementary skills",
                "structure": "Domain-specific teams with cross-team coordination",
                "advantages": ["Expertise focus", "Parallel processing", "Skill optimization"],
                "disadvantages": ["Integration challenges", "Team boundaries", "Communication overhead"]
            }
        ]
        
        for pattern in multiagent_patterns:
            with st.expander(f"🤝 {pattern['pattern']}"):
                st.markdown(pattern['description'])
                st.markdown(f"**Structure:** {pattern['structure']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Advantages:**")
                    for advantage in pattern['advantages']:
                        st.markdown(f"• {advantage}")
                with col2:
                    st.markdown("**Disadvantages:**")
                    for disadvantage in pattern['disadvantages']:
                        st.markdown(f"• {disadvantage}")

with concept_tabs[2]:
    st.subheader("🧩 Agent Components")
    
    component_tabs = st.tabs([
        "🧠 Memory Systems",
        "🎯 Goal Management",
        "🔧 Tool Integration",
        "📊 Decision Making"
    ])
    
    with component_tabs[0]:
        st.markdown("### 🧠 Memory Systems")
        
        memory_types = [
            {
                "type": "Working Memory",
                "description": "Short-term context for current task execution",
                "characteristics": [
                    "Limited capacity (context window)",
                    "Temporary storage",
                    "Fast access",
                    "Task-specific information"
                ],
                "implementation": """
class WorkingMemory:
    def __init__(self, max_tokens=4000):
        self.max_tokens = max_tokens
        self.context = []
        self.current_tokens = 0
    
    def add(self, item, tokens):
        while self.current_tokens + tokens > self.max_tokens and self.context:
            removed = self.context.pop(0)
            self.current_tokens -= removed['tokens']
        
        self.context.append({'content': item, 'tokens': tokens})
        self.current_tokens += tokens
    
    def get_context(self):
        return [item['content'] for item in self.context]
"""
            },
            {
                "type": "Episodic Memory",
                "description": "Memory of specific experiences and events",
                "characteristics": [
                    "Event-based storage",
                    "Temporal organization",
                    "Rich contextual details",
                    "Personal experiences"
                ],
                "implementation": """
class EpisodicMemory:
    def __init__(self):
        self.episodes = []
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def store_episode(self, event, context, outcome, timestamp):
        episode = {
            'event': event,
            'context': context,
            'outcome': outcome,
            'timestamp': timestamp,
            'embedding': self.embedder.encode([event + ' ' + context])[0]
        }
        self.episodes.append(episode)
    
    def retrieve_similar_episodes(self, query, top_k=5):
        query_embedding = self.embedder.encode([query])[0]
        
        similarities = []
        for episode in self.episodes:
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                episode['embedding'].reshape(1, -1)
            )[0][0]
            similarities.append((similarity, episode))
        
        similarities.sort(reverse=True)
        return [ep for _, ep in similarities[:top_k]]
"""
            },
            {
                "type": "Semantic Memory",
                "description": "General knowledge and learned concepts",
                "characteristics": [
                    "Fact-based storage",
                    "Conceptual relationships",
                    "Domain knowledge",
                    "Generalized learning"
                ],
                "implementation": """
class SemanticMemory:
    def __init__(self):
        self.knowledge_graph = {}
        self.concepts = {}
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_concept(self, concept, description, relations=None):
        self.concepts[concept] = {
            'description': description,
            'embedding': self.embedder.encode([description])[0],
            'relations': relations or {}
        }
    
    def add_relation(self, concept1, concept2, relation_type):
        if concept1 not in self.knowledge_graph:
            self.knowledge_graph[concept1] = {}
        self.knowledge_graph[concept1][concept2] = relation_type
    
    def find_related_concepts(self, query, threshold=0.7):
        query_embedding = self.embedder.encode([query])[0]
        
        related = []
        for concept, data in self.concepts.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                data['embedding'].reshape(1, -1)
            )[0][0]
            
            if similarity > threshold:
                related.append((concept, similarity))
        
        return sorted(related, key=lambda x: x[1], reverse=True)
"""
            }
        ]
        
        for memory_type in memory_types:
            with st.expander(f"🧠 {memory_type['type']}"):
                st.markdown(memory_type['description'])
                
                st.markdown("**Key Characteristics:**")
                for char in memory_type['characteristics']:
                    st.markdown(f"• {char}")
                
                st.markdown("**Implementation Example:**")
                st.code(memory_type['implementation'], language='python')
    
    with component_tabs[1]:
        st.markdown("### 🎯 Goal Management")
        
        goal_management_aspects = [
            {
                "aspect": "Goal Representation",
                "description": "How agents represent and structure their objectives",
                "approaches": [
                    "Logical predicates and conditions",
                    "Natural language descriptions",
                    "Hierarchical goal trees", 
                    "Utility functions and rewards"
                ],
                "example": """
class Goal:
    def __init__(self, description, priority=1.0, deadline=None):
        self.description = description
        self.priority = priority
        self.deadline = deadline
        self.status = "active"  # active, completed, failed, paused
        self.sub_goals = []
        self.dependencies = []
        self.success_criteria = []
    
    def add_sub_goal(self, sub_goal):
        self.sub_goals.append(sub_goal)
    
    def add_dependency(self, dependency):
        self.dependencies.append(dependency)
    
    def is_achievable(self):
        return all(dep.status == "completed" for dep in self.dependencies)
    
    def evaluate_progress(self):
        if not self.sub_goals:
            return self.check_success_criteria()
        
        completed_subgoals = sum(1 for sg in self.sub_goals if sg.status == "completed")
        return completed_subgoals / len(self.sub_goals)
"""
            },
            {
                "aspect": "Goal Planning",
                "description": "Breaking down goals into actionable steps",
                "approaches": [
                    "Forward chaining from current state",
                    "Backward chaining from goal state",
                    "Means-ends analysis",
                    "Hierarchical task networks"
                ],
                "example": """
class GoalPlanner:
    def __init__(self, actions, world_state):
        self.actions = actions
        self.world_state = world_state
    
    def plan(self, goal):
        return self.backward_chain(goal, [])
    
    def backward_chain(self, goal, plan):
        if self.is_satisfied(goal):
            return plan
        
        for action in self.actions:
            if self.can_achieve(action, goal):
                new_plan = [action] + plan
                preconditions = action.get_preconditions()
                
                sub_plan = []
                for precondition in preconditions:
                    if not self.is_satisfied(precondition):
                        sub_sub_plan = self.backward_chain(precondition, [])
                        if sub_sub_plan is None:
                            break
                        sub_plan.extend(sub_sub_plan)
                else:
                    return sub_plan + new_plan
        
        return None  # No plan found
"""
            },
            {
                "aspect": "Goal Prioritization",
                "description": "Managing multiple competing objectives",
                "approaches": [
                    "Priority scoring and ranking",
                    "Deadline-based scheduling",
                    "Resource optimization",
                    "Multi-objective optimization"
                ],
                "example": """
class GoalScheduler:
    def __init__(self):
        self.goals = []
        self.resources = {}
    
    def add_goal(self, goal):
        self.goals.append(goal)
        self.reorder_goals()
    
    def reorder_goals(self):
        def priority_score(goal):
            urgency = self.calculate_urgency(goal)
            importance = goal.priority
            feasibility = self.calculate_feasibility(goal)
            return urgency * importance * feasibility
        
        self.goals.sort(key=priority_score, reverse=True)
    
    def calculate_urgency(self, goal):
        if goal.deadline is None:
            return 1.0
        
        time_remaining = (goal.deadline - datetime.now()).total_seconds()
        estimated_time = self.estimate_completion_time(goal)
        
        if time_remaining <= 0:
            return float('inf')  # Overdue
        
        return estimated_time / time_remaining
    
    def get_next_goal(self):
        for goal in self.goals:
            if goal.status == "active" and goal.is_achievable():
                return goal
        return None
"""
            }
        ]
        
        for aspect in goal_management_aspects:
            with st.expander(f"🎯 {aspect['aspect']}"):
                st.markdown(aspect['description'])
                
                st.markdown("**Approaches:**")
                for approach in aspect['approaches']:
                    st.markdown(f"• {approach}")
                
                st.markdown("**Implementation Example:**")
                st.code(aspect['example'], language='python')
    
    with component_tabs[2]:
        st.markdown("### 🔧 Tool Integration")
        
        st.markdown("""
        Tool integration allows agents to extend their capabilities beyond text generation
        by interacting with external systems, APIs, and services.
        """)
        
        integration_patterns = [
            {
                "pattern": "Function Calling",
                "description": "Direct invocation of predefined functions",
                "pros": ["Type safety", "Clear interfaces", "Easy debugging"],
                "cons": ["Limited flexibility", "Predefined tools only", "Static binding"],
                "code": """
class FunctionCallingAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tools = {
            'search_web': self.search_web,
            'calculate': self.calculate,
            'send_email': self.send_email
        }
    
    def search_web(self, query: str) -> str:
        # Implementation for web search
        return f"Search results for: {query}"
    
    def calculate(self, expression: str) -> float:
        # Safe mathematical calculation
        try:
            return eval(expression)  # In practice, use a safe evaluator
        except:
            return "Error in calculation"
    
    def run(self, query):
        function_descriptions = self.get_function_descriptions()
        
        prompt = f'''
        You have access to the following functions:
        {function_descriptions}
        
        User query: {query}
        
        Choose the appropriate function to call and provide parameters.
        Format: function_name(param1="value1", param2="value2")
        '''
        
        response = self.llm.generate(prompt)
        function_call = self.parse_function_call(response)
        
        if function_call:
            result = self.execute_function(function_call)
            return result
        
        return "No appropriate function found"
"""
            },
            {
                "pattern": "Natural Language APIs",
                "description": "Tools accessed through natural language interfaces",
                "pros": ["Flexible usage", "Natural interaction", "Easy extension"],
                "cons": ["Parsing complexity", "Error handling", "Ambiguity issues"],
                "code": """
class NLAPIAgent:
    def __init__(self, llm):
        self.llm = llm
        self.api_registry = {}
    
    def register_api(self, name, endpoint, description, examples):
        self.api_registry[name] = {
            'endpoint': endpoint,
            'description': description,
            'examples': examples
        }
    
    def run(self, query):
        api_descriptions = self.build_api_descriptions()
        
        prompt = f'''
        Available APIs:
        {api_descriptions}
        
        User request: {query}
        
        Choose the best API and format a natural language request.
        Format: API_NAME: natural language request
        '''
        
        response = self.llm.generate(prompt)
        api_call = self.parse_api_call(response)
        
        if api_call:
            api_name, nl_request = api_call
            result = self.execute_nl_api(api_name, nl_request)
            return result
        
        return "Could not determine appropriate API"
    
    def execute_nl_api(self, api_name, request):
        api_info = self.api_registry[api_name]
        
        # Convert natural language to API parameters
        params = self.nl_to_params(request, api_info['examples'])
        
        # Make API call
        response = requests.post(api_info['endpoint'], json=params)
        return response.json()
"""
            },
            {
                "pattern": "Code Generation",
                "description": "Generate and execute code to use tools",
                "pros": ["Maximum flexibility", "Dynamic capabilities", "Full programming power"],
                "cons": ["Security risks", "Execution overhead", "Error complexity"],
                "code": """
class CodeGenAgent:
    def __init__(self, llm):
        self.llm = llm
        self.sandbox = CodeSandbox()
        self.available_libraries = [
            'requests', 'pandas', 'numpy', 'matplotlib', 'json'
        ]
    
    def run(self, query):
        code_prompt = f'''
        Generate Python code to answer this query: {query}
        
        Available libraries: {', '.join(self.available_libraries)}
        
        Rules:
        - Only use approved libraries
        - Handle errors gracefully
        - Return results as strings
        - No file system access
        
        Code:
        '''
        
        code = self.llm.generate(code_prompt)
        cleaned_code = self.clean_and_validate_code(code)
        
        try:
            result = self.sandbox.execute(cleaned_code)
            return result
        except Exception as e:
            return f"Execution error: {str(e)}"
    
    def clean_and_validate_code(self, code):
        # Remove dangerous operations
        forbidden_patterns = ['import os', 'import sys', 'eval(', 'exec(']
        
        for pattern in forbidden_patterns:
            if pattern in code:
                raise SecurityError(f"Forbidden pattern: {pattern}")
        
        return code
"""
            }
        ]
        
        for pattern in integration_patterns:
            with st.expander(f"🔧 {pattern['pattern']}"):
                st.markdown(pattern['description'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Pros:**")
                    for pro in pattern['pros']:
                        st.markdown(f"• {pro}")
                with col2:
                    st.markdown("**Cons:**")
                    for con in pattern['cons']:
                        st.markdown(f"• {con}")
                
                st.markdown("**Implementation:**")
                st.code(pattern['code'], language='python')
    
    with component_tabs[3]:
        st.markdown("### 📊 Decision Making")
        
        decision_frameworks = [
            {
                "framework": "Rule-Based Decisions",
                "description": "Decisions based on predefined rules and conditions",
                "when_to_use": "Clear, well-defined scenarios with known optimal responses",
                "advantages": ["Predictable behavior", "Fast execution", "Easy to debug"],
                "disadvantages": ["Limited flexibility", "Rule maintenance overhead", "Poor adaptation"],
                "implementation": """
class RuleBasedDecisionMaker:
    def __init__(self):
        self.rules = []
    
    def add_rule(self, condition, action, priority=1):
        self.rules.append({
            'condition': condition,
            'action': action,
            'priority': priority
        })
    
    def decide(self, context):
        applicable_rules = []
        
        for rule in self.rules:
            if rule['condition'](context):
                applicable_rules.append(rule)
        
        if not applicable_rules:
            return None
        
        # Select highest priority rule
        best_rule = max(applicable_rules, key=lambda r: r['priority'])
        return best_rule['action']
    
    def execute_decision(self, context):
        decision = self.decide(context)
        if decision:
            return decision(context)
        return "No applicable rule found"
"""
            },
            {
                "framework": "Utility-Based Decisions",
                "description": "Decisions that maximize expected utility or value",
                "when_to_use": "Trade-off scenarios with quantifiable outcomes",
                "advantages": ["Optimal decisions", "Handles uncertainty", "Quantitative reasoning"],
                "disadvantages": ["Utility function design", "Computational complexity", "Preference modeling"],
                "implementation": """
class UtilityBasedDecisionMaker:
    def __init__(self):
        self.utility_functions = {}
        self.probability_model = None
    
    def add_utility_function(self, outcome_type, function):
        self.utility_functions[outcome_type] = function
    
    def calculate_expected_utility(self, action, context):
        possible_outcomes = self.get_possible_outcomes(action, context)
        expected_utility = 0
        
        for outcome in possible_outcomes:
            probability = self.probability_model.get_probability(outcome, action, context)
            utility = 0
            
            for outcome_type, value in outcome.items():
                if outcome_type in self.utility_functions:
                    utility += self.utility_functions[outcome_type](value)
            
            expected_utility += probability * utility
        
        return expected_utility
    
    def decide(self, actions, context):
        best_action = None
        best_utility = float('-inf')
        
        for action in actions:
            utility = self.calculate_expected_utility(action, context)
            if utility > best_utility:
                best_utility = utility
                best_action = action
        
        return best_action, best_utility
"""
            },
            {
                "framework": "Learning-Based Decisions",
                "description": "Decisions improved through experience and learning",
                "when_to_use": "Dynamic environments with feedback availability",
                "advantages": ["Adaptive behavior", "Performance improvement", "Handles novelty"],
                "disadvantages": ["Learning time", "Exploration vs exploitation", "Data requirements"],
                "implementation": """
class LearningBasedDecisionMaker:
    def __init__(self, learning_rate=0.1, epsilon=0.1):
        self.q_table = {}  # State-action values
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Exploration rate
        self.experience_buffer = []
    
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)
    
    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        
        # Get max Q-value for next state
        next_actions = self.get_possible_actions(next_state)
        max_next_q = max([self.get_q_value(next_state, a) for a in next_actions], default=0)
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + 0.9 * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
    
    def decide(self, state):
        actions = self.get_possible_actions(state)
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        # Exploit: choose action with highest Q-value
        best_action = max(actions, key=lambda a: self.get_q_value(state, a))
        return best_action
    
    def learn_from_experience(self, state, action, reward, next_state):
        self.update_q_value(state, action, reward, next_state)
        
        # Store experience for potential replay
        self.experience_buffer.append((state, action, reward, next_state))
        if len(self.experience_buffer) > 10000:
            self.experience_buffer.pop(0)
"""
            }
        ]
        
        for framework in decision_frameworks:
            with st.expander(f"📊 {framework['framework']}"):
                st.markdown(framework['description'])
                st.markdown(f"**When to use:** {framework['when_to_use']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Advantages:**")
                    for advantage in framework['advantages']:
                        st.markdown(f"• {advantage}")
                with col2:
                    st.markdown("**Disadvantages:**")
                    for disadvantage in framework['disadvantages']:
                        st.markdown(f"• {disadvantage}")
                
                st.markdown("**Implementation:**")
                st.code(framework['implementation'], language='python')

with concept_tabs[3]:
    st.subheader("🔄 Agent Workflows")
    
    st.markdown("""
    Agent workflows define how agents process information, make decisions, 
    and take actions in their environment over time.
    """)
    
    # Workflow visualization
    workflow_stages = [
        {
            "stage": "Perception",
            "description": "Gathering and processing environmental information",
            "activities": [
                "Sensor data collection",
                "Information parsing",
                "Context understanding",
                "Situational awareness"
            ]
        },
        {
            "stage": "Reasoning",
            "description": "Analyzing information and generating insights",
            "activities": [
                "Pattern recognition",
                "Causal analysis",
                "Hypothesis generation",
                "Logical inference"
            ]
        },
        {
            "stage": "Planning",
            "description": "Developing strategies and action sequences",
            "activities": [
                "Goal formulation",
                "Strategy selection",
                "Resource allocation",
                "Timeline creation"
            ]
        },
        {
            "stage": "Action",
            "description": "Executing planned activities and interventions",
            "activities": [
                "Tool utilization",
                "Communication",
                "Environment modification",
                "Progress monitoring"
            ]
        },
        {
            "stage": "Learning",
            "description": "Updating knowledge and improving performance",
            "activities": [
                "Outcome evaluation",
                "Model updating",
                "Experience storage",
                "Strategy refinement"
            ]
        }
    ]
    
    for stage in workflow_stages:
        with st.expander(f"🔄 {stage['stage']}"):
            st.markdown(stage['description'])
            st.markdown("**Key Activities:**")
            for activity in stage['activities']:
                st.markdown(f"• {activity}")

# Implementation guide
st.header("🛠️ Implementation Guide")

implementation_tabs = st.tabs([
    "🚀 Building Your First Agent",
    "🔧 Advanced Agent Patterns",
    "📊 Agent Evaluation",
    "🏭 Production Deployment"
])

with implementation_tabs[0]:
    st.subheader("🚀 Building Your First Agent")
    
    st.markdown("### Step-by-Step Agent Creation")
    
    basic_agent_code = """
# Simple Agent Implementation
import openai
import json
import time

class SimpleAgent:
    def __init__(self, api_key, system_prompt):
        self.client = openai.OpenAI(api_key=api_key)
        self.system_prompt = system_prompt
        self.conversation_history = []
        self.tools = {}
    
    def add_tool(self, name, function, description):
        self.tools[name] = {
            'function': function,
            'description': description
        }
    
    def run(self, user_input, max_iterations=5):
        self.conversation_history.append({
            'role': 'user',
            'content': user_input
        })
        
        for iteration in range(max_iterations):
            # Get agent response
            response = self.get_agent_response()
            
            # Check if agent wants to use a tool
            if self.should_use_tool(response):
                tool_result = self.execute_tool(response)
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': response
                })
                self.conversation_history.append({
                    'role': 'system',
                    'content': f"Tool result: {tool_result}"
                })
                continue
            
            # Final response
            self.conversation_history.append({
                'role': 'assistant', 
                'content': response
            })
            return response
        
        return "Max iterations reached"
    
    def get_agent_response(self):
        messages = [
            {'role': 'system', 'content': self.build_system_message()}
        ] + self.conversation_history
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.1
        )
        
        return response.choices[0].message.content
    
    def build_system_message(self):
        tool_descriptions = "\\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])
        
        return f'''
{self.system_prompt}

Available tools:
{tool_descriptions}

To use a tool, respond with: TOOL_USE: tool_name(parameters)
Otherwise, provide your final answer.
'''
    
    def should_use_tool(self, response):
        return response.startswith("TOOL_USE:")
    
    def execute_tool(self, response):
        # Parse tool call from response
        tool_call = response.replace("TOOL_USE:", "").strip()
        tool_name = tool_call.split("(")[0]
        
        if tool_name in self.tools:
            try:
                # Simple parameter parsing (in practice, use proper parsing)
                params = tool_call.split("(")[1].rstrip(")")
                result = self.tools[tool_name]['function'](params)
                return result
            except Exception as e:
                return f"Tool execution error: {str(e)}"
        
        return "Tool not found"

# Example usage
def web_search(query):
    # Simulate web search
    return f"Search results for '{query}': Found 5 relevant articles about the topic."

def calculator(expression):
    try:
        return str(eval(expression))  # In practice, use safe evaluation
    except:
        return "Invalid mathematical expression"

# Create and configure agent
agent = SimpleAgent(
    api_key="your-openai-api-key",
    system_prompt="You are a helpful research assistant that can search the web and perform calculations."
)

agent.add_tool("search", web_search, "Search the web for information")
agent.add_tool("calculate", calculator, "Perform mathematical calculations")

# Run the agent
result = agent.run("What is 15% of 250, and can you find information about AI agents?")
print(result)
"""
    
    st.code(basic_agent_code, language='python')
    
    st.markdown("### 🔧 Key Components Explained")
    
    components = [
        {
            "component": "System Prompt",
            "purpose": "Defines the agent's role, capabilities, and behavior",
            "best_practices": [
                "Be specific about the agent's role and expertise",
                "Include clear instructions for tool usage",
                "Specify output format requirements",
                "Add safety and ethical guidelines"
            ]
        },
        {
            "component": "Tool Integration", 
            "purpose": "Extends agent capabilities beyond text generation",
            "best_practices": [
                "Provide clear tool descriptions",
                "Implement robust error handling",
                "Use consistent parameter formats",
                "Add input validation and sanitization"
            ]
        },
        {
            "component": "Conversation Management",
            "purpose": "Maintains context and manages multi-turn interactions",
            "best_practices": [
                "Limit context window to prevent token overflow",
                "Summarize long conversations",
                "Preserve important information across turns",
                "Implement graceful degradation"
            ]
        },
        {
            "component": "Iteration Control",
            "purpose": "Prevents infinite loops and manages execution flow",
            "best_practices": [
                "Set reasonable iteration limits",
                "Implement convergence detection",
                "Add timeout mechanisms",
                "Provide progress feedback"
            ]
        }
    ]
    
    for component in components:
        with st.expander(f"🔧 {component['component']}"):
            st.markdown(f"**Purpose:** {component['purpose']}")
            st.markdown("**Best Practices:**")
            for practice in component['best_practices']:
                st.markdown(f"• {practice}")

with implementation_tabs[1]:
    st.subheader("🔧 Advanced Agent Patterns")
    
    advanced_patterns = [
        {
            "pattern": "Self-Reflection Agent",
            "description": "Agent that evaluates and improves its own performance",
            "key_features": [
                "Performance self-assessment",
                "Strategy modification",
                "Error analysis and correction",
                "Continuous improvement"
            ],
            "implementation": """
class SelfReflectiveAgent:
    def __init__(self, llm):
        self.llm = llm
        self.performance_history = []
        self.strategies = {}
        self.current_strategy = "default"
    
    def run_with_reflection(self, task):
        # Initial attempt
        result = self.execute_task(task)
        
        # Self-evaluation
        evaluation = self.self_evaluate(task, result)
        
        # If performance is poor, reflect and improve
        if evaluation['score'] < 0.7:
            reflection = self.reflect_on_performance(task, result, evaluation)
            improved_result = self.retry_with_improvements(task, reflection)
            return improved_result
        
        return result
    
    def self_evaluate(self, task, result):
        eval_prompt = f'''
        Task: {task}
        Result: {result}
        
        Evaluate this performance on a scale of 0-1 considering:
        - Completeness: Did it fully address the task?
        - Accuracy: Is the information correct?
        - Clarity: Is the response clear and well-structured?
        - Efficiency: Was this the best approach?
        
        Return JSON: {{"score": 0.8, "strengths": ["..."], "weaknesses": ["..."]}}
        '''
        
        evaluation_text = self.llm.generate(eval_prompt)
        return json.loads(evaluation_text)
    
    def reflect_on_performance(self, task, result, evaluation):
        reflection_prompt = f'''
        Task: {task}
        My result: {result}
        Evaluation: {evaluation}
        
        Analyze what went wrong and suggest specific improvements:
        1. What was the root cause of poor performance?
        2. What alternative approach could work better?
        3. What additional information or tools might help?
        4. How should I modify my strategy?
        
        Provide specific, actionable recommendations.
        '''
        
        return self.llm.generate(reflection_prompt)
    
    def retry_with_improvements(self, task, reflection):
        improved_prompt = f'''
        Original task: {task}
        
        Based on reflection: {reflection}
        
        Now attempt the task again with improvements:
        '''
        
        return self.llm.generate(improved_prompt)
"""
        },
        {
            "pattern": "Collaborative Agent Swarm",
            "description": "Multiple specialized agents working together",
            "key_features": [
                "Role specialization",
                "Inter-agent communication",
                "Task distribution",
                "Collective decision making"
            ],
            "implementation": """
class AgentSwarm:
    def __init__(self):
        self.agents = {}
        self.communication_protocol = MessageProtocol()
        self.task_distributor = TaskDistributor()
        self.coordinator = SwarmCoordinator()
    
    def add_agent(self, name, agent, specialization):
        self.agents[name] = {
            'agent': agent,
            'specialization': specialization,
            'workload': 0,
            'performance_history': []
        }
    
    def solve_complex_task(self, task):
        # Analyze task and identify required capabilities
        task_analysis = self.analyze_task_requirements(task)
        
        # Assign subtasks to appropriate agents
        assignments = self.task_distributor.assign_subtasks(
            task_analysis, self.agents
        )
        
        # Execute subtasks in parallel
        results = {}
        for agent_name, subtasks in assignments.items():
            agent = self.agents[agent_name]['agent']
            agent_results = []
            
            for subtask in subtasks:
                result = agent.execute(subtask)
                agent_results.append(result)
            
            results[agent_name] = agent_results
        
        # Coordinate and synthesize results
        final_result = self.coordinator.synthesize_results(
            task, results, self.agents
        )
        
        return final_result
    
    def facilitate_agent_communication(self, sender, message, recipients=None):
        if recipients is None:
            recipients = [name for name in self.agents.keys() if name != sender]
        
        for recipient in recipients:
            self.communication_protocol.send_message(
                sender, recipient, message
            )
    
    def update_agent_performance(self, agent_name, task, performance_score):
        self.agents[agent_name]['performance_history'].append({
            'task': task,
            'score': performance_score,
            'timestamp': time.time()
        })

class SwarmCoordinator:
    def synthesize_results(self, original_task, agent_results, agents):
        synthesis_prompt = f'''
        Original task: {original_task}
        
        Results from specialized agents:
        '''
        
        for agent_name, results in agent_results.items():
            specialization = agents[agent_name]['specialization']
            synthesis_prompt += f'''
        
        {agent_name} ({specialization}):
        {json.dumps(results, indent=2)}
        '''
        
        synthesis_prompt += '''
        
        Synthesize these results into a comprehensive, coherent response that addresses the original task.
        Resolve any conflicts between agent outputs and ensure consistency.
        '''
        
        # Use a coordinator LLM to synthesize
        coordinator_llm = OpenAI(api_key="your-key")
        final_result = coordinator_llm.generate(synthesis_prompt)
        
        return final_result
"""
        },
        {
            "pattern": "Adaptive Learning Agent",
            "description": "Agent that learns and adapts from experience",
            "key_features": [
                "Experience collection",
                "Pattern learning",
                "Strategy adaptation",
                "Performance optimization"
            ],
            "implementation": """
class AdaptiveLearningAgent:
    def __init__(self, base_llm):
        self.base_llm = base_llm
        self.experience_db = ExperienceDatabase()
        self.pattern_learner = PatternLearner()
        self.strategy_adapter = StrategyAdapter()
        self.performance_tracker = PerformanceTracker()
    
    def execute_task(self, task):
        # Retrieve relevant past experiences
        similar_experiences = self.experience_db.find_similar(task)
        
        # Adapt strategy based on past experiences
        adapted_strategy = self.strategy_adapter.adapt_strategy(
            task, similar_experiences
        )
        
        # Execute with adapted strategy
        start_time = time.time()
        result = self.execute_with_strategy(task, adapted_strategy)
        execution_time = time.time() - start_time
        
        # Evaluate performance
        performance = self.evaluate_performance(task, result)
        
        # Store experience for future learning
        experience = {
            'task': task,
            'strategy': adapted_strategy,
            'result': result,
            'performance': performance,
            'execution_time': execution_time,
            'timestamp': time.time()
        }
        
        self.experience_db.store(experience)
        self.performance_tracker.update(performance)
        
        # Learn patterns from new experience
        self.pattern_learner.update_patterns(experience)
        
        return result
    
    def execute_with_strategy(self, task, strategy):
        strategy_prompt = f'''
        Task: {task}
        
        Strategy to follow: {strategy}
        
        Execute the task following this strategy:
        '''
        
        return self.base_llm.generate(strategy_prompt)

class PatternLearner:
    def __init__(self):
        self.patterns = {}
        self.pattern_embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def update_patterns(self, experience):
        # Extract features from the experience
        features = self.extract_features(experience)
        
        # Find or create pattern cluster
        pattern_cluster = self.find_pattern_cluster(features)
        
        if pattern_cluster is None:
            # Create new pattern
            pattern_id = f"pattern_{len(self.patterns)}"
            self.patterns[pattern_id] = {
                'features': features,
                'experiences': [experience],
                'success_rate': experience['performance']['score']
            }
        else:
            # Update existing pattern
            pattern = self.patterns[pattern_cluster]
            pattern['experiences'].append(experience)
            
            # Update success rate
            scores = [exp['performance']['score'] for exp in pattern['experiences']]
            pattern['success_rate'] = sum(scores) / len(scores)
    
    def extract_features(self, experience):
        task_embedding = self.pattern_embedder.encode([experience['task']])[0]
        
        return {
            'task_embedding': task_embedding,
            'task_length': len(experience['task']),
            'execution_time': experience['execution_time'],
            'strategy_type': self.classify_strategy(experience['strategy'])
        }
    
    def find_pattern_cluster(self, features, similarity_threshold=0.8):
        for pattern_id, pattern in self.patterns.items():
            similarity = self.calculate_similarity(features, pattern['features'])
            if similarity > similarity_threshold:
                return pattern_id
        return None
"""
        }
    ]
    
    for pattern in advanced_patterns:
        with st.expander(f"🔧 {pattern['pattern']}"):
            st.markdown(pattern['description'])
            
            st.markdown("**Key Features:**")
            for feature in pattern['key_features']:
                st.markdown(f"• {feature}")
            
            st.markdown("**Implementation:**")
            st.code(pattern['implementation'], language='python')

with implementation_tabs[2]:
    st.subheader("📊 Agent Evaluation")
    
    evaluation_categories = [
        {
            "category": "Task Performance",
            "metrics": [
                {
                    "metric": "Success Rate",
                    "description": "Percentage of tasks completed successfully",
                    "measurement": "Binary success/failure classification",
                    "code": """
def calculate_success_rate(agent_results):
    successful_tasks = sum(1 for result in agent_results if result['success'])
    total_tasks = len(agent_results)
    return successful_tasks / total_tasks if total_tasks > 0 else 0

def evaluate_task_success(task, result, ground_truth=None):
    if ground_truth:
        return evaluate_against_ground_truth(result, ground_truth)
    else:
        return evaluate_with_llm_judge(task, result)

def evaluate_with_llm_judge(task, result):
    judge_prompt = f'''
    Task: {task}
    Agent Result: {result}
    
    Evaluate if the agent successfully completed the task.
    Consider:
    - Did it address all parts of the task?
    - Is the result accurate and appropriate?
    - Would this be considered a successful completion?
    
    Respond with just: SUCCESS or FAILURE
    '''
    
    judgment = llm_judge.generate(judge_prompt)
    return "SUCCESS" in judgment.upper()
"""
                },
                {
                    "metric": "Quality Score",
                    "description": "Qualitative assessment of output quality",
                    "measurement": "Multi-dimensional scoring (1-10 scale)",
                    "code": """
def calculate_quality_score(task, result):
    dimensions = ['accuracy', 'completeness', 'clarity', 'relevance']
    scores = {}
    
    for dimension in dimensions:
        score = evaluate_dimension(task, result, dimension)
        scores[dimension] = score
    
    # Weighted average
    weights = {'accuracy': 0.3, 'completeness': 0.3, 'clarity': 0.2, 'relevance': 0.2}
    weighted_score = sum(scores[dim] * weights[dim] for dim in dimensions)
    
    return weighted_score, scores

def evaluate_dimension(task, result, dimension):
    eval_prompt = f'''
    Task: {task}
    Result: {result}
    
    Rate the {dimension} of this result on a scale of 1-10, where:
    1 = Very poor {dimension}
    10 = Excellent {dimension}
    
    Consider only {dimension} in your evaluation.
    Respond with just the number.
    '''
    
    response = llm_evaluator.generate(eval_prompt)
    try:
        return float(response.strip())
    except:
        return 5.0  # Default middle score if parsing fails
"""
                },
                {
                    "metric": "Efficiency",
                    "description": "Resource usage and time efficiency",
                    "measurement": "Time, token usage, API calls",
                    "code": """
class EfficiencyTracker:
    def __init__(self):
        self.metrics = {}
    
    def start_tracking(self, task_id):
        self.metrics[task_id] = {
            'start_time': time.time(),
            'token_count': 0,
            'api_calls': 0,
            'tool_calls': 0
        }
    
    def record_api_call(self, task_id, tokens_used):
        if task_id in self.metrics:
            self.metrics[task_id]['api_calls'] += 1
            self.metrics[task_id]['token_count'] += tokens_used
    
    def record_tool_call(self, task_id):
        if task_id in self.metrics:
            self.metrics[task_id]['tool_calls'] += 1
    
    def finish_tracking(self, task_id):
        if task_id in self.metrics:
            self.metrics[task_id]['total_time'] = time.time() - self.metrics[task_id]['start_time']
            return self.metrics[task_id]
        return None
    
    def calculate_efficiency_score(self, task_id, baseline_metrics=None):
        metrics = self.metrics.get(task_id)
        if not metrics:
            return 0
        
        # Normalize metrics (lower is better for efficiency)
        time_score = min(10, 100 / max(1, metrics['total_time']))
        token_score = min(10, 1000 / max(1, metrics['token_count']))
        call_score = min(10, 50 / max(1, metrics['api_calls']))
        
        return (time_score + token_score + call_score) / 3
"""
                }
            ]
        },
        {
            "category": "Robustness",
            "metrics": [
                {
                    "metric": "Error Handling",
                    "description": "Ability to handle errors and edge cases gracefully",
                    "measurement": "Error recovery rate and graceful degradation",
                    "code": """
def test_error_handling(agent, error_scenarios):
    results = []
    
    for scenario in error_scenarios:
        try:
            # Inject error condition
            with ErrorInjector(scenario['error_type']):
                result = agent.run(scenario['task'])
                
            # Evaluate how well the agent handled the error
            handling_score = evaluate_error_handling(
                scenario, result, agent.get_error_logs()
            )
            
            results.append({
                'scenario': scenario['name'],
                'handled_gracefully': handling_score > 0.7,
                'score': handling_score
            })
            
        except Exception as e:
            results.append({
                'scenario': scenario['name'],
                'handled_gracefully': False,
                'score': 0.0,
                'exception': str(e)
            })
    
    return results

def evaluate_error_handling(scenario, result, error_logs):
    eval_prompt = f'''
    Error scenario: {scenario['description']}
    Agent response: {result}
    Error logs: {error_logs}
    
    Rate how well the agent handled this error situation (1-10):
    - Did it recognize the error?
    - Did it provide helpful feedback?
    - Did it attempt recovery or alternative approaches?
    - Was the response appropriate for the error type?
    
    Score (1-10):
    '''
    
    response = llm_evaluator.generate(eval_prompt)
    try:
        return float(response.strip()) / 10.0
    except:
        return 0.5
"""
                },
                {
                    "metric": "Consistency",
                    "description": "Consistent performance across similar tasks",
                    "measurement": "Standard deviation of performance scores",
                    "code": """
def measure_consistency(agent, task_variations, num_runs=5):
    all_scores = []
    
    for task in task_variations:
        task_scores = []
        
        for run in range(num_runs):
            result = agent.run(task)
            score = evaluate_task_quality(task, result)
            task_scores.append(score)
        
        all_scores.extend(task_scores)
    
    # Calculate consistency metrics
    mean_score = np.mean(all_scores)
    std_deviation = np.std(all_scores)
    consistency_score = max(0, 1 - (std_deviation / mean_score))
    
    return {
        'mean_performance': mean_score,
        'standard_deviation': std_deviation,
        'consistency_score': consistency_score,
        'coefficient_of_variation': std_deviation / mean_score
    }
"""
                }
            ]
        },
        {
            "category": "Adaptability",
            "metrics": [
                {
                    "metric": "Learning Rate",
                    "description": "Speed of improvement with experience",
                    "measurement": "Performance improvement over time",
                    "code": """
def measure_learning_rate(agent, learning_tasks, evaluation_tasks):
    performance_history = []
    
    for i, learning_task in enumerate(learning_tasks):
        # Agent learns from this task
        agent.learn_from_task(learning_task)
        
        # Evaluate on held-out evaluation tasks
        eval_scores = []
        for eval_task in evaluation_tasks:
            result = agent.run(eval_task)
            score = evaluate_task_quality(eval_task, result)
            eval_scores.append(score)
        
        avg_performance = np.mean(eval_scores)
        performance_history.append(avg_performance)
    
    # Calculate learning rate (slope of improvement)
    if len(performance_history) > 1:
        x = np.arange(len(performance_history))
        y = np.array(performance_history)
        learning_rate = np.polyfit(x, y, 1)[0]  # Linear regression slope
    else:
        learning_rate = 0
    
    return {
        'performance_history': performance_history,
        'learning_rate': learning_rate,
        'initial_performance': performance_history[0] if performance_history else 0,
        'final_performance': performance_history[-1] if performance_history else 0
    }
"""
                }
            ]
        }
    ]
    
    for category in evaluation_categories:
        with st.expander(f"📊 {category['category']}"):
            for metric in category['metrics']:
                st.markdown(f"### {metric['metric']}")
                st.markdown(metric['description'])
                st.markdown(f"**Measurement:** {metric['measurement']}")
                st.code(metric['code'], language='python')
                st.markdown("---")

with implementation_tabs[3]:
    st.subheader("🏭 Production Deployment")
    
    deployment_considerations = [
        {
            "aspect": "Scalability Architecture",
            "description": "Design for handling multiple concurrent users and requests",
            "strategies": [
                "Microservices architecture for agent components",
                "Load balancing across agent instances",
                "Asynchronous processing for long-running tasks",
                "Resource pooling and auto-scaling"
            ],
            "implementation": """
from fastapi import FastAPI, BackgroundTasks
import asyncio
import redis
from celery import Celery

app = FastAPI()
redis_client = redis.Redis(host='localhost', port=6379)
celery_app = Celery('agent_service', broker='redis://localhost:6379')

class AgentService:
    def __init__(self):
        self.agent_pool = AgentPool(max_agents=10)
        self.task_queue = TaskQueue()
        self.result_store = ResultStore()
    
    async def process_request(self, request):
        # Check if request is already being processed
        existing_task = await self.check_duplicate_request(request)
        if existing_task:
            return await self.get_cached_result(existing_task)
        
        # Get available agent from pool
        agent = await self.agent_pool.get_agent()
        
        try:
            # Process request
            task_id = self.generate_task_id()
            result = await self.execute_with_timeout(agent, request, timeout=300)
            
            # Cache result
            await self.result_store.store(task_id, result)
            return result
            
        finally:
            # Return agent to pool
            await self.agent_pool.return_agent(agent)

@celery_app.task
def process_long_running_task(agent_config, task_data):
    agent = create_agent_from_config(agent_config)
    result = agent.run(task_data)
    return result

@app.post("/agent/task")
async def submit_task(task_data: dict, background_tasks: BackgroundTasks):
    if task_data.get('async', False):
        # Submit to background processing
        task = process_long_running_task.delay(agent_config, task_data)
        return {"task_id": task.id, "status": "queued"}
    else:
        # Synchronous processing
        service = AgentService()
        result = await service.process_request(task_data)
        return {"result": result, "status": "completed"}
"""
        },
        {
            "aspect": "Monitoring & Observability",
            "description": "Track agent performance and system health in production",
            "strategies": [
                "Real-time performance metrics",
                "Error tracking and alerting",
                "Usage analytics and optimization",
                "A/B testing for agent improvements"
            ],
            "implementation": """
import logging
import prometheus_client
from datetime import datetime
import json

# Prometheus metrics
TASK_DURATION = prometheus_client.Histogram('agent_task_duration_seconds', 'Time spent on tasks')
TASK_SUCCESS_RATE = prometheus_client.Counter('agent_task_success_total', 'Successful tasks')
TASK_FAILURE_RATE = prometheus_client.Counter('agent_task_failure_total', 'Failed tasks')
ACTIVE_AGENTS = prometheus_client.Gauge('agent_active_count', 'Number of active agents')

class AgentMonitor:
    def __init__(self):
        self.logger = logging.getLogger('agent_monitor')
        self.metrics_store = MetricsStore()
        self.alert_manager = AlertManager()
    
    def track_task_execution(self, agent_id, task_data, result, duration):
        # Log detailed execution info
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent_id': agent_id,
            'task_type': task_data.get('type', 'unknown'),
            'duration': duration,
            'success': result.get('success', False),
            'tokens_used': result.get('tokens_used', 0),
            'error': result.get('error')
        }
        
        self.logger.info(json.dumps(log_entry))
        
        # Update Prometheus metrics
        TASK_DURATION.observe(duration)
        if result.get('success'):
            TASK_SUCCESS_RATE.inc()
        else:
            TASK_FAILURE_RATE.inc()
        
        # Store for analytics
        self.metrics_store.store_execution_metrics(log_entry)
        
        # Check for alerts
        self.check_performance_alerts(agent_id, log_entry)
    
    def check_performance_alerts(self, agent_id, metrics):
        # Alert on high failure rate
        recent_failures = self.metrics_store.get_recent_failure_rate(agent_id, minutes=10)
        if recent_failures > 0.2:  # 20% failure rate
            self.alert_manager.send_alert(
                f"High failure rate for agent {agent_id}: {recent_failures:.2%}"
            )
        
        # Alert on high latency
        if metrics['duration'] > 60:  # 60 seconds
            self.alert_manager.send_alert(
                f"High latency detected for agent {agent_id}: {metrics['duration']:.2f}s"
            )
        
        # Alert on excessive token usage
        if metrics['tokens_used'] > 10000:
            self.alert_manager.send_alert(
                f"High token usage for agent {agent_id}: {metrics['tokens_used']} tokens"
            )

class PerformanceAnalyzer:
    def __init__(self, metrics_store):
        self.metrics_store = metrics_store
    
    def generate_daily_report(self):
        metrics = self.metrics_store.get_daily_metrics()
        
        report = {
            'total_tasks': metrics['task_count'],
            'success_rate': metrics['success_rate'],
            'avg_duration': metrics['avg_duration'],
            'total_tokens': metrics['total_tokens'],
            'cost_estimate': metrics['total_tokens'] * 0.002 / 1000,  # Rough cost estimate
            'peak_concurrency': metrics['peak_concurrent_agents'],
            'error_breakdown': metrics['error_types']
        }
        
        return report
    
    def identify_optimization_opportunities(self):
        # Analyze patterns for optimization
        slow_tasks = self.metrics_store.get_slow_tasks(threshold=30)
        high_token_tasks = self.metrics_store.get_high_token_tasks(threshold=5000)
        
        recommendations = []
        
        if slow_tasks:
            recommendations.append({
                'type': 'latency_optimization',
                'description': f"Optimize {len(slow_tasks)} slow task types",
                'impact': 'high'
            })
        
        if high_token_tasks:
            recommendations.append({
                'type': 'token_optimization', 
                'description': f"Reduce token usage for {len(high_token_tasks)} task types",
                'impact': 'medium'
            })
        
        return recommendations
"""
        },
        {
            "aspect": "Security & Safety",
            "description": "Ensure safe and secure agent operation in production",
            "strategies": [
                "Input validation and sanitization",
                "Output filtering and safety checks",
                "Access control and authentication",
                "Audit logging and compliance"
            ],
            "implementation": """
import re
import hashlib
from typing import List, Dict, Any

class AgentSecurityManager:
    def __init__(self):
        self.input_validator = InputValidator()
        self.output_filter = OutputFilter()
        self.access_controller = AccessController()
        self.audit_logger = AuditLogger()
    
    def validate_request(self, user_id: str, request_data: Dict[str, Any]) -> bool:
        # Authenticate user
        if not self.access_controller.is_authenticated(user_id):
            self.audit_logger.log_unauthorized_access(user_id, request_data)
            return False
        
        # Validate input
        if not self.input_validator.validate(request_data):
            self.audit_logger.log_invalid_input(user_id, request_data)
            return False
        
        # Check rate limits
        if not self.access_controller.check_rate_limit(user_id):
            self.audit_logger.log_rate_limit_exceeded(user_id)
            return False
        
        return True
    
    def filter_response(self, user_id: str, response: str) -> str:
        # Filter potentially harmful content
        filtered_response = self.output_filter.filter_harmful_content(response)
        
        # Remove sensitive information
        filtered_response = self.output_filter.remove_sensitive_info(filtered_response)
        
        # Log the interaction
        self.audit_logger.log_interaction(user_id, response, filtered_response)
        
        return filtered_response

class InputValidator:
    def __init__(self):
        self.dangerous_patterns = [
            r'<script.*?</script>',  # Script injection
            r'javascript:',          # JavaScript URLs
            r'on\w+\s*=',           # Event handlers
            r'eval\s*\(',           # Code execution
            r'exec\s*\(',           # Code execution
        ]
        
        self.max_input_length = 10000
        self.allowed_file_types = ['.txt', '.md', '.json', '.csv']
    
    def validate(self, request_data: Dict[str, Any]) -> bool:
        # Check input length
        if isinstance(request_data.get('input'), str):
            if len(request_data['input']) > self.max_input_length:
                return False
        
        # Check for dangerous patterns
        input_text = str(request_data.get('input', ''))
        for pattern in self.dangerous_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                return False
        
        # Validate file uploads
        if 'files' in request_data:
            for file_info in request_data['files']:
                if not any(file_info['name'].endswith(ext) for ext in self.allowed_file_types):
                    return False
        
        return True

class OutputFilter:
    def __init__(self):
        self.sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',    # SSN
            r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'  # Phone
        ]
        
        self.harmful_content_classifer = HarmfulContentClassifier()
    
    def filter_harmful_content(self, text: str) -> str:
        # Use ML classifier to detect harmful content
        if self.harmful_content_classifer.is_harmful(text):
            return "I cannot provide that information as it may be harmful."
        
        return text
    
    def remove_sensitive_info(self, text: str) -> str:
        filtered_text = text
        
        for pattern in self.sensitive_patterns:
            filtered_text = re.sub(pattern, '[REDACTED]', filtered_text)
        
        return filtered_text

class AuditLogger:
    def __init__(self):
        self.logger = logging.getLogger('agent_audit')
        self.logger.setLevel(logging.INFO)
        
        # Configure secure logging
        handler = logging.FileHandler('/var/log/agent_audit.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_interaction(self, user_id: str, original_response: str, filtered_response: str):
        log_entry = {
            'event_type': 'agent_interaction',
            'user_id': hashlib.sha256(user_id.encode()).hexdigest()[:16],  # Anonymized
            'response_hash': hashlib.sha256(original_response.encode()).hexdigest(),
            'filtered': original_response != filtered_response,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_unauthorized_access(self, user_id: str, request_data: Dict[str, Any]):
        log_entry = {
            'event_type': 'unauthorized_access',
            'user_id': hashlib.sha256(user_id.encode()).hexdigest()[:16],
            'request_type': request_data.get('type', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.warning(json.dumps(log_entry))
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
            st.code(consideration['implementation'], language='python')

# Use cases and applications
st.header("🎯 Real-World Applications")

application_tabs = st.tabs([
    "💼 Business Automation",
    "🔬 Research & Analysis", 
    "🎓 Education & Training",
    "🛠️ Software Development"
])

with application_tabs[0]:
    st.subheader("💼 Business Automation")
    
    business_cases = [
        {
            "use_case": "Customer Service Agent",
            "description": "Automated customer support with human-like interaction",
            "capabilities": [
                "Multi-turn conversation handling",
                "Knowledge base integration",
                "Escalation to human agents",
                "Sentiment analysis and response adaptation"
            ],
            "implementation_overview": """
1. Intent Recognition: Classify customer queries and route appropriately
2. Knowledge Retrieval: Search internal knowledge base for relevant information
3. Response Generation: Create personalized, contextually appropriate responses
4. Action Execution: Update customer records, create tickets, schedule callbacks
5. Escalation Logic: Transfer complex issues to human agents with context
""",
            "business_impact": [
                "24/7 customer service availability",
                "Reduced response times from hours to seconds",
                "Consistent service quality across all interactions",
                "Cost reduction of 40-60% compared to human-only support"
            ]
        },
        {
            "use_case": "Sales Intelligence Agent",
            "description": "AI-powered sales research and lead qualification",
            "capabilities": [
                "Prospect research and profiling",
                "Competitive analysis automation",
                "Personalized outreach generation",
                "Deal scoring and prioritization"
            ],
            "implementation_overview": """
1. Data Collection: Gather prospect information from multiple sources
2. Analysis: Analyze company financials, recent news, and decision makers
3. Scoring: Rank leads based on fit and likelihood to convert
4. Outreach: Generate personalized email sequences and talking points
5. CRM Integration: Update records and schedule follow-up activities
""",
            "business_impact": [
                "50% increase in qualified lead generation",
                "Higher conversion rates through personalization",
                "Sales team focus on high-value activities",
                "Improved sales cycle efficiency"
            ]
        },
        {
            "use_case": "Financial Analysis Agent",
            "description": "Automated financial reporting and analysis",
            "capabilities": [
                "Financial data extraction and processing",
                "Trend analysis and forecasting",
                "Risk assessment and monitoring",
                "Automated report generation"
            ],
            "implementation_overview": """
1. Data Integration: Connect to financial systems and databases
2. Analysis: Perform ratio analysis, trend identification, and variance analysis
3. Modeling: Create financial models and forecasts
4. Reporting: Generate executive summaries and detailed reports
5. Alerting: Monitor key metrics and alert on anomalies
""",
            "business_impact": [
                "Faster financial close processes",
                "More accurate forecasting and planning",
                "Early detection of financial risks",
                "Reduced manual analysis workload"
            ]
        }
    ]
    
    for case in business_cases:
        with st.expander(f"💼 {case['use_case']}"):
            st.markdown(case['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Key Capabilities:**")
                for capability in case['capabilities']:
                    st.markdown(f"• {capability}")
                
                st.markdown("**Implementation Overview:**")
                st.code(case['implementation_overview'])
            
            with col2:
                st.markdown("**Business Impact:**")
                for impact in case['business_impact']:
                    st.markdown(f"• {impact}")

with application_tabs[1]:
    st.subheader("🔬 Research & Analysis")
    
    research_applications = [
        {
            "application": "Scientific Literature Review Agent",
            "description": "Automated analysis of scientific papers and research trends",
            "workflow": [
                "Query scientific databases (PubMed, arXiv, Google Scholar)",
                "Extract and summarize key findings from papers",
                "Identify research gaps and emerging trends",
                "Generate comprehensive literature reviews",
                "Track citation networks and research impact"
            ],
            "tools_used": [
                "PubMed API for biomedical literature",
                "arXiv API for preprint access",
                "Citation analysis tools",
                "Text summarization models",
                "Graph analysis for citation networks"
            ]
        },
        {
            "application": "Market Research Agent",
            "description": "Comprehensive market analysis and competitive intelligence",
            "workflow": [
                "Collect data from multiple market sources",
                "Analyze competitor strategies and positioning",
                "Identify market trends and opportunities",
                "Generate market sizing and forecasts",
                "Create strategic recommendations"
            ],
            "tools_used": [
                "Web scraping for public company data",
                "Social media monitoring APIs",
                "Financial data providers",
                "Survey and polling platforms",
                "Statistical analysis tools"
            ]
        },
        {
            "application": "Legal Research Agent",
            "description": "Automated legal research and case analysis",
            "workflow": [
                "Search legal databases and case law",
                "Analyze precedents and legal arguments",
                "Identify relevant statutes and regulations",
                "Generate legal memos and briefs",
                "Track regulatory changes and updates"
            ],
            "tools_used": [
                "Legal database APIs (Westlaw, LexisNexis)",
                "Court record systems",
                "Regulatory tracking services",
                "Legal document analysis tools",
                "Citation verification systems"
            ]
        }
    ]
    
    for app in research_applications:
        with st.expander(f"🔬 {app['application']}"):
            st.markdown(app['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Workflow Steps:**")
                for i, step in enumerate(app['workflow'], 1):
                    st.markdown(f"{i}. {step}")
            
            with col2:
                st.markdown("**Tools & Integrations:**")
                for tool in app['tools_used']:
                    st.markdown(f"• {tool}")

with application_tabs[2]:
    st.subheader("🎓 Education & Training")
    
    education_cases = [
        {
            "case": "Personalized Tutoring Agent",
            "description": "Adaptive learning system that adjusts to student needs",
            "features": [
                "Learning style assessment and adaptation",
                "Personalized curriculum generation",
                "Real-time feedback and assessment",
                "Progress tracking and analytics",
                "Parent and teacher reporting"
            ],
            "example_interaction": """
Student: "I don't understand how photosynthesis works"

Agent Analysis:
- Student Level: 8th grade biology
- Learning Style: Visual learner (from assessment)
- Previous Struggles: Chemical processes

Agent Response:
- Generates visual diagram of photosynthesis
- Uses analogy to cooking (familiar concept)
- Breaks down into simple steps
- Provides interactive simulation
- Follows up with practice questions
- Adjusts explanation based on comprehension
"""
        },
        {
            "case": "Corporate Training Agent",
            "description": "Automated employee training and skill development",
            "features": [
                "Role-based training customization",
                "Interactive scenario simulations",
                "Skill gap analysis and recommendations",
                "Compliance training automation",
                "Performance tracking and certification"
            ],
            "example_interaction": """
Employee: "I need to learn about data privacy regulations"

Agent Analysis:
- Role: Marketing Manager
- Experience Level: Intermediate
- Compliance Requirements: GDPR, CCPA
- Department: Marketing (handles customer data)

Agent Response:
- Provides role-specific privacy training
- Uses marketing scenarios for examples
- Interactive GDPR compliance checkup
- Real case studies from marketing context
- Quiz with immediate feedback
- Certificate upon completion
"""
        }
    ]
    
    for case in education_cases:
        with st.expander(f"🎓 {case['case']}"):
            st.markdown(case['description'])
            
            st.markdown("**Key Features:**")
            for feature in case['features']:
                st.markdown(f"• {feature}")
            
            st.markdown("**Example Interaction:**")
            st.code(case['example_interaction'])

with application_tabs[3]:
    st.subheader("🛠️ Software Development")
    
    dev_applications = [
        {
            "application": "Code Review Agent",
            "description": "Automated code analysis and review assistance",
            "capabilities": [
                "Bug detection and security vulnerability scanning",
                "Code quality assessment and style checking",
                "Performance optimization suggestions",
                "Documentation and comment generation",
                "Test case recommendations"
            ],
            "integration_points": [
                "GitHub/GitLab pull request integration",
                "IDE plugins and extensions",
                "CI/CD pipeline integration",
                "Issue tracking system connectivity",
                "Code repository analysis"
            ]
        },
        {
            "application": "DevOps Automation Agent",
            "description": "Infrastructure management and deployment automation",
            "capabilities": [
                "Infrastructure as Code generation",
                "Deployment pipeline optimization",
                "Monitoring and alerting setup",
                "Incident response automation",
                "Cost optimization recommendations"
            ],
            "integration_points": [
                "Cloud provider APIs (AWS, Azure, GCP)",
                "Container orchestration platforms",
                "Monitoring and logging systems",
                "Configuration management tools",
                "Security scanning and compliance tools"
            ]
        },
        {
            "application": "Technical Documentation Agent",
            "description": "Automated documentation generation and maintenance",
            "capabilities": [
                "API documentation from code analysis",
                "User guide generation from features",
                "Code comment and docstring creation",
                "Architecture diagram generation",
                "Knowledge base maintenance"
            ],
            "integration_points": [
                "Code repository integration",
                "Documentation platforms (Confluence, Notion)",
                "API specification tools (OpenAPI, GraphQL)",
                "Diagram generation tools",
                "Content management systems"
            ]
        }
    ]
    
    for app in dev_applications:
        with st.expander(f"🛠️ {app['application']}"):
            st.markdown(app['description'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Core Capabilities:**")
                for capability in app['capabilities']:
                    st.markdown(f"• {capability}")
            
            with col2:
                st.markdown("**Integration Points:**")
                for integration in app['integration_points']:
                    st.markdown(f"• {integration}")

# Best practices
st.header("💡 Agent Development Best Practices")

best_practices = [
    "**Define Clear Objectives**: Establish specific, measurable goals for your agent",
    "**Start Simple**: Begin with basic functionality and iterate to add complexity",
    "**Design for Failure**: Implement robust error handling and graceful degradation",
    "**Monitor Continuously**: Track performance, usage, and user satisfaction metrics",
    "**Ensure Safety**: Implement content filtering and safety measures",
    "**Plan for Scale**: Design architecture that can handle increased load",
    "**Maintain Transparency**: Provide clear explanations for agent decisions",
    "**Regular Updates**: Keep knowledge bases and models current",
    "**User-Centric Design**: Focus on user experience and practical utility",
    "**Ethical Considerations**: Address bias, privacy, and fairness concerns"
]

for practice in best_practices:
    st.markdown(f"• {practice}")

# Resources
st.header("📚 Learning Resources")

resources = [
    {
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "type": "Research Paper",
        "description": "Foundational paper on reasoning and acting agents",
        "difficulty": "Advanced"
    },
    {
        "title": "Langchain Agents Documentation",
        "type": "Documentation",
        "description": "Practical guide to building agents with Langchain",
        "difficulty": "Intermediate"
    },
    {
        "title": "Multi-Agent Systems: A Modern Approach",
        "type": "Book",
        "description": "Comprehensive textbook on multi-agent systems",
        "difficulty": "Advanced"
    },
    {
        "title": "Building LLM-Powered Agents",
        "type": "Tutorial",
        "description": "Hands-on guide to agent development",
        "difficulty": "Beginner"
    }
]

for resource in resources:
    with st.expander(f"📖 {resource['title']}"):
        st.markdown(f"**Type:** {resource['type']}")
        st.markdown(f"**Description:** {resource['description']}")
        st.markdown(f"**Difficulty:** {resource['difficulty']}")