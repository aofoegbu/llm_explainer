import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Data Engineering for LLMs", page_icon="🔧", layout="wide")

st.title("🔧 Data Engineering for LLMs")
st.markdown("### Building Robust Data Infrastructure for Large Language Models")

# Overview
st.header("🎯 Overview")
st.markdown("""
Data Engineering for LLMs involves designing and building systems to collect, process, store, and serve 
the massive datasets required for training and fine-tuning language models. This includes handling 
billions of tokens, ensuring data quality, and maintaining efficient data pipelines.
""")

# Core concepts
st.header("🧠 Core Concepts")

concept_tabs = st.tabs([
    "📊 Data Requirements",
    "🏗️ Pipeline Architecture", 
    "⚙️ ETL vs ELT",
    "🔄 Data Processing"
])

with concept_tabs[0]:
    st.subheader("📊 LLM Data Requirements")
    
    st.markdown("""
    Large Language Models require massive, high-quality datasets with specific characteristics
    for effective training and deployment.
    """)
    
    # Data scale requirements
    data_requirements = [
        {
            "model_size": "GPT-3 Scale (175B)",
            "training_tokens": "300B tokens",
            "raw_data_size": "~45TB text",
            "processed_size": "~570GB",
            "training_time": "~34 days (V100)",
            "compute_cost": "$4.6M (estimated)"
        },
        {
            "model_size": "GPT-4 Scale (1.7T+)",
            "training_tokens": "13T+ tokens",
            "raw_data_size": "~2PB text",
            "processed_size": "~25TB",
            "training_time": "~90 days (A100)",
            "compute_cost": "$63M+ (estimated)"
        },
        {
            "model_size": "LLaMA-7B",
            "training_tokens": "1T tokens",
            "raw_data_size": "~150TB text",
            "processed_size": "~1.9TB",
            "training_time": "~21 days (A100)",
            "compute_cost": "$500K (estimated)"
        }
    ]
    
    df_requirements = pd.DataFrame(data_requirements)
    st.dataframe(df_requirements, use_container_width=True)
    
    # Data quality characteristics
    st.markdown("### 🎯 Data Quality Requirements")
    
    quality_aspects = [
        {
            "aspect": "Diversity",
            "description": "Representation across languages, domains, styles, and demographics",
            "importance": "Prevents model bias and improves generalization",
            "metrics": [
                "Language distribution coverage",
                "Domain representation balance",
                "Geographic and cultural diversity",
                "Temporal coverage across years"
            ],
            "implementation": """
def assess_data_diversity(dataset):
    diversity_metrics = {}
    
    # Language diversity
    languages = detect_languages(dataset)
    diversity_metrics['language_entropy'] = calculate_entropy(languages)
    
    # Domain diversity  
    domains = classify_domains(dataset)
    diversity_metrics['domain_distribution'] = calculate_gini_coefficient(domains)
    
    # Temporal diversity
    dates = extract_dates(dataset)
    diversity_metrics['temporal_span'] = max(dates) - min(dates)
    
    return diversity_metrics
"""
        },
        {
            "aspect": "Quality",
            "description": "Clean, well-formatted, and meaningful content",
            "importance": "Ensures model learns from high-quality examples",
            "metrics": [
                "Grammar and spelling accuracy",
                "Coherence and readability scores",
                "Information density measures",
                "Duplicate content percentage"
            ],
            "implementation": """
def assess_text_quality(text_batch):
    quality_scores = {}
    
    # Grammar and spelling
    quality_scores['grammar_score'] = check_grammar(text_batch)
    
    # Readability
    quality_scores['readability'] = calculate_flesch_score(text_batch)
    
    # Information density
    quality_scores['info_density'] = calculate_compression_ratio(text_batch)
    
    # Coherence
    quality_scores['coherence'] = measure_topic_coherence(text_batch)
    
    return quality_scores
"""
        },
        {
            "aspect": "Scale",
            "description": "Sufficient volume for effective model training",
            "importance": "More data generally leads to better model performance",
            "metrics": [
                "Total token count",
                "Unique n-gram coverage",
                "Vocabulary size and distribution",
                "Data-to-parameter ratio"
            ],
            "implementation": """
def calculate_scale_metrics(dataset, model_params):
    scale_metrics = {}
    
    # Token statistics
    total_tokens = count_tokens(dataset)
    scale_metrics['total_tokens'] = total_tokens
    scale_metrics['tokens_per_param'] = total_tokens / model_params
    
    # Vocabulary coverage
    vocab = extract_vocabulary(dataset)
    scale_metrics['vocab_size'] = len(vocab)
    scale_metrics['vocab_diversity'] = calculate_vocab_entropy(vocab)
    
    return scale_metrics
"""
        },
        {
            "aspect": "Freshness",
            "description": "Recent and up-to-date information",
            "importance": "Ensures model knowledge reflects current state of world",
            "metrics": [
                "Average content age",
                "Recency distribution",
                "Update frequency",
                "Temporal coverage gaps"
            ],
            "implementation": """
def analyze_data_freshness(dataset):
    freshness_metrics = {}
    
    # Extract timestamps
    timestamps = extract_creation_dates(dataset)
    
    # Calculate freshness scores
    current_time = datetime.now()
    ages = [(current_time - ts).days for ts in timestamps]
    
    freshness_metrics['avg_age_days'] = np.mean(ages)
    freshness_metrics['recency_score'] = calculate_recency_weight(ages)
    
    return freshness_metrics
"""
        }
    ]
    
    for aspect in quality_aspects:
        with st.expander(f"🎯 {aspect['aspect']}"):
            st.markdown(aspect['description'])
            st.markdown(f"**Importance:** {aspect['importance']}")
            
            st.markdown("**Key Metrics:**")
            for metric in aspect['metrics']:
                st.markdown(f"• {metric}")
            
            st.markdown("**Implementation Example:**")
            st.code(aspect['implementation'], language='python')

with concept_tabs[1]:
    st.subheader("🏗️ Data Pipeline Architecture")
    
    st.markdown("""
    LLM data pipelines must handle massive scale, ensure reliability, and maintain 
    data quality throughout the ingestion, processing, and serving stages.
    """)
    
    architecture_components = [
        {
            "component": "Data Ingestion Layer",
            "description": "Collect data from diverse sources at scale",
            "technologies": [
                "Apache Kafka for streaming ingestion",
                "Apache Airflow for batch orchestration", 
                "AWS Kinesis for real-time streams",
                "Custom crawlers and APIs"
            ],
            "challenges": [
                "Rate limiting and throttling",
                "Source reliability and availability",
                "Format standardization",
                "Duplicate detection across sources"
            ],
            "implementation": """
# Kafka-based streaming ingestion
from kafka import KafkaConsumer, KafkaProducer
import asyncio
import aiohttp

class DataIngestionService:
    def __init__(self, kafka_config):
        self.producer = KafkaProducer(**kafka_config)
        self.source_configs = {}
        self.rate_limiters = {}
    
    async def ingest_from_api(self, source_config):
        rate_limiter = self.rate_limiters[source_config['name']]
        
        async with aiohttp.ClientSession() as session:
            async with rate_limiter:
                async with session.get(source_config['url']) as response:
                    data = await response.text()
                    
                    # Send to Kafka topic
                    self.producer.send(
                        topic=source_config['topic'],
                        value=data.encode('utf-8'),
                        key=source_config['source_id'].encode('utf-8')
                    )
    
    def start_batch_ingestion(self, source_configs):
        for config in source_configs:
            self.schedule_batch_job(config)
    
    def schedule_batch_job(self, config):
        # Use Airflow or similar for scheduling
        dag = DAG(
            f"ingest_{config['name']}",
            schedule_interval=config['schedule'],
            catchup=False
        )
        
        ingest_task = PythonOperator(
            task_id='ingest_data',
            python_callable=self.run_batch_ingestion,
            op_kwargs={'config': config},
            dag=dag
        )
"""
        },
        {
            "component": "Data Processing Layer",
            "description": "Transform and clean raw data for model consumption",
            "technologies": [
                "Apache Spark for distributed processing",
                "Dask for Python-native scaling",
                "Ray for ML workloads",
                "Custom processing frameworks"
            ],
            "challenges": [
                "Memory management for large files",
                "Fault tolerance and recovery",
                "Consistent processing across nodes",
                "Quality control at scale"
            ],
            "implementation": """
# Spark-based distributed processing
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, FloatType

class LLMDataProcessor:
    def __init__(self, spark_config):
        self.spark = SparkSession.builder.config(spark_config).getOrCreate()
        self.quality_filters = QualityFilters()
    
    def process_text_batch(self, input_path, output_path):
        # Read raw text data
        df = self.spark.read.text(input_path)
        
        # Apply cleaning and filtering
        df_cleaned = df.filter(col("value").isNotNull()) \\
                      .filter(self.quality_filters.length_filter()) \\
                      .filter(self.quality_filters.language_filter()) \\
                      .withColumn("cleaned_text", self.clean_text_udf(col("value"))) \\
                      .withColumn("quality_score", self.quality_score_udf(col("cleaned_text")))
        
        # Filter by quality threshold
        df_filtered = df_cleaned.filter(col("quality_score") > 0.7)
        
        # Tokenize and prepare for training
        df_tokenized = df_filtered.withColumn("tokens", self.tokenize_udf(col("cleaned_text"))) \\
                                 .withColumn("token_count", self.count_tokens_udf(col("tokens")))
        
        # Write processed data
        df_tokenized.write.mode("overwrite").parquet(output_path)
    
    @udf(returnType=StringType())
    def clean_text_udf(self, text):
        return self.quality_filters.clean_text(text)
    
    @udf(returnType=FloatType())
    def quality_score_udf(self, text):
        return self.quality_filters.calculate_quality_score(text)
"""
        },
        {
            "component": "Data Storage Layer",
            "description": "Efficient storage and retrieval of processed training data",
            "technologies": [
                "Distributed file systems (HDFS, S3)",
                "Column stores (Parquet, Delta Lake)",
                "Object storage for raw data",
                "Metadata catalogs (Hive, Glue)"
            ],
            "challenges": [
                "Storage cost optimization",
                "Access pattern optimization",
                "Data lifecycle management",
                "Backup and disaster recovery"
            ],
            "implementation": """
# Delta Lake for data versioning and ACID transactions
from delta.tables import DeltaTable
import delta

class LLMDataStorage:
    def __init__(self, storage_config):
        self.storage_path = storage_config['base_path']
        self.spark = SparkSession.builder \\
                    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \\
                    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \\
                    .getOrCreate()
    
    def create_training_dataset(self, processed_data_path, dataset_name):
        # Read processed data
        df = self.spark.read.parquet(processed_data_path)
        
        # Add metadata columns
        df_with_metadata = df.withColumn("dataset_version", lit(self.get_next_version())) \\
                            .withColumn("created_at", current_timestamp()) \\
                            .withColumn("data_source", col("source_id"))
        
        # Write as Delta table for versioning
        table_path = f"{self.storage_path}/{dataset_name}"
        df_with_metadata.write.format("delta").mode("append").save(table_path)
        
        # Update metadata catalog
        self.update_catalog_metadata(dataset_name, df_with_metadata.count())
    
    def create_training_splits(self, dataset_name, split_ratios):
        table_path = f"{self.storage_path}/{dataset_name}"
        df = self.spark.read.format("delta").load(table_path)
        
        # Create deterministic splits based on hash
        splits = {}
        cumulative_ratio = 0
        
        for split_name, ratio in split_ratios.items():
            if split_name == 'test' and cumulative_ratio < 1.0:
                # Remainder goes to test
                splits[split_name] = df.filter(col("hash_bucket") >= cumulative_ratio)
            else:
                splits[split_name] = df.filter(
                    (col("hash_bucket") >= cumulative_ratio) & 
                    (col("hash_bucket") < cumulative_ratio + ratio)
                )
                cumulative_ratio += ratio
        
        return splits
"""
        },
        {
            "component": "Data Serving Layer",
            "description": "Efficient data loading and streaming for model training",
            "technologies": [
                "PyTorch DataLoader optimization",
                "TensorFlow tf.data API",
                "Ray Data for distributed loading",
                "Custom data streaming protocols"
            ],
            "challenges": [
                "I/O bottleneck elimination",
                "Memory-efficient loading",
                "Shuffling at scale",
                "Multi-GPU data distribution"
            ],
            "implementation": """
# Optimized PyTorch DataLoader for LLM training
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import datasets
from transformers import AutoTokenizer

class LLMDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048, streaming=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if streaming:
            self.dataset = datasets.load_dataset('parquet', data_files=data_path, streaming=True)['train']
        else:
            self.dataset = datasets.load_dataset('parquet', data_files=data_path)['train']
        
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
    
    def tokenize_function(self, examples):
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': item['input_ids']  # For causal LM
        }

class OptimizedDataLoader:
    def __init__(self, dataset, batch_size, num_workers=8, distributed=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        sampler = DistributedSampler(dataset) if distributed else None
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
    
    def __iter__(self):
        return iter(self.dataloader)
"""
        }
    ]
    
    for component in architecture_components:
        with st.expander(f"🏗️ {component['component']}"):
            st.markdown(component['description'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Technologies:**")
                for tech in component['technologies']:
                    st.markdown(f"• {tech}")
            with col2:
                st.markdown("**Key Challenges:**")
                for challenge in component['challenges']:
                    st.markdown(f"• {challenge}")
            
            st.markdown("**Implementation Example:**")
            st.code(component['implementation'], language='python')

with concept_tabs[2]:
    st.subheader("⚙️ ETL vs ELT for LLMs")
    
    st.markdown("""
    The choice between ETL (Extract, Transform, Load) and ELT (Extract, Load, Transform) 
    significantly impacts LLM data pipeline design, performance, and maintainability.
    """)
    
    # ETL vs ELT comparison
    comparison_data = {
        'Aspect': [
            'Processing Location',
            'Storage Requirements', 
            'Flexibility',
            'Performance',
            'Cost Model',
            'Debugging',
            'Schema Evolution',
            'Real-time Capability'
        ],
        'ETL (Extract, Transform, Load)': [
            'External processing cluster',
            'Lower storage (only final data)',
            'Limited (pre-defined transforms)',
            'Better for complex transformations',
            'Higher compute, lower storage',
            'More complex (distributed)',
            'Requires pipeline changes',
            'More complex to implement'
        ],
        'ELT (Extract, Load, Transform)': [
            'Data warehouse/lake engine',
            'Higher storage (raw + processed)',
            'High (transform at query time)',
            'Better for simple transformations',
            'Lower compute, higher storage',
            'Easier (SQL-based)',
            'Schema-on-read flexibility',
            'Easier with streaming'
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Implementation patterns
    implementation_tabs = st.tabs(["🔄 ETL Pattern", "📊 ELT Pattern", "🔀 Hybrid Approach"])
    
    with implementation_tabs[0]:
        st.markdown("### 🔄 ETL Pattern for LLMs")
        
        st.markdown("""
        Traditional ETL works well when you have well-defined transformation requirements
        and want to minimize storage costs by only keeping processed data.
        """)
        
        etl_code = """
# ETL Pipeline for LLM Data Processing
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from transformers import AutoTokenizer

class LLMETLPipeline:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
        self.quality_thresholds = config['quality_thresholds']
    
    def extract_data(self, **context):
        \"\"\"Extract raw text data from various sources\"\"\"
        extracted_data = []
        
        for source in self.config['data_sources']:
            if source['type'] == 'web_crawl':
                data = self.crawl_websites(source['urls'])
            elif source['type'] == 'api':
                data = self.fetch_from_api(source['endpoint'])
            elif source['type'] == 'file':
                data = self.read_files(source['path'])
            
            extracted_data.extend(data)
        
        # Store raw data temporarily
        raw_data_path = f"/tmp/raw_data_{context['ds']}.jsonl"
        self.save_data(extracted_data, raw_data_path)
        
        return raw_data_path
    
    def transform_data(self, raw_data_path, **context):
        \"\"\"Apply comprehensive transformations to raw data\"\"\"
        
        # Load raw data
        raw_data = self.load_data(raw_data_path)
        
        transformed_data = []
        
        for item in raw_data:
            # Quality filtering
            if not self.passes_quality_checks(item['text']):
                continue
            
            # Text cleaning and normalization
            cleaned_text = self.clean_text(item['text'])
            
            # Tokenization and length filtering
            tokens = self.tokenizer.encode(cleaned_text)
            if len(tokens) < self.config['min_tokens'] or len(tokens) > self.config['max_tokens']:
                continue
            
            # Deduplication check
            if self.is_duplicate(cleaned_text):
                continue
            
            # Language detection and filtering
            if not self.is_target_language(cleaned_text):
                continue
            
            transformed_item = {
                'text': cleaned_text,
                'tokens': tokens,
                'token_count': len(tokens),
                'source': item['source'],
                'quality_score': self.calculate_quality_score(cleaned_text),
                'processed_at': datetime.now().isoformat()
            }
            
            transformed_data.append(transformed_item)
        
        # Store transformed data
        transformed_path = f"/tmp/transformed_data_{context['ds']}.jsonl"
        self.save_data(transformed_data, transformed_path)
        
        return transformed_path
    
    def load_data(self, transformed_data_path, **context):
        \"\"\"Load processed data into final storage\"\"\"
        
        # Load transformed data
        data = self.load_data(transformed_data_path)
        
        # Create training batches
        batches = self.create_training_batches(data)
        
        # Store in final format (e.g., HuggingFace datasets)
        final_dataset_path = f"{self.config['output_path']}/dataset_{context['ds']}"
        
        dataset = datasets.Dataset.from_list(batches)
        dataset.save_to_disk(final_dataset_path)
        
        # Update metadata
        self.update_dataset_metadata(final_dataset_path, len(data))
        
        # Cleanup temporary files
        self.cleanup_temp_files([transformed_data_path])
        
        return final_dataset_path

# Airflow DAG definition
default_args = {
    'owner': 'llm-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'llm_etl_pipeline',
    default_args=default_args,
    description='ETL pipeline for LLM training data',
    schedule_interval='@daily',
    catchup=False
)

pipeline = LLMETLPipeline(config)

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=pipeline.extract_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=pipeline.transform_data,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=pipeline.load_data,
    dag=dag
)

extract_task >> transform_task >> load_task
"""
        
        st.code(etl_code, language='python')
    
    with implementation_tabs[1]:
        st.markdown("### 📊 ELT Pattern for LLMs")
        
        st.markdown("""
        ELT patterns are often preferred for LLM data pipelines due to their flexibility
        and ability to handle schema evolution and experimental transformations.
        """)
        
        elt_code = """
# ELT Pipeline using Delta Lake and Spark
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

class LLMELTPipeline:
    def __init__(self, spark_config):
        self.spark = SparkSession.builder \\
                    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \\
                    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \\
                    .config(spark_config) \\
                    .getOrCreate()
        
        self.bronze_path = "/data/bronze/raw_text"
        self.silver_path = "/data/silver/cleaned_text" 
        self.gold_path = "/data/gold/training_ready"
    
    def extract_and_load_raw(self, source_configs):
        \"\"\"Extract from sources and load directly into bronze layer\"\"\"
        
        for source in source_configs:
            if source['type'] == 'streaming':
                # Stream directly to Delta table
                stream_df = self.spark.readStream \\
                           .format("kafka") \\
                           .option("kafka.bootstrap.servers", source['kafka_servers']) \\
                           .option("subscribe", source['topic']) \\
                           .load()
                
                # Parse and write to bronze
                parsed_df = stream_df.select(
                    col("key").cast("string").alias("source_key"),
                    col("value").cast("string").alias("raw_text"),
                    col("timestamp").alias("ingestion_time"),
                    lit(source['name']).alias("source_name")
                )
                
                query = parsed_df.writeStream \\
                        .format("delta") \\
                        .outputMode("append") \\
                        .option("checkpointLocation", f"/checkpoints/{source['name']}") \\
                        .start(self.bronze_path)
                
            elif source['type'] == 'batch':
                # Batch load to Delta table
                batch_df = self.spark.read \\
                          .option("multiline", "true") \\
                          .text(source['path'])
                
                batch_df_with_metadata = batch_df.select(
                    col("value").alias("raw_text"),
                    current_timestamp().alias("ingestion_time"),
                    lit(source['name']).alias("source_name"),
                    input_file_name().alias("source_file")
                )
                
                batch_df_with_metadata.write \\
                                     .format("delta") \\
                                     .mode("append") \\
                                     .save(self.bronze_path)
    
    def transform_to_silver(self):
        \"\"\"Transform bronze data to silver (cleaned) layer\"\"\"
        
        # Read from bronze layer
        bronze_df = self.spark.read.format("delta").load(self.bronze_path)
        
        # Apply transformations using SQL for flexibility
        self.spark.sql(f\"\"\"
        CREATE OR REPLACE TEMPORARY VIEW bronze_data 
        USING DELTA 
        LOCATION '{self.bronze_path}'
        \"\"\")
        
        silver_df = self.spark.sql(\"\"\"
        SELECT 
            raw_text,
            source_name,
            ingestion_time,
            
            -- Text cleaning
            regexp_replace(raw_text, '[\\\\r\\\\n\\\\t]+', ' ') as cleaned_text,
            
            -- Quality metrics
            length(raw_text) as char_count,
            size(split(raw_text, ' ')) as word_count,
            
            -- Language detection (simplified)
            CASE 
                WHEN raw_text RLIKE '[a-zA-Z]' THEN 'en'
                ELSE 'other' 
            END as detected_language,
            
            -- Quality score (simplified)
            CASE 
                WHEN length(raw_text) > 100 AND size(split(raw_text, ' ')) > 20 THEN 0.8
                WHEN length(raw_text) > 50 AND size(split(raw_text, ' ')) > 10 THEN 0.6
                ELSE 0.3
            END as quality_score,
            
            current_timestamp() as processed_at
            
        FROM bronze_data
        WHERE 
            raw_text IS NOT NULL 
            AND length(raw_text) > 10
            AND raw_text NOT RLIKE '^[0-9\\\\s\\\\p{Punct}]*$'  -- Not just numbers/punctuation
        \"\"\")
        
        # Write to silver layer with data quality constraints
        silver_df.write \\
                .format("delta") \\
                .mode("overwrite") \\
                .option("overwriteSchema", "true") \\
                .save(self.silver_path)
    
    def transform_to_gold(self, tokenizer_name="gpt2"):
        \"\"\"Transform silver data to gold (training-ready) layer\"\"\"
        
        # Register UDFs for tokenization
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        def tokenize_text(text):
            if text is None:
                return None
            try:
                tokens = tokenizer.encode(text, max_length=2048, truncation=True)
                return tokens
            except:
                return None
        
        def count_tokens(tokens):
            return len(tokens) if tokens else 0
        
        tokenize_udf = udf(tokenize_text, ArrayType(IntegerType()))
        count_tokens_udf = udf(count_tokens, IntegerType())
        
        # Read from silver layer
        silver_df = self.spark.read.format("delta").load(self.silver_path)
        
        # Create training-ready data
        gold_df = silver_df \\
                 .filter(col("quality_score") >= 0.7) \\
                 .filter(col("detected_language") == "en") \\
                 .withColumn("tokens", tokenize_udf(col("cleaned_text"))) \\
                 .withColumn("token_count", count_tokens_udf(col("tokens"))) \\
                 .filter(col("token_count").between(50, 2048)) \\
                 .select(
                     col("cleaned_text").alias("text"),
                     col("tokens"),
                     col("token_count"),
                     col("source_name"),
                     col("quality_score"),
                     current_timestamp().alias("training_ready_at")
                 )
        
        # Add sequence-level features for training
        gold_df_enhanced = gold_df \\
                          .withColumn("text_hash", sha2(col("text"), 256)) \\
                          .withColumn("training_weight", 
                                    when(col("quality_score") > 0.9, 1.2)
                                    .when(col("quality_score") > 0.8, 1.0)
                                    .otherwise(0.8)) \\
                          .withColumn("batch_id", monotonically_increasing_id() % 1000)
        
        # Write to gold layer partitioned by batch_id for efficient training
        gold_df_enhanced.write \\
                       .format("delta") \\
                       .mode("overwrite") \\
                       .partitionBy("batch_id") \\
                       .save(self.gold_path)
    
    def create_training_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        \"\"\"Create deterministic train/val/test splits\"\"\"
        
        gold_df = self.spark.read.format("delta").load(self.gold_path)
        
        # Create splits based on hash for deterministic results
        splits_df = gold_df.withColumn("split_hash", 
                                     abs(hash(col("text_hash"))) % 100) \\
                          .withColumn("split",
                                    when(col("split_hash") < train_ratio * 100, "train")
                                    .when(col("split_hash") < (train_ratio + val_ratio) * 100, "val")
                                    .otherwise("test"))
        
        # Write splits
        for split in ["train", "val", "test"]:
            split_df = splits_df.filter(col("split") == split)
            split_df.write \\
                   .format("delta") \\
                   .mode("overwrite") \\
                   .save(f"{self.gold_path}_splits/{split}")
"""
        
        st.code(elt_code, language='python')
    
    with implementation_tabs[2]:
        st.markdown("### 🔀 Hybrid Approach")
        
        st.markdown("""
        Many production LLM data pipelines use a hybrid approach, combining ETL for 
        well-understood transformations with ELT for exploratory and experimental processing.
        """)
        
        hybrid_code = """
# Hybrid ETL/ELT Pipeline for LLM Data
class HybridLLMPipeline:
    def __init__(self, config):
        self.config = config
        self.spark = self.init_spark()
        self.airflow_dag = self.init_airflow()
    
    def stage_1_etl_critical_cleaning(self, raw_data_path):
        \"\"\"
        ETL for critical, well-understood transformations
        that must be done before storage due to cost/compliance
        \"\"\"
        
        # These transformations are expensive to redo and well-understood
        transformations = [
            self.remove_pii_data,           # Privacy compliance - must be done early
            self.deduplicate_exact_matches, # Storage cost optimization
            self.filter_by_language,        # Clear business requirement
            self.remove_malformed_data      # Data integrity requirement
        ]
        
        processed_data = raw_data_path
        for transform in transformations:
            processed_data = transform(processed_data)
        
        # Store in data lake for ELT processing
        self.store_in_data_lake(processed_data, "bronze_clean")
        
        return processed_data
    
    def stage_2_elt_experimental_processing(self):
        \"\"\"
        ELT for experimental transformations that may change
        based on model requirements or research findings
        \"\"\"
        
        # Use SQL-based transformations for flexibility
        experimental_queries = {
            "quality_scoring_v1": \"\"\"
                SELECT *,
                       (char_count * 0.3 + word_count * 0.4 + readability_score * 0.3) as quality_v1
                FROM bronze_clean
            \"\"\",
            
            "quality_scoring_v2": \"\"\"
                SELECT *,
                       CASE 
                           WHEN source_type = 'academic' THEN quality_v1 * 1.2
                           WHEN source_type = 'social' THEN quality_v1 * 0.8
                           ELSE quality_v1
                       END as quality_v2
                FROM silver_quality_v1
            \"\"\",
            
            "domain_classification": \"\"\"
                SELECT *,
                       CASE 
                           WHEN text RLIKE '(?i)(science|research|study)' THEN 'scientific'
                           WHEN text RLIKE '(?i)(news|report|article)' THEN 'news'
                           WHEN text RLIKE '(?i)(tutorial|how.to|guide)' THEN 'educational'
                           ELSE 'general'
                       END as domain
                FROM silver_quality_v2
            \"\"\"
        }
        
        # Execute experimental transformations
        for version, query in experimental_queries.items():
            self.spark.sql(f"CREATE OR REPLACE TABLE silver_{version} AS {query}")
    
    def stage_3_etl_training_preparation(self, final_silver_table):
        \"\"\"
        ETL for final training data preparation
        Performance-critical operations that benefit from optimization
        \"\"\"
        
        # Use optimized Python code for performance-critical operations
        df = self.spark.table(final_silver_table)
        
        # Tokenization (expensive operation, worth optimizing)
        tokenized_df = self.distributed_tokenization(df)
        
        # Sequence packing for efficient training
        packed_df = self.pack_sequences(tokenized_df)
        
        # Create optimized training format
        training_ready_df = self.create_training_batches(packed_df)
        
        # Write in optimized format for training
        training_ready_df.write \\
                        .format("delta") \\
                        .mode("overwrite") \\
                        .option("optimizeWrite", "true") \\
                        .option("dataChange", "false") \\
                        .save(self.config['training_data_path'])
    
    def create_adaptive_pipeline(self):
        \"\"\"
        Create pipeline that adapts processing strategy based on data characteristics
        \"\"\"
        
        # Analyze incoming data to choose processing strategy
        data_characteristics = self.analyze_data_batch()
        
        if data_characteristics['is_well_structured']:
            # Use ETL for structured data
            return self.create_etl_pipeline()
        elif data_characteristics['requires_experimentation']:
            # Use ELT for experimental processing
            return self.create_elt_pipeline()
        else:
            # Use hybrid approach
            return self.create_hybrid_pipeline()
    
    def orchestrate_hybrid_pipeline(self):
        \"\"\"
        Orchestrate the complete hybrid pipeline
        \"\"\"
        
        from airflow import DAG
        from airflow.operators.python_operator import PythonOperator
        from airflow.operators.bash_operator import BashOperator
        
        # Stage 1: ETL Critical Cleaning
        etl_cleaning = PythonOperator(
            task_id='etl_critical_cleaning',
            python_callable=self.stage_1_etl_critical_cleaning,
            dag=self.airflow_dag
        )
        
        # Stage 2: ELT Experimental Processing
        elt_experimental = BashOperator(
            task_id='elt_experimental_processing',
            bash_command='spark-submit --class ELTProcessor experimental_transforms.py',
            dag=self.airflow_dag
        )
        
        # Stage 3: ETL Training Preparation
        etl_training_prep = PythonOperator(
            task_id='etl_training_preparation', 
            python_callable=self.stage_3_etl_training_preparation,
            dag=self.airflow_dag
        )
        
        # Define dependencies
        etl_cleaning >> elt_experimental >> etl_training_prep
        
        return self.airflow_dag
"""
        
        st.code(hybrid_code, language='python')

with concept_tabs[3]:
    st.subheader("🔄 Data Processing Techniques")
    
    processing_tabs = st.tabs([
        "🧹 Data Cleaning",
        "🔍 Quality Assessment",
        "📊 Deduplication",
        "🌐 Language Processing"
    ])
    
    with processing_tabs[0]:
        st.markdown("### 🧹 Data Cleaning for LLMs")
        
        cleaning_techniques = [
            {
                "technique": "Text Normalization",
                "purpose": "Standardize text format and encoding",
                "methods": [
                    "Unicode normalization (NFC, NFD, NFKC, NFKD)",
                    "Character encoding standardization (UTF-8)",
                    "Whitespace normalization and removal",
                    "Special character handling"
                ],
                "code": """
import unicodedata
import re
from ftfy import fix_text

class TextNormalizer:
    def __init__(self):
        self.whitespace_pattern = re.compile(r'\\s+')
        self.control_char_pattern = re.compile(r'[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]')
    
    def normalize_text(self, text):
        if not text:
            return ""
        
        # Fix common encoding issues
        text = fix_text(text)
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters
        text = self.control_char_pattern.sub('', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_html_markup(self, text):
        from bs4 import BeautifulSoup
        
        # Parse HTML and extract text
        soup = BeautifulSoup(text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get clean text
        clean_text = soup.get_text()
        
        # Normalize spaces and line breaks
        clean_text = re.sub(r'\\n+', '\\n', clean_text)
        clean_text = re.sub(r' +', ' ', clean_text)
        
        return clean_text.strip()
"""
            },
            {
                "technique": "Content Filtering",
                "purpose": "Remove unwanted or harmful content",
                "methods": [
                    "Explicit content detection and removal",
                    "Spam and low-quality content filtering",
                    "Promotional content identification",
                    "Personally identifiable information (PII) removal"
                ],
                "code": """
import re
from transformers import pipeline

class ContentFilter:
    def __init__(self):
        self.toxicity_classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert"
        )
        
        # Common PII patterns
        self.pii_patterns = {
            'email': re.compile(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'),
            'phone': re.compile(r'\\b(?:\\+?1[-. ]?)?(?:\\(?[0-9]{3}\\)?[-. ]?)?[0-9]{3}[-. ]?[0-9]{4}\\b'),
            'ssn': re.compile(r'\\b\\d{3}-\\d{2}-\\d{4}\\b'),
            'credit_card': re.compile(r'\\b(?:\\d{4}[-\\s]?){3}\\d{4}\\b')
        }
        
        # Spam indicators
        self.spam_patterns = [
            re.compile(r'(?i)click here'),
            re.compile(r'(?i)buy now'),
            re.compile(r'(?i)limited time offer'),
            re.compile(r'(?i)earn \\$\\d+ per day')
        ]
    
    def is_toxic(self, text, threshold=0.8):
        if not text or len(text.strip()) < 10:
            return False
        
        try:
            result = self.toxicity_classifier(text[:512])  # Limit length for efficiency
            toxicity_score = result[0]['score'] if result[0]['label'] == 'TOXIC' else 1 - result[0]['score']
            return toxicity_score > threshold
        except:
            return False
    
    def remove_pii(self, text):
        cleaned_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            cleaned_text = pattern.sub(f'[{pii_type.upper()}_REDACTED]', cleaned_text)
        
        return cleaned_text
    
    def is_spam(self, text):
        text_lower = text.lower()
        
        # Check for spam patterns
        spam_score = 0
        for pattern in self.spam_patterns:
            if pattern.search(text):
                spam_score += 1
        
        # Check for excessive capitalization
        if len(text) > 50:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.3:
                spam_score += 1
        
        # Check for excessive punctuation
        punct_ratio = sum(1 for c in text if c in '!?') / max(len(text), 1)
        if punct_ratio > 0.05:
            spam_score += 1
        
        return spam_score >= 2
    
    def filter_content(self, text):
        if self.is_toxic(text) or self.is_spam(text):
            return None
        
        return self.remove_pii(text)
"""
            }
        ]
        
        for technique in cleaning_techniques:
            with st.expander(f"🧹 {technique['technique']}"):
                st.markdown(f"**Purpose:** {technique['purpose']}")
                
                st.markdown("**Methods:**")
                for method in technique['methods']:
                    st.markdown(f"• {method}")
                
                st.markdown("**Implementation:**")
                st.code(technique['code'], language='python')
    
    with processing_tabs[1]:
        st.markdown("### 🔍 Quality Assessment")
        
        quality_code = """
# Comprehensive Quality Assessment for LLM Training Data
import numpy as np
from textstat import flesch_reading_ease, flesch_kincaid_grade
import language_tool_python
from collections import Counter
import re

class QualityAssessor:
    def __init__(self):
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
        self.min_chars = 50
        self.max_chars = 10000
        self.min_words = 10
        self.max_words = 2000
    
    def assess_text_quality(self, text):
        if not text or not isinstance(text, str):
            return {'overall_score': 0.0, 'reasons': ['Invalid text input']}
        
        scores = {}
        issues = []
        
        # Length checks
        char_count = len(text)
        word_count = len(text.split())
        
        if char_count < self.min_chars:
            issues.append(f'Text too short ({char_count} chars)')
            scores['length'] = 0.0
        elif char_count > self.max_chars:
            issues.append(f'Text too long ({char_count} chars)')
            scores['length'] = 0.5
        else:
            scores['length'] = 1.0
        
        # Word count check
        if word_count < self.min_words:
            issues.append(f'Too few words ({word_count})')
            scores['word_count'] = 0.0
        elif word_count > self.max_words:
            issues.append(f'Too many words ({word_count})')
            scores['word_count'] = 0.5
        else:
            scores['word_count'] = 1.0
        
        # Readability assessment
        try:
            readability = flesch_reading_ease(text)
            if readability < 30:  # Very difficult
                scores['readability'] = 0.3
                issues.append('Text very difficult to read')
            elif readability < 50:  # Difficult
                scores['readability'] = 0.6
            elif readability < 70:  # Standard
                scores['readability'] = 0.8
            else:  # Easy
                scores['readability'] = 1.0
        except:
            scores['readability'] = 0.5
            issues.append('Could not assess readability')
        
        # Grammar and language quality
        try:
            grammar_errors = self.grammar_tool.check(text[:1000])  # Limit for performance
            error_ratio = len(grammar_errors) / max(word_count, 1)
            
            if error_ratio > 0.1:  # More than 10% error rate
                scores['grammar'] = 0.2
                issues.append(f'High grammar error rate ({error_ratio:.2%})')
            elif error_ratio > 0.05:  # 5-10% error rate
                scores['grammar'] = 0.6
            else:
                scores['grammar'] = 1.0
        except:
            scores['grammar'] = 0.7  # Default if grammar check fails
        
        # Information density
        scores['information_density'] = self.calculate_information_density(text)
        if scores['information_density'] < 0.3:
            issues.append('Low information density')
        
        # Repetition check
        scores['repetition'] = self.check_repetition(text)
        if scores['repetition'] < 0.5:
            issues.append('High repetition detected')
        
        # Language detection confidence
        scores['language_confidence'] = self.check_language_confidence(text)
        if scores['language_confidence'] < 0.8:
            issues.append('Uncertain language detection')
        
        # Calculate overall score (weighted average)
        weights = {
            'length': 0.1,
            'word_count': 0.1,
            'readability': 0.2,
            'grammar': 0.25,
            'information_density': 0.2,
            'repetition': 0.1,
            'language_confidence': 0.05
        }
        
        overall_score = sum(scores[metric] * weight for metric, weight in weights.items())
        
        return {
            'overall_score': overall_score,
            'component_scores': scores,
            'issues': issues,
            'metrics': {
                'char_count': char_count,
                'word_count': word_count,
                'readability_score': readability if 'readability' in locals() else None
            }
        }
    
    def calculate_information_density(self, text):
        # Simple heuristic: ratio of unique words to total words
        words = text.lower().split()
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        # Adjust for very short texts
        if total_words < 20:
            return 0.5
        
        density = unique_words / total_words
        
        # Normalize to 0-1 scale (0.5-0.9 range maps to 0-1)
        normalized_density = max(0, min(1, (density - 0.5) / 0.4))
        
        return normalized_density
    
    def check_repetition(self, text):
        # Check for repetitive patterns
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) < 3:
            return 1.0
        
        # Check for repeated sentences
        sentence_counts = Counter(s.strip().lower() for s in sentences if s.strip())
        repeated_sentences = sum(1 for count in sentence_counts.values() if count > 1)
        
        repetition_ratio = repeated_sentences / len(sentences)
        
        # Return quality score (inverse of repetition)
        return max(0, 1 - repetition_ratio * 2)
    
    def check_language_confidence(self, text):
        # Simple language detection confidence
        # In practice, use langdetect or similar library
        try:
            from langdetect import detect, detect_langs
            langs = detect_langs(text)
            if langs and langs[0].lang == 'en':
                return langs[0].prob
            else:
                return 0.0
        except:
            # Fallback: check for English characteristics
            english_chars = sum(1 for c in text if c.isascii())
            total_chars = len(text)
            return english_chars / max(total_chars, 1)

# Batch quality assessment
class BatchQualityProcessor:
    def __init__(self, assessor=None):
        self.assessor = assessor or QualityAssessor()
    
    def process_batch(self, texts, quality_threshold=0.7):
        results = []
        
        for i, text in enumerate(texts):
            quality_result = self.assessor.assess_text_quality(text)
            
            results.append({
                'index': i,
                'text': text,
                'quality_score': quality_result['overall_score'],
                'passed_quality': quality_result['overall_score'] >= quality_threshold,
                'issues': quality_result['issues'],
                'metrics': quality_result['metrics']
            })
        
        return results
    
    def generate_quality_report(self, results):
        passed = sum(1 for r in results if r['passed_quality'])
        total = len(results)
        
        avg_score = np.mean([r['quality_score'] for r in results])
        
        # Most common issues
        all_issues = []
        for r in results:
            all_issues.extend(r['issues'])
        
        issue_counts = Counter(all_issues)
        
        report = {
            'total_texts': total,
            'passed_quality': passed,
            'pass_rate': passed / total if total > 0 else 0,
            'average_quality_score': avg_score,
            'common_issues': issue_counts.most_common(5)
        }
        
        return report
"""
        
        st.code(quality_code, language='python')
    
    with processing_tabs[2]:
        st.markdown("### 📊 Deduplication Strategies")
        
        dedup_code = """
# Advanced Deduplication for LLM Training Data
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHashLSH, MinHash
import re

class LLMDeduplicator:
    def __init__(self, similarity_threshold=0.85):
        self.similarity_threshold = similarity_threshold
        self.exact_hashes = set()
        self.minhash_lsh = MinHashLSH(threshold=similarity_threshold, num_perm=128)
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        
    def exact_deduplication(self, texts):
        \"\"\"Remove exact duplicates using hash comparison\"\"\"
        seen_hashes = set()
        unique_texts = []
        duplicate_count = 0
        
        for text in texts:
            # Normalize text for hashing
            normalized = self.normalize_for_hashing(text)
            text_hash = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_texts.append(text)
            else:
                duplicate_count += 1
        
        return unique_texts, duplicate_count
    
    def near_duplicate_detection_minhash(self, texts, chunk_size=1000):
        \"\"\"Detect near-duplicates using MinHash LSH\"\"\"
        unique_texts = []
        duplicate_indices = set()
        
        # Process in chunks for memory efficiency
        for chunk_start in range(0, len(texts), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(texts))
            chunk_texts = texts[chunk_start:chunk_end]
            
            # Create MinHash for each text in chunk
            chunk_minhashes = []
            for i, text in enumerate(chunk_texts):
                global_idx = chunk_start + i
                if global_idx in duplicate_indices:
                    continue
                
                minhash = self.create_minhash(text)
                
                # Check against existing LSH
                similar_items = self.minhash_lsh.query(minhash)
                
                if similar_items:
                    # Mark as duplicate
                    duplicate_indices.add(global_idx)
                else:
                    # Add to LSH and keep text
                    self.minhash_lsh.insert(global_idx, minhash)
                    unique_texts.append(text)
        
        return unique_texts, len(duplicate_indices)
    
    def semantic_deduplication(self, texts, batch_size=100):
        \"\"\"Remove semantically similar texts using TF-IDF and cosine similarity\"\"\"
        unique_texts = []
        duplicate_indices = set()
        
        # Process in batches for memory efficiency
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = [texts[i] for i in range(batch_start, batch_end) 
                          if i not in duplicate_indices]
            
            if not batch_texts:
                continue
            
            # Vectorize batch
            try:
                tfidf_matrix = self.vectorizer.fit_transform(batch_texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Find similar pairs
                batch_duplicates = set()
                for i in range(len(batch_texts)):
                    if i in batch_duplicates:
                        continue
                    
                    for j in range(i + 1, len(batch_texts)):
                        if similarity_matrix[i, j] > self.similarity_threshold:
                            # Keep the longer text (assuming more information)
                            if len(batch_texts[i]) >= len(batch_texts[j]):
                                batch_duplicates.add(j)
                            else:
                                batch_duplicates.add(i)
                                break
                
                # Add non-duplicates to unique texts
                for i, text in enumerate(batch_texts):
                    if i not in batch_duplicates:
                        unique_texts.append(text)
                    else:
                        duplicate_indices.add(batch_start + i)
                        
            except Exception as e:
                # If vectorization fails, keep all texts in batch
                unique_texts.extend(batch_texts)
        
        return unique_texts, len(duplicate_indices)
    
    def create_minhash(self, text):
        \"\"\"Create MinHash signature for text\"\"\"
        minhash = MinHash(num_perm=128)
        
        # Tokenize text into shingles (n-grams)
        shingles = self.create_shingles(text, n=3)
        
        for shingle in shingles:
            minhash.update(shingle.encode('utf-8'))
        
        return minhash
    
    def create_shingles(self, text, n=3):
        \"\"\"Create character n-grams (shingles) from text\"\"\"
        # Clean and normalize text
        cleaned = re.sub(r'\\s+', ' ', text.lower().strip())
        
        # Create character n-grams
        shingles = []
        for i in range(len(cleaned) - n + 1):
            shingles.append(cleaned[i:i + n])
        
        return set(shingles)  # Remove duplicates
    
    def normalize_for_hashing(self, text):
        \"\"\"Normalize text for consistent hashing\"\"\"
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\\s+', ' ', normalized)
        
        # Remove punctuation for fuzzy matching
        normalized = re.sub(r'[^\\w\\s]', '', normalized)
        
        # Strip leading/trailing whitespace
        normalized = normalized.strip()
        
        return normalized
    
    def comprehensive_deduplication(self, texts):
        \"\"\"Apply multiple deduplication strategies in sequence\"\"\"
        print(f"Starting deduplication with {len(texts)} texts")
        
        # Stage 1: Exact deduplication
        texts, exact_dups = self.exact_deduplication(texts)
        print(f"After exact deduplication: {len(texts)} texts ({exact_dups} exact duplicates removed)")
        
        # Stage 2: Near-duplicate detection with MinHash
        if len(texts) > 1000:  # Only for large datasets
            texts, near_dups = self.near_duplicate_detection_minhash(texts)
            print(f"After MinHash deduplication: {len(texts)} texts ({near_dups} near-duplicates removed)")
        
        # Stage 3: Semantic deduplication (for smaller remaining set)
        if len(texts) <= 10000:  # Semantic similarity is expensive
            texts, semantic_dups = self.semantic_deduplication(texts)
            print(f"After semantic deduplication: {len(texts)} texts ({semantic_dups} semantic duplicates removed)")
        
        return texts

# Distributed deduplication for large datasets
class DistributedDeduplicator:
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def spark_exact_deduplication(self, df, text_column='text'):
        \"\"\"Exact deduplication using Spark\"\"\"
        from pyspark.sql.functions import sha2, col
        
        # Add hash column
        df_with_hash = df.withColumn('text_hash', sha2(col(text_column), 256))
        
        # Remove duplicates based on hash
        df_deduplicated = df_with_hash.dropDuplicates(['text_hash'])
        
        # Remove hash column
        df_final = df_deduplicated.drop('text_hash')
        
        return df_final
    
    def spark_near_duplicate_detection(self, df, text_column='text'):
        \"\"\"Near-duplicate detection using Spark and MinHash\"\"\"
        from pyspark.sql.functions import udf
        from pyspark.sql.types import ArrayType, IntegerType
        
        def text_to_minhash(text):
            deduplicator = LLMDeduplicator()
            minhash = deduplicator.create_minhash(text)
            return minhash.hashvalues.tolist()
        
        minhash_udf = udf(text_to_minhash, ArrayType(IntegerType()))
        
        # Add MinHash signatures
        df_with_minhash = df.withColumn('minhash', minhash_udf(col(text_column)))
        
        # Group by similar MinHash signatures (simplified)
        # In practice, would use more sophisticated LSH implementation
        df_deduplicated = df_with_minhash.dropDuplicates(['minhash'])
        
        return df_deduplicated.drop('minhash')
"""
        
        st.code(dedup_code, language='python')
    
    with processing_tabs[3]:
        st.markdown("### 🌐 Language Processing")
        
        language_code = """
# Language Detection and Processing for Multilingual LLM Data
from langdetect import detect, detect_langs, LangDetectException
import fasttext
from collections import Counter
import unicodedata

class LanguageProcessor:
    def __init__(self, fasttext_model_path=None):
        # Initialize language detection models
        try:
            if fasttext_model_path:
                self.fasttext_model = fasttext.load_model(fasttext_model_path)
            else:
                # Download if not available: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
                self.fasttext_model = None
        except:
            self.fasttext_model = None
        
        # Language confidence thresholds
        self.confidence_thresholds = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }
        
        # Supported languages for LLM training
        self.supported_languages = {
            'en': 'English',
            'es': 'Spanish', 
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'hi': 'Hindi'
        }
    
    def detect_language_ensemble(self, text):
        \"\"\"Use ensemble of language detection methods for better accuracy\"\"\"
        if not text or len(text.strip()) < 20:
            return {'language': 'unknown', 'confidence': 0.0, 'method': 'insufficient_text'}
        
        results = []
        
        # Method 1: langdetect library
        try:
            lang_probs = detect_langs(text)
            if lang_probs:
                results.append({
                    'language': lang_probs[0].lang,
                    'confidence': lang_probs[0].prob,
                    'method': 'langdetect'
                })
        except LangDetectException:
            pass
        
        # Method 2: FastText (if available)
        if self.fasttext_model:
            try:
                predictions = self.fasttext_model.predict(text.replace('\\n', ' '), k=1)
                if predictions and len(predictions[0]) > 0:
                    lang_code = predictions[0][0].replace('__label__', '')
                    confidence = float(predictions[1][0])
                    results.append({
                        'language': lang_code,
                        'confidence': confidence,
                        'method': 'fasttext'
                    })
            except:
                pass
        
        # Method 3: Character-based heuristics
        char_based_result = self.detect_language_char_based(text)
        if char_based_result['confidence'] > 0.5:
            results.append(char_based_result)
        
        # Ensemble decision
        if not results:
            return {'language': 'unknown', 'confidence': 0.0, 'method': 'no_detection'}
        
        # Weight results and choose best
        weighted_results = {}
        method_weights = {'langdetect': 0.4, 'fasttext': 0.5, 'char_based': 0.1}
        
        for result in results:
            lang = result['language']
            weight = method_weights.get(result['method'], 0.3)
            weighted_score = result['confidence'] * weight
            
            if lang in weighted_results:
                weighted_results[lang] += weighted_score
            else:
                weighted_results[lang] = weighted_score
        
        # Get best result
        best_lang = max(weighted_results, key=weighted_results.get)
        best_confidence = weighted_results[best_lang]
        
        return {
            'language': best_lang,
            'confidence': best_confidence,
            'method': 'ensemble',
            'all_results': results
        }
    
    def detect_language_char_based(self, text):
        \"\"\"Simple character-based language detection\"\"\"
        char_counts = Counter()
        
        for char in text:
            # Categorize character by script
            char_name = unicodedata.name(char, '')
            
            if 'LATIN' in char_name:
                char_counts['latin'] += 1
            elif 'CYRILLIC' in char_name:
                char_counts['cyrillic'] += 1
            elif 'CJK' in char_name or 'HIRAGANA' in char_name or 'KATAKANA' in char_name:
                char_counts['cjk'] += 1
            elif 'ARABIC' in char_name:
                char_counts['arabic'] += 1
            elif 'DEVANAGARI' in char_name:
                char_counts['devanagari'] += 1
        
        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return {'language': 'unknown', 'confidence': 0.0, 'method': 'char_based'}
        
        # Map character scripts to languages
        script_to_lang = {
            'latin': 'en',  # Default to English for Latin script
            'cyrillic': 'ru',
            'cjk': 'zh',
            'arabic': 'ar',
            'devanagari': 'hi'
        }
        
        # Find dominant script
        dominant_script = max(char_counts, key=char_counts.get)
        confidence = char_counts[dominant_script] / total_chars
        
        return {
            'language': script_to_lang.get(dominant_script, 'unknown'),
            'confidence': confidence,
            'method': 'char_based'
        }
    
    def filter_by_language(self, texts, target_languages=None, min_confidence=0.7):
        \"\"\"Filter texts by language with confidence threshold\"\"\"
        if target_languages is None:
            target_languages = ['en']  # Default to English
        
        filtered_texts = []
        language_stats = Counter()
        
        for text in texts:
            detection_result = self.detect_language_ensemble(text)
            
            language = detection_result['language']
            confidence = detection_result['confidence']
            
            language_stats[language] += 1
            
            # Keep text if it matches target language and confidence criteria
            if (language in target_languages and 
                confidence >= min_confidence):
                filtered_texts.append({
                    'text': text,
                    'language': language,
                    'confidence': confidence
                })
        
        return filtered_texts, dict(language_stats)
    
    def balance_multilingual_dataset(self, texts_by_language, target_distribution):
        \"\"\"Balance dataset across multiple languages\"\"\"
        balanced_texts = []
        
        # Calculate total target samples
        total_target = sum(target_distribution.values())
        
        for language, target_ratio in target_distribution.items():
            if language not in texts_by_language:
                continue
            
            available_texts = texts_by_language[language]
            target_count = int(total_target * target_ratio)
            
            # Sample texts for this language
            if len(available_texts) >= target_count:
                # Randomly sample target_count texts
                import random
                sampled_texts = random.sample(available_texts, target_count)
            else:
                # Use all available texts
                sampled_texts = available_texts
                print(f"Warning: Only {len(available_texts)} texts available for {language}, target was {target_count}")
            
            balanced_texts.extend(sampled_texts)
        
        return balanced_texts
    
    def create_language_quality_report(self, detection_results):
        \"\"\"Create comprehensive language quality report\"\"\"
        total_texts = len(detection_results)
        
        # Language distribution
        language_counts = Counter(r['language'] for r in detection_results)
        
        # Confidence distribution
        confidence_buckets = {
            'high': sum(1 for r in detection_results if r['confidence'] >= self.confidence_thresholds['high']),
            'medium': sum(1 for r in detection_results if self.confidence_thresholds['medium'] <= r['confidence'] < self.confidence_thresholds['high']),
            'low': sum(1 for r in detection_results if self.confidence_thresholds['low'] <= r['confidence'] < self.confidence_thresholds['medium']),
            'very_low': sum(1 for r in detection_results if r['confidence'] < self.confidence_thresholds['low'])
        }
        
        # Supported language coverage
        supported_count = sum(language_counts[lang] for lang in language_counts if lang in self.supported_languages)
        
        report = {
            'total_texts': total_texts,
            'language_distribution': dict(language_counts),
            'confidence_distribution': confidence_buckets,
            'supported_languages_count': supported_count,
            'supported_languages_ratio': supported_count / total_texts if total_texts > 0 else 0,
            'average_confidence': sum(r['confidence'] for r in detection_results) / total_texts if total_texts > 0 else 0
        }
        
        return report

# Distributed language processing
class DistributedLanguageProcessor:
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def process_languages_at_scale(self, df, text_column='text'):
        \"\"\"Process language detection at scale using Spark\"\"\"
        from pyspark.sql.functions import udf
        from pyspark.sql.types import StructType, StructField, StringType, FloatType
        
        # Define return schema for language detection UDF
        language_schema = StructType([
            StructField("language", StringType(), True),
            StructField("confidence", FloatType(), True)
        ])
        
        def detect_language_udf(text):
            processor = LanguageProcessor()
            result = processor.detect_language_ensemble(text)
            return (result['language'], result['confidence'])
        
        language_detect_udf = udf(detect_language_udf, language_schema)
        
        # Apply language detection
        df_with_language = df.withColumn('language_info', language_detect_udf(col(text_column)))
        
        # Extract language and confidence into separate columns
        df_processed = df_with_language.withColumn('detected_language', col('language_info.language')) \\
                                      .withColumn('language_confidence', col('language_info.confidence')) \\
                                      .drop('language_info')
        
        return df_processed
"""
        
        st.code(language_code, language='python')

# Implementation examples
st.header("🛠️ Implementation Examples")

example_tabs = st.tabs([
    "🚀 Basic Pipeline",
    "🏭 Production Pipeline",
    "📊 Monitoring & Quality",
    "⚡ Performance Optimization"
])

with example_tabs[0]:
    st.subheader("🚀 Basic Data Pipeline")
    
    basic_pipeline_code = """
# Basic LLM Data Engineering Pipeline
import os
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd
from transformers import AutoTokenizer
import datasets

class BasicLLMDataPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['tokenizer_name'])
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processing components
        self.text_normalizer = TextNormalizer()
        self.content_filter = ContentFilter()
        self.quality_assessor = QualityAssessor()
        self.deduplicator = LLMDeduplicator()
    
    def run_pipeline(self, input_sources: List[str]):
        \"\"\"Run the complete data pipeline\"\"\"
        print("Starting LLM data pipeline...")
        
        # Step 1: Data Collection
        raw_texts = self.collect_data(input_sources)
        print(f"Collected {len(raw_texts)} raw texts")
        
        # Step 2: Text Cleaning and Normalization
        cleaned_texts = self.clean_and_normalize(raw_texts)
        print(f"Cleaned texts: {len(cleaned_texts)} remaining")
        
        # Step 3: Quality Filtering
        quality_texts = self.filter_by_quality(cleaned_texts)
        print(f"After quality filtering: {len(quality_texts)} remaining")
        
        # Step 4: Deduplication
        unique_texts = self.deduplicator.comprehensive_deduplication(quality_texts)
        print(f"After deduplication: {len(unique_texts)} unique texts")
        
        # Step 5: Tokenization and Formatting
        processed_data = self.tokenize_and_format(unique_texts)
        print(f"Tokenized {len(processed_data)} examples")
        
        # Step 6: Create Dataset Splits
        train_data, val_data, test_data = self.create_splits(processed_data)
        
        # Step 7: Save Processed Data
        self.save_datasets(train_data, val_data, test_data)
        
        # Step 8: Generate Report
        self.generate_pipeline_report(raw_texts, unique_texts, processed_data)
        
        print("Pipeline completed successfully!")
        
        return {
            'train_size': len(train_data),
            'val_size': len(val_data), 
            'test_size': len(test_data),
            'total_tokens': sum(len(item['input_ids']) for item in processed_data)
        }
    
    def collect_data(self, input_sources: List[str]) -> List[str]:
        \"\"\"Collect text data from various sources\"\"\"
        texts = []
        
        for source in input_sources:
            if source.endswith('.txt'):
                with open(source, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Split into paragraphs or documents
                    documents = content.split('\\n\\n')
                    texts.extend([doc.strip() for doc in documents if doc.strip()])
            
            elif source.endswith('.jsonl'):
                with open(source, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        if 'text' in data:
                            texts.append(data['text'])
            
            elif source.endswith('.csv'):
                df = pd.read_csv(source)
                if 'text' in df.columns:
                    texts.extend(df['text'].dropna().tolist())
        
        return texts
    
    def clean_and_normalize(self, texts: List[str]) -> List[str]:
        \"\"\"Clean and normalize text data\"\"\"
        cleaned_texts = []
        
        for text in texts:
            # Normalize text
            normalized = self.text_normalizer.normalize_text(text)
            
            if not normalized:
                continue
            
            # Apply content filtering
            filtered = self.content_filter.filter_content(normalized)
            
            if filtered:
                cleaned_texts.append(filtered)
        
        return cleaned_texts
    
    def filter_by_quality(self, texts: List[str]) -> List[str]:
        \"\"\"Filter texts based on quality scores\"\"\"
        quality_texts = []
        quality_threshold = self.config.get('quality_threshold', 0.7)
        
        for text in texts:
            quality_result = self.quality_assessor.assess_text_quality(text)
            
            if quality_result['overall_score'] >= quality_threshold:
                quality_texts.append(text)
        
        return quality_texts
    
    def tokenize_and_format(self, texts: List[str]) -> List[Dict]:
        \"\"\"Tokenize texts and format for training\"\"\"
        processed_data = []
        max_length = self.config.get('max_length', 2048)
        
        for text in texts:
            # Tokenize
            tokens = self.tokenizer.encode(
                text,
                max_length=max_length,
                truncation=True,
                padding=False
            )
            
            # Skip if too short
            if len(tokens) < self.config.get('min_length', 50):
                continue
            
            # Format for training
            processed_data.append({
                'input_ids': tokens,
                'attention_mask': [1] * len(tokens),
                'labels': tokens,  # For causal language modeling
                'text': text,
                'length': len(tokens)
            })
        
        return processed_data
    
    def create_splits(self, data: List[Dict]):
        \"\"\"Create train/validation/test splits\"\"\"
        import random
        random.seed(42)  # For reproducibility
        
        # Shuffle data
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        # Calculate split sizes
        total_size = len(shuffled_data)
        train_size = int(total_size * self.config.get('train_ratio', 0.8))
        val_size = int(total_size * self.config.get('val_ratio', 0.1))
        
        # Create splits
        train_data = shuffled_data[:train_size]
        val_data = shuffled_data[train_size:train_size + val_size]
        test_data = shuffled_data[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def save_datasets(self, train_data, val_data, test_data):
        \"\"\"Save processed datasets\"\"\"
        # Convert to HuggingFace datasets format
        train_dataset = datasets.Dataset.from_list(train_data)
        val_dataset = datasets.Dataset.from_list(val_data)
        test_dataset = datasets.Dataset.from_list(test_data)
        
        # Save datasets
        train_dataset.save_to_disk(self.output_dir / 'train')
        val_dataset.save_to_disk(self.output_dir / 'validation')
        test_dataset.save_to_disk(self.output_dir / 'test')
        
        # Save metadata
        metadata = {
            'tokenizer_name': self.config['tokenizer_name'],
            'max_length': self.config.get('max_length', 2048),
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'total_tokens': sum(len(item['input_ids']) for item in train_data + val_data + test_data)
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def generate_pipeline_report(self, raw_texts, final_texts, processed_data):
        \"\"\"Generate pipeline processing report\"\"\"
        report = {
            'pipeline_config': self.config,
            'processing_stats': {
                'raw_texts_count': len(raw_texts),
                'final_texts_count': len(final_texts),
                'processed_examples_count': len(processed_data),
                'total_tokens': sum(len(item['input_ids']) for item in processed_data),
                'avg_tokens_per_example': sum(len(item['input_ids']) for item in processed_data) / len(processed_data) if processed_data else 0,
                'retention_rate': len(final_texts) / len(raw_texts) if raw_texts else 0
            },
            'quality_metrics': {
                'examples_by_length': self.analyze_length_distribution(processed_data),
                'token_distribution': self.analyze_token_distribution(processed_data)
            }
        }
        
        with open(self.output_dir / 'pipeline_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\\nPipeline Report:")
        print(f"Raw texts: {report['processing_stats']['raw_texts_count']}")
        print(f"Final texts: {report['processing_stats']['final_texts_count']}")
        print(f"Retention rate: {report['processing_stats']['retention_rate']:.2%}")
        print(f"Total tokens: {report['processing_stats']['total_tokens']:,}")
        print(f"Average tokens per example: {report['processing_stats']['avg_tokens_per_example']:.1f}")
    
    def analyze_length_distribution(self, data):
        \"\"\"Analyze token length distribution\"\"\"
        lengths = [len(item['input_ids']) for item in data]
        
        return {
            'min': min(lengths) if lengths else 0,
            'max': max(lengths) if lengths else 0,
            'mean': sum(lengths) / len(lengths) if lengths else 0,
            'median': sorted(lengths)[len(lengths)//2] if lengths else 0
        }
    
    def analyze_token_distribution(self, data):
        \"\"\"Analyze token usage patterns\"\"\"
        from collections import Counter
        
        token_counts = Counter()
        for item in data:
            for token_id in item['input_ids']:
                token_counts[token_id] += 1
        
        return {
            'unique_tokens': len(token_counts),
            'most_common_tokens': token_counts.most_common(10),
            'total_token_occurrences': sum(token_counts.values())
        }

# Usage example
if __name__ == "__main__":
    # Configuration
    config = {
        "tokenizer_name": "gpt2",
        "output_dir": "./processed_data",
        "max_length": 2048,
        "min_length": 50,
        "quality_threshold": 0.7,
        "train_ratio": 0.8,
        "val_ratio": 0.1
    }
    
    # Save config
    with open('pipeline_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run pipeline
    pipeline = BasicLLMDataPipeline('pipeline_config.json')
    results = pipeline.run_pipeline([
        'data/raw_text_1.txt',
        'data/raw_text_2.jsonl',
        'data/raw_text_3.csv'
    ])
    
    print("Pipeline results:", results)
"""
    
    st.code(basic_pipeline_code, language='python')

# Continue with additional comprehensive content as requested...
# I need to continue building this comprehensive educational app with all the remaining topics

st.markdown("### 📚 Additional Resources")

resources = [
    {
        "title": "The BigScience Workshop Data Pipeline",
        "type": "Technical Report",
        "description": "Comprehensive documentation of data engineering for BLOOM model",
        "difficulty": "Advanced"
    },
    {
        "title": "Data Engineering for Machine Learning",
        "type": "Book", 
        "description": "Best practices for ML data pipelines and infrastructure",
        "difficulty": "Intermediate"
    },
    {
        "title": "Apache Spark for Large Scale Data Processing",
        "type": "Documentation",
        "description": "Official guide to distributed data processing",
        "difficulty": "Intermediate"
    }
]

for resource in resources:
    with st.expander(f"📖 {resource['title']}"):
        st.markdown(f"**Type:** {resource['type']}")
        st.markdown(f"**Description:** {resource['description']}")
        st.markdown(f"**Difficulty:** {resource['difficulty']}")