import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Data Engineering for LLMs", page_icon="🏗️", layout="wide")

st.title("🏗️ Data Engineering for LLMs")
st.markdown("### Building Robust Data Infrastructure for Language Model Training")

# Overview
st.header("🎯 Overview")
st.markdown("""
Data Engineering for LLMs involves designing and implementing scalable data pipelines that can handle 
the massive datasets required for training large language models. This includes data collection, 
processing, storage, and serving infrastructure that can handle petabytes of text data efficiently.
""")

# Core concepts
st.header("🧠 Core Concepts")

concept_tabs = st.tabs([
    "📊 Data Pipeline Architecture",
    "⚡ ETL vs ELT", 
    "🗄️ Storage Systems",
    "🔄 Real-time Processing"
])

with concept_tabs[0]:
    st.subheader("📊 Data Pipeline Architecture")
    
    st.markdown("""
    LLM data pipelines must handle diverse data sources, massive scale, and complex processing 
    requirements while maintaining data quality and lineage.
    """)
    
    # Pipeline components
    pipeline_components = [
        {
            "component": "Data Ingestion",
            "description": "Collecting data from various sources at scale",
            "technologies": [
                "Apache Kafka for streaming data",
                "Apache NiFi for data flow management",
                "Cloud storage APIs (S3, GCS, Azure Blob)",
                "Web scraping frameworks (Scrapy, Beautiful Soup)"
            ],
            "challenges": [
                "Handling diverse data formats and encodings",
                "Managing rate limits and API quotas",
                "Ensuring data freshness and timeliness",
                "Dealing with unreliable data sources"
            ],
            "implementation": """
# Scalable data ingestion pipeline
from kafka import KafkaProducer
import asyncio
import aiohttp
from typing import AsyncGenerator

class DataIngestionPipeline:
    def __init__(self, kafka_bootstrap_servers, batch_size=1000):
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.batch_size = batch_size
        self.rate_limiter = RateLimiter(requests_per_second=100)
    
    async def ingest_from_api(self, api_endpoint: str, topic: str):
        async with aiohttp.ClientSession() as session:
            async for batch in self.fetch_data_batches(session, api_endpoint):
                for record in batch:
                    await self.rate_limiter.acquire()
                    self.producer.send(topic, record)
                
                # Commit batch
                self.producer.flush()
    
    async def fetch_data_batches(self, session, endpoint) -> AsyncGenerator:
        offset = 0
        while True:
            try:
                async with session.get(f"{endpoint}?offset={offset}&limit={self.batch_size}") as response:
                    if response.status == 200:
                        data = await response.json()
                        if not data:
                            break
                        yield data
                        offset += self.batch_size
                    else:
                        await asyncio.sleep(60)  # Backoff on error
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                await asyncio.sleep(30)

# Web scraping with distributed processing
class DistributedWebScraper:
    def __init__(self, worker_count=10, proxy_pool=None):
        self.worker_count = worker_count
        self.proxy_pool = proxy_pool or []
        self.url_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
    
    async def scrape_urls(self, urls: list):
        # Add URLs to queue
        for url in urls:
            await self.url_queue.put(url)
        
        # Start workers
        workers = [
            asyncio.create_task(self.worker(worker_id))
            for worker_id in range(self.worker_count)
        ]
        
        # Wait for completion
        await self.url_queue.join()
        
        # Cancel workers
        for worker in workers:
            worker.cancel()
    
    async def worker(self, worker_id: int):
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    url = await self.url_queue.get()
                    proxy = self.get_proxy() if self.proxy_pool else None
                    
                    async with session.get(url, proxy=proxy, timeout=30) as response:
                        if response.status == 200:
                            content = await response.text()
                            extracted_data = self.extract_content(content, url)
                            await self.result_queue.put(extracted_data)
                    
                    self.url_queue.task_done()
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} error: {e}")
                    self.url_queue.task_done()
"""
        },
        {
            "component": "Data Processing",
            "description": "Transforming raw data into training-ready format",
            "technologies": [
                "Apache Spark for distributed processing",
                "Dask for Python-based parallel computing",
                "Ray for ML workload distribution",
                "Apache Beam for unified batch/stream processing"
            ],
            "challenges": [
                "Text normalization and cleaning at scale",
                "Language detection and filtering",
                "Duplicate detection and deduplication",
                "Quality assessment and filtering"
            ],
            "implementation": """
# Distributed text processing with Spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import re

class LLMDataProcessor:
    def __init__(self, app_name="LLM_Data_Processing"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        self.text_quality_threshold = 0.7
        self.min_text_length = 100
        self.max_text_length = 1000000
    
    def process_text_corpus(self, input_path: str, output_path: str):
        # Read raw text data
        df = self.spark.read.text(input_path)
        
        # Apply processing pipeline
        processed_df = df \
            .withColumn("text_length", length(col("value"))) \
            .filter(col("text_length").between(self.min_text_length, self.max_text_length)) \
            .withColumn("cleaned_text", self.clean_text_udf(col("value"))) \
            .withColumn("language", self.detect_language_udf(col("cleaned_text"))) \
            .filter(col("language") == "en") \
            .withColumn("quality_score", self.assess_quality_udf(col("cleaned_text"))) \
            .filter(col("quality_score") > self.text_quality_threshold) \
            .withColumn("hash", sha2(col("cleaned_text"), 256)) \
            .dropDuplicates(["hash"]) \
            .select("cleaned_text", "quality_score", "language")
        
        # Write processed data
        processed_df.write \
            .mode("overwrite") \
            .parquet(output_path)
        
        return processed_df.count()
    
    @staticmethod
    def clean_text(text: str) -> str:
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text.strip()
    
    def assess_text_quality(self, text: str) -> float:
        if not text or len(text) < 10:
            return 0.0
        
        score = 1.0
        
        # Penalize excessive repetition
        words = text.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                score -= 0.4
        
        # Penalize excessive punctuation
        punct_ratio = sum(1 for c in text if c in '!@#$%^&*()') / len(text)
        if punct_ratio > 0.1:
            score -= 0.3
        
        # Penalize non-alphabetic content
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio < 0.5:
            score -= 0.3
        
        return max(0.0, score)

# Real-time stream processing with Kafka
class StreamProcessor:
    def __init__(self, kafka_config):
        self.kafka_config = kafka_config
        self.text_processor = TextProcessor()
    
    async def process_stream(self, input_topic: str, output_topic: str):
        from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
        
        consumer = AIOKafkaConsumer(
            input_topic,
            **self.kafka_config,
            group_id="llm_data_processor"
        )
        
        producer = AIOKafkaProducer(**self.kafka_config)
        
        await consumer.start()
        await producer.start()
        
        try:
            async for message in consumer:
                try:
                    # Decode and process message
                    raw_text = message.value.decode('utf-8')
                    processed_text = self.text_processor.process(raw_text)
                    
                    if processed_text:
                        await producer.send(
                            output_topic,
                            processed_text.encode('utf-8')
                        )
                
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
        
        finally:
            await consumer.stop()
            await producer.stop()
"""
        },
        {
            "component": "Data Storage",
            "description": "Efficient storage and retrieval of massive datasets",
            "technologies": [
                "Apache Parquet for columnar storage",
                "Delta Lake for ACID transactions",
                "Apache Iceberg for table format",
                "Object storage (S3, GCS, Azure Blob)"
            ],
            "challenges": [
                "Optimizing storage formats for ML workloads",
                "Managing data versioning and lineage",
                "Ensuring data consistency and integrity",
                "Balancing cost and performance"
            ],
            "implementation": """
# Optimized data storage for LLM training
import pyarrow as pa
import pyarrow.parquet as pq
from deltalake import DeltaTable, write_deltalake
import pandas as pd

class LLMDataStorage:
    def __init__(self, storage_path: str, partition_columns=None):
        self.storage_path = storage_path
        self.partition_columns = partition_columns or ["language", "source"]
        
    def write_training_data(self, df: pd.DataFrame, mode="append"):
        # Optimize schema for training data
        schema = pa.schema([
            ("text", pa.string()),
            ("tokens", pa.int64()),
            ("language", pa.string()),
            ("source", pa.string()),
            ("quality_score", pa.float64()),
            ("created_at", pa.timestamp('ns'))
        ])
        
        # Convert to Arrow table with optimized schema
        table = pa.Table.from_pandas(df, schema=schema)
        
        # Write with Delta Lake for ACID properties
        write_deltalake(
            self.storage_path,
            table,
            mode=mode,
            partition_by=self.partition_columns,
            engine="rust",  # Use faster Rust engine
            storage_options={"AWS_S3_ALLOW_UNSAFE_RENAME": "true"}
        )
    
    def optimize_storage(self):
        # Optimize Delta table
        dt = DeltaTable(self.storage_path)
        
        # Compact small files
        dt.optimize.compact()
        
        # Z-order by frequently filtered columns
        dt.optimize.z_order(["language", "quality_score"])
        
        # Vacuum old files (careful in production)
        dt.vacuum(retention_hours=168)  # 7 days
    
    def create_training_splits(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        dt = DeltaTable(self.storage_path)
        df = dt.to_pandas()
        
        # Stratified sampling by language
        splits = {}
        for language in df['language'].unique():
            lang_df = df[df['language'] == language]
            
            # Shuffle and split
            shuffled = lang_df.sample(frac=1, random_state=42)
            n = len(shuffled)
            
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            splits[f'train_{language}'] = shuffled[:train_end]
            splits[f'val_{language}'] = shuffled[train_end:val_end]
            splits[f'test_{language}'] = shuffled[val_end:]
        
        # Write splits to separate tables
        for split_name, split_df in splits.items():
            split_path = f"{self.storage_path}_splits/{split_name}"
            write_deltalake(split_path, split_df, mode="overwrite")
        
        return splits

# Efficient data loading for training
class DataLoader:
    def __init__(self, data_path: str, batch_size=1000):
        self.data_path = data_path
        self.batch_size = batch_size
        
    def load_batches(self, filters=None):
        dt = DeltaTable(self.data_path)
        
        # Apply filters if provided
        if filters:
            dt = dt.filter(filters)
        
        # Read in batches to manage memory
        parquet_files = dt.file_uris()
        
        for file_batch in self.chunk_files(parquet_files, self.batch_size):
            # Read batch of files
            tables = []
            for file_uri in file_batch:
                table = pq.read_table(file_uri)
                tables.append(table)
            
            # Combine and yield
            combined_table = pa.concat_tables(tables)
            yield combined_table.to_pandas()
    
    def chunk_files(self, files, chunk_size):
        for i in range(0, len(files), chunk_size):
            yield files[i:i + chunk_size]
"""
        },
        {
            "component": "Data Quality",
            "description": "Ensuring high-quality training data through validation and monitoring",
            "technologies": [
                "Great Expectations for data validation",
                "Apache Griffin for data quality",
                "Deequ for data quality metrics",
                "Custom ML-based quality assessment"
            ],
            "challenges": [
                "Defining quality metrics for text data",
                "Scaling quality checks to petabyte datasets",
                "Balancing quality vs. quantity trade-offs",
                "Continuous monitoring of data drift"
            ],
            "implementation": """
# Comprehensive data quality framework
import great_expectations as ge
from typing import Dict, List, Tuple
import numpy as np

class LLMDataQuality:
    def __init__(self):
        self.quality_checks = {
            'completeness': self.check_completeness,
            'uniqueness': self.check_uniqueness,
            'validity': self.check_validity,
            'consistency': self.check_consistency,
            'accuracy': self.check_accuracy
        }
        
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, float]:
        results = {}
        
        for check_name, check_func in self.quality_checks.items():
            try:
                score = check_func(df)
                results[check_name] = score
            except Exception as e:
                logger.error(f"Quality check {check_name} failed: {e}")
                results[check_name] = 0.0
        
        # Calculate overall quality score
        results['overall'] = np.mean(list(results.values()))
        
        return results
    
    def check_completeness(self, df: pd.DataFrame) -> float:
        # Check for missing or empty text
        total_rows = len(df)
        if total_rows == 0:
            return 0.0
        
        complete_rows = df['text'].notna().sum()
        non_empty_rows = (df['text'].str.len() > 0).sum()
        
        return min(complete_rows / total_rows, non_empty_rows / total_rows)
    
    def check_uniqueness(self, df: pd.DataFrame) -> float:
        # Check for duplicate content
        total_rows = len(df)
        if total_rows == 0:
            return 1.0
        
        unique_rows = df['text'].nunique()
        return unique_rows / total_rows
    
    def check_validity(self, df: pd.DataFrame) -> float:
        # Check text validity (encoding, format, etc.)
        valid_count = 0
        total_count = len(df)
        
        for text in df['text']:
            if self.is_valid_text(text):
                valid_count += 1
        
        return valid_count / total_count if total_count > 0 else 0.0
    
    def is_valid_text(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        
        # Check minimum length
        if len(text) < 10:
            return False
        
        # Check for excessive special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.3:
            return False
        
        # Check for reasonable character distribution
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio < 0.5:
            return False
        
        return True
    
    def check_consistency(self, df: pd.DataFrame) -> float:
        # Check consistency in formatting, language, etc.
        consistency_scores = []
        
        # Language consistency
        if 'language' in df.columns:
            detected_langs = df['text'].apply(self.detect_language)
            consistency = (detected_langs == df['language']).mean()
            consistency_scores.append(consistency)
        
        # Encoding consistency
        encoding_consistency = df['text'].apply(self.check_encoding_consistency).mean()
        consistency_scores.append(encoding_consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def check_accuracy(self, df: pd.DataFrame) -> float:
        # Use ML models to assess content accuracy
        accuracy_scores = []
        
        # Factual consistency check (simplified)
        for text in df['text'].sample(min(100, len(df))):
            accuracy_score = self.assess_factual_accuracy(text)
            accuracy_scores.append(accuracy_score)
        
        return np.mean(accuracy_scores) if accuracy_scores else 0.5

# Data drift monitoring
class DataDriftMonitor:
    def __init__(self, reference_dataset_path: str):
        self.reference_stats = self.compute_dataset_statistics(reference_dataset_path)
        
    def compute_dataset_statistics(self, dataset_path: str) -> Dict:
        dt = DeltaTable(dataset_path)
        df = dt.to_pandas()
        
        stats = {
            'text_length_distribution': df['text'].str.len().describe().to_dict(),
            'language_distribution': df['language'].value_counts(normalize=True).to_dict(),
            'quality_score_distribution': df['quality_score'].describe().to_dict(),
            'vocab_size': len(set(' '.join(df['text']).split())),
            'avg_sentence_length': df['text'].str.split('.').str.len().mean()
        }
        
        return stats
    
    def detect_drift(self, new_dataset_path: str, threshold=0.1) -> Dict[str, bool]:
        new_stats = self.compute_dataset_statistics(new_dataset_path)
        drift_detected = {}
        
        # Compare distributions
        for stat_name in self.reference_stats:
            if isinstance(self.reference_stats[stat_name], dict):
                # Compare distribution metrics
                ref_mean = self.reference_stats[stat_name].get('mean', 0)
                new_mean = new_stats[stat_name].get('mean', 0)
                
                if ref_mean > 0:
                    drift_ratio = abs(new_mean - ref_mean) / ref_mean
                    drift_detected[stat_name] = drift_ratio > threshold
            else:
                # Compare single values
                ref_val = self.reference_stats[stat_name]
                new_val = new_stats[stat_name]
                
                if ref_val > 0:
                    drift_ratio = abs(new_val - ref_val) / ref_val
                    drift_detected[stat_name] = drift_ratio > threshold
        
        return drift_detected
"""
        }
    ]
    
    for component in pipeline_components:
        with st.expander(f"🔧 {component['component']}"):
            st.markdown(component['description'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Technologies:**")
                for tech in component['technologies']:
                    st.markdown(f"• {tech}")
            with col2:
                st.markdown("**Challenges:**")
                for challenge in component['challenges']:
                    st.markdown(f"• {challenge}")
            
            st.markdown("**Implementation Example:**")
            st.code(component['implementation'], language='python')

with concept_tabs[1]:
    st.subheader("⚡ ETL vs ELT for LLM Data")
    
    st.markdown("""
    The choice between ETL (Extract, Transform, Load) and ELT (Extract, Load, Transform) 
    approaches significantly impacts LLM data pipeline design and performance.
    """)
    
    # ETL vs ELT comparison
    etl_elt_comparison = {
        'Aspect': [
            'Processing Location',
            'Data Movement',
            'Scalability',
            'Flexibility',
            'Performance',
            'Cost',
            'Complexity',
            'Real-time Support'
        ],
        'ETL (Extract-Transform-Load)': [
            'External processing engine',
            'Raw data → Processing → Clean data → Storage',
            'Limited by processing engine capacity',
            'Requires schema definition upfront',
            'Fast queries on pre-processed data',
            'Higher processing infrastructure costs',
            'More complex pipeline management',
            'Challenging for real-time processing'
        ],
        'ELT (Extract-Load-Transform)': [
            'Data warehouse/lake processing',
            'Raw data → Storage → Transform in place',
            'Leverages warehouse compute scalability',
            'Schema-on-read flexibility',
            'Processing cost scales with usage',
            'Lower upfront costs, pay-per-use',
            'Simpler initial pipeline',
            'Better support for streaming data'
        ]
    }
    
    comparison_df = pd.DataFrame(etl_elt_comparison)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Implementation examples
    approach_tabs = st.tabs(["🔧 ETL Implementation", "📊 ELT Implementation", "🔄 Hybrid Approach"])
    
    with approach_tabs[0]:
        st.markdown("### 🔧 ETL Implementation for LLM Data")
        
        etl_code = """
# ETL Pipeline for LLM Training Data
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from transformers import pipeline

class LLMDataETL:
    def __init__(self):
        self.text_classifier = pipeline("text-classification", model="unitary/toxic-bert")
        self.language_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
        
    def extract_data(self, **context):
        # Extract from multiple sources
        sources = [
            {'type': 'web_scraping', 'config': {...}},
            {'type': 'api', 'config': {...}},
            {'type': 'files', 'config': {...}}
        ]
        
        raw_data = []
        for source in sources:
            extractor = self.get_extractor(source['type'])
            data = extractor.extract(source['config'])
            raw_data.extend(data)
        
        # Store raw data temporarily
        raw_df = pd.DataFrame(raw_data)
        raw_df.to_parquet('/tmp/raw_data.parquet')
        
        return '/tmp/raw_data.parquet'
    
    def transform_data(self, raw_data_path: str, **context):
        # Load raw data
        df = pd.read_parquet(raw_data_path)
        
        # Apply transformations
        transformed_data = []
        
        for _, row in df.iterrows():
            text = row['text']
            
            # Text cleaning
            cleaned_text = self.clean_text(text)
            
            # Quality filtering
            if not self.passes_quality_checks(cleaned_text):
                continue
            
            # Language detection
            language = self.detect_language(cleaned_text)
            if language not in ['en', 'es', 'fr']:  # Target languages
                continue
            
            # Toxicity filtering
            toxicity_score = self.assess_toxicity(cleaned_text)
            if toxicity_score > 0.8:
                continue
            
            # Tokenization and length filtering
            tokens = self.tokenize(cleaned_text)
            if len(tokens) < 50 or len(tokens) > 2048:
                continue
            
            # Create processed record
            processed_record = {
                'text': cleaned_text,
                'language': language,
                'token_count': len(tokens),
                'quality_score': self.calculate_quality_score(cleaned_text),
                'source': row['source'],
                'processed_at': datetime.now(),
                'hash': hashlib.sha256(cleaned_text.encode()).hexdigest()
            }
            
            transformed_data.append(processed_record)
        
        # Create processed DataFrame
        processed_df = pd.DataFrame(transformed_data)
        
        # Deduplication
        processed_df = processed_df.drop_duplicates(subset=['hash'])
        
        # Save transformed data
        output_path = '/tmp/transformed_data.parquet'
        processed_df.to_parquet(output_path)
        
        return output_path
    
    def load_data(self, transformed_data_path: str, **context):
        # Load transformed data
        df = pd.read_parquet(transformed_data_path)
        
        # Load to final storage with partitioning
        storage_engine = DataStorageEngine()
        
        # Partition by language and date
        for language in df['language'].unique():
            lang_df = df[df['language'] == language]
            
            storage_engine.write_partition(
                data=lang_df,
                partition_keys={'language': language, 'date': datetime.now().date()},
                table_name='llm_training_data'
            )
        
        # Update metadata and catalog
        catalog = DataCatalog()
        catalog.register_dataset(
            name='llm_training_data',
            schema=df.dtypes.to_dict(),
            row_count=len(df),
            size_mb=df.memory_usage(deep=True).sum() / (1024*1024),
            created_at=datetime.now()
        )
        
        return f"Loaded {len(df)} records to training dataset"

# Airflow DAG definition
default_args = {
    'owner': 'llm-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'llm_data_etl',
    default_args=default_args,
    description='ETL pipeline for LLM training data',
    schedule_interval='@daily',
    catchup=False,
    tags=['llm', 'data-engineering']
)

etl = LLMDataETL()

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=etl.extract_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=etl.transform_data,
    op_kwargs={'raw_data_path': '{{ ti.xcom_pull(task_ids="extract_data") }}'},
    dag=dag
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=etl.load_data,
    op_kwargs={'transformed_data_path': '{{ ti.xcom_pull(task_ids="transform_data") }}'},
    dag=dag
)

extract_task >> transform_task >> load_task
"""
        
        st.code(etl_code, language='python')
        
        st.markdown("**ETL Benefits for LLM Data:**")
        etl_benefits = [
            "Pre-processed data is immediately ready for training",
            "Consistent data quality through upfront validation",
            "Reduced storage costs through early filtering",
            "Better performance for repeated data access"
        ]
        for benefit in etl_benefits:
            st.markdown(f"• {benefit}")
    
    with approach_tabs[1]:
        st.markdown("### 📊 ELT Implementation for LLM Data")
        
        elt_code = """
# ELT Pipeline using Modern Data Stack
import dbt
from dbt.cli.main import dbtRunner
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator

class LLMDataELT:
    def __init__(self, warehouse_config):
        self.warehouse = SnowflakeConnection(warehouse_config)
        self.dbt_runner = dbtRunner()
        
    def extract_and_load_raw(self):
        # Extract from sources and load directly to data warehouse
        sources = [
            {'name': 'web_data', 'connector': WebScrapingConnector()},
            {'name': 'api_data', 'connector': APIConnector()},
            {'name': 'file_data', 'connector': FileConnector()}
        ]
        
        for source in sources:
            # Stream data directly to warehouse
            raw_table = f"raw_{source['name']}"
            
            for batch in source['connector'].extract_batches():
                self.warehouse.load_batch(
                    data=batch,
                    table=raw_table,
                    schema='raw_data'
                )
    
    def transform_with_dbt(self):
        # Use dbt for transformations in the warehouse
        dbt_commands = [
            ['run', '--models', 'staging'],
            ['test', '--models', 'staging'],
            ['run', '--models', 'intermediate'],
            ['test', '--models', 'intermediate'],
            ['run', '--models', 'marts'],
            ['test', '--models', 'marts']
        ]
        
        for command in dbt_commands:
            result = self.dbt_runner.invoke(command)
            if not result.success:
                raise Exception(f"dbt command failed: {command}")

# dbt models for LLM data transformation

# models/staging/stg_raw_text.sql
'''
{{ config(materialized='view') }}

SELECT 
    raw_text,
    source_url,
    extracted_at,
    source_type,
    -- Basic cleaning
    TRIM(REGEXP_REPLACE(raw_text, '\\s+', ' ')) as cleaned_text,
    LENGTH(raw_text) as text_length,
    -- Generate unique identifier
    SHA2(raw_text) as content_hash
FROM {{ ref('raw_web_data') }}
WHERE raw_text IS NOT NULL
  AND LENGTH(raw_text) > 10
'''

# models/intermediate/int_text_quality.sql
'''
{{ config(materialized='table') }}

WITH text_metrics AS (
    SELECT *,
        -- Calculate quality metrics
        REGEXP_COUNT(cleaned_text, '[a-zA-Z]') / text_length as alpha_ratio,
        REGEXP_COUNT(cleaned_text, '[0-9]') / text_length as digit_ratio,
        REGEXP_COUNT(cleaned_text, '[^a-zA-Z0-9\\s]') / text_length as special_ratio,
        -- Language detection (simplified)
        CASE 
            WHEN REGEXP_LIKE(cleaned_text, '\\b(the|and|or|but|in|on|at|to|for|of|with|by)\\b', 'i') 
            THEN 'en'
            ELSE 'other'
        END as detected_language
    FROM {{ ref('stg_raw_text') }}
),

quality_scored AS (
    SELECT *,
        CASE 
            WHEN alpha_ratio >= 0.6 
             AND special_ratio <= 0.2 
             AND text_length BETWEEN 100 AND 10000
             AND detected_language = 'en'
            THEN 1.0
            WHEN alpha_ratio >= 0.4 
             AND special_ratio <= 0.4 
             AND text_length BETWEEN 50 AND 20000
            THEN 0.7
            ELSE 0.3
        END as quality_score
    FROM text_metrics
)

SELECT * FROM quality_scored
WHERE quality_score >= 0.7
'''

# models/marts/llm_training_data.sql
'''
{{ config(
    materialized='incremental',
    unique_key='content_hash',
    cluster_by=['detected_language', 'quality_score']
) }}

SELECT 
    content_hash,
    cleaned_text as text,
    detected_language as language,
    quality_score,
    text_length as token_count,
    source_type,
    extracted_at,
    CURRENT_TIMESTAMP() as processed_at
FROM {{ ref('int_text_quality') }}
WHERE quality_score >= 0.8

{% if is_incremental() %}
  AND extracted_at > (SELECT MAX(extracted_at) FROM {{ this }})
{% endif %}
'''

# Airflow DAG for ELT
elt_dag = DAG(
    'llm_data_elt',
    default_args=default_args,
    description='ELT pipeline for LLM training data',
    schedule_interval='@hourly',
    catchup=False
)

# Extract and Load raw data
extract_load_task = PythonOperator(
    task_id='extract_and_load_raw',
    python_callable=LLMDataELT().extract_and_load_raw,
    dag=elt_dag
)

# Transform with dbt
dbt_run_staging = BashOperator(
    task_id='dbt_run_staging',
    bash_command='dbt run --models staging --target prod',
    dag=elt_dag
)

dbt_test_staging = BashOperator(
    task_id='dbt_test_staging',
    bash_command='dbt test --models staging --target prod',
    dag=elt_dag
)

dbt_run_marts = BashOperator(
    task_id='dbt_run_marts',
    bash_command='dbt run --models marts --target prod',
    dag=elt_dag
)

# Data quality checks
quality_check_task = SnowflakeOperator(
    task_id='quality_checks',
    sql='''
    SELECT 
        COUNT(*) as total_records,
        AVG(quality_score) as avg_quality,
        COUNT(DISTINCT language) as language_count
    FROM marts.llm_training_data
    WHERE processed_at >= CURRENT_DATE()
    ''',
    dag=elt_dag
)

extract_load_task >> dbt_run_staging >> dbt_test_staging >> dbt_run_marts >> quality_check_task
"""
        
        st.code(elt_code, language='python')
        
        st.markdown("**ELT Benefits for LLM Data:**")
        elt_benefits = [
            "Faster data ingestion with minimal upfront processing",
            "Flexibility to reprocess data with new transformation logic",
            "Leverages warehouse compute power for heavy transformations",
            "Better support for exploratory data analysis and iteration"
        ]
        for benefit in elt_benefits:
            st.markdown(f"• {benefit}")
    
    with approach_tabs[2]:
        st.markdown("### 🔄 Hybrid Approach")
        
        st.markdown("""
        A hybrid approach combines the best of both ETL and ELT, using ETL for critical 
        preprocessing and ELT for flexible analytics and model preparation.
        """)
        
        hybrid_code = """
# Hybrid ETL/ELT Pipeline for LLM Data
class HybridLLMPipeline:
    def __init__(self):
        self.stream_processor = StreamProcessor()  # For real-time ETL
        self.warehouse = DataWarehouse()           # For ELT analytics
        
    def process_streaming_data(self):
        # ETL for real-time data processing
        kafka_consumer = KafkaConsumer('raw_text_stream')
        
        for message in kafka_consumer:
            # Immediate ETL processing for critical data
            raw_text = message.value
            
            # Quick quality filtering
            if self.passes_basic_quality(raw_text):
                # Basic cleaning and enrichment
                processed = {
                    'text': self.basic_clean(raw_text),
                    'length': len(raw_text),
                    'timestamp': datetime.now(),
                    'source': message.topic
                }
                
                # Load to both operational store and warehouse
                self.operational_store.insert(processed)
                self.warehouse.stream_insert('raw_processed', processed)
    
    def batch_analytics_processing(self):
        # ELT for complex analytics and model preparation
        dbt_commands = [
            # Advanced text analysis
            'run --models advanced_text_metrics',
            
            # Cross-document analysis
            'run --models document_similarity',
            
            # Topic modeling
            'run --models topic_extraction',
            
            # Final training data preparation
            'run --models training_data_marts'
        ]
        
        for cmd in dbt_commands:
            self.run_dbt_command(cmd)
    
    def create_training_datasets(self):
        # Use warehouse for flexible dataset creation
        training_queries = {
            'general': '''
                SELECT text, language, quality_score
                FROM marts.processed_text
                WHERE quality_score >= 0.8
                  AND language IN ('en', 'es', 'fr')
                SAMPLE(1000000 ROWS)
            ''',
            
            'domain_specific': '''
                SELECT text, domain, topic_distribution
                FROM marts.processed_text pt
                JOIN marts.document_topics dt ON pt.doc_id = dt.doc_id
                WHERE domain = 'technical'
                  AND topic_confidence >= 0.7
            ''',
            
            'high_quality': '''
                SELECT text, all_quality_metrics
                FROM marts.processed_text
                WHERE quality_score >= 0.95
                  AND coherence_score >= 0.8
                  AND factual_accuracy >= 0.9
            '''
        }
        
        datasets = {}
        for name, query in training_queries.items():
            result = self.warehouse.execute(query)
            datasets[name] = result.to_pandas()
        
        return datasets

# Configuration for hybrid approach
hybrid_config = {
    'etl_processing': {
        'real_time_filters': ['length', 'encoding', 'basic_quality'],
        'batch_size': 1000,
        'processing_timeout': 30
    },
    'elt_processing': {
        'warehouse_compute': 'auto_scaling',
        'dbt_threads': 8,
        'quality_thresholds': {
            'minimum_quality': 0.7,
            'premium_quality': 0.9
        }
    },
    'storage_strategy': {
        'hot_storage': 'last_30_days',
        'warm_storage': 'last_6_months', 
        'cold_storage': 'older_than_6_months'
    }
}
"""
        
        st.code(hybrid_code, language='python')

with concept_tabs[2]:
    st.subheader("🗄️ Storage Systems for LLM Data")
    
    st.markdown("""
    Choosing the right storage system is crucial for LLM data pipelines, 
    as they must handle massive datasets efficiently while supporting 
    various access patterns.
    """)
    
    storage_options = [
        {
            "system": "Data Lakes",
            "description": "Store raw data in its native format",
            "best_for": [
                "Unstructured text data",
                "Multi-format data ingestion",
                "Exploratory data analysis",
                "Cost-effective long-term storage"
            ],
            "technologies": ["Amazon S3", "Google Cloud Storage", "Azure Data Lake", "Apache Hadoop HDFS"],
            "pros": ["Schema flexibility", "Cost effective", "Scalable", "Multi-format support"],
            "cons": ["No ACID guarantees", "Potential data swamps", "Query performance", "Governance challenges"]
        },
        {
            "system": "Data Warehouses",
            "description": "Structured, optimized for analytical queries",
            "best_for": [
                "Structured training data",
                "Complex analytical queries",
                "Data transformations",
                "Reporting and dashboards"
            ],
            "technologies": ["Snowflake", "BigQuery", "Redshift", "Azure Synapse"],
            "pros": ["Optimized for analytics", "ACID compliance", "SQL interface", "Performance"],
            "cons": ["Higher costs", "Schema rigidity", "ETL complexity", "Vendor lock-in"]
        },
        {
            "system": "Lakehouse",
            "description": "Combines data lake flexibility with warehouse performance",
            "best_for": [
                "Mixed workload requirements",
                "ML and analytics together",
                "ACID transactions on data lakes",
                "Unified data platform"
            ],
            "technologies": ["Delta Lake", "Apache Iceberg", "Apache Hudi", "Databricks"],
            "pros": ["Best of both worlds", "ACID on object storage", "ML optimized", "Open standards"],
            "cons": ["Complexity", "Newer technology", "Tool compatibility", "Learning curve"]
        },
        {
            "system": "Vector Databases",
            "description": "Optimized for similarity search and embeddings",
            "best_for": [
                "Embedding storage",
                "Similarity search",
                "RAG applications",
                "Semantic clustering"
            ],
            "technologies": ["Pinecone", "Weaviate", "Qdrant", "Milvus"],
            "pros": ["Fast similarity search", "ML optimized", "Real-time queries", "Semantic search"],
            "cons": ["Limited scope", "Cost", "Specialized use cases", "New ecosystem"]
        }
    ]
    
    for option in storage_options:
        with st.expander(f"🗄️ {option['system']}"):
            st.markdown(option['description'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Best For:**")
                for use_case in option['best_for']:
                    st.markdown(f"• {use_case}")
            
            with col2:
                st.markdown("**Technologies:**")
                for tech in option['technologies']:
                    st.markdown(f"• {tech}")
            
            with col3:
                st.markdown("**Pros:**")
                for pro in option['pros']:
                    st.markdown(f"• {pro}")
                st.markdown("**Cons:**")
                for con in option['cons']:
                    st.markdown(f"• {con}")

with concept_tabs[3]:
    st.subheader("🔄 Real-time Processing")
    
    st.markdown("""
    Real-time data processing enables LLM systems to work with fresh data 
    and provide up-to-date responses, crucial for applications requiring 
    current information.
    """)
    
    realtime_architectures = [
        {
            "architecture": "Lambda Architecture",
            "description": "Combines batch and stream processing for comprehensive data handling",
            "components": {
                "Batch Layer": "Processes complete datasets periodically",
                "Speed Layer": "Handles real-time data streams",
                "Serving Layer": "Merges batch and real-time results"
            },
            "implementation": """
# Lambda Architecture for LLM Data Processing
import asyncio
from kafka import KafkaConsumer, KafkaProducer
import redis

class LambdaArchitecture:
    def __init__(self):
        self.batch_processor = BatchProcessor()
        self.stream_processor = StreamProcessor()
        self.serving_layer = ServingLayer()
        
    async def run_architecture(self):
        # Start batch processing (scheduled)
        batch_task = asyncio.create_task(self.run_batch_processing())
        
        # Start stream processing (continuous)
        stream_task = asyncio.create_task(self.run_stream_processing())
        
        # Start serving layer
        serving_task = asyncio.create_task(self.run_serving_layer())
        
        await asyncio.gather(batch_task, stream_task, serving_task)
    
    async def run_batch_processing(self):
        while True:
            # Process complete dataset every 6 hours
            try:
                self.batch_processor.process_full_dataset()
                await asyncio.sleep(6 * 3600)  # 6 hours
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    async def run_stream_processing(self):
        consumer = KafkaConsumer('text_stream')
        
        async for message in consumer:
            try:
                # Process real-time data
                result = self.stream_processor.process_message(message.value)
                
                # Store in speed layer
                self.serving_layer.update_realtime_view(result)
                
            except Exception as e:
                logger.error(f"Stream processing error: {e}")

class StreamProcessor:
    def __init__(self):
        self.text_classifier = TextClassifier()
        self.quality_assessor = QualityAssessor()
        
    def process_message(self, raw_text: str) -> dict:
        # Fast processing pipeline
        processed = {
            'text': self.quick_clean(raw_text),
            'timestamp': time.time(),
            'quality': self.quality_assessor.quick_score(raw_text),
            'language': self.detect_language_fast(raw_text),
            'topics': self.extract_topics_fast(raw_text)
        }
        
        return processed
    
    def quick_clean(self, text: str) -> str:
        # Lightweight cleaning for real-time processing
        return re.sub(r'\\s+', ' ', text.strip())

class ServingLayer:
    def __init__(self):
        self.redis_client = redis.Redis()
        self.batch_store = BatchDataStore()
        
    def update_realtime_view(self, data: dict):
        # Update real-time view in Redis
        key = f"realtime:{data['timestamp']}"
        self.redis_client.setex(key, 3600, json.dumps(data))
    
    def query_data(self, query_params: dict):
        # Merge batch and real-time data
        batch_results = self.batch_store.query(query_params)
        
        # Get recent real-time data
        realtime_keys = self.redis_client.keys("realtime:*")
        realtime_data = []
        
        for key in realtime_keys:
            data = json.loads(self.redis_client.get(key))
            if self.matches_query(data, query_params):
                realtime_data.append(data)
        
        # Merge and return
        return self.merge_results(batch_results, realtime_data)
"""
        },
        {
            "architecture": "Kappa Architecture",
            "description": "Stream-only processing architecture for simplified real-time analytics",
            "components": {
                "Stream Processing": "Single stream processing layer",
                "Reprocessing": "Replay streams for historical analysis",
                "Storage": "Stream-first storage design"
            },
            "implementation": """
# Kappa Architecture Implementation
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

class KappaArchitecture:
    def __init__(self):
        self.pipeline_options = PipelineOptions([
            '--streaming',
            '--runner=DataflowRunner',
            '--project=your-project',
            '--region=us-central1'
        ])
    
    def create_processing_pipeline(self):
        with beam.Pipeline(options=self.pipeline_options) as pipeline:
            
            # Read from Kafka stream
            text_stream = (pipeline
                | 'Read from Kafka' >> beam.io.ReadFromKafka(
                    consumer_config={'bootstrap.servers': 'localhost:9092'},
                    topics=['text-input']
                )
                | 'Extract text' >> beam.Map(lambda x: x[1])  # Get message value
            )
            
            # Process text data
            processed_stream = (text_stream
                | 'Clean text' >> beam.Map(self.clean_text)
                | 'Add timestamp' >> beam.Map(self.add_timestamp)
                | 'Assess quality' >> beam.Map(self.assess_quality)
                | 'Filter quality' >> beam.Filter(lambda x: x['quality'] > 0.7)
            )
            
            # Multiple outputs from single stream
            # Training data
            training_data = (processed_stream
                | 'Format for training' >> beam.Map(self.format_for_training)
                | 'Write training data' >> beam.io.WriteToBigQuery(
                    'project:dataset.training_data',
                    schema=self.get_training_schema()
                )
            )
            
            # Analytics data
            analytics_data = (processed_stream
                | 'Windowed aggregation' >> beam.WindowInto(
                    beam.transforms.window.FixedWindows(300)  # 5 minute windows
                )
                | 'Count by language' >> beam.CombinePerKey(beam.combiners.CountCombineFn())
                | 'Write analytics' >> beam.io.WriteToBigQuery(
                    'project:dataset.text_analytics',
                    schema=self.get_analytics_schema()
                )
            )
            
            # Real-time serving
            serving_data = (processed_stream
                | 'Format for serving' >> beam.Map(self.format_for_serving)
                | 'Write to Firestore' >> beam.ParDo(FirestoreWriteDoFn())
            )
    
    def clean_text(self, text: str) -> str:
        # Stateless text cleaning function
        return re.sub(r'[^\\w\\s]', '', text.lower().strip())
    
    def assess_quality(self, record: dict) -> dict:
        text = record['text']
        
        # Fast quality assessment
        word_count = len(text.split())
        char_diversity = len(set(text)) / len(text) if text else 0
        
        quality_score = min(1.0, (word_count / 100) * char_diversity)
        
        record['quality'] = quality_score
        return record

# Stream reprocessing for historical analysis
class StreamReprocessor:
    def __init__(self, kafka_cluster):
        self.kafka_cluster = kafka_cluster
        
    def reprocess_historical_data(self, start_timestamp, end_timestamp):
        # Reset consumer to historical position
        consumer = KafkaConsumer(
            'text-input',
            auto_offset_reset='earliest',
            bootstrap_servers=self.kafka_cluster
        )
        
        # Seek to specific timestamp
        partitions = consumer.assignment()
        timestamps = {partition: start_timestamp for partition in partitions}
        offset_dict = consumer.offsets_for_times(timestamps)
        
        for partition, offset_info in offset_dict.items():
            if offset_info:
                consumer.seek(partition, offset_info.offset)
        
        # Process historical data with current logic
        for message in consumer:
            if message.timestamp > end_timestamp:
                break
                
            # Apply current processing logic to historical data
            processed = self.current_processing_logic(message.value)
            self.store_reprocessed_result(processed)
"""
        }
    ]
    
    for arch in realtime_architectures:
        with st.expander(f"🔄 {arch['architecture']}"):
            st.markdown(arch['description'])
            
            st.markdown("**Key Components:**")
            for component, desc in arch['components'].items():
                st.markdown(f"• **{component}**: {desc}")
            
            st.markdown("**Implementation Example:**")
            st.code(arch['implementation'], language='python')

# Performance metrics visualization
st.header("📊 Data Pipeline Performance")

# Simulate performance data
pipeline_metrics = pd.DataFrame({
    'Stage': ['Ingestion', 'Processing', 'Storage', 'Serving'],
    'Throughput (GB/hr)': [1000, 800, 1200, 600],
    'Latency (ms)': [50, 200, 100, 10],
    'Success Rate (%)': [99.5, 97.8, 99.9, 99.7],
    'Cost per GB ($)': [0.01, 0.05, 0.02, 0.03]
})

col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(pipeline_metrics, x='Stage', y='Throughput (GB/hr)', 
                  title="Pipeline Throughput by Stage")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.line(pipeline_metrics, x='Stage', y='Latency (ms)', 
                   title="Pipeline Latency by Stage", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

# Best practices
st.header("💡 Data Engineering Best Practices")

best_practices = [
    "**Design for Scale**: Plan for 10x data growth from day one",
    "**Data Quality First**: Implement quality checks at every stage",
    "**Monitor Everything**: Track performance, quality, and costs continuously",
    "**Automate Recovery**: Build self-healing pipelines with automatic retry logic",
    "**Version Your Data**: Maintain data lineage and versioning for reproducibility",
    "**Security by Design**: Implement encryption, access controls, and audit logging",
    "**Cost Optimization**: Use appropriate storage tiers and compute resources",
    "**Documentation**: Maintain clear documentation of schemas and transformations",
    "**Testing**: Implement comprehensive testing for data pipelines",
    "**Incremental Processing**: Design for incremental updates to handle large datasets"
]

for practice in best_practices:
    st.markdown(f"• {practice}")

# Resources
st.header("📚 Learning Resources")

resources = [
    {
        "title": "Designing Data-Intensive Applications",
        "type": "Book",
        "description": "Comprehensive guide to modern data system architecture",
        "difficulty": "Advanced"
    },
    {
        "title": "Apache Spark: The Definitive Guide",
        "type": "Book", 
        "description": "In-depth coverage of distributed data processing",
        "difficulty": "Intermediate"
    },
    {
        "title": "Data Engineering on AWS",
        "type": "Course",
        "description": "Practical data engineering using AWS services",
        "difficulty": "Intermediate"
    },
    {
        "title": "Modern Data Stack",
        "type": "Documentation",
        "description": "Guide to building modern data pipelines",
        "difficulty": "Beginner"
    }
]

for resource in resources:
    with st.expander(f"📖 {resource['title']}"):
        st.markdown(f"**Type:** {resource['type']}")
        st.markdown(f"**Description:** {resource['description']}")
        st.markdown(f"**Difficulty:** {resource['difficulty']}")