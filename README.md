﻿﻿
# 📰 IT Newsfeed platform

## 📝 Description

A scalable, real-time newsfeed platform that aggregates IT-related events from multiple public sources, filters them using semantic analysis, and provides dynamic ranking capabilities. 

The system uses machine learning-based content filtering to identify relevant IT operations news and offers both programmatic APIs and an interactive dashboard.


## ✨ Features

- **Multi-Source Aggregation:** Reddit (r/sysadmin, r/outages, r/cybersecurity), RSS feeds (Krebs Security, AWS/Azure status)
- **Semantic Content Filtering:** ML-based relevance detection using sentence embeddings and cosine similarity.
- **Dynamic Ranking:** Ranking engine that combines events importance and recency, enabling real-time adjustments to prioritize either high-value or fresh events depending on user preference.
- **Vector Storage:** FAISS-based storage with efficient similarity search
- **Interactive Dashboard:** Streamlit web interface with real-time ranking preference controls
- **REST API:** Mock Newsfeed API compliance for automated testing

## 🛠️ Tools

 **Core Framework**
- **Python 3.12** - Primary development language
- **Poetry** - Dependency management and virtual environment handling

 **Data Sources & APIs**
- **PRAW** - Python Reddit API Wrapper for subreddit aggregation
- **Feedparser** - RSS feed parsing for news sources
- **Requests** - HTTP client for API interactions
- **FastAPI** - REST API framework for Mock Newsfeed compliance

 **Machine Learning & NLP**
- **Sentence Transformers** - all-MiniLM-L6-v2 model for semantic embeddings 

 **Storage & Search**
- **FAISS Index** - Vector similarity search and storage
- **Pickle** - Event serialization for hashing 

 **User Interface**
- **Streamlit** - Interactive web dashboard with real-time controls
- **Plotly** - Data visualization and charts
- **HTML/CSS** - Custom styling and responsive design

 **Development & Deployment**
- **Docker** - Containerization for consistent deployment
- **Docker Compose** - for multi-container set up
- **Git** - Version control and collaboration

## 🎯 Key Design Decisions

### 1. Fetching method:

Opted for a **plugin-based architecture** with asynchronous processing to efficiently aggregate data from multiple IT sources.

**Plugin System:**
Defined two core plugins:
- **RedditSourcePlugin:** Fetches posts from Reddit's JSON API, extracting titles, content (selftext/URLs), timestamps, and post IDs
- **RSSSourcePlugin:** Parses RSS/XML feeds using feedparser, extracting article titles, summaries, publication dates, and links

**Concurrent Processing:**
- Uses `asyncio.gather()` to fetch from all 7 sources simultaneously
- Shared HTTP session for optimal performance (2-5 second total response time)
- Individual source failures don't break the pipeline

**Extensible Design:**
- Plugin registry system allows easy addition of new source types
- Clean separation between source configuration and implementation
- Standardized `Event` objects across all data sources
- New plugins only need to implement `fetch_events()` method from `BaseSourcePlugin`

**Current Data Sources:**
- **Reddit Communities (3):** r/sysadmin, r/outages, r/cybersecurity
- **RSS Feeds (4):** 
  - Security: Ars Technica Security, Krebs on Security
  - Cloud Status: AWS Status, Azure Status

### 2. Filtering method:

Opted for semantic embeddings instead of traditional keyword matching, for deeper context understanding, improved relevance, and fewer false positives.

Implemented a multi-stage filtering approach to ensure content quality and eliminate duplicates:

1. **Deduplication using Content Hashing:**

- Generate SHA-256 hashes from concatenated event title and body text
- Compare hashes against existing stored events to filter out duplicates before storage
- Only new events (not found in storage) proceed to the next step

2. **Semantic Relevance Filtering:**

- Events passing deduplication are evaluated using all-MiniLM-L6-v2 sentence embeddings
- Cosine similarity comparison against 39 curated IT reference phrases
- Maximum similarity score across all reference phrases is used as the event's semantic relevance score
- Configurable threshold (default: 0.5) to control filtering strictness
- Only semantically relevant events proceed to ranking stage

### 3. Ranking method:
Opted for a two-stage approach, balancing importance and recency:
1. **Importance score calculation:**

**formula:** `Importance_score = Semantic_score (60%) + Urgency_score (30%) + Source_score (10%)`
  Where      :
- *Semantic relevance (60%):*  ML-Based event relevance using **all-MiniLM-L6-v2** sentence embeddings and cosine similarity against 39 IT reference phrases such as "security breach", "server outage" ... etc.
- *Urgency (30%):* Keyword-based criticality detection scanning for terms like "critical", "emergency", ... etc.
- *Source credibility (10%):* Weighted trust scores based on source reliability. Helps prioritize official channels over community discussions.
2. **Recency score:**
- Time-based scoring using exponential decay with a 12-hour half-life. 
- Formula: `Recency_score = e^(-age_in_hours/12)`, where newer events score closer to 1.0 and older events approach 0.
3. **Final ranking score calculation:**

- formula : `Final Score = Importance(70%) + Recency(30%)`

### 4. Modular Architecture: 

Separated core concerns (aggregation, embedding, filtering, ranking, storage, orchestration, UI, and scheduling) into independent components for improved testability, maintainability, and extensibility.

### 5. Vector Storage
Opted for FAISS over traditional databases to provide native similarity search, efficient handling of sentence embeddings, and scalable vector operations essential for semantic filtering.

##  📊 Architecture Diagram

<img width="921" height="487" alt="newsfeed_platfrom_architecture" src="https://github.com/user-attachments/assets/ee1508d7-b0f1-4ceb-bf98-a110a05fcab5" />



## 🚀 Installation & Setup

### 📌 Prerequisites

- Docker
- Docker Compose

### 🛠️ Installation

1. Clone the Git repository:
   ``` bash
   git clone https://github.com/zakariajaadi/newsfeed_platform.git
   cd newsfeed_platform
   ```
2. Start docker compose services:
   ```bash
   docker-compose up
   ```
   > Note : Services run with default settings: 0.5 filtering threshold and 5-minute scheduler interval for ingestion. These values can be customized in the configuration file under the config directory if needed.
3. Check services status:  Check if all services are running:
    ```bash
   docker ps
   ```
You should see three services:
- **Dashboard:** Streamlit Web interface for event visualization
- **Api:** REST API for event ingestion and retrieval
- **Scheduler:** Background service for periodic news collection
4. Access the services

- **Dashboard:** http://localhost:8501
- **API doc:** : http://localhost:8000/
- **Scheduler**: Running in the background (automatically fetches and processes news every 5 minutes, keeping only events with relevance scores ≥ 0.5)

### 📡 API Endpoints (Mock Newsfeed)

**POST /ingest:**  
Accepts raw events for processing

**GET /retrieve:**  
Returns filtered and ranked events in default order

**GET /health:**  
Api healthcheck

**GET /:**  
Api doc

## 🔮 Future Perspectives

### Configuration Management

###  Re-ranking 

Current State: Consumers can only adjust a single combined importance × recency weight, limiting fine-grained control over how results are prioritized.

Proposed Enhancement: Multi-factor weight control with granular customization:

- Separate controls for semantic relevance (currently 60% of importance)
- Independent urgency factor adjustment (currently 30% of importance)  
- Source credibility weighting (currently 10% of importance)
- Recency decay rate configuration (currently fixed 12-hour exponential)

##  💡 Answers to bonus questions

### Answer to Question 1 (Scalability considerations):

I would opt for :  

**Distributed Processing:**

- Message queues (e.g., Kafka, RabbitMQ, or Pulsar) to handle high-frequency updates from hundreds of channels. This ensures that updates are buffered and can be processed asynchronously, preventing the system from being overwhelmed

**Storage Scaling:**
- Move from a single in-memory index to a distributed vector store like (Pinecone, Weaviate) to support high volume and concurrency.
- Horizontal partitioning by source/time
- Caching layers for hot data

**Orchestration & Workflow Management:**
- Replace APScheduler with an enterprise-grade orchestration platforms like **Prefect** or **Airflow** for advanced scheduling, dependency management, retry logic, and distributed task execution.
- Deploy the system on Kubernetes to handle containerized services, automatic scaling, self-healing, and rolling updates, ensuring resilience and elasticity under heavy load

### Answer for Question 2 (False Alarms/Fake News Detection):

**ML-Based Approach:**
- Curate a labeled dataset of legitimate vs fake IT news over time
- Fine-tune a BERT classifier for IT-specific fake news detection
- Integrate the classifier as an additional filtering stage in the pipeline

> **Prior Experience:**
Previously fine-tuned a BERT model for phishing URL detection, achieving high accuracy with minimal training data. The power of fine-tuning is that we don't need massive datasets since pre-trained models already understand language patterns, requiring only few domain-specific examples to adapt.

**Complementary Techniques:**
- Source reputation scoring based on historical accuracy
- Cross-source validation (multiple independent sources reporting same event)
- Optional feedback loop: continuously update the dataset with confirmed false positives to improve future detection










