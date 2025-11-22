# pgvector-tutorial
A practical reference guide for installing **pgvector**, generating vector datasets, loading data into PostgreSQL, building vector indexes, benchmarking performance, and tuning PostgreSQL for high-performance similarity search.

Full documentation is available in the **Wiki**.

---

## 📘 Documentation (Wiki)

The Wiki covers the following topics:

### 1. pgvector Installation
- Build from source  
- Enable the `vector` extension  

### 2. Data Generation
- Synthetic vector creation (Python)  
- Dataset export for benchmarking  

### 3. Data Loading
- Table schema design  
- Bulk loading using `COPY`  

### 4. Index Creation & Evaluation
- Index creation  
- Query examples  
- Recall/latency evaluation  
- Recommended index parameters  

### 5. PostgreSQL Performance Tuning
- `shared_buffers`  
- `maintenance_work_mem`  
- `effective_io_concurrency`  
- Parallel workers (`max_parallel_workers`, etc.)  
- Example optimized configs  

---

## 🚀 Purpose

This repository serves as a practical playbook for PostgreSQL vector search workloads using pgvector.  
It provides structured guidance for ANN search, benchmarking, and system tuning.

---

## 📄 Wiki

➡️ https://github.com/yourname/pgvector-playbook/wiki

---

## 📬 Contributions

Issues, suggestions, and pull requests are welcome.
