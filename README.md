# LlamaIndex Learning Repository

This repository contains experiments and tests with LlamaIndex, exploring two main learning paths: data analysis with Pandas and database integration using SQLAlchemy.

## 📋 About

A personal learning project diving deep into LlamaIndex capabilities, covering structured data processing, natural language querying, and database interactions.

## 🛠️ Technologies Used

- **Python 3.x**
- **LlamaIndex**: Framework for creating applications based on LLMs and custom data
- **Groq**: Integration with the Groq API for access to language models
- **Pandas**: Data manipulation and analysis
- **SQLAlchemy**: SQL database connectivity and ORM
- **Matplotlib**: Data visualization and plotting
- **Gradio**: Creation of user interfaces for demonstrations
- **FPDF**: Generation of PDF documents
- **python-dotenv**: Management of environment variables

## 📦 Dependencies

```
llama-index==0.12.24
llama-index-llms-groq==0.3.1
llama-index-experimental==0.5.4
llama-index-embeddings-huggingface==0.5.2
gradio==5.21.0
fpdf==1.7.2
python-dotenv==1.0.1
pandas==2.2.3
matplotlib==3.10.1
SQLAlchemy==2.0.39
```

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/repository-name.git
cd repository-name
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - Windows:
   ```bash
   venv\Scripts\activate
   ```
   - Linux/Mac:
   ```bash
   source venv/bin/activate
   ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Configure environment variables:
   - Create a `.env` file in the project root
   - Add your API keys (example):
   ```
   GROQ_API_KEY=your-groq-api-key
   ```

## 📖 Learning Paths

### Pandas Data Analysis Journey

#### Lesson 1
* Configuring an LLM for data interaction using a Groq API key
* Exploring LlamaIndex capabilities to make natural language queries on structured data
* Using PandasQueryEngine to transform natural language queries into Python code operations
* Validating and interpreting PandasQueryEngine responses by comparing them with direct Pandas operations

#### Lesson 2
* Creating data visualizations and plotting graphs with Matplotlib
* Enhancing output with more detailed natural language descriptions
* Processing and simplifying query outputs
* Controlling language settings for responses

#### Lesson 3
* Building query pipelines to streamline data processing and analysis
* Creating sequential processing chains for complex queries
* Implementing multi-stage data transformation pipelines

#### Lesson 4
* Developing simple user interfaces with Gradio
* Connecting LlamaIndex query engines to interactive web interfaces
* Creating intuitive input forms for natural language queries
* Displaying query results and visualizations in a user-friendly format

### Database Integration Journey

#### Lesson 1
* Connecting databases using SQLAlchemy
* Using LlamaIndex to connect data to natural language models
* Configuring LlamaIndex with specific models and embeddings
* Performing natural language queries with a LlamaIndex engine

#### Lesson 2
* Providing database table context to improve query accuracy
* Using language models to automatically generate descriptions for tables and columns
* Updating schema lists in LlamaIndex with detailed descriptions for more precise queries

## 🔍 Explored Features

- Document and database indexing with LlamaIndex
- Creation of indexes for efficient querying
- LLM queries using custom data
- SQL database connectivity
- Schema enrichment with automated descriptions
- Data visualization with Matplotlib
- Natural language enhancement of query results
- Language control for outputs
- Query pipeline construction and optimization
- Interactive web interface development with Gradio
- Exporting results to PDF