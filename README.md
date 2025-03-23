# LlamaIndex Learning Repository

This repository contains experiments and tests with LlamaIndex, a platform for building applications using Large Language Models (LLMs) connected to external data sources.

## 📋 About

This is a personal learning and experimentation project with LlamaIndex, exploring its indexing capabilities, information retrieval, and interface with LLMs.

## 🛠️ Technologies Used

- **Python 3.x**
- **LlamaIndex**: Framework for creating applications based on LLMs and custom data
- **Groq**: Integration with the Groq API for access to language models
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Data visualization and plotting
- **Gradio**: Creation of user interfaces for demonstrations
- **FPDF**: Generation of PDF documents
- **python-dotenv**: Management of environment variables

## 📦 Dependencies

```
pandas==2.2.3
llama-index==0.12.24
llama-index-llms-groq==0.3.1
llama-index-experimental==0.5.4
gradio==5.21.0
fpdf==1.7.2
matplotlib==3.10.1
python-dotenv==1.0.1
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

## 📖 What I've Learned So Far

### Lesson 1
* Configuring an LLM for data interaction using a Groq API key
* Exploring LlamaIndex capabilities to make natural language queries on structured data
* Using PandasQueryEngine to transform natural language queries into Python code operations
* Validating and interpreting PandasQueryEngine responses by comparing them with direct Pandas operations

### Lesson 2
* Creating data visualizations and plotting graphs with Matplotlib
* Enhancing output with more detailed natural language descriptions
* Processing and simplifying query outputs
* Controlling language settings for responses

### Lesson 3
* Building query pipelines to streamline data processing and analysis
* Creating sequential processing chains for complex queries
* Implementing multi-stage data transformation pipelines

### Lesson 4
* Developing simple user interfaces with Gradio
* Connecting LlamaIndex query engines to interactive web interfaces
* Creating intuitive input forms for natural language queries
* Displaying query results and visualizations in a user-friendly format
* Generating a PDF file with the history of user interactions

## 🔍 Explored Features

- Document indexing with LlamaIndex
- Creation of indexes for efficient querying
- LLM queries using custom data
- Data visualization with Matplotlib
- Natural language enhancement of query results
- Language control for outputs
- Query pipeline construction and optimization
- Interactive web interface development with Gradio
- Exporting results to PDF

## 🌟 Examples
