# LlamaIndex Learning Repository

This repository contains experiments and tests with LlamaIndex, a platform for building applications using Large Language Models (LLMs) connected to external data sources.

## 📋 About

This is a personal learning and experimentation project with LlamaIndex, exploring its indexing capabilities, information retrieval, and interface with LLMs.

## 🛠️ Technologies Used

- **Python 3.x**
- **LlamaIndex**: Framework for creating applications based on LLMs and custom data
- **Groq**: Integration with the Groq API for access to language models
- **Pandas**: Data manipulation and analysis
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
python-dotenv==1.0.1
```

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/josielborges/llama-index-study.git
cd llama-index-study
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

* Configuring an LLM for data interaction using a Groq API key
* Exploring LlamaIndex capabilities to make natural language queries on structured data
* Using PandasQueryEngine to transform natural language queries into Python code operations
* Validating and interpreting PandasQueryEngine responses by comparing them with direct Pandas operations

## 🔍 Explored Features

- Document indexing with LlamaIndex
- Creation of indexes for efficient querying
- LLM queries using custom data
- Interface generation with Gradio
- Exporting results to PDF

## 🌟 Examples

[TODO]