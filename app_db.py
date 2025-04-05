# Standard library imports
import os
from typing import List

# Third-party imports
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData

# LlamaIndex imports
from llama_index.core import Settings, VectorStoreIndex, SQLDatabase, PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline, InputComponent, FnComponent
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema, ObjectIndex
from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.core.llms import ChatResponse
from llama_index.core.retrievers import SQLRetriever
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables
load_dotenv()

# Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
LLAMA_MODEL = 'llama-3.3-70b-versatile'
HF_EMBEDDINGS_MODEL = 'BAAI/bge-m3'
DB_PATH = 'data/ecommerce.db'

# Initialize LlamaIndex settings
def initialize_settings():
    Settings.llm = Groq(model=LLAMA_MODEL, api_key=GROQ_API_KEY)
    Settings.embed_model = HuggingFaceEmbedding(model_name=HF_EMBEDDINGS_MODEL)

# Database setup
def setup_database():
    engine = create_engine(f'sqlite:///{DB_PATH}')
    metadata_obj = MetaData()
    metadata_obj.reflect(bind=engine)
    
    return engine, metadata_obj

# Generate table descriptions
def generate_table_description(table_name, sample_df, llm):
    prompt = f'''
    Analyze the sample of the table '{table_name}' below and give a short and concise description of the table contents.
    Inform up to 5 unique values of each column.

    Table sample:
    {sample_df}

    Description:
    '''
    response = llm.complete(prompt)
    return response.text

def get_table_descriptions(engine, metadata_obj, llm):
    tables_names = metadata_obj.tables.keys()
    tables_dict = {}

    for table_name in tables_names:
        df = pd.read_sql_table(table_name, engine)
        sample_df = df.head(5).to_string()

        description = generate_table_description(table_name, sample_df, llm)
        tables_dict[table_name] = description

        print(f'Table: {table_name}\nDescription:\n{description}')
        print('-'*15)
    
    return tables_dict, tables_names

# Create query engine
def create_query_engine(engine, tables_dict, tables_names):
    sql_database = SQLDatabase(engine=engine)
    table_node_map = SQLTableNodeMapping(sql_database=sql_database)
    
    table_schema_obj = [
        SQLTableSchema(table_name=table_name, context_str=tables_dict[table_name])
        for table_name in tables_names
    ]

    obj_index = ObjectIndex.from_objects(table_schema_obj, table_node_map, VectorStoreIndex)
    obj_retriever = obj_index.as_retriever(similarity_top_k=1)
    
    return sql_database, obj_retriever

# Create query pipeline components
def table_description(tables_schema: List[SQLTableSchema], sql_database):
    descriptions = []
    for table_schema in tables_schema:
        table_info = sql_database.get_single_table_info(table_schema.table_name)
        table_info += (' The table\'s description is: ' + table_schema.context_str) 
        descriptions.append(table_info)
    return '\n\n'.join(descriptions)

def sql_response(response: ChatResponse) -> str:
    response_content = response.message.content
    sql_query = response_content.split("SQLQuery: ", 1)[-1].split("SQLResult: ", 1)[0]
    return sql_query.strip().strip('```').strip()

# Setup query pipeline
def setup_query_pipeline(sql_database, obj_retriever, llm, engine):
    # SQL query generation prompt
    prompt_1_str = '''
    Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. You can order the results by a relevant column to return the most interesting examples in the database.

    Never query for all the columns from a specific table, only ask for a few relevant columns given the question.

    Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Pay attention to which column is in which table. Also, qualify column names with the table name when needed. You are required to use the following format, each taking one line:

    Question: Question here
    SQLQuery: SQL Query to run
    SQLResult: Result of the SQLQuery
    Answer: Final answer here

    Only use tables listed below.
    {schema}

    Question: {query_str}
    SQLQuery: 
    '''
    prompt_1 = PromptTemplate(prompt_1_str, dialect=engine.dialect.name)

    # Response formatting prompt
    prompt_2_str = '''
    You are a Database query assistant.
    Given the follow question, the SQL query and the SQL result, respond the user question in a nice way and be objective.
    Avoid to start the conversation with apresentations or salutations, like "Hello".

    Question: {user_question}
    SQL query: {query}
    SQL result: {result}
    Response:
    '''
    prompt_2 = PromptTemplate(prompt_2_str)

    # Create partial function for table_description
    table_context_fn = lambda tables_schema: table_description(tables_schema, sql_database)
    table_context = FnComponent(fn=table_context_fn)
    
    # SQL response extraction
    sql_query_component = FnComponent(fn=sql_response)
    
    # SQL result retriever
    sql_result = SQLRetriever(sql_database=sql_database)

    # Build the query pipeline
    qp = QueryPipeline(
        modules={
            "input": InputComponent(),
            "table_access": obj_retriever,
            "table_context": table_context,
            "prompt_1": prompt_1,
            "llm_1": llm,
            "sql_query": sql_query_component,
            "sql_result": sql_result,
            "prompt_2": prompt_2,
            "llm_2": llm
        },
        verbose=False
    )

    qp.add_chain(['input', 'table_access', 'table_context'])
    qp.add_link('input', 'prompt_1', dest_key='query_str')
    qp.add_link('table_context', 'prompt_1', dest_key='schema')

    qp.add_chain(['prompt_1', 'llm_1', 'sql_query', 'sql_result'])
    qp.add_link('input', 'prompt_2', dest_key='user_question')
    qp.add_link('sql_query', 'prompt_2', dest_key='query')
    qp.add_link('sql_result', 'prompt_2', dest_key='result')
    qp.add_link('prompt_2', 'llm_2')
    
    return qp

# Function to process user queries
def process_query(user_message: str, query_pipeline):
    response = query_pipeline.run(query=user_message)
    return str(response.message.content)

# Add message to history
def add_to_history(user_message: str, history, query_pipeline):
    assistant_message = process_query(user_message, query_pipeline)
    history = history or []
    history.append([user_message, assistant_message])
    return assistant_message, history

# Create Gradio interface
def create_gradio_interface(query_pipeline):
    with gr.Blocks() as demo:
        gr.Markdown("## Database Query Assistant")
        gr.Markdown("""Ask anything about the database. 
                    You can query information about Clientes (customers), Fornecedores (suppliers), and Funcionarios (employees). 
                    Try questions like 'Quais os estados mais frequentes na tabela de Clientes?' or 'Mostre os fornecedores com seus contatos'.""")
        
        chatbot = gr.Chatbot(label="Chat with the database")
        msg = gr.Textbox(label="Enter a message and press enter", placeholder="Enter a message")
        clear = gr.ClearButton([msg, chatbot])
        
        def update_history(user_message, history):
            _, updated_history = add_to_history(user_message, history, query_pipeline)
            return "", updated_history

        msg.submit(update_history, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False)
        clear.click(lambda: None, inputs=None, outputs=[chatbot], queue=False)

    return demo

# Main function
def main():
    # Initialize
    initialize_settings()
    engine, metadata_obj = setup_database()
    
    # Get LLM instance
    llm = Groq(model=LLAMA_MODEL, api_key=GROQ_API_KEY)
    
    # Generate table descriptions
    tables_dict, tables_names = get_table_descriptions(engine, metadata_obj, llm)
    
    # Create query engine components
    sql_database, obj_retriever = create_query_engine(engine, tables_dict, tables_names)
    
    # Setup query pipeline
    query_pipeline = setup_query_pipeline(sql_database, obj_retriever, llm, engine)
    
    # Create and launch Gradio interface
    demo = create_gradio_interface(query_pipeline)
    demo.queue()
    demo.launch(debug=True)

if __name__ == "__main__":
    main()