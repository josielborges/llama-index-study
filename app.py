import os
from datetime import datetime

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from fpdf import FPDF
from llama_index.core import PromptTemplate, Settings
from llama_index.core.query_pipeline import QueryPipeline, Link, InputComponent
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.llms.groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_LLAMA_MODEL = "llama3-70b-8192"

Settings.llm = Groq(model=GROQ_LLAMA_MODEL, api_key=GROQ_API_KEY)


def columns_description(df):
    return "\n".join([f"'{col}': {str(df[col].dtype)}" for col in df.columns])


def create_query_pipeline(df):
    # Instructions to guide the model in converting a natural language query into executable Python code using the Pandas library, in Portuguese, to retrieve responses in this language.
    instruction_str = (
        "1. Converta a consulta para código Python executável usando Pandas.\n"
        "2. A linha final do código deve ser uma expressão Python que possa ser chamada com a função `eval()`.\n"
        "3. O código deve representar uma solução para a consulta.\n"
        "4. IMPRIMA APENAS A EXPRESSÃO.\n"
        "5. Não coloque a expressão entre aspas.\n")

    # Prompt that will be sent to the model so that it generates the desired Pandas code.
    pandas_prompt_str = (
        "Você está trabalhando com um dataframe do pandas em Python chamado `df`.\n"
        "{colunas_detalhes}\n\n"
        "Este é o resultado de `print(df.head())`:\n"
        "{df_str}\n\n"
        "Siga estas instruções:\n"
        "{instruction_str}\n"
        "Consulta: {query_str}\n\n"
        "Expressão:"
    )

    # Prompt to guide the model in synthesizing a response based on the results obtained from the Pandas query.
    response_synthesis_prompt_str = (
        "Dada uma pergunta de entrada, atue como analista de dados e elabore uma resposta a partir dos resultados da consulta.\n"
        "Responda de forma natural, sem introduções como 'A resposta é:' ou algo semelhante.\n"
        "Consulta: {query_str}\n\n"
        "Instruções do Pandas (opcional):\n{pandas_instructions}\n\n"
        "Saída do Pandas: {pandas_output}\n\n"
        "Resposta:"
        "Ao final, exibir o código usado para gerar a resposta, no formato: O código utilizado foi {pandas_instructions}"
    )

    # Module to obtain Pandas instructions.
    pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
        instruction_str=instruction_str, colunas_detalhes=columns_description(df), df_str=df.head(5)
    )
    # Module to execute the Pandas instructions.
    pandas_output_parser = PandasInstructionParser(df)

    # Module to synthesize the response.
    response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

    # Model
    llm = Groq(model='llama3-70b-8192', api_key=GROQ_API_KEY)

    qp = QueryPipeline(
        modules={
            "input": InputComponent(),
            "pandas_prompt": pandas_prompt,
            "llm1": llm,
            "pandas_output_parser": pandas_output_parser,
            "response_synthesis_prompt": response_synthesis_prompt,
            "llm2": llm
        },
        verbose=True
    )

    qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])

    qp.add_links(
        [
            Link("input", "response_synthesis_prompt", dest_key="query_str"),
            Link("llm1", "response_synthesis_prompt", dest_key="pandas_instructions"),
            Link("pandas_output_parser", "response_synthesis_prompt", dest_key="pandas_output")
        ]
    )

    qp.add_link("response_synthesis_prompt", "llm2")

    return qp


def load_data(file_path, df_state):
    if file_path is None and df_state is None:
        return 'Please upload a CSV file.', pd.DataFrame(), df_state
    try:
        df = pd.read_csv(file_path)
        return 'File load successfully', df.head(), df
    except Exception as e:
        return f'Error on loading the file: {str(e)}', pd.DataFrame(), df_state


def proccess_query(query, df_state):
    if df_state is not None and query is not None:
        qp = create_query_pipeline(df_state)
        response = qp.run(query_str=query)
        return response.message.content

    return ''


def add_history(question, answer, history_state):
    if question and answer:
        history_state.append((question, answer))
        gr.Info("Added to PDF.", duration=2)
        return history_state


def generate_pdf(history_state):
    if not history_state:
        return "There is not information to generate a PFD file", None
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"data/question_answer_report_{timestamp}.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    for question, answer in history_state:
        pdf.set_font('Arial', "B", 14)
        pdf.multi_cell(0, 8, txt=question)
        pdf.ln(2)
        pdf.set_font('Arial', "", 12)
        pdf.multi_cell(0, 8, txt=answer)
        pdf.ln(6)

    pdf.output(output_file)
    return output_file


def clear_questions_answers():
    return "", ""


def reset_application():
    return None, "Application reseted. Upload a new CSV file.", pd.DataFrame(), "", None, [], ""


with gr.Blocks(theme=gr.themes.Glass()) as app:
    gr.Markdown("# Analyzind data with LlamaIndex and Pandas")

    gr.Markdown('This application allows you to upload a CSV file and analyze it using LlamaIndex and Pandas.')

    file_input = gr.File(file_count='single', type='filepath', label='Upload a CSV file')
    upload_status = gr.Textbox(label='Upload status')
    data_table = gr.DataFrame()

    gr.Markdown('''
                Questions examples:
                1. How many rows are in the file?
                2. What are the column types?
                3. What are the descriptive statistics of the numeric columns?
                ''')

    query_input = gr.Textbox(label='Type yout question about the dataset')
    submit_button = gr.Button('Send')
    response_output = gr.Textbox(label='Response')
    with gr.Row():
        add_pdf_button = gr.Button('Add history to PDF')
        clear_button = gr.Button("Clear questions and answers.")
        generate_pdf_button = gr.Button('Generate PDF')
    pdf_file = gr.File(label='PDF Download')
    reset_button = gr.Button('Analyze other dataset.')

    df_state = gr.State(value=None)
    history_state = gr.State(value=[])

    file_input.change(fn=load_data,
                      inputs=[file_input, df_state],
                      outputs=[upload_status, data_table, df_state])

    submit_button.click(fn=proccess_query,
                        inputs=[query_input, df_state],
                        outputs=[response_output])

    clear_button.click(fn=clear_questions_answers,
                       inputs=[],
                       outputs=[query_input, response_output])

    add_pdf_button.click(fn=add_history,
                         inputs=[query_input, response_output, history_state],
                         outputs=history_state)

    generate_pdf_button.click(fn=generate_pdf,
                              inputs=[history_state],
                              outputs=pdf_file)

    reset_button.click(fn=reset_application,
                       inputs=[],
                       outputs=[file_input, upload_status, data_table, response_output, pdf_file, history_state,
                                query_input])

if __name__ == "__main__":
    app.launch(debug=True)
