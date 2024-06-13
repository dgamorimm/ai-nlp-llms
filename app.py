import os
import openai
import langchain
import chromadb
from flask import Flask, request, render_template
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms.openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from decouple import config
import warnings
warnings.filterwarnings('ignore')

# Inicializa o Flask
app = Flask(__name__)

# Função para ler os arquivos em PDF
def read_pdf(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

# Executa a função para ler os arquivos em pdf
doc = read_pdf('documents/')

# Cria o gerador de embeddings (Coneverter o texto em representações numéricas = Tokenizador)
embeddings = OpenAIEmbeddings(api_key = config('OPENAI_API_KEY'))

# Aplica o gerador de embeddings aos documentos e armazena no banco de dados vetorial ( quando inicializa o banco ele vai ficar em memória(demanda muito recurso computacional))
index_name = 'dgamorimm-index'
index = Chroma.from_documents(documents = doc, embedding = embeddings, collection_name= index_name)

# Define a função de busca por similaridade
def search_similar(query, k=2):
    matching_results = index.similarity_search(query, k)
    return matching_results

# Cria a instancia do LLM
llm = OpenAI(api_key = config('OPENAI_API_KEY'), temperature= 0.3)

# Cria a chai para o sistema de QA (Perguntas e Respostas)
chain = load_qa_chain(llm, chain_type="stuff")

# Define a função para obter a respota
def get_answer(query):
    results = search_similar(query)
    return chain.run(input_documents = results, question = query)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/llms', methods=['POST'])
def llms():
    query = request.form['query']
    response = get_answer(query)
    return render_template('index.html', query=query, response=response)

if __name__ == '__main__':
    app.run(debug=False)