!pip3 install --no-deps torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
!pip3 install openai
!pip3 install -q transformers==4.33.0 
!pip3 install -q accelerate==0.22.0 
!pip3 install -q einops==0.6.1 
!pip3 install -q langchain==0.0.300 
!pip3 install -q xformers==0.0.21
!pip3 install -q bitsandbytes==0.41.1 
!pip3 install -q sentence_transformers==2.2.2
!pip3 install arxiv
!pip3 install -q ipykernel jupyter
!pip3 install -q --upgrade huggingface_hub

!pip3 install unstructured
!pip3 install "unstructured[pdf]"
!pip3 install pytesseract
!pip install faiss-cpu
!pip3 install WhisperSpeech

## If you use linux. If the other, then just search how to install poppler-utils, tesseract-ocr on your OS
apt-get install -y poppler-utils
apt-get install -y tesseract-ocr

import os
import sys
import arxiv
from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader,DirectoryLoader,WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from pathlib import Path
from openai import OpenAI
from IPython.display import Audio, display
from whisperspeech.pipeline import Pipeline

import os
import arxiv
import time
from urllib.error import HTTPError
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from time import time
from IPython.display import display, Markdown

# Directory setup for downloading papers
dirpath = "arxiv_papers"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

# Search and download arxiv papers
search = arxiv.Search(
    query="Depth estimation",
    max_results=10,
    sort_by=arxiv.SortCriterion.LastUpdatedDate,
    sort_order=arxiv.SortOrder.Descending
)

for result in search.results():
    while True:
        try:
            result.download_pdf(dirpath=dirpath)
            print(f"-> Paper id {result.get_short_id()} with title '{result.title}' is downloaded.")
            break
        except FileNotFoundError:
            print("File not found")
            break
        except HTTPError:
            print("Forbidden")
            break
        except ConnectionResetError as e:
            print("Connection reset by peer")
            time.sleep(5)

# Load papers from directory
loader = DirectoryLoader(dirpath, glob="./*.pdf", loader_cls=PyPDFLoader)
papers = loader.load()
print("Total number of pages loaded:", len(papers))

# Merge all papers into a single text block for chunking
full_text = ''
for paper in papers:
    full_text += paper.page_content

full_text = " ".join(l for l in full_text.splitlines() if l)
print(len(full_text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

paper_chunks = text_splitter.create_documents([full_text])

# Initialize model and tokenizer
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

query_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    #max_length=1024,
    device_map="auto",
)

llm = HuggingFacePipeline(pipeline=query_pipeline)

# Initialize embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
faiss_vector_store = FAISS.from_documents(paper_chunks, embeddings)

retriever = faiss_vector_store.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

def colorize_text(text):
    for word, color in zip(["Reasoning", "Question", "Answer", "Total time"], ["blue", "red", "green", "magenta"]):
        text = text.replace(f"{word}:", f"\n\n**<font color='{color}'>{word}:</font>**")
    return text

def test_rag(qa, query):
    time_start = time()
    response = qa.run(query)
    time_end = time()
    total_time = f"{round(time_end-time_start, 3)} sec."

    full_response = f"Question: {query}\nAnswer: {response}\nTotal time: {total_time}"
    display(Markdown(colorize_text(full_response)))
    return response

# Example query
query = "Please, expain about latest depth estimation. Expain detail as you can possible"
aud = test_rag(qa, query)




