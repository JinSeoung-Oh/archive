import time
import torch
import requests
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = 'sk-...'

import logging
import sys

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  

from llama_index import (
    KnowledgeGraphIndex,
    LLMPredictor,
    ServiceContext,
    SimpleDirectoryReader,
)
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore
#from llama_index.llms import OpenAI
# pip install llama-cpp-python
#from llama_index.llms import LlamaCPP

from IPython.display import Markdown, display
#from transformers import AutoModel, AutoModelForCausalLM
#from llama_index.llms import HuggingFaceLLM, HuggingFaceInferenceAPI

#model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

#llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
#    model_url=model_url,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
#    model_path=None,
#    temperature=0.1,
#    max_new_tokens=256,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
#    context_window=3900,
    # kwargs to pass to __call__()
#    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
#    model_kwargs={"n_gpu_layers": 1},
#    verbose=True,
#)

#llm = HuggingFaceLLM(
    #context_window=4096,
    #max_new_tokens=256,
    #generate_kwargs={"temperature": 0.7, "do_sample": False},
    #system_prompt=system_prompt,
    #query_wrapper_prompt=query_wrapper_prompt,
#    tokenizer_name="Kaeri-Jenti/LDCC-with-openorca",
#    model_name="Kaeri-Jenti/LDCC-with-openorca",
#    device_map="auto",
    #stopping_ids=[50278, 50279, 50277, 1, 0],
    #tokenizer_kwargs={"max_length": 4096},
    # uncomment this if using CUDA to reduce memory usage
#     model_kwargs={"torch_dtype": torch.float16}
#)
from llama_index.llms import OpenAI
llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")

service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=512)

# To set up NebulaGraph locally, begin by establishing a connection using its default credentials
# Install go --> sudo snap install go --classic
# Install nebula-console from https://github.com/vesoft-inc/nebula-console#from-source-code
# ./nebula-console -addr 127.0.0.1 -port 9669 -u root -p nebula
# CREATE SPACE llamaindex(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);
# ADD HOSTS 127.0.0.1:9779;
# :sleep 10;
# USE llamaindex;
# CREATE TAG entity(name string);
# CREATE EDGE relationship(relationship string);
# CREATE TAG INDEX entity_index ON entity(name(256));

os.environ["NEBULA_USER"] = "root"
os.environ["NEBULA_PASSWORD"] = "nebula"
os.environ[
    "NEBULA_ADDRESS"
] = "127.0.0.1:9669"

space_name = "llamaindex"
edge_types, rel_prop_names = ["relationship"], [
    "relationship"
] 
tags = ["entity"]

graph_store = NebulaGraphStore(
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

## Next, the data is loaded into the system using LlamaIndex’s SimpleDirectoryReader, 
## which reads documents from a specified directory. A Knowledge Graph index, kg_index, is then constructed using these documents
from llama_index import SimpleDirectoryReader

print(os.getcwd())
reader = SimpleDirectoryReader(input_dir=os.getcwd() + "/kg/data/knowledge graphs/koean_llamaindex/")
documents = reader.load_data()
#print(type(documents)) # <-- list, check time sleep.. try except...?
#print(documents)

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=5,
    service_context=service_context,
    space_name=space_name,
    edge_types=edge_types,
    rel_prop_names=rel_prop_names,
    tags=tags,
    include_embeddings=True,
)

#save kg
storage_context.persist(persist_dir=os.getcwd() + "/kg/graph/test")

with open(os.getcwd()+'/b.txt') as f:
    lines = [line.rstrip() for line in f]

re_line=[]
for line in lines:
    if line == '':
        continue
    else:
        re_line.append(line)

sen = []
for data in re_line:
    print('non-error', data)
    tri = extract_korean_triplet(data)
    if tri == []:
        time.sleep(30)
        re_tri = extract_korean_triplet(data)
        sen.append(re_tri)
    else:
        sen.append(tri)

---------------------------------------------------------------------------------------
import re
pattern = re.compile(r'"node_1": "(.*?)",\s*"node_2": "(.*?)",\s*"edge": "(.*?)"')
triplet=[]

for sentence in sen:
    tri = sentence[1:-1]
    #print(tri)
    matches = pattern.findall(tri)
    for match in matches:
        node_1, node_2, edge = match
        if node_1 == '=':
            new_node = edge
            edge = node_1
            new_node = convert_math_symbol_to_str(new_node)
            edge = convert_math_symbol_to_str(edge)
            node_2 = convert_math_symbol_to_str(node_2)
            triplet.append([new_node, edge, node_2])
        elif node_2 == '=':
            new_node = edge
            edge = node_2
            node_1 = convert_math_symbol_to_str(node_1)
            edge = convert_math_symbol_to_str(edge)
            new_node = convert_math_symbol_to_str(new_node)
            triplet.append([node_1, edge, new_node])
        else:
            node_1 = convert_math_symbol_to_str(node_1)
            edge = convert_math_symbol_to_str(edge)
            node_2 = convert_math_symbol_to_str(node_2)
            triplet.append([node_1, edge, node_2])

for tri in triplet:
    print('...')
    print(tri)
    if len(tri) == 3:
        graph_store.upsert_triplet(tri[0], tri[1], tri[2])
        kg_index.upsert_triplet((tri[0], tri[1], tri[2]))
    else:
        new_triplet=[]
        for i in range(0, len(tri), 3):
            new_triplet.append(tri[i:i + 3])
        for new in new_triplet:
            if len(new) ==3:
                graph_store.upsert_triplet(new[0], new[1], new[2])
                kg_index.upsert_triplet((new[0], new[1], new[2]))
            else:
                continue

----------------------------------------------------------------------------------------
from llama_index.llms import OpenAI
os.environ["OPENAI_API_KEY"] = 'sk-...'
# pip install llama-cpp-python
#from llama_index.llms import LlamaCPP

from IPython.display import Markdown, display

llm = OpenAI(temperature=0, model="gpt-3.5-turbo-0613")
#model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=512)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import KnowledgeGraphRAGRetriever
import openai

graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    service_context=service_context,
    llm=llm,
    verbose=True,
)

query_engine = RetrieverQueryEngine.from_args(
    graph_rag_retriever, service_context=service_context
)

import logging
logging.basicConfig(level=logging.DEBUG)

core_key = '운행정보 확인장치 장착 이전 기간의 주행거리는 어떻게 계산 되나요?'

import time
start = time.time()

print(graph_store.get('운행정보 확인장치'))

response = query_engine.query(
    f"Please answer about '{core_key}' more detail in korean. Do not sumary. If existed graph node conatin keyword extracted from given query, please contain this for your answer")
#display(Markdown(f"<b>{response}</b>"))

kg_response = response.response
print(kg_response)
