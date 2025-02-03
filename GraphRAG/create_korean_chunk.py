import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from konlpy.tag import Kkma

file = open('./text_text.txt', 'r')
string = file.readlines()

new_string = ''

for info in string:
    print(info)
    new_string += info + '\n\n'

test_text = new_string

para = test_text.split('\n\n')
para = para[:-1]
#print(para)

#print(para_)
test_para = [item.replace("\n", "") for item in para]
para_ = [item for item in test_para if item != '']
para_ = [item for item in test_para if item != ' ']
#print(para_)
    

sen = []
for i, para in enumerate(para_):
    #print(para)
    sentences = kkma.sentences(para)
    for sentence in sentences:
        sen.append(sentence)

# BERT 모델 및 토크나이저 로드
model_name = "kykim/bert-kor-base"  # 사용할 BERT 모델의 이름
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 요약할 텍스트 입력
summarize = []
for text in sen:
    tokens = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")

    # BERT 모델로 요약 생성
    summary_id = model.generate(tokens, max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    # 생성된 요약 텍스트 디코딩
    summary_text = tokenizer.decode(summary_id[0], skip_special_tokens=True)
    summarize.append(summary_text)

with open(os.getcwd()+'/a.txt') as f:
    lines = [line.rstrip() for line in f]

re_line = []
for line in lines:
    if line == '<그림>' or line =='<표>':
        continue
    else:
        re_line.append(line)

test_list =[]
max_iter = len(re_line)

for i in range(0, max_iter-1):
    ele_1 = len(re_line[i])
    ele_text = re_line[i]
    ele_2 = len(re_line[i+1])
    ele_2_text = re_line[i+1]
    if ele_1==0 and ele_2==0:
        continue
    elif ele_1 ==0 and ele_2!=0:
        test_list.append(ele_2_text)
    elif ele_1 !=0 and ele_2 ==0:
        continue
    else:
        test_list.append([ele_text]+[ele_2_text])
        test_list.append([])

split_index = []
for i in range(len(test_list)):
    ele = test_list[i]
    if ele == []:
        split_index.append(i)

max_iter = len(test_list)
print(max_iter)

test = []
for i in range(0, max_iter-1):
    ele_1 = test_list[i]
    ele_2 = test_list[i+1]
    if type(ele_2) == list:
        test.append(ele_2)
    else:
        test.append([ele_1])

doc = []
for info in test:
    if len(info) == 0:
        continue
    elif len(info[0]) == 0:
        continue
    else:
        doc.append(info)

document = []

for info in doc:
    if len(info) == 1:
        continue
    else:
        document.append(info)

data = []
for i in range(len(document)):
    #print(i)
    key = i
    text = document[i]
    test = {'id':key, 'text':text}
    data.append([test])

------------------------------------------------------------------------------------------
import json
with open(os.getcwd() + '/kg/data/korean_chunk.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False)

with open(os.getcwd() + '/kg/data/korean_chunk.json', 'r') as file:
    test_data = json.load(file)

extract_doc = []
for dict in test_data:
    tt = dict[0]
    text = tt['text']
    extract_doc.append(text)
------------------------------------------------------------------------------------------

import time
import torch
import requests
import os
import ray
from openai import ChatCompletion

def extract_korean_triplet(text):
    api_key = 'sk-...'
    url = "https://api.openai.com/v1/chat/completions"

    print('start extracting triplet...')

    SYS_PROMPT = (
    "주어진 맥락에서 용어 및 관계를 추출하는 네트워크 그래프 생성기로서 당신은 용어와 그들의 관계를 추출하는 작업을 수행합니다. "
    "주어진 텍스트에서 한 문장에 두 개 이상의 관계가 포함된 경우, 각 관계에 대한 triplet을 추출하세요."
    "주어진 맥락 청크가 제공됩니다. 당신의 작업은 주어진 맥락에서 ontology를 추출하는 것입니다."
    "추출된 온톨로지의 노드 중 하나는 반드시 주어진 맥락에서 설명하고 있는 key concept이여야 하며 명사여야 합니다."
    "추출된 온톨로지를 문장으로 바꾸었을 때 의미가 있는 문장이 생성되어야 합니다.\n"
    "Thought 1: 각 문장을 통과하면서 해당 문장에서 언급된 주요 용어를 생각해보세요.\n"
        "\t용어에는 객체, 엔터티, 위치, 조직, 사람, \n"
        "\t조건, 약어, 문서, 서비스, 개념 등이 포함될 수 있습니다.\n"
        "\t용어는 최대한 형태소여야 합니다.\n\n"
    "Thought 2: 이러한 용어가 다른 용어와 일대일 관계를 가질 수 있는 방법을 생각해보세요.\n"
        "\t같은 문장이나 같은 단락에 언급된 용어는 일반적으로 서로 관련이 있을 것입니다.\n"
        "\t용어는 다른 여러 용어와 관련이 있을 수 있습니다.\n\n"
    "Thought 3: 각 관련된 용어 쌍 간의 관계를 찾아보세요. \n\n"
    "Thought 4: 추출한 triplet에서 관계가 정말로 1개만 있는지 생각해보세요. \n\n"
    "Thought 5: 추출된 온톨로지를 문장으로 바꾸었을 때 그 문장이 의미를 가진 문장인지 잘 생각해보세요. \n\n"
    "출력 형식은 용어 쌍 및 그들 간의 관계를 포함하며 다음과 같습니다: \n"
    "[\n"
    "   {\n"
    '       "node_1": "A concept from extracted ontology", it is noun \n'
    '       "node_2": "A related concept from extracted ontology", it is noun \n'
    '       "edge": "Key verb phrase that explains the relationship between node1 and node2 "\n'
    "   }, {...}\n"
    "]"
)
    # one or two sentences
    input_text = f"이 텍스트에서 관계들을 추출해줘:'{text}'"

    #print(input_text)
    
    messages = [
        {"role": "system", "content":SYS_PROMPT},
        {"role": "user", "content": input_text}
    ]

    headers = {"Authorization":f"Bearer {api_key}"}

    call_data_ = {"model": "gpt-3.5-turbo", "messages": messages}

    #requests.post(url, headers=headers, json=call_data_).json()

    data = requests.post(url, headers=headers, json=call_data_).json()

    print('decode', data)

    try:
        triplet = data['choices'][0]['message']['content']
        #print(triplet)
    except:
           triplet = []
           #print(triplet)

    
    return triplet


sen = []
for data in document:
    print('non-error', data)
    tri = extract_korean_triplet(data)
    print(tri)
    if tri == []:
        print('test', data)
        time.sleep(30)
        re_tri = extract_korean_triplet(data)
        print('check',re_tri)
        sen.append(re_tri)
    else:
        sen.append(tri)

------------------------------------------------------------
## Testing Ray
@ray.remote
def extract_korean_triplet(text):
    import time
    api_key = 'sk-...'
    url = "https://api.openai.com/v1/chat/completions"

    print('start extracting triplet...')
    max = 10

    SYS_PROMPT = (
    "주어진 맥락에서 용어 및 관계를 추출하는 네트워크 그래프 생성기로서 당신은 용어와 그들의 관계를 추출하는 작업을 수행합니다."
    "If given text contain '=', then make triplet like [All text before '=' , '=', rest part of given text] from given text. output format like: \n"
    "[\n"
    "   {\n"
    '       "node_1": "first element of triplet" \n'
    '       "node_2": "last element of triplet" \n'
    '       "edge": "second element of triplet" \n'
    "   }\n"
    "]"
    "And then follw next. If given text is not contain '=', then follow next."
    "주어진 텍스트에서 한 문장에 두 개 이상의 관계가 포함된 경우, 각 관계에 대한 triplet을 추출하세요."
    "주어진 맥락 청크가 제공됩니다. 당신의 작업은 주어진 맥락에서 ontology를 추출하는 것입니다."
    "추출된 온톨로지의 노드 중 하나는 반드시 주어진 맥락에서 설명하고 있는 key concept이여야 하며 명사여야 합니다."
    "추출된 온톨로지를 문장으로 바꾸었을 때 의미가 있는 문장이 생성되어야 합니다.\n"
    "Thought 1: 각 문장을 통과하면서 해당 문장에서 언급된 주요 용어를 생각해보세요.\n"
        "\t용어에는 객체, 엔터티, 위치, 조직, 사람, \n"
        "\t조건, 약어, 문서, 서비스, 개념 등이 포함될 수 있습니다.\n"
        "\t용어는 최대한 형태소여야 합니다.\n\n"
    "Thought 2: 이러한 용어가 다른 용어와 일대일 관계를 가질 수 있는 방법을 생각해보세요.\n"
        "\t같은 문장이나 같은 단락에 언급된 용어는 일반적으로 서로 관련이 있을 것입니다.\n"
        "\t용어는 다른 여러 용어와 관련이 있을 수 있습니다.\n\n"
    "Thought 3: 각 관련된 용어 쌍 간의 관계를 찾아보세요. \n\n"
    "Thought 4: 추출한 triplet에서 관계가 정말로 1개만 있는지 생각해보세요. \n\n"
    "Thought 5: 추출된 온톨로지를 문장으로 바꾸었을 때 그 문장이 의미를 가진 문장인지 잘 생각해보세요. \n\n"
    "출력 형식은 용어 쌍 및 그들 간의 관계를 포함하며 다음과 같습니다: \n"
    "[\n"
    "   {\n"
    '       "node_1": "A concept from extracted ontology", \n'
    '       "node_2": "A related concept from extracted ontology", \n'
    '       "edge": "Key verb phrase that explains the relationship between node1 and node2 "\n'
    "   }, {...}\n"
    "]"
)
    # one or two sentences
    input_text = f"A question is provided below.Given the question, extract up to '{max}' keywords from the text.Focus on extracting the keywords that we can use  to best lookup answers to the question.Avoid stopwords from '{text}'.With this extracted keywords, please extraction relationship from '{text}'"
    #input_text = f"이 텍스트에서 관계들을 추출해줘:'{text}'"
    #print(input_text)
    
    messages = [
        {"role": "system", "content":SYS_PROMPT},
        {"role": "user", "content": input_text}
    ]

    headers = {"Authorization":f"Bearer {api_key}"}

    call_data_ = {"model":"gpt-3.5-turbo-0613", "messages": messages}
    #call_data_ = {"model": "gpt-3.5-turbo", "messages": messages}

    #requests.post(url, headers=headers, json=call_data_).json()

    print(text)
    data = requests.post(url, headers=headers, json=call_data_).json()

    #print('type', type(data))

    #triplet = data['choices'][0]['message']['content'] <-- error occure in this line

    while True:
        try:
            if 'error' in data:
                time.sleep(60)
                logging.error('502 error occurred. Retrying...')
                continue
                
            return_triplet = data.get('choices', [])[0].get('message', {}).get('content', '')
            print('decode', return_triplet)
            break
        
        except requests.exceptions.RequestException as e:
            print(f"Error in request: {e}")
            time.sleep(60) 
    
        
    return return_triplet
---------------------------------------------------------------------------
post_sen = []
for senten in sen:
    sent = senten[1:-1]
    if sent == []:
        continue
    else:
        post_sen.append(sent)

import re
pattern = re.compile(r'"node_1": "(.*?)",\s*"node_2": "(.*?)",\s*"edge": "(.*?)"')
triplet=[]

for sentence in post_sen:
    tri = sentence[1:-1]
    print(tri)
    matches = pattern.findall(tri)
    for match in matches:
        node_1, node_2, edge = match
        triplet.append([node_1, edge, node_2])
