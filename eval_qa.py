from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric,FaithfulnessMetric
from deepeval.test_case import LLMTestCase
import os

import anthropic
import os
import http.client
import requests
import re
import requests
import time
import json
from json_repair import repair_json

import math
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

import getpass
import os

os.environ['OPENAI_API_KEY'] = "sk-..."
from openai import OpenAI

client = OpenAI()

api_key = "sk-ant-..."
client_ = anthropic.Anthropic(api_key=api_key)
 
def get_response_from_claude(context,qs, sys_prompt):
    result_text = ""
    
    # Claude에 메시지 생성 요청을 보냅니다.
    response = client_.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
        temperature=0.0,
        system= sys_prompt,
        messages=[{"role": "user", "content": f"'{context}'로부터 추출된 {qs}에 대한 평가를 수행해주세요"}]
    )
    
    # 응답 객체에서 텍스트 내용만 추출합니다.
    if not response.content or not isinstance(response.content, list):
        result_text = "No response or unexpected response format."
    else:
        response_texts = [block.text for block in response.content if hasattr(block, 'text')]
        result_text = " ".join(response_texts)
 
    return result_text

fluency_templet = """
당신의 역할은 주어진 context를 분석하여 추출된 각 질문(Q)의 Fluency를 개별적으로 평가하고, 낮은 점수를 받은 질문들을 식별하는 것입니다.
Fluency는 질문의 관련성, 의미, 다양성을 의미합니다.

평가 단계:
1. Context 이해:
   - 제공된 {context}를 주의 깊게 읽으세요.
   - context의 주요 주제, 핵심 개념, 중요 정보를 파악하세요.
2. 개별 질문 평가:
   - {qs}의 각 질문에 대해 다음 기준으로 Fluency를 평가하세요:
     1: 매우 낮음 - context와 거의 관련이 없거나 의미가 불분명한 질문
     2: 낮음 - context와 약간 관련이 있지만 중요도가 낮은 질문
     3: 보통 - context와 관련이 있고 어느 정도 의미 있는 질문
     4: 높음 - context와 밀접하게 관련되어 있고 중요한 내용을 다루는 질문
     5: 매우 높음 - context의 핵심을 정확히 짚어내고 깊이 있는 사고를 요구하는 질문
3. 평가 요소:
   - 각 질문을 평가할 때 다음 요소들을 고려하세요:
     a) 관련성: 질문이 context와 얼마나 밀접하게 관련되어 있는가?
     b) 중요도: 질문이 context의 핵심 내용을 얼마나 잘 다루고 있는가?
     c) 다양성: 질문이 다른 질문들과 얼마나 다른 측면을 다루고 있는가?
     d) 질문 유형: 사실 확인, 분석, 추론, 비판적 사고 중 어떤 유형인가?
     e) 개방성: 질문이 얼마나 깊이 있는 사고나 다양한 관점을 요구하는가?
4. 낮은 점수 질문 식별:
   - Fluency 점수가 2점 이하인 질문들을 "낮은 점수 질문"으로 분류하세요.

평가 결과를 다음 형식으로 제출하세요:
[
개별 질문 평가:

[질문 내용]: [점수] - [간단한 평가 이유]
[질문 내용]: [점수] - [간단한 평가 이유]
...

낮은 점수 질문들:

[질문 내용]: [점수] - [개선 제안]
[질문 내용]: [점수] - [개선 제안]
...
]
"""

flexibility_templet = """
당신의 역할은 주어진 context를 분석하여 추출된 각 질문(Q)의 Flexibility를 개별적으로 평가하고, 낮은 점수를 받은 질문들을 식별하는 것입니다. 
Flexibility는 질문이 다양한 카테고리, 관점, 접근 방식을 포함하는 정도를 의미합니다.
평가 단계:
1. Context 이해:
   - 제공된 {context}를 주의 깊게 읽으세요.
   - context의 주요 주제, 핵심 개념, 다룰 수 있는 잠재적 카테고리를 파악하세요.
2. 개별 질문 평가:
   - {qs}의 각 질문에 대해 다음 기준으로 Flexibility를 평가하세요:
     1: 매우 낮음 - 하나의 카테고리나 관점에만 국한된 질문
     2: 낮음 - 두 개의 카테고리나 관점을 다루는 질문
     3: 보통 - 세 개의 카테고리나 관점을 다루는 질문
     4: 높음 - 네 개의 카테고리나 관점을 다루는 질문
     5: 매우 높음 - 다섯 개 이상의 카테고리나 관점을 다루는 질문
3. 평가 요소:
   - 각 질문을 평가할 때 다음 요소들을 고려하세요:
     a) 질문 카테고리: 사실적, 개념적, 분석적, 평가적, 창의적, 실용적/적용 질문 등
     b) 관점 다양성: 기술적, 사회적, 경제적, 윤리적, 환경적 관점 등
     c) 접근 방식: 연대기적, 비교/대조, 원인/결과, 문제/해결책 접근 등
     d) 다학제성: 질문이 여러 학문 분야나 영역을 연결하는 정도
     e) 사고의 확장: 질문이 기존의 사고 패턴을 벗어나는 정도
4. 낮은 점수 질문 식별:
   - Flexibility 점수가 2점 이하인 질문들을 "낮은 점수 질문"으로 분류하세요.
   
평가 결과를 다음 형식으로 제출하세요:
[
개별 질문 평가:

[질문 내용]: [점수] - [간단한 평가 이유]
[질문 내용]: [점수] - [간단한 평가 이유]
...

낮은 점수 질문들:

[질문 내용]: [점수] - [개선 제안]
[질문 내용]: [점수] - [개선 제안]
...
]
"""

originality_prompt = """
당신의 역할은 주어진 context를 분석하여 추출된 각 질문(Q)의 Originality를 개별적으로 평가하고, 낮은 점수를 받은 질문들을 식별하는 것입니다. 
Originality는 질문의 독창성, 희소성, 그리고 관습적인 사고에서 벗어난 정도를 의미합니다.

평가 단계:
1. Context 이해:
   - 제공된 {context}를 주의 깊게 읽으세요.
   - context의 주요 주제, 핵심 개념, 일반적으로 예상되는 질문 유형을 파악하세요.
2. 개별 질문 평가:
   - {qs}의 각 질문에 대해 다음 기준으로 Originality를 평가하세요:
     1: 매우 낮음 - 매우 일반적이고 예측 가능한 질문
     2: 낮음 - 일반적이지만 약간의 독특한 요소가 있는 질문
     3: 보통 - 어느 정도 독창적이며 새로운 관점을 제시하는 질문
     4: 높음 - 독창적이고 예상치 못한 접근을 보이는 질문
     5: 매우 높음 - 매우 독창적이고 혁신적인 사고를 보여주는 질문
3. 평가 요소:
   - 각 질문을 평가할 때 다음 요소들을 고려하세요:
     a) 독특성: 질문이 얼마나 흔하지 않은가?
     b) 관점의 새로움: 질문이 context를 얼마나 새로운 각도에서 바라보는가?
     c) 창의적 응용: 질문이 context의 내용을 다른 영역이나 상황에 어떻게 적용하는가?
     d) 사고의 확장: 질문이 기존의 사고 패턴을 얼마나 벗어나는가?
4. 낮은 점수 질문 식별:
   - Originality 점수가 2점 이하인 질문들을 "낮은 점수 질문"으로 분류하세요.

평가 결과를 다음 형식으로 제출하세요:
[
개별 질문 평가:

[질문 내용]: [점수] - [간단한 평가 이유]
[질문 내용]: [점수] - [간단한 평가 이유]
...

낮은 점수 질문들:

[질문 내용]: [점수] - [개선 제안]
[질문 내용]: [점수] - [개선 제안]
...
]
"""

elaboration_templet = """
당신의 역할은 주어진 context를 분석하여 추출된 각 질문(Q)의 Elaboration을 개별적으로 평가하고, 낮은 점수를 받은 질문들을 식별하는 것입니다. 
Elaboration은 아이디어를 확장, 정제, 그리고 꾸며내는 능력을 의미합니다. 이는 세부 사항을 추가하고, 뉘앙스를 발전시키며, 기본 개념을 더 복잡하거나 정교하게 만드는 것을 포함합니다.

평가 단계:
1. Context 이해:
   - 제공된 {context}를 주의 깊게 읽으세요.
   - context의 주요 주제, 핵심 개념, 세부 사항들을 파악하세요.
2. 개별 질문 평가:
   - {qs}의 각 질문에 대해 다음 기준으로 Elaboration을 평가하세요:
     1: 매우 낮음 - 기본적인 사실 확인 수준의 질문으로, 세부 사항이나 깊이가 거의 없음
     2: 낮음 - 약간의 세부 사항을 요구하지만, 개념을 더 발전시키지는 않음
     3: 보통 - 기본 개념에 대한 일정 수준의 확장이나 세부 사항을 요구함
     4: 높음 - 개념을 상당히 확장하고, 여러 측면이나 세부 사항을 고려하도록 요구함
     5: 매우 높음 - 복잡한 아이디어를 정교화하고, 다양한 측면을 깊이 있게 탐구하도록 요구함
3. 평가 요소:
   - 각 질문을 평가할 때 다음 요소들을 고려하세요:
     a) 세부 사항 요구: 질문이 얼마나 구체적인 세부 사항을 요구하는가?
     b) 개념 확장: 질문이 기본 개념을 얼마나 확장하거나 발전시키는가?
     c) 복잡성: 질문이 얼마나 복잡한 사고나 다면적 접근을 요구하는가?
     d) 깊이: 질문이 주제에 대해 얼마나 깊이 있는 탐구를 요구하는가?
     e) 연결성: 질문이 여러 아이디어나 개념을 어떻게 연결하는가?
4. 낮은 점수 질문 식별:
   - Elaboration 점수가 2점 이하인 질문들을 "낮은 점수 질문"으로 분류하세요.

평가 결과를 다음 형식으로 제출하세요:
[
개별 질문 평가:

[질문 내용]: [점수] - [간단한 평가 이유]
[질문 내용]: [점수] - [간단한 평가 이유]
...

낮은 점수 질문들:

[질문 내용]: [점수] - [개선 제안]
[질문 내용]: [점수] - [개선 제안]
...
]
"""

check_templet = """
당신의 역할은 주어진 Context를 분석하고, 이에 대해 생성된 질문들(Q)이 얼마나 적합한지 평가하는 것입니다.
이 평가는 질문들이 Context의 내용을 얼마나 잘 반영하고 있는지, 그리고 Context로부터 중요한 정보를 얼마나 효과적으로 추출하고 있는지를 판단합니다.

평가 단계:
1. Context 이해:
   - 제공된 {context}를 주의 깊게 읽으세요.
   - Context의 주요 주제, 핵심 개념, 중요 정보, 등장인물, 사건, 배경 등을 파악하세요.
2. 생성된 질문(Q) 검토:
   - 주어진 {qs}들을 면밀히 검토하세요.
   - 각 질문이 Context의 어떤 부분과 관련되어 있는지 확인하세요.
3. 개별 질문 평가:
   - 각 질문에 대해 다음 기준으로 적합성을 평가하세요:
     1: 매우 낮음 - Context와 거의 관련이 없거나 잘못된 정보를 포함한 질문
     2: 낮음 - Context와 약간 관련이 있지만 중요도가 낮거나 부적절한 질문
     3: 보통 - Context와 관련이 있고 어느 정도 중요한 정보를 다루는 질문
     4: 높음 - Context의 중요한 측면을 정확히 다루는 질문
     5: 매우 높음 - Context의 핵심을 정확히 짚어내고 깊이 있는 이해를 요구하는 질문
4. 평가 요소:
   - 각 질문을 평가할 때 다음 요소들을 고려하세요:
     a) 관련성: 질문이 Context의 내용과 얼마나 밀접하게 관련되어 있는가?
     b) 정확성: 질문이 Context의 정보를 정확하게 반영하고 있는가?
     c) 중요도: 질문이 Context의 핵심적인 내용을 다루고 있는가?
     d) 포괄성: 질문들이 Context의 다양한 측면을 고루 다루고 있는가?
     e) 깊이: 질문이 Context에 대한 표면적 이해를 넘어 깊이 있는 사고를 요구하는가?
5. 부적절한 질문 식별:
   - 적합성 점수가 2점 이하인 질문들을 "부적절한 질문"으로 분류하세요.

평가 결과를 다음 형식으로 제출하세요:
[
개별 질문 평가:

[질문 내용]: [점수] - [간단한 평가 이유]
[질문 내용]: [점수] - [간단한 평가 이유]
...

부적절한 질문들:

[질문 내용]: [점수] - [개선 제안]
[질문 내용]: [점수] - [개선 제안]
...
]
"""

def faithfulness(prompt, LLM_response, retrieval_context, model):
    from deepeval import evaluate
    faithfulness_metric = FaithfulnessMetric(threshold=0.7,
                                             model = model,
                                             include_reason = True)
    test_case = LLMTestCase(input=prompt, actual_output = LLM_response, retrieval_context = retrieval_context)
    faithfulness_metric.measure(test_case)
    score = faithfulness_metric.score
    reason = faithfulness_metric.reason
    
    return score, reason

def get_embeddings(texts):
    return np.array([model.encode(text) for text in texts])

def calculate_entropy(sim_matrix):
    return -np.sum(sim_matrix * np.log2(sim_matrix + 1e-10)) / sim_matrix.shape[0]

def calculate_conditional_entropy(total_sim_matrix, i):
    n = total_sim_matrix.shape[0]
    conditional_sim = total_sim_matrix[i] / np.sum(total_sim_matrix[i])
    return -np.sum(conditional_sim * np.log2(conditional_sim + 1e-10))

def calculate_metrics(questions):
    embeddings = get_embeddings(questions)
    total_sim_matrix = cosine_similarity(embeddings)
    total_sim_matrix = (total_sim_matrix + 1) / 2  # 유사도를 0~1 범위로 정규화
    np.fill_diagonal(total_sim_matrix, 1)
    
    total_entropy = calculate_entropy(total_sim_matrix)
    
    conditional_entropies = []
    information_gains = []
    
    for i in range(len(questions)):
        cond_entropy = calculate_conditional_entropy(total_sim_matrix, i)
        conditional_entropies.append(cond_entropy)
        
        subset_sim_matrix = np.delete(np.delete(total_sim_matrix, i, axis=0), i, axis=1)
        subset_entropy = calculate_entropy(subset_sim_matrix)
        ig = total_entropy - subset_entropy
        information_gains.append(ig)
    
    return total_entropy, conditional_entropies, information_gains

def fluency_eval(context, qs):
    sys_prompt = fluency_templet.format(context = context , qs = qs)
    result = get_response_from_claude(context,qs, sys_prompt)

    return result

def flexibility_eval(context, qs):
    sys_prompt = flexibility_templet.format(context = context , qs = qs)
    result = get_response_from_claude(context,qs, sys_prompt)
    
    return result

def originality_eval(context, qs):
    sys_prompt = originality_prompt.format(context = context , qs = qs)
    result = get_response_from_claude(context,qs, sys_prompt)
    
    return result

def Elaboration_eval(context, qs):
    sys_prompt = elaboration_templet.format(context = context , qs = qs)
    result = get_response_from_claude(context,qs, sys_prompt)
    
    return result

def context_question(context, qs):
    sys_prompt = check_templet.format(context = context , qs = qs)
    result = get_response_from_claude(context,qs, sys_prompt)
    
    return result

model = 'gpt-4o'
context = ["소녀 미나는 작은 해안 마을에 살고 있었습니다. 그녀는 바다를 사랑했지만, 수영을 할 줄 몰라 늘 물가에서만 머물러야 했습니다. 어느 날, 마을에 큰 태풍이 몰아치면서 미나의 어린 동생이 파도에 휩쓸려 갔습니다. 주변 어른들은 모두 무서워서 나서지 못했지만, 미나는 용기를 내어 물속으로 뛰어들었습니다. 그녀는 공포와 싸우며 필사적으로 팔다리를 움직여 동생에게 다가갔고, 기적적으로 동생을 구해냈습니다. 이 사건 이후 미나는 마을의 영웅이 되었고, 수영에 대한 두려움도 극복했습니다. 그녀는 이제 다른 아이들에게 수영을 가르치며, 바다의 아름다움과 위험성을 동시에 일깨워주고 있습니다."]

questions = [
    "이야기의 주요 테마는 무엇인가요?",
    "주인공은 누구인가요?",
    "이야기는 어디에서 일어나나요?",
    "이야기의 갈등은 무엇인가요?",
    "이야기는 어떻게 끝나나요?"
]

answers = ["용기와 극복",
           "소녀 미나",
           "작은 해안 마을",
           "미나의 수영 두려움과 동생을 구해야 하는 상황",
           "미나가 동생을 구하고 마을의 영웅이 되며, 수영 두려움을 극복하고 다른 아이들에게 수영을 가르침"]

all_result = []
for q,a in zip(questions, answers):
    result = {}
    score, reason = faithfulness('None', q, context, model)
    result['faithfulness_score'] = score
    result['faithfulness_reason'] = reason

    all_result.append(result)

# 메트릭 계산
total_entropy, cond_entropies, igs = calculate_metrics(questions)

# 결과 출력
print(f"전체 질문 세트의 엔트로피: {total_entropy:.4f}")
if total_entropy > 0:
    print("모든 질문들이 어느 정도의 고유한 정보를 제공하고 있습니다")
    
for question, ce, ig in zip(questions, cond_entropies, igs):
    print(f"질문: {question}")
    print(f"  조건부 엔트로피: {ce:.4f}")
    print(f"  Information Gain: {ig:.4f}")
    print()

fluency = fluency_eval(context, questions)
flexibility = flexibility_eval(context, questions)
originality = originality_eval(context, questions)
Elaboration = Elaboration_eval(context, questions)
context_check = context_question(context, questions)
