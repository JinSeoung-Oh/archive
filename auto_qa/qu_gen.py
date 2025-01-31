### upload your task (upload your data)
### context = If you want run this code without context, then just set context=None

task = "다음 주어진 컨텍스트로부터 QA 쌍 10개씩 추출해주고 요약도 해줘. 보고서 형식으로 다시 재작성해줘.자체 피드백을 통해 향상된 결과 값을 리턴해줘"
context = """경남 거제시는 코로나19로 인해 초등학교 돌봄교실 급식에 차질이 생기자 지역 농산물로 만든 도시락을 제공했다. 도시락을 통해 취약계층·맞벌이 부부의 육아 부담을 경감하고 지역경제 활성화에 기여하였다는 평을 받아 지난해 최우수상을 수상했다. (경남 거제시, 사회적 가치를 더한 우리 아이 건강밥상 / '20년 최우수상)
강원 화천군은 지역 내 교통인프라 부족으로 학생, 임신부 등 교통약자의 이동에 어려움이 많았다. 이를 해소하고자 학생 등교 및 방과후 이동을 지원하는 학생 무상 통학 셔틀과 임신부 등 의료원 방문을 지원하는 메디컬 셔틀을 운영하여 주민 편의 증진에 기여한 공을 인정받았다.(강원 화천군, 화천 행복 셔틀 / '20년 우수상)
지역 맞춤형 저출산 대응을 추진하고 있는 11곳의 지자체 중 최우수상을 선정하는 자리가 마련된다.
이와 관련해 행정안전부(장관 전해철)는 {2021년 지자체 저출산 대응 우수사례 경진대회}를 11월 4일(목) 정부서울청사 별관 국제회의실에서 개최한다고 밝혔다.
  * 올해 6회째를 맞이하는 '지자체 저출산 대응 우수사례 경진대회'는 초저출산 문제를 극복해 나가는 자치단체들의 노력을 격려하고 우수사례를 공유하는 자리이다.
이번 대회는 시·도 심사를 통해 추천된 우수사례 51건을 대상으로 온라인 국민심사\*와 전문가 서류심사를 통해 1차로 11건의 우수 사례를 선정했으며, 경진대회 발표심사를 거쳐 최종순위를 가리게 된다.
\* 국민이 광화문 1번가에 접속하여 온라인 심사 참여(시도 3건, 시군구 3건 선택)
  * 올해 선정된 11곳은 시·도 3곳(울산,강원,전남)과 시·군·구 8곳(서울 서초구, 서울 강동구, 부산 수영구, 경기 시흥시, 강원 양구군, 충남 당진시, 전남 광양시, 경북 포항시)이다.
  * 경진대회에서는 11건의 우수사례 중 최우수 2건, 우수 4건, 장려 5건을 선정하고, 총 특별교부세 7억 원이 지원된다.
{t_8.png}
올해 선정된 우수사례를 살펴보면 증가하는 돌봄 수요에 대응한 공공 돌봄 서비스 강화, 육아기 부담 경감을 위한 재정지원, 지역 특성을 고려한 출산인프라 지원, 청년 및 신혼부부를 위한 주거지원 등의 유형이 많았다.
  * 이를 통해 단순히 출산과 양육을 지원하는 저출산 개념에서 주민의 삶의 질 전반을 향상시키기 위한 사업으로 확장되었음을 알 수 있었다.
  * 또한, 정부 정책의 틈새를 촘촘히 메우는 지역별 맞춤형 정책들이 눈길을 끌었다.
1차심사에 참여한 조용남 심사위원(한국보육진흥원 교직원지원국장)은 "우수사례들을 살펴보면, 저출산 문제에 대해 통합적·근본적 관점에서 해결하고자 하는 고민과 의지가 보였다"라며, "많은 우수사례들이 있었지만, 그중 주요 출산계충인 젊은 세대의 출산 결심을 독려할 수 있고, 또 실제 출산과 육아 부담을 덜어 줄 수 있는 국민이 체감할 수 있는 사업을 중심으로 1차 선정했다"고 평가했다.
고규창 행정안전부 차관은 “이번 경진대회는 국정 우선과제인 저출산 및 인구 위기를 극복하기 위해 현장에서 주민들과 소통하며 정책 실현에 앞장서는 지자체 역할의 중요성을 되새기는 자리"라며, “지역 맞춤형 정책을 추진하고 있는 11곳 지자체 모두에게 감사와 격려를 보낸다."고 말했다.
  * 덧붙여 "앞으로도 행정안전부는 주민들에게 체감도 높은 지역별 우수사례를 발굴·확산하고, 급변하는 인구 상황에 대응하여 지역사회의 대응력을 강화할 수 있도록 적극 지원하겠다."라고 말했다."""
  
### Enter or upload your detail requirement
### If you have any guideline, then just setup it None like below
qa_additioanl_requirement = None
su_additioanl_requirement = None
re_additioanl_requirement = "제시된 문단의 세 번째 문장을 음슴체로 바꿔줘"

### setup parameter
multi_task = True
default_context=context

### Load Agent_list
import json

with open("./agent_library.json", "r") as f:
    sys_msg_list = json.load(f)

agent_list = []
for info in sys_msg_list:
    #print(info)
    for agent_info in info:
        #print(agent_info)
        a_name = agent_info['name']
        a_de = agent_info['description']
        agent = a_name + ':' + a_de + '\n'
        agent_list.append(agent)
        
 ### select Agent manually, below is example
 selected_agnet = ['GeneralQAExpert', 'GeneralSummaryExpert', 'GeneralRewriteExpert', 'ComprehensiveFeedbackExpert', 'QAImprovementExpert', 'SummaryImprovementExpert', 'RewriteImprovementExpert']
 
import getpass
import os

#os.environ['OPENAI_API_KEY'] = "sk-..."
os.environ['OPENAI_API_KEY'] = 'sk-...'
from openai import OpenAI
from json_ko import repair_json
import json

client = OpenAI()

from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import time
from agent_or_prompt import *

def add_context_to_queries(queries, context):
    return [
        {**query, "context": context}
        for query in queries
    ]
    
    
 def add_additional_requirements(queries, qa_additioanl_requirement, su_additioanl_requirement, re_additioanl_requirement):
    new = []
    for query in queries:
        selected = query['agents']
        for agent in selected:
            if 'QA' in agent:
                query['additioanl_requirement'] = qa_additioanl_requirement
                new.append(query)
            if 'Summary' in agent:
                query['additioanl_requirement'] = su_additioanl_requirement
                new.append(query)
            if 'Rewrite' in agent:
                query['additioanl_requirement'] = re_additioanl_requirement
                new.append(query)
    return new
    
 def or_agent(prompt_for_query_analizer,task, agent_list):
    sys_prompt = prompt_for_query_analizer
    user_prompt = task_example
    add = f'주어진 질문를 세부 질문으로 분석해주고, 해당 세부 질문을 수행하는데 필요한 Agent를 {agent_list}로부터 찾아서 리턴해줘'
   
    generate_params = {
            'model': 'gpt-4o',
            'temperature': 0,
        }


    ## Summary Sentence:
    messages = [
            {
                'role': 'system',
                'content': sys_prompt,
            },
            {
                'role': 'user',
                'content': user_prompt + add
            }
        ]

    response = client.chat.completions.create(
        messages=messages,
        **generate_params,
    )
    or_agent_list = response.choices[0].message.content
   
    return or_agent_list
    

multi_task = True
if multi_task:
    analyzed_query = or_agent(prompt_for_query_analizer,task, selected_agnet )
else:
    analyzed_query = [{"question" : task, "agents":selected_agnet}]
    
json_check = repair_json(analyzed_query)
analyzed_query  = json.loads(json_check)

analyzed_query = add_context_to_queries(analyzed_query, default_context)
analyzed_query= add_additional_requirements(analyzed_query, qa_additioanl_requirement, su_additioanl_requirement, re_additioanl_requirement)


### build pipline
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

llm = ChatOpenAI(model='gpt-4o')

class LLMAgent:
    def __init__(self, name, system_message, description):
        self.name = name
        self.system_message = system_message
        self.description = description

    def __str__(self):
        return f"Agent: {self.name}\nDescription: {self.description}"

def DynamicAgent_response(llm_agent, llm, state):
    query = state.current_task
    context = state.subtasks[0].context if state.subtasks else ''
    requirement = state.subtasks[0].additioanl_requirement if state.subtasks else ''
       
    messages = [
        SystemMessage(content=llm_agent.system_message),
        HumanMessage(content=f"Context: {context}\nQuestion: {query}\nGuidelines:{requirement}")
    ]

    #print(messages)
    response = llm(messages)
       
    state.agent_outputs[llm_agent.name.lower()] = response.content
    state.last_agent = llm_agent.name
    return state

def create_agent_response(agent_info, llm, state):
    llm_agent = LLMAgent(agent_info['name'], agent_info['system_message'], agent_info['description'])
    new_state = DynamicAgent_response(llm_agent, llm, state)
    return new_state

class Subtask(BaseModel):
    question: str
    agents: List[str]
    context: Optional[str] = None
    additioanl_requirement: Optional[str] = None

class State(BaseModel):
    subtasks: List[Subtask] = Field(default_factory=list)
    current_task: Optional[str] = None
    current_agents: List[str] = Field(default_factory=list)
    next_agent: Optional[str] = None
    agent_outputs: dict = Field(default_factory=dict)
    last_agent: Optional[str] = None
    task_complete: bool = False
    subtask_complete: bool = False

def orchestrator(state):
    if state.subtasks:
        current_subtask = state.subtasks[0]
        state.current_task = current_subtask.question
        state.current_agents = current_subtask.agents.copy()
        state.next_agent = state.current_agents.pop(0)
        state.subtask_complete = False
    else:
        state.task_complete = True
    return state

def route_to_agent(state):
    agent_type = state.next_agent
    if agent_type not in graph.nodes:
        agent_info = next((agent for agent in flattened_sys_msg_list if agent['name'] == agent_type), None)
        if agent_info:
            state = create_agent_response(agent_info, llm, state)
    return state

def agent_completed(state):
    if state.current_agents:
        state.next_agent = state.current_agents.pop(0)
    else:
        state.subtasks.pop(0)
        state.subtask_complete = True
    return state

# Initialize the graph
graph = StateGraph(State)

# Add nodes
graph.add_node("orchestrator", orchestrator)
graph.add_node("router", route_to_agent)
graph.add_node("agent_completed", agent_completed)

# Set entry point
graph.set_entry_point("orchestrator")

# Add conditional edges
graph.add_conditional_edges(
    "orchestrator",
    lambda state: "router" if not state.task_complete else END)
graph.add_edge("router", "agent_completed")
graph.add_conditional_edges(
    "agent_completed",
    lambda state: "router" if not state.subtask_complete else "orchestrator")

# Compile the graph
chain = graph.compile()

# Flatten sys_msg_list
flattened_sys_msg_list = [agent for sublist in sys_msg_list for agent in sublist]

# Select Agents
selected_agents = set()
for query in analyzed_query:
    selected_agents.update(query['agents'])

# Create LLM Agents
llm_agents = []
for agent_info in flattened_sys_msg_list:
    if agent_info['name'] in selected_agents:
        agent = LLMAgent(agent_info['name'], agent_info['system_message'], agent_info['description'])
        llm_agents.append(agent)

# Run the chain with the analyzed query
results = []
for query in analyzed_query:
    initial_state = State(subtasks=[Subtask(**query)])
    final_state = chain.invoke(initial_state)
    results.append({
        "question": query["question"],
        "output": final_state["agent_outputs"]
    })
    time.sleep(5)

# Create the final response as JSON Object
final_response = json.dumps(results, ensure_ascii=False, indent=2)

# Print the final response
print(final_response)

## save
json.dump(final_response, open("./result.json", "w"), ensure_ascii=False, indent=4)





 
 
