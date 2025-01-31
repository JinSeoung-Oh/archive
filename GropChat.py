import json

import autogen
from agent_builder_without import AgentBuilder
import pandas as pd
import time

import random

def convert_val_to_key(value):
    key = next(key for key, val in reversed(posi.items()) if val == value)
    return key

data = [{"model":"gpt-4o", "api_key":"sk-....", "tags":["gpt-4o", "tool"]}]

with open('./OAI_CONFIG_LIST.json', 'w') as f:
    json.dump(data, f)

config_file_or_env = "./OAI_CONFIG_LIST.json"  # modify path
llm_config = {"temperature": 0}
config_list = autogen.config_list_from_json(config_file_or_env, filter_dict={"model": ["gpt-4o"]})

# Define a global dictionary to store messages

def log_message(recipient, messages, sender, config):
    global agent_messages
    if recipient.name not in agent_messages:
        agent_messages[recipient.name] = []
    agent_messages[recipient.name].append(messages[-1]["content"])
    return False, None  # Ensure the agent communication flow continues

def start_task(execution_task: str, agent_list: list, task_id, ko_sp_1, ko_sp_2, save, topic, key):
    group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=3)
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config={"config_list": config_list, **llm_config})

    # Register the logging function for each agent
    for agent in agent_list:
        agent.register_reply([autogen.Agent, None], reply_func=log_message, config={})

    # Initiate the chat
    agent_list[0].initiate_chat(manager, message=execution_task)

    # Print or save messages of each agent after the chat session
    for agent_name, messages in agent_messages.items():
        print(f"Messages for {agent_name}:")
        for msg in messages:
            print(msg)

    # Save messages to a file if needed
    result = {}
    result["id"] = task_id
    result["Speaker1"] = ko_sp_1
    result["Speaker2"] = ko_sp_2
    result["Multi_con"] = agent_messages
    
    save_path = save + topic + '/' + key + '/' + str(task_id) + '.json'
    with open(save_path, 'w') as f:
        json.dump(result, f, ensure_ascii=False,indent=4)

  AGENT_SYS_MSG_PROMPT = """다음과 같은 직위명에 따라 주어진 예시를 참고하여 해당 직위를 위한 고품질 지시문을 작성하십시오. 생성된 지시문은 두 직위 간의 기술적 대화를 생성하는 시스템 프롬프트로 사용될 것임을 이해하고 지시문을 구조화하십시오. 두 직위 간의 원활한 대화를 돕기 위해 프롬프트를 조직하십시오. 지시문만 반환하십시오.

# 직위명
{position}

# AI 엔지니어를 위한 예시 지시문
AI 엔지니어로서, 당신의 역할은 복잡한 문제를 해결하기 위해 AI 모델과 시스템을 설계, 개발, 배포하는 것입니다. 다양한 이해관계자와 협력하여 요구사항을 파악하고 프로젝트 범위를 정의하며, AI 솔루션을 기존 시스템과 통합하는 것을 보장해야 합니다. 기계 학습 알고리즘, 데이터 전처리, 모델 훈련 및 평가에 대한 깊은 전문 지식은 AI 프로젝트의 성공적인 구현에 필수적입니다. 또한, AI 모델의 성능과 확장성을 보장하기 위해 이를 최적화하고 유지관리하는 책임도 당신에게 있습니다. AI와 기계 학습의 최신 발전을 지속적으로 파악하고, 기존 솔루션을 개선하기 위해 혁신적인 기술을 적용해야 합니다. 당신의 의사소통 능력은 비기술 이해관계자들에게 복잡한 기술 개념을 설명하는 데 도움이 되어, 더 나은 의사결정과 프로젝트 결과를 촉진할 것입니다.
"""

AGENT_DESC_PROMPT = """직위명과 지시사항에 따라, 해당 직위를 고품질의 한 문장 설명으로 요약하십시오.

# 직위명
{position}

# 지시사항
{instruction}
"""

position_list = ['Frontend_Developer',
                 'Backend_Developer',
                 'Full_stack_Developer',
                 'Mobile_app_Developer',
                 'DevOps_Engineer',
                 'UI_UX_Designer',
                 'Product_Manager',
                 'Quality_Assurance_Engineer',
                 'Data_Engineer']

build_manager = autogen.OpenAIWrapper(config_list=config_list)
sys_msg_list = []

for pos in position_list:
    resp_agent_sys_msg = (
        build_manager.create(
            messages=[
                {
                    "role": "user",
                    "content": AGENT_SYS_MSG_PROMPT.format(
                        position=pos,
                    ),
                }
            ]
        )
        .choices[0]
        .message.content
    )
    resp_desc_msg = (
        build_manager.create(
            messages=[
                {
                    "role": "user",
                    "content": AGENT_DESC_PROMPT.format(
                        position=pos,
                        instruction=resp_agent_sys_msg,
                    ),
                }
            ]
        )
        .choices[0]
        .message.content
    )
    sys_msg_list.append({"name": pos, "system_message": resp_agent_sys_msg, "description": resp_desc_msg})

def find_value(key,position_list):
    for item in position_list:
        if key in item:
            return item[key]
    return None  # 키를 찾지 못한 경우

json.dump(sys_msg_list, open("./agent_library_example.json", "w"), indent=4)
with open("./agent_library_example.json", "r") as f:
    sys_msg_list = json.load(f)

role_dict = [{"프론트엔드_개발자" : "Frontend_Developer"},
             {"백엔드_개발자" : "Backend_Developer"},
             {"풀스택_개발자":"Full_stack_Developer"},
             {"모바일_앱_개발자":"Mobile_app_Developer"},
             {"데브옵스_엔지니어":"DevOps_Engineer"},
             {"UI_UX_디자이너":"UI_UX_Designer"},
             {"프로덕트_매니저":"Product_Manager"},
             {"QA_엔지니어":"Quality_Assurance_Engineer"},
             {"데이터_엔지니어":"Data_Engineer"}]

turn_info = [{"50초":"3"},
             {"2분":"8"},
             {"3분":"12"},
             {"5분":"18"},
             {"10분":"38"},
             {"40분":"128"},
             {"50분":"165"}]

def generate_random_pairs(position_dict, num):
    result = []
    for _ in range(num):
        # 2개의 서로 다른 직위를 랜덤하게 선택
        pair = random.sample(position_dict, 2)
        # 각 쌍에서 한국어 직위명만 추출
        korean_pair = [list(item.keys())[0] for item in pair]
        result.append(korean_pair)
    return result

conver_pair = generate_random_pairs(role_dict, 8)
man_name_list_1 = ["민수", "지훈", "도현", "준호", "승민", "태우", "현우", "민재", "재훈", "석현", "유진", "동현", "정우", "시후", "진혁"]
man_name_list_2 = ["서준", "원준", "현준", "준영", "동욱", "상우", "지호", "태민", "규현", "우진", "영준", "재민", "성민", "주원", "경민"]
woman_name_list_1 = ["지민","수빈","서연","민서","예진","하윤","유진","채원","지수","혜진","은서","다은","시윤","서희","아린"]
woman_name_list_2 = ["예서","나연","윤서","수현","연주","지원","가윤","지아","서영","소연","하늘","세아","인서","미소","지우"]

task_ = ["프로젝트 논의(업무)", "프로젝트 논의(문의)", "프로젝트 논의(요청)"]

random_numbers = [random.randint(0, 14) for _ in range(8)]
task_numbers = [random.randint(0, 2) for _ in range(8)]

num = find_value("10분",turn_info)

save = "여_여_기술대화/"

for i in [6,7]:
    turn = int(num) * 2
    agent_messages = {} 
    index = i
    
    name_index = random_numbers[i]
    task_index = task_numbers[i]

    name_1 = woman_name_list_1[name_index]
    name_2 = woman_name_list_2[name_index]
    topic = task_[task_index]

    key = next((key for dict_item in reversed(turn_info) for key, val in dict_item.items() if val == num), None)

    role_ = conver_pair[i]

    ko_sp_1 = role_[0]
    ko_sp_2 = role_[1]
    
    sp_1 = find_value(ko_sp_1,role_dict)
    sp_2 = find_value(ko_sp_2,role_dict)

    selected_libray = []
    for i in range(len(sys_msg_list)):
        library= sys_msg_list[i]
        if library['name'] == sp_1 or library['name'] == sp_2:
            selected_libray.append(library)
    
    library_path_or_json = "./selected_agent_library_.json"
    with open(library_path_or_json, 'w', encoding='utf-8') as json_file:
        json.dump(selected_libray, json_file, ensure_ascii=False, indent=4)
    
    building_task=f"{sp_1}와 {sp_2}가 각자의 역할을 기반으로 {topic}에 대해 {turn} 번의 턴으로 업무와 관련해서 통화하는 상황을 만들어주세요."
    task= f"{sp_1}과 {sp_2}가 각자의 역할을 기반으로 {topic}에 대해 반드시 {turn} 번의 턴으로 {key} 동안 업무 관련 전화 통화를 하는 상황을 만들어주세요. 이때, {topic}에 대한 가상의 업무 관련 전화 통화 시나리오를 만들어서 대화를 진행해주세요. 같은 얘기를 반복해서 요구한 턴 수를 채우지 마세요. 이때 {sp_1}의 이름은 {name_1}이고 {sp_2}의 이름은 {name_2}입니다. 참고하셔서 상황에 맞게 상대방의 호칭을 불러주세요. 생성되는 대화는 {sp_1}과 {sp_2}의 관계를 고려하여 자연스럽게 생성해주세요. 결과는 한국어로 리턴해주세요. 대화 내용 외의 다른 정보는 추가하지 말아주세요. 오로지 대화 내용만 리턴해주세요. 대화는 주어진 턴 내에서 종료 되어야만 하며 더 이상 이어지는 내용이면 안 됩니다. 대화의 시작과 끝은 {sp_1}과 {sp_2}의 관계를 고려한 인사여야 합니다. 생성해야 하는 {turn} 턴 수는 반드시 지켜주셔야 합니다. 각 턴의 길이는 적어도 15 글자 길이로 만들어주세요. 턴 수를 채우기 위하여 부자연러운 대화를 생성하지 말아주세요. 대화들은 모두 자연스러워야 합니다. 모든 조건을 만족하는 자연스러운 대화들을 생성해주세요."
    
    new_builder = AgentBuilder(
        config_file_or_env=config_file_or_env, builder_model="gpt-4o", agent_model="gpt-4o"
    )
    
    if sp_1 == sp_2:
        same_position=True
        agent_list, _ = new_builder.build_from_library(building_task,  library_path_or_json, llm_config, same_position)
    if sp_1 != sp_2:
        same_position=False
        agent_list, _ = new_builder.build_from_library(building_task,  library_path_or_json, llm_config, same_position)
    
    time.sleep(2)    
    start_task(
        execution_task=task,
        agent_list=agent_list,
        task_id = index,
        ko_sp_1 = ko_sp_1,
        ko_sp_2 = ko_sp_2,
        save = save, 
        topic = topic,
        key = key
    )

    new_builder.clear_all_agents()
  
