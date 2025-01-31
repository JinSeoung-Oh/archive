import re
import requests
import time
import json
from json_repair import repair_json
import os
import anthropic
import base64
import cv2

def find_all_folder_paths(directory):
    folder_list = []
    items = os.listdir(directory)

    for item in items:
        item_path = os.path.join(directory, item)
        folder_list.append(item_path)
    
    return folder_list

def get_image_size(image_path):
    # 이미지 읽기
    img = cv2.imread(image_path)
    
    # 이미지가 제대로 로드되었는지 확인
    if img is None:
        raise Exception("이미지를 불러올 수 없습니다.")
    
    # 이미지의 높이, 너비, 채널 수 구하기
    h, w, c = img.shape
    
    return h, w

# Anthropic API 키 설정
client = anthropic.Anthropic(api_key="sk-ant-...")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_qa_from_image(image_path):
    base64_image = encode_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                },
                {
                    "type": "text",
                    "text": """이 이미지를 바탕으로 5개의 Q&A를 만들어주세요. 각 Q&A는 이미지의 다양한 측면을 다루어야 합니다.
반드시 다음 형식을 정확히 따라주세요:
[
    {"질문 (질문이 무엇을 묻고 있는지)": "답변"},
    {"질문 (질문이 무엇을 묻고 있는지)": "답변"},
    {"질문 (질문이 무엇을 묻고 있는지)": "답변"},
    {"질문 (질문이 무엇을 묻고 있는지)": "답변"},
    {"질문 (질문이 무엇을 묻고 있는지)": "답변"}
]
질문은 반드시 '질문 (무엇을 물어보는 것인지)'의 형식이어야 하며, 괄호 안에는 질문이 묻고 있는 내용을 자유롭게 명시해야 합니다. 예를 들어, '질문 (배경)', '질문 (객체)', '질문 (행동)', '질문 (분위기)', '질문 (세부사항)' 등으로 다양하게 표현할 수 있습니다.
답변은 해당 질문에 대해 매우 자세하게 작성해야 합니다.
반드시 이 형식을 지켜주세요. 다른 설명이나 추가 텍스트 없이 오직 요청된 JSON 형식만 반환해 주세요."""
                }
            ]
        }
    ]

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=2000,
        messages=messages
    )

    # 응답에서 JSON 부분만 추출
    json_match = re.search(r'\[.*\]', response.content[0].text, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            qa_list = json.loads(json_str)
            
            # 형식 검증
            if len(qa_list) != 5 or not all(len(qa) == 1 for qa in qa_list):
                raise ValueError("Invalid QA format")
            
            for qa in qa_list:
                question = list(qa.keys())[0]
                if not question.startswith("질문 (") or ")" not in question:
                    raise ValueError(f"Invalid question format: {question}")
            
            return json.dumps(qa_list, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, ValueError) as e:
            return f"Error: {str(e)}"
    else:
        return "Error: No JSON found in the response"

imgs_path = "./natural_sample"
check_result = find_all_folder_paths(imgs_path)

import re

all_ = []
id_ = 0

for index, path in enumerate(check_result):
    print(f"처리 중인 이미지 {index + 1}: {path}")
    print(f"현재 id_ 값: {id_}")

    result = generate_qa_from_image(path)
    result_ = repair_json(result)
    result__ = json.loads(result_)
    print(f"생성된 QA: {result__}")
    
    result = {}
    img_name = path.split('/')[-1]
    img_id = path.split('/')[-1].split('.')[-2]
    h, w = get_image_size(path)
    
    result['image'] = {
        'image_id': img_id,
        'file_name': img_name
    }
    result['width'] = w
    result['height'] = h
    result['type'] = "Natural scene"
    
    q_list = []
    a_list = []
    
    for i, response in enumerate(result__):
        for k, v in response.items():
            print(k)
            # 새로운 파싱 로직
            match = re.match(r'질문\s*\((.*?)\)\s*:?\s*(.*)', k)
            if match:
                type_info = match.group(1).strip()
                question = match.group(2).strip()
                if not question:  # 질문이 비어있으면 전체를 질문으로 취급
                    question = k
            else:
                type_info = "Unknown"
                question = k
            
            q_info = {"instruction_id": id_, "instruction_type": type_info, "instruction": question}
            a_info = {"instruction_id": id_, "answer": v}
            q_list.append(q_info)
            a_list.append(a_info)
            id_ += 1
        print(f"  질문-답변 쌍 {i + 1} 추가 완료, 현재 id_: {id_}")
    
    result['instructions'] = q_list
    result['generated'] = a_list
    
    all_.append(result.copy())
    print(f"이미지 {index + 1} 처리 완료. all_ 리스트 현재 길이: {len(all_)}")
    print(f"추가된 결과의 image_id: {result['image']['image_id']}")
    print("---")

print(f"처리 완료. 최종 id_ 값: {id_}")
print(f"all_ 리스트의 최종 길이: {len(all_)}")

# 결과 확인
for i, res in enumerate(all_):
    print(f"결과 {i + 1}의 image_id: {res['image']['image_id']}")
    print(f"결과 {i + 1}의 질문-답변 쌍 수: {len(res['instructions'])}")
    print("---")

# 결과를 JSON 파일로 저장
with open('./task_6_test_result.json', 'w', encoding='utf-8') as f:
    json.dump(all_, f, ensure_ascii=False, indent=4)

