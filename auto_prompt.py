import anthropic
import os
import http.client
import requests
import re
import requests
import time
import json
from json_repair import repair_json

an_cluade = "sk-..."

INIT_PROMPT_TEMPLATE = """
당신은 AI 어시스턴트를 위한 초기 프롬프트를 만드는 숙련된 프롬프트 엔지니어입니다.
Prompt 엔지니어로서, 당신의 역할은 고품질의 지시문을 설계하고 작성함으로써 다양한 AI 모델과 시스템의 성능을 최적화하는 것입니다. 
당신은 애플리케이션과 플랫폼 또는 사용자의 특정 요구사항에 맞는 프롬프트를 정의하고, 이를 통해 AI 모델의 반응을 유도하며, 예상되는 출력을 보장해야 합니다.
정확하고 명확한 지시문을 통해 AI의 응답 품질을 극대화하는 능력은 핵심입니다. 다양하고 복잡한 문제를 해결하기 위해 AI 시스템과 인간 사용자 간의 원활한 상호작용을 촉진하기 위해,
컨텍스트를 적절히 설정하고 사용자 요구사항을 반영하며, 의도를 명확하게 전달하는 기술이 중요합니다.
주어진 사용자 요구사항을 바탕으로 명확하고 간결하며 효과적인 프롬프트를 만드는 것이 목표입니다.

사용자 요구사항: {user_requirements}

이 요구사항을 충족하는 프롬프트를 생성해주세요.
사용자의 요구사항을 세밀하게 분석하여 해당 요구사항을 충족시키기 위한 프롬프트를 작성해주세요.
프롬프트는 구체적이고 실행 가능해야 하며, AI로부터 원하는 행동을 이끌어낼 수 있도록 설계되어야 합니다.

반드시 한국어로 작성해 주세요.

초기 프롬프트:
"""

IMPROVE_PROMPT_TEMPLATE = """
당신은 전문 프롬프트 엔지니어입니다. 현재 프롬프트를 사용자의 피드백을 바탕으로 개선하는 것이 당신의 임무입니다.
Prompt 엔지니어로서, 당신의 역할은 고품질의 지시문을 설계하고 작성함으로써 다양한 AI 모델과 시스템의 성능을 최적화하는 것입니다. 
당신은 애플리케이션과 플랫폼 또는 사용자의 특정 요구사항에 맞는 프롬프트를 정의하고, 이를 통해 AI 모델의 반응을 유도하며, 예상되는 출력을 보장해야 합니다.
정확하고 명확한 지시문을 통해 AI의 응답 품질을 극대화하는 능력은 핵심입니다. 다양하고 복잡한 문제를 해결하기 위해 AI 시스템과 인간 사용자 간의 원활한 상호작용을 촉진하기 위해,
컨텍스트를 적절히 설정하고 사용자 요구사항을 반영하며, 의도를 명확하게 전달하는 기술이 중요합니다.
현재 프롬프트와 사용자의 피드백을 주의 깊게 분석한 후, 개선된 버전을 제공해 주세요.

현재 프롬프트: {current_prompt}

사용자 피드백: {feedback}

사용자의 피드백을 고려하여 프롬프트를 개선해 주세요. 개선된 프롬프트는 명확하고 구체적이어야 하며, 사용자의 의도에 부합해야 합니다.

반드시 한국어로 작성해 주세요.

개선된 프롬프트:
"""

def init_generate_prompt(user_requirements):
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": INIT_PROMPT_TEMPLATE.format(user_requirements=user_requirements)}
        ]
    )
    return response.content[0].text

def improve_prompt(current_prompt, feedback):
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": IMPROVE_PROMPT_TEMPLATE.format(current_prompt=current_prompt, feedback=feedback)}
        ]
    )
    return response.content[0].text

def prompt_improvement_loop():
    user_requirements = input("AI 어시스턴트에 대한 요구사항을 입력하세요: ")
    current_prompt = init_generate_prompt(user_requirements)
    print(f"초기 프롬프트: {current_prompt}")
    
    while True:
        feedback = input("피드백을 입력하세요 ('종료'를 입력하면 프로그램이 종료됩니다): ")
        if feedback.lower() == '종료':
            break
        
        current_prompt = improve_prompt(current_prompt, feedback)
        print(f"업데이트된 프롬프트: {current_prompt}")

  prompt_improvement_loop()
