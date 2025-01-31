# QA expert list
qa_agent_roles = {
    "GeneralQAExpert": "일반 주제 QA 생성 전문가",
    "TechnicalQAExpert": "기술 문서 QA 생성 전문가",
    "EducationalQAExpert": "교육용 QA 생성 전문가",
    "AnalyticalQAExpert": "분석적 사고 훈련용 QA 생성 전문가"
}

# summary expert list
summary_agent_roles = {
    "GeneralSummaryExpert": "일반 텍스트 요약 전문가",
    "AcademicSummaryExpert": "학술 논문 요약 전문가",
    "NewsSummaryExpert": "뉴스 기사 요약 전문가",
    "TechnicalSummaryExpert": "기술 문서 요약 전문가"
}

# rewirte expert list
rewrite_agent_roles = {
    "GeneralRewriteExpert": "일반 텍스트 재작성 전문가",
    "SEOContentRewriteExpert": "SEO 최적화 콘텐츠 재작성 전문가",
    "AcademicRewriteExpert": "학술 텍스트 재작성 전문가",
    "CreativeRewriteExpert": "창의적 글쓰기 재작성 전문가"
}

# feedback expert list
feedback_agent_roles = {
    "QAFeedbackExpert": "QA셋 평가 및 피드백 전문가",
    "SummaryFeedbackExpert": "요약문 평가 및 피드백 전문가",
    "RewriteFeedbackExpert": "재작성 텍스트 평가 및 피드백 전문가",
    "ComprehensiveFeedbackExpert": "종합적 콘텐츠 평가 및 피드백 전문가"
}

# improvement expert list
feedback_reflection_agent_roles = {
    "QAImprovementExpert": "QA셋 개선 전문가",
    "SummaryImprovementExpert": "요약문 개선 전문가",
    "RewriteImprovementExpert": "재작성 텍스트 개선 전문가",
    "ComprehensiveImprovementExpert": "종합적 콘텐츠 개선 전문가"
}


import getpass
import os

#os.environ['OPENAI_API_KEY'] = "sk-...."
os.environ['OPENAI_API_KEY'] = 'sk-....'
from openai import OpenAI
import autogen
from json_ko import repair_json
import json
from agent_prompt import *

client = OpenAI()

config = [{"model":"gpt-4o", "api_key":"sk-...", "tags":["gpt-4o", "tool"]}]

def merge_dynamic_lists(lists):
    merged_list = []
    for lst in lists:
        merged_list.extend(lst)
    return merged_list
    
def generate_agent_prompt(sys_templet, desc_templet, role_list):
    pos = []
    role = []

    for k,v in role_list.items():
        pos.append(k)
        role.append(v)

    agent_prompt = []
    build_manager = autogen.OpenAIWrapper(config_list=config)
    for i in range(len(pos)):
        position_=pos[i]
        role_ = role[i]
        resp_agent_sys_msg = (
            build_manager.create(
                messages=[
                    {
                        "role": "user",
                        "content": sys_templet.format(
                            position=position_, role = role_),
                    }]).choices[0].message.content)
        resp_desc_msg = (
            build_manager.create(
                messages=[
                    {
                        "role": "user",
                        "content": desc_templet.format(
                            position=position_,
                        instruction=resp_agent_sys_msg,
                        ),
                    }]).choices[0].message.content)
        agent_prompt.append({"name": position_, "system_message": resp_agent_sys_msg, "description": resp_desc_msg})

    return agent_prompt 
    
    
qa_sys_msg_list = generate_agent_prompt(qa_AGENT_SYS_MSG_PROMPT, qa_AGENT_DESC_PROMPT, qa_agent_roles)
su_sys_msg_list = generate_agent_prompt(su_AGENT_SYS_MSG_PROMPT, su_AGENT_DESC_PROMPT, summary_agent_roles)
re_sys_msg_list = generate_agent_prompt(re_AGENT_SYS_MSG_PROMPT, re_AGENT_DESC_PROMPT, rewrite_agent_roles)
fe_sys_msg_list = generate_agent_prompt(fe_AGENT_SYS_MSG_PROMPT, fe_AGENT_DESC_PROMPT, feedback_agent_roles)
im_sys_msg_list = generate_agent_prompt(im_AGENT_SYS_MSG_PROMPT, im_AGENT_DESC_PROMPT, feedback_reflection_agent_roles)

gen_agent = []

if qa_sys_msg_list is not None:
    gen_agent.append(qa_sys_msg_list)
if su_sys_msg_list is not None:
    gen_agent.append(su_sys_msg_list)
if re_sys_msg_list is not None:
    gen_agent.append(re_sys_msg_list)
if fe_sys_msg_list is not None:
    gen_agent.append(fe_sys_msg_list)
if im_sys_msg_list is not None:
    gen_agent.append(im_sys_msg_list)
    
all_agent = merge_dynamic_lists(gen_agent)
json.dump(all_agent, open("./agent_library.json", "w"), ensure_ascii=False, indent=4)
 
