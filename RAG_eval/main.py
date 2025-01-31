import argparse
import json

from dotenv import load_dotenv
import os
import re
from openai import OpenAI
load_dotenv()

from deepeval.test_case import LLMTestCase
from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric, FaithfulnessMetric, HallucinationMetric, KnowledgeRetentionMetric, GEval, ToxicityMetric, BiasMetric
from datasets import Dataset
from ragas.metrics import answer_similarity
from summary_eval import *
from answer_correctness_eval import *
from custom_eval import *

def answer_relevancy(prompt, LLM_response, model):
    from deepeval import evaluate
    answer_relevancy_metric = AnswerRelevancyMetric(
                                  threshold=0.7,
                                  model = model,
                                  include_reason=True)
    test_case = LLMTestCase(
                   input = prompt,
                   actual_output = LLM_response)
                   
    answer_relevancy_metric.measure(test_case)
    score = answer_relevancy_metric.score
    reason = answer_relevancy_metric.reason
    
    return score, reason


def contextual_precision(prompt, LLM_response, expected_answer, retrieval_context, model):
    from deepeval import evaluate
    contextual_precision_metric = ContextualPrecisionMetric(
                                      threshold=0.7,
                                      model = model,
                                      include_reason = True)
    test_case = LLMTestCase(
                   input = prompt,
                   actual_output = LLM_response,
                   expected_answer = expected_answer,
                   retrieval_context = retrieval_context)
    contextual_precision_metric.measure(test_case)
    score = contextual_precision_metric.score
    reason = contextual_precision_metric.reason
    
    return score, reason
    
    
def contextual_recall(prompt, LLM_response, expected_answer, retrieval_context, model):
    from deepeval import evaluate
    contextual_recall_metric = ContextualRecallMetric(
                                     threshold=0.7,
                                     model = model,
                                     include_reason = True)
    test_case = LLMTestCase(
                   input = prompt,
                   model = model,
                   include_reason = True)
    contextual_recall_metric.measure(test_case)
    score = contextual_recall_metric.score
    reason = contextual_recall_metric.reason
    
    return score, reason


def contextual_relevancy(prompt, LLM_response, retrieval_context, model):
    from deepeval import evaluate
    contextual_relevancy_metric = ContextualRelevancyMetric(
                                     threshold=0.7,
                                     model = model,
                                     include_reason = True)
    test_case = LLMTestCase(
                   input = prompt,
                   actual_output = LLM_response,
                   retrieval_context = retrieval_context)
    contextual_relevancy_metric.measure(test_case)
    score = contextual_relevancy_metric.score
    reason = contextual_relevancy_metric.reason
    
    return score, reason

def faithfulness(prompt, LLM_response, retrieval_context, model):
    from deepeval import evaluate
    faithfulness_metric = FaithfulnessMetric(threshold=0.7,
                                             model = model,
                                             include_reason = True)
    test_case = LLMTestCase(input=prompt, actual_output = LLM_response, retrieval_context = retrieval_context, model = model)
    faithfulness_metric.measure(test_case)
    score = faithfulness_metric.score
    reason = faithfulness_metric.reason
    
    return score, reason

def hallucination(prompt, LLM_response, context):
    from deepeval import evaluate
    hallucination_metric = HallucinationMetric(threshold=0.5)
    test_case = LLMTestCase(input=prompt,
                            actual_output = LLM_response,
                            context = context)
    hallucination_metric.measure(test_case)
    score = hallucination_metric.score
    reason = hallucination_metric.reason
    
    return score, reason
    
    
def knowledge_retention(messages, model):
    from deepeval import evaluate
    from deepeval.test_case import ConversationalTestCase
    
    messages_info = []
    for i in range(len(messages)):
        mes = messages[i]
        prompt = mes[0]
        LLM_response = mes[1]
        messages_info.appned(LLMTestCase(input=prompt, actual_output = LLM_response))
    test_case = ConversationalTestCase(messages = messages_info)
    knowledge_retention_metric = KnowledgeRetentionMetric(threshold=0.5, model = model)
    
    knowledge_retention_metric.measure(test_case)
    score = knowledge_retention_metric.score
    reason = knowledge_retention_metric.reason
    
    return score, reason

def summarization(prompt, LLM_response, model):
    from deepeval import evaluate
    summarization_metric = SummarizationMetric(
                               threshold=0.5,
                               model = model
                               )
    test_case = LLMTestCase(input = prompt, actual_output = LLM_response)
    summarization_metric.measure(test_case)
    score = summarization_metric.score
    reason = summarization_metric.reason
    
    return score, reason


def toxicity(prompt, LLM_response, model):
    from deepeval import evaluate
    toxicity_metric = ToxicityMetric(threshold=0.5, model = model)
    test_case = LLMTestCase(input = prompt,
                            actual_output = LLM_response)
    toxicity_metric.measure(test_case)
    score = toxicity_metric.score
    reason = toxicity_metric.reason
    
    return score, reason

def bias(prompt, LLM_response, model):
    from deepeval import evaluate
    bias_metric = BiasMetric(threshold=0.5, model = model)
    test_case = LLMTestCase(input = prompt,
                            actual_output = LLM_response)
    score = bias_metric.score
    reason = bias_metric.reason
    
    return score, reason

def g_eval(name, criteria, evaluation_step, prompt, LLM_response, context, retrieval_context):
    from deepeval import evaluate
    test_case = LLMTestCase(input=prompt, actual_output = LLM_response, context = context, retrieval_context = retrieval_context)
    params = [test_case.INPUT, test_case.ACTUAL_OUTPUT, test_case.CONTEXT, test_case.RETRICAL_CONTEXT]
    custom_metric = GEval(name = name,
                          model = model,
                          threshold = 0.5,
                          evaluation_step = evaluation_step,
                          evaluation_params = params)
    score = custom_metric.score
    reason = custom_metric.reason
    
    return score, reason

def answer_similarity(prompt, LLM_response, context, expected_answer, model):
    from ragas import evaluate
    data_samples = {'question':prompt, 'answer':LLM_response, 'contexts':context, 'ground_truth':expected_answer}
    dataset = Dataset.from_dict(data_samples)
    
    score = evaluate(dataset,metrics=[answer_similarity], model = model)
    
    return score

def answer_correctness(prompt, LLM_response, context, expected_answer, model):
    from ragas import evaluate
    data_samples = {'question':prompt, 'answer':LLM_response, 'contexts':context, 'ground_truth':expected_answer}
    dataset = Dataset.from_dict(data_samples)
    
    score = cus_evaluate(dataset,metrics=[answer_correctness], model = model)
    
    return score
    
# 함수 매핑 딕셔너리와 각 함수에 필요한 파라미터 리스트, 그리고 반환 값 형식
function_map = {
    "answer_relevancy": (answer_relevancy, ["prompt", "LLM_response", "model"], ("score", "reason")),
    "contextual_precision": (contextual_precision, ["prompt", "LLM_response", "expected_answer", "retrieval_context", "model"], ("score", "reason")),
    "contextual_recall": (contextual_recall, ["prompt", "LLM_response", "expected_answer", "retrieval_context", "model"], ("score", "reason")),
    "contextual_relevancy": (contextual_relevancy, ["prompt", "LLM_response", "retrieval_context", "model"], ("score", "reason")),
    "faithfulness": (faithfulness, ["prompt", "LLM_response", "retrieval_context", "model"], ("score", "reason")),
    "hallucination": (hallucination, ["prompt", "LLM_response", "context"], ("score", "reason")),
    "knowledge_retention": (knowledge_retention, ["messages", "model"], ("score", "reason")),
    "summarization": (summarization, ["prompt", "LLM_response", "assessment_question", "model"], ("score", "reason")),
    "toxicity": (toxicity, ["prompt", "LLM_response", "model"], ("score", "reason")),
    "bias": (bias, ["prompt", "LLM_response", "model"], ("score", "reason")),
    "g_eval": (g_eval, ["name", "criteria", "evaluation_step", "prompt", "LLM_response", "context", "retrieval_context"], ("score", "reason")),
    "answer_similarity": (answer_similarity, ["prompt", "LLM_response", "context", "expected_answer", "model"], ("score",)),
    "answer_correctness": (answer_correctness, ["prompt", "LLM_response", "context", "expected_answer", "model"], ("score",))
}

def parse_and_call_functions(mod_string, **kwargs):
    results = {}
    
    # mod_string에서 함수 이름들을 추출합니다
    function_names = re.findall(r'\b(\w+)\b', mod_string)
    
    for func_name in function_names:
        if func_name in function_map:
            func, required_params, return_format = function_map[func_name]
            
            # 필요한 파라미터만 선택합니다
            params = {param: kwargs.get(param) for param in required_params if param in kwargs}
            
            # 필요한 모든 파라미터가 제공되었는지 확인합니다
            if all(param in params for param in required_params):
                # 함수를 호출하고 결과를 저장합니다
                result = func(**params)
                
                # 반환 값을 형식에 맞게 정리합니다
                if isinstance(result, tuple) and len(result) == len(return_format):
                    results[func_name] = dict(zip(return_format, result))
                else:
                    results[func_name] = {"error": "Unexpected return format"}
            else:
                missing_params = [param for param in required_params if param not in params]
                results[func_name] = {"error": f"Missing parameters: {', '.join(missing_params)}"}
    
    return results
    
def main():
    parser = argparse.ArgumentParser(description='RAG 성능 테스트 모듈입니다. 디테일한 설명은 https://docs.confident-ai.com/docs/metrics-introduction를 참고하세요')
    parser.add_argument('--key', type=str, help='키를 입력하세요.', required=True)
    parser.add_argument('--mod', type=str, help='''평가 항목을 입력하세요. 아래 항목 중에 여러개를 string으로 골라도 됩니다. ex. answer_relevancy contextual_precision hallucination
가능 리스트:
- answer_relevancy(prompt, LLM_response, model 필요) - score, reason
- contextual_precision(prompt, LLM_response, expected_answer, retrieval_context, model 필요) - score, reason
- contextual_recall(prompt, LLM_response, expected_answer, retrieval_context, model 필요) - score, reason
- contextual_relevancy(prompt, LLM_response, retrieval_context, model 필요) - score, reason
- faithfulness(prompt, LLM_response, retrieval_context, model 필요) - score, reason
- hallucination(prompt, LLM_response, context 필요) - score, reason
- knowledge_retention(messages, model 필요) - score, reason
- summarization(prompt, LLM_response, assessment_qeustion, model 필요) - score, reason
- toxicity(prompt, LLM_response, model 필요) - score, reason
- bias(prompt, LLM_response, model 필요) - score, reason
- g_eval(name, criteria, evaluation_step, prompt, LLM_response, context, retrieval_context 필요) - score, reason
- answer_similarity(prompt, LLM_response, context, expected_answer, model 필요) - score
- answer_correctness(prompt, LLM_response, context, expected_answer, model 필요) - score''', required=True)
    parser.add_argument('--prompt', type=str, help='프롬프트를 입력하세요', default=None)
    parser.add_argument('--response', type=str, help='LLM 결과 값을 입력하세요', default=None)
    parser.add_argument('--gt', type=str, help='실제로 나와야 하는 결과 값(gt)를 입력하세요', default=None)
    parser.add_argument('--context', type=str, help='본문 내용을 입력하세요')
    parser.add_argument('--model', type=str, help='모델 명을 입력하세요', default='gpt-4o')
    parser.add_argument('--retrive', type=str, help='검색된 데이터를 입력하세요', default=None)
    parser.add_argument('--message', type=str, help='LLM과 대화한 모든 내용을 list로 넣어주세요. [[prompt_1, LLM_answer], [prompt_2, LL_answer_2], ... , [prompt_n, LL_answer_n]] 입니다.', default=None)
    parser.add_argument('--assessment_question', type=str, help='LLM 요약을 평가하는 추가적인 기준을 질문 형태로 넣어주세요. 안 넣어으면 자동으로 생성 돼요. 해당 argument에 대한 구체적인 설명입니다 - A list of close-ended questions that can be answered with either a yes or a no. These are questions you want your summary to be able to ideally answer, and is especially helpful if you already know what a good summary for your use case looks like. ex) Does a higher score mean a more comprehensive summary?', default=None)
    parser.add_argument('--name' , type=str, help='COT 형식으로 평가 하기 위해 필요한 변수입니다. 하고자 하는 작업의 이름을 입력하세요. 해당 변수는 평가 매트릭스의 이름이 됩니다. ex) Correctness', default=None)
    parser.add_argument('--criteria', type=str, help='COT 형식으로 평가 하기 위해 필요한 변수입니다. 평가 매트릭스에서 정확하게 무엇을 평가하고 싶은지를 구체적으로 적으시면 됩니다. ex) Determine whether the actual output is factually correct based on the expected output', default=None)
    parser.add_argument('--eval_step', type=str, help='COT 형식으로 평가 하기 위해 필요한 변수입니다. 구체적인 프로세스를 입력하시면 됩니다. ex) Check whether the facts in actual output contradicts any facts in expected output, You should also heavily penalize omission of detail, Vague language, or contradicting OPINIONS, are OK', default=None)
    parser.add_argument('--save', type=str, help='저장 폴더 경로를 입력해주세요')
    args = parser.parse_args()
    
    os.environ["OPENAI_API_KEY"] = args.key
    client = OpenAI()
    
    prompt = args.prompt
    response = args.response
    gt = args.gt
    model = args.model
    retive = args.retrive
    message = args.message
    context = args.context
    assessment_qeustion = args.assessment_question
    name = args.name
    criteria = args.criteria
    eval_step = args.eval_step
    mode = args.mod
    save = args.save
    
    params = {
             "prompt": prompt,
              "LLM_response": response,
              "model": model,
              "expected_answer": gt,
              "retrieval_context": retive,
              "context": context,
              "messages": message,
              "assessment_question": assessment_qeustion,
              "name": name,
              "criteria": criteria,
              "evaluation_step": eval_step
              }
              
    final_result = []
    results = parse_and_call_functions(mode, **params)
    for func_name, result in results.items():
        sub_result = f"{func_name}: {result}"
        final_result.append(sub_result)
    
    print(final_result)
    with open(save + '/result.json', 'w') as f:
         json.dump(final_result, f)
         
if __name__ == '__main__':
   main()
