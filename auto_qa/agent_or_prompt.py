prompt_for_query_analizer = """
주어진 질문에서 오직 실제 작업을 요구하는 주요 태스크만을 세부 질문(sub-questions)으로 추출하세요. 각 세부 질문에 대해 필요한 RAG 에이전트를 "agents" 키로 리스트에 포함시켜 JSON 형식으로 반환해주세요.

결과 형식:    
[
  {
    "question": "세부 질문 1",
    "agents": ["Agent1", "Agent2", ...]
  },
  {
    "question": "세부 질문 2",
    "agents": ["Agent1", "Agent2", ...]
  },
  ...    
]

참고사항:
1. 오직 실제 작업을 수행하는 주요 태스크만 세부 질문으로 추출하세요. 예를 들어, "QA 쌍 추출", "요약", "재작성" 등의 실제 작업만 포함합니다.
2. 각 세부 질문에 대해, 해당 질문에 답하기 위해 필요한 에이전트들을 선택하여 "agents" 리스트에 포함시켜주세요.
3. 사용 가능한 에이전트 목록은 user_query에 제공됩니다. 이 목록에서만 에이전트를 선택하세요.
4. 피드백이나 개선에 관한 명시적인 요청이 있는 경우에만 다음과 같이 처리하세요:
   - 해당 작업과 피드백 및 향상 과정을 하나의 세부 질문으로 통합하세요.
   - 각 세부 질문에 대해 주 에이전트(예: GeneralQAExpert, GeneralSummaryExpert, GeneralRewriteExpert)를 선택하세요.
   - ComprehensiveFeedbackExpert를 해당 세부 질문에 추가하세요.
   - 각 세부 질문의 특성에 맞는 개선 에이전트(예: QAImprovementExpert, SummaryImprovementExpert, RewriteImprovementExpert)를 추가하세요.
5. 피드백이나 개선에 관한 명시적인 요청이 없는 경우:
   - 각 세부 질문에 대해 가장 적합한 주 에이전트만 선택하세요.
   - ComprehensiveFeedbackExpert, QAImprovementExpert, SummaryImprovementExpert, RewriteImprovementExpert 등의 피드백 및 개선 관련 에이전트는 포함하지 마세요.
6. 다음 유형의 요청은 절대 세부 질문으로 포함하지 마세요:
   - 질문 분석에 관한 요청
   - 에이전트 선택에 관한 요청
   - 메타 태스크나 프로세스에 관한 요청
7. 중복되는 작업이나 피드백 과정을 별도의 세부 질문으로 분리하지 마세요. 대신, 관련된 작업과 피드백을 하나의 세부 질문으로 통합하세요

이 지침을 철저히 따라 주어진 질문을 분석하고 결과를 제공해주세요. 실제 작업을 요구하는 주요 태스크 외의 모든 요청은 무시하고 결과에 포함하지 마세요.
"""
