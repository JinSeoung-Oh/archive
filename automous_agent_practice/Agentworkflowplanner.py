from typing import Optional, Dict, Any, List
import json
import asyncio
import aiohttp
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    # 다른 프로바이더 추가 가능

@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Dict[str, int]
    created_at: datetime

class LLMError(Exception):
    def __init__(self, message: str, provider: ModelProvider, details: Optional[Dict] = None):
        self.message = message
        self.provider = provider
        self.details = details or {}
        super().__init__(self.message)

class BaseLLMClient(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        pass

    @abstractmethod
    async def generate_json(self, prompt: str, **kwargs) -> Dict:
        pass

class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": kwargs.get("model", self.model),
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant focused on generating code and specifications for autonomous agents."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 4000)
            }
            
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        raise LLMError(
                            f"OpenAI API error: {error_data.get('error', {}).get('message', 'Unknown error')}",
                            ModelProvider.OPENAI,
                            error_data
                        )
                    
                    result = await response.json()
                    
                    return LLMResponse(
                        content=result["choices"][0]["message"]["content"],
                        model=result["model"],
                        usage=result["usage"],
                        created_at=datetime.fromtimestamp(result["created"])
                    )
                    
            except aiohttp.ClientError as e:
                raise LLMError(f"Network error: {str(e)}", ModelProvider.OPENAI)
            except Exception as e:
                raise LLMError(f"Unexpected error: {str(e)}", ModelProvider.OPENAI)

    async def generate_json(self, prompt: str, **kwargs) -> Dict:
        response = await self.generate(prompt, **kwargs)
        try:
            if isinstance(response, LLMResponse):
                response_content = response.content  # LLMResponse 객체에서 content 추출
            else:
                response_content = response
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            raise LLMError(
                f"Failed to parse JSON response: {str(e)}", 
                ModelProvider.ANTHROPIC,
                {"response_content": response_content}
            )

class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-latest"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": kwargs.get("model", self.model),
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": kwargs.get("max_tokens", 4000)
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status != 200:
                        error_data = await response.json()
                        raise LLMError(
                            f"Anthropic API error: {error_data.get('error', {}).get('message', 'Unknown error')}",
                            ModelProvider.ANTHROPIC,
                            error_data
                        )
                    
                    result = await response.json()
                    
                    return LLMResponse(
                        content=result["content"][0]["text"],
                        model=result["model"],
                        usage={
                            "prompt_tokens": result.get("usage", {}).get("input_tokens", 0),
                            "completion_tokens": result.get("usage", {}).get("output_tokens", 0)
                        },
                        created_at=datetime.now()  # Anthropic doesn't provide creation timestamp
                    )
                    
        except aiohttp.ClientError as e:
            raise LLMError(f"Network error: {str(e)}", ModelProvider.ANTHROPIC)
        except Exception as e:
            raise LLMError(f"Unexpected error: {str(e)}", ModelProvider.ANTHROPIC)

    async def generate_json(self, prompt: str, **kwargs) -> Dict:
        response = await self.generate(prompt, **kwargs)
        try:
            # LLMResponse 객체에서 content 추출
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            raise LLMError(
                f"Failed to parse JSON response: {str(e)}", 
                ModelProvider.ANTHROPIC,
                {"response_content": response.content}
            )

class LLMClient:
    def __init__(self, 
                 provider: ModelProvider = ModelProvider.OPENAI,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 fallback_provider: Optional[ModelProvider] = None,
                 fallback_api_key: Optional[str] = None):
        
        self.primary_client = self._create_client(provider, api_key, model)
        self.fallback_client = None
        if fallback_provider and fallback_api_key:
            self.fallback_client = self._create_client(fallback_provider, fallback_api_key)
        
        self.retry_count = 3
        self.retry_delay = 1  # seconds

    def _create_client(self, 
                      provider: ModelProvider, 
                      api_key: Optional[str],
                      model: Optional[str] = None) -> BaseLLMClient:
        
        if api_key is None:
            raise ValueError(f"API key is required for {provider.value}")
            
        if provider == ModelProvider.OPENAI:
            return OpenAIClient(api_key, model or "gpt-4o")
        elif provider == ModelProvider.ANTHROPIC:
            return AnthropicClient(api_key, model or "claude-3-5-sonnet-latest")
        else:
            raise ValueError(f"Unsupported provider: {provider.value}")

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        last_error = None
        
        for attempt in range(self.retry_count):
            try:
                return await self.primary_client.generate(prompt, **kwargs)
            except LLMError as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if self.fallback_client and attempt == self.retry_count - 1:
                    try:
                        logger.info("Trying fallback provider...")
                        return await self.fallback_client.generate(prompt, **kwargs)
                    except LLMError as fallback_error:
                        last_error = fallback_error
                        
                await asyncio.sleep(self.retry_delay * (attempt + 1))
                
        raise last_error

    async def generate_json(self, prompt: str, **kwargs) -> Dict:
        last_error = None
        
        for attempt in range(self.retry_count):
            try:
                return await self.primary_client.generate_json(prompt, **kwargs)
            except LLMError as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if self.fallback_client and attempt == self.retry_count - 1:
                    try:
                        logger.info("Trying fallback provider...")
                        return await self.fallback_client.generate_json(prompt, **kwargs)
                    except LLMError as fallback_error:
                        last_error = fallback_error
                        
                await asyncio.sleep(self.retry_delay * (attempt + 1))
                
        raise last_error

  from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import logging
from datetime import datetime
from json_repair import repair_json
import ast
from itertools import chain

class Agent_planner:
    def __init__(self, llm_client):
        self.llm = llm_client

    async def build_plan(self, task_info: dict, agent_info: dict, context: str) -> Dict:
        prompt = f"""You are a Workflow Planning Agent. Your role is to create a comprehensive workflow plan by matching tasks with the appropriate agents from the GIVEN LIST of available agents.

        # tasks
        {task_info}
        # A Fixed List of Available Agents
        {agent_info}

        1. Core Rules
          - You MUST ONLY select agents from the provided agent_info list
          - You MUST NOT create any new agents
          - You MUST NOT modify any agent's capabilities or roles
          - Each selected agent MUST have matching responsibilities in their documented role
          - You MUST include ALL necessary agents required to complete EVERY aspect of the tasks
          - Each task MUST be fully covered by the selected agents' responsibilities
          - No task requirements should be left unaddressed
          - The SAME agent CAN appear MULTIPLE times if their skills are needed at different stages
          - There is NO LIMIT on how many times an agent can be used in a workflow

        2. Analysis Process
          For each task:
          a) Detailed Task Breakdown:
             - Break down each task into ALL required sub-tasks and components
             - Identify EVERY specific action, tool, and skill needed
             - Ensure no task aspects are overlooked
             - Consider if same agent needs to perform multiple steps
  
          b) Comprehensive Agent Matching:
             - Match EACH sub-task with ALL relevant agents
             - Check agent responsibilities against EVERY task requirement
             - Ensure complete coverage of task requirements
             - Include supporting agents needed for task completion
             - Consider dependencies between tasks and agents
             - Don't hesitate to use same agent multiple times if needed
  
          c) Verification Per Task:
             - Review if ALL components of THIS SPECIFIC task are addressed
             - Confirm EVERY required skill and tool for THIS task is covered
             - Validate that no essential agents for THIS task are missing
             - Double-check for completeness of THIS task's workflow
             - Ensure agents selected can handle THIS task's specific requirements
             - Verify if repeated agent appearances are necessary and appropriate

        3. Output Format
          - Return a LIST OF LISTS, where each inner list represents agents for a specific task:
            [
              [Agent1, Agent2, Agent3, Agent1],  # Agents for Task 1 (Agent1 appears twice)
              [Agent2, Agent4, Agent1, Agent2],  # Agents for Task 2 (Agent2 appears twice)
              [Agent3, Agent1, Agent5, Agent3]   # Agents for Task 3 (Agent3 appears twice)
            ]
          - Each inner list MUST include ALL agents needed for that specific task's completion
          - The order within each inner list should reflect the workflow sequence for that task
          - Each task's agent list should be complete and independent
          - The SAME agent CAN appear multiple times in the same inner list if needed
          - The SAME agent CAN appear across different task lists as required
          - "Return your response as a valid JSON array, with all names in quotes"

        CRITICAL:
        - You MUST ensure NO required agents are omitted from any task's list
        - Each task MUST have its own complete list of required agents
        - Each task's agent list MUST be verified for completeness independently
        - Agents CAN and SHOULD be repeated when their skills are needed multiple times
        - Return ONLY the nested list structure without additional text
        - Your response should ONLY contain the list of lists in the specified format
        - Each inner list MUST fully address its corresponding task's requirements
        """

        response = await self.llm.generate(prompt)
        analysis = response.content if hasattr(response, 'content') else response

        return analysis

with open('./task.json', 'r', encoding='utf-8') as f:
  task_info = json.load(f)
    
with open('./complex_pipeline.json', 'r', encoding='utf-8') as f:
  agent_info = json.load(f)

for agent in agent_info:
    agent_ = agent['sub_agents']
    for info in agent_:
        print(info['Agent'])

  llm_client = LLMClient(
        provider=ModelProvider.ANTHROPIC,
        api_key="sk-...",
        fallback_provider=ModelProvider.OPENAI,
        fallback_api_key="sk-..."
    )

context = """
    *2020년 6월 축구경기 결과 *
    | 팀1 결과 | 팀2 |1차전 |2차전| 
    | 알제리 |5-3|튀니지 |2-1|3-2| 
    | 에티오피아 |0-5| 가나 | 0-2 | 0-3| 
    | 코트디부아르|3-3(a)| 적도 기니 |1-1|2-2|

    알제리와 튀니지, 치열한 경기 끝에 알제리 승리 알제리와 튀니지는 5-3의 스코어로 알제리가 승리했다. 첫 번째 경기에서는 2-1로 이겼으며, 두 번째 경기에서는 3-2로 승리를 거두며 두 경기 모두에서 우위를 점했다. 양 팀은 치열한 경기를 펼쳤지만, 알제리가 결정적 순간마다 득점을 올리며 최종 승리를 거머쥐었다.
    가나, 에티오피아 상대로 완승 가나는 에티오피아와의 두 경기 합산 5-0의 압도적인 스코어로 승리했다. 첫 번째 경기에서는 1-0으로 승리했으며, 두 번째 경기에서는 3-0으로 에티오피아를 완벽하게 제압했다. 가나의 공격력과 수비력 모두 훌륭한 경기였다.
    코트디부아르와 적도 기니, 치열한 접전 끝에 무승부 코트디부아르와 적도 기니는 두 경기 합산 3-3의 무승부를 기록했다. 첫 번째 경기에서는 1-1로 비겼고, 두 번째 경기에서도 2-2로 팽팽한 승부를 펼쳤다. 원정 다득점 원칙에 따라 코트디부아르가 다음 라운드에 진출하게 되었다.
"""
planner = Agent_planner(llm_client)
result = await planner.build_plan(task_info, agent_info, context)
print(result)
