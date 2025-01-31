from typing import Optional, Dict, List
import json
from datetime import datetime
import sqlite3
import aiosqlite
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class StorageClient:
    def __init__(self, db_path: str = "agent_system.db"):
        # 디렉토리 경로 확인 및 생성
        directory = os.path.dirname(db_path)
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """데이터베이스 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_systems (
                    id TEXT PRIMARY KEY,
                    spec TEXT,
                    implementation TEXT,
                    created_at TIMESTAMP,
                    status TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tools (
                    name TEXT PRIMARY KEY,
                    spec TEXT,
                    code TEXT,
                    created_at TIMESTAMP,
                    agent_id TEXT,
                    FOREIGN KEY(agent_id) REFERENCES agent_systems(id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_results (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT,
                    input_data TEXT,
                    result TEXT,
                    executed_at TIMESTAMP,
                    status TEXT,
                    FOREIGN KEY(agent_id) REFERENCES agent_systems(id)
                )
            """)

    async def save_agent_system(self, system_id: str, system_data: dict):
        """Agent 시스템 저장"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO agent_systems
                (id, spec, implementation, created_at, status)
                VALUES (?, ?, ?, ?, ?)
            """, (
                system_id,
                json.dumps(system_data['spec'].__dict__),
                system_data['implementation'],
                system_data['created_at'],
                'active'
            ))
            await db.commit()

    async def save_tool(self, tool_name: str, tool_data: dict):
        """Tool 저장"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO tools
                (name, spec, code, created_at, agent_id)
                VALUES (?, ?, ?, ?, ?)
            """, (
                tool_name,
                json.dumps(tool_data['spec'].__dict__),
                tool_data['code'],
                tool_data['created_at'],
                tool_data.get('agent_id')
            ))
            await db.commit()

    async def load_agent_system(self, system_id: str) -> Optional[dict]:
        """Agent 시스템 로드"""
        async with aiosqlite.connect(self.db_path) as db:
            # Agent 시스템 정보 로드
            async with db.execute(
                "SELECT * FROM agent_systems WHERE id = ?",
                (system_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                
                # 관련된 Tool들 로드
                tools = {}
                async with db.execute(
                    "SELECT * FROM tools WHERE agent_id = ?",
                    (system_id,)
                ) as tool_cursor:
                    tool_rows = await tool_cursor.fetchall()
                    for tool_row in tool_rows:
                        tools[tool_row[0]] = {
                            'spec': json.loads(tool_row[1]),
                            'code': tool_row[2],
                            'created_at': tool_row[3]
                        }
                
                return {
                    'id': row[0],
                    'spec': json.loads(row[1]),
                    'implementation': row[2],
                    'created_at': row[3],
                    'status': row[4],
                    'tools': tools
                }

    async def list_agents(self) -> List[Dict]:
        """모든 Agent 시스템 목록 조회"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT id, spec, status, created_at FROM agent_systems"
            ) as cursor:
                rows = await cursor.fetchall()
                return [{
                    'id': row[0],
                    'spec': json.loads(row[1]),
                    'status': row[3],
                    'created_at': row[4]
                } for row in rows]

    async def get_agent_tools(self, agent_id: str) -> List[Dict]:
        """특정 Agent의 Tool 목록 조회"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT name, spec, created_at FROM tools WHERE agent_id = ?",
                (agent_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [{
                    'name': row[0],
                    'spec': json.loads(row[1]),
                    'created_at': row[2]
                } for row in rows]

    async def save_execution_result(self, 
                                  agent_id: str, 
                                  input_data: dict, 
                                  result: dict):
        """실행 결과 저장"""
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO execution_results
                (id, agent_id, input_data, result, executed_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                execution_id,
                agent_id,
                json.dumps(input_data),
                json.dumps(result),
                datetime.now(),
                'completed'
            ))
            await db.commit()

    async def get_agent_execution_history(self, 
                                        agent_id: str, 
                                        limit: int = 10) -> List[Dict]:
        """Agent의 실행 히스토리 조회"""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("""
                SELECT id, input_data, result, executed_at, status 
                FROM execution_results 
                WHERE agent_id = ?
                ORDER BY executed_at DESC
                LIMIT ?
            """, (agent_id, limit)) as cursor:
                rows = await cursor.fetchall()
                return [{
                    'id': row[0],
                    'input_data': json.loads(row[1]),
                    'result': json.loads(row[2]),
                    'executed_at': row[3],
                    'status': row[4]
                } for row in rows]


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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ToolSpec:
    """Tool 구현 스펙"""
    name: str
    purpose: str
    parameters: Dict[str, Dict[str, str]]  # {"param_name": {"type": type, "description": desc}}
    return_type: Dict[str, str]  # {"type": type, "description": desc}
    example_implementation: str  # 실제 구현 예시 코드

@dataclass
class AgentSpec:
    """Agent 스펙 정의"""
    name: str
    role: str  # Agent의 주요 역할
    responsibilities: List[str]  # 담당 업무 목록
    tools: List[ToolSpec]  # 필요한 도구들
    proposal_prompt: str

class ToolImplementor:
    """Tool 구현 생성기"""
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def implement_tool(self, tool_spec: ToolSpec) -> str:
        """Tool 스펙을 바탕으로 실제 구현 코드 생성"""
        prompt = f"""Implement the tool based on this specification:

        Name: {tool_spec.name}
        Purpose: {tool_spec.purpose}
        Parameters: {json.dumps(tool_spec.parameters, indent=2)}
        Return Type: {json.dumps(tool_spec.return_type, indent=2)}

        The implementation should be a single async function that: 
        1. Takes the specified parameters directly 
        2. Returns the specified return type 
        3. Includes error handling and logging 
        4. Uses type hints

        Example format: 
        async def {tool_spec.name}(param1: type1, param2: type2) -> return_type: 
            # Implementation 
            return result 
        
        DO NOT include any class definitions or test cases. Return ONLY the function implementation. 
        AND DO NOT INCLUDE ANY EXTRA INFORMATION(LIKE pytho or '''). JUST RETURN ONLY THE FUNCTION IMPLEMENTATION.
        """
        
        response = await self.llm.generate(prompt)
        return response.content if hasattr(response, 'content') else response

class userqueryAnalyzer:
    """User requirement 분석"""
    def __init__(self, llm_client):
        self.llm = llm_client

    async def analyze_user_requirements(self, requirements: str, context: str) -> List:
        """
        User 요구사항을 분석하여 Task 도출
        """
        prompt = f"""당신은 Requirements Analysis Planner입니다. 주어진 Requirements와 Context 분석하여 개별적인 Core Task를 추출하고 각 Task의 세부사항을 정의하는 것이 당신의 역할입니다.
        
        Requirements: {requirements}
        Context: {context}
        
        분석 프로세스:
        1. Requirements와 Context를 주의 깊게 읽고 독립적으로 수행 가능한 모든 Core Task를 식별합니다.
        2. 각 Core Task에 대해 다음 정보를 명확하게 정의합니다.

        리턴 포맷:
        [
            {{
                "task_name": "Task의 명칭",
                "purpose": "Task의 목적",
                "detail": "Task 수행을 위한 세부 작업 목록",
                "limitation": "Task 수행 시 제약사항",
                "success_criteria": "Task 성공 판단 기준"
            }},
            {{
                "task_name": "다음 Task의 명칭",
                ...
            }}
        ]

        각 필드별 작성 가이드:
        1. task_name: 해당 Task를 명확하게 식별할 수 있는 간단한 이름
        2. purpose: Task가 왜 필요하고 무엇을 달성하고자 하는지 명확한 설명
        3. detail: Task가 수행해야 할 구체적인 작업들을 순서대로 나열
        4. limitation: Task 수행 시 반드시 지켜야 할 제약사항이나 고려사항
        5. success_criteria: Task가 성공적으로 수행되었는지 판단할 수 있는 구체적인 기준

        주의사항:
        1. 각 Task는 독립적으로 수행 가능해야 합니다.
        2. Task 간 의존성이 있다면 detail에 명시합니다.
        3. 모든 제약사항과 요구사항이 누락없이 포함되어야 합니다.
        4. 성공 기준은 가능한 한 구체적이고 측정 가능해야 합니다.
        5. 반드시 주어진 요구사항에 있는 정보만으로 리턴해야 하며, 없는 내용을 임의로 추가하지 마십시오. 특정 필드에 해당하는 정보가 요구사항에 없는 경우 해당 필드는 None으로 리턴하십시오.
        
        요구사항을 분석하여 위 포맷에 맞게 결과를 반환해주세요.
        """
        response = await self.llm.generate(prompt)
        TaskAnalysis = repair_json(response.content if hasattr(response, 'content') else response)
        TaskAnalysis = json.loads(TaskAnalysis)

        return TaskAnalysis
        
class MultiAgentAnalyzer:
    """Chain-of-Thought 기반 Multi-Agent 시스템 분석"""
    def __init__(self, llm_client):
        self.llm = llm_client

    async def analyze_requirements(self, requirements: dict) -> List[AgentSpec]:
        """
        요구사항을 분석하여 필요한 Agent와 Tool 도출
        """
        # 1. 요구사항 분석 및 필요한 Agent 도출
        prompt = f"""Let's analyze the requirements step by step to identify necessary agents and their tools:
        Requirements: {requirements}

        Think through this step by step:
        1) What are the main objectives of the system?
           - Break down input requirements into:
             * Primary task (what needs to be done)
             * Success criteria (how to verify completion)
             * Expected output format (what form the answer should take)
           - Ensure the generated question aligns with ALL these components
        2) What distinct responsibilities can be identified?
           - Map each component from the requirements analysis to specific responsibilities
           - Verify that the proposed question captures the core task AND matches success criteria
           - Flag any misalignment between task description and success criteria
        3) What specialized agents would be needed for each responsibility?
        4) What prompt are right for agents including action plan?
        5) What specific tools would each agent need?

        For each identified agent, consider:
        - What is their specific role?
        - What tasks must they perform?
        - What action plans they should follow?
        - What tools would they need to execute these plans?
        - How would these tools work in detail?

        For each agent's proposal prompt, ensure it includes:
        - Clear role definition and operation boundaries
        - Comprehensive step-by-step process for every task
        - Explicit rules for tool selection and usage
        - Detailed conditions and triggers for each action
        - DO NOT CHANGE "Memory and Context Management" PART and Please return the given example content(only "Memory and Context Management" PART) exactly as it is in your answer 
        - Clear success/failure criteria for tasks
        - Error handling and recovery procedures

        Provide your complete analysis in the following JSON format:
        {{
            "agents": [
                {{
                    "name": "descriptive_agent_name",
                    "role": "specific_role_description",
                    "responsibilities": [
                        "responsibility1",
                        "responsibility2",
                        ...
                    ],
                    "tools": [
                        {{
                            "name": "tool_name",
                            "purpose": "specific_purpose",
                            "parameters": {{
                                "param_name": {{
                                    "type": "exact_type",
                                    "description": "parameter_description"
                                }},
                                ...
                            }},
                            "return_type": {{
                                "type": "exact_return_type",
                                "description": "what_is_returned"
                            }},
                            "example_implementation": "detailed implementation example"
                        }},
                        ...
                    ],
                    "proposal_prompt": "You are the [agent_name]. Your role is [role_description].

        CORE OBJECTIVES:
        1. [Primary objective 1]
        2. [Primary objective 2]
        ...

        OPERATION PROTOCOL:

        1. Task Analysis:
        - ALWAYS start by analyzing the current request/situation
        - Check conversation history for relevant context
        - Identify which responsibility this falls under

        2. Decision Making Process:
        For each task you encounter:
        a) First, evaluate:
           - Task type and priority
           - Available context and history
           - Required tools and resources
   
        b) Then, determine:
           - Primary approach
           - Fallback options
           - Success criteria
           - Potential failure points

        3. Tool Usage Guidelines:
        [tool_name1]:
        - USE WHEN: [specific conditions]
        - REQUIRED INPUT: [what must be checked/prepared]
        - EXPECTED OUTPUT: [what to expect]
        - FAILURE HANDLING: [what to do if tool fails]

        [tool_name2]:
        [Repeat for each tool]

        4. Memory and Context Management:
        - ALWAYS check your historical performance records for similar tasks
        - STORE detailed records of your actions and their outcomes, including:
          * What tools you used and why
          * How effective your decisions were
          * Where your approach succeeded or failed
        - UPDATE your approach based on historical effectiveness patterns

        5. Error Recovery:
        - If [error_condition1]: [specific_recovery_action1]
        - If [error_condition2]: [specific_recovery_action2]

        6. Interaction Rules:
        - When working with [other_agent1]:
          * Share [specific_information]
          * Request [specific_assistance]
          * Wait for [specific_response]

        7. Success Criteria:
        - Task is complete when:
          * [criterion1]
          * [criterion2]

        THINKING PROCESS:
        For every action, follow this exact sequence:
        1. Assess current state and requirements
        2. Review relevant history and context
        3. Select appropriate tools and approaches
        4. Execute action plan
        5. Verify results
        6. Update memory as needed

        Remember:
        - NEVER skip steps in the thinking process
        - ALWAYS verify tool outputs
        - MAINTAIN consistency in responses
        - FOLLOW error recovery procedures precisely"
                }},
                ...
            ]
        }}

        Ensure that:
        1. Each agent has a clear, focused role
        2. Tools are defined with enough detail to be implemented
        3. The example implementation shows actual working code
        4. Parameter types and return types are precise
        5. RETURN ALL REQUIRED AGNETS FOR SATISFIYING GIVEN USER REQUIREMENT
        6. Please CAREFULLY CHECK IF ANY ADDITIONAL AGENTS ARE NEEDED besides the agents you've suggested. 
        7. Please consider whether THE AGENTS YOU'VE PROVIDED CAN SUFFICIENTLY SOLVE THE GIVEN PROBLEMS.
        8. DO NOT include any explanatory text. Your entire response must be ONLY a valid JSON object
        9. HAVE TO FOLLOW RETURN FORMAT. DO NOT ADD ANY EXTRA INFORMATION AND FILL ALL REQUIRED FILED
        10. Please provide ALL information without omitting any details, even if they might seem redundant or obvious

        VERIFICATION STEPS:
        1. Before returning, verify that NO content uses [...] or similar abbreviations
        2. Verify that THE AGENTS PROVIDED CAN SUFFICIENTLY SOLVE THE GIVEN PROBLEMS
        3. Validate that the entire prompt template is included without omissions
        4. CONFIRM THAT ALL SECTIONS ARE COMPLETELY FILLED OUT
        """

        response = await self.llm.generate(prompt)
        analysis = repair_json(response.content if hasattr(response, 'content') else response)
        analysis = json.loads(analysis)

        # 2. Agent 스펙 생성
        agent_specs = []
        for agent_data in analysis["agents"]:
            # Tool 스펙 생성
            tools = [ToolSpec(**tool) for tool in agent_data["tools"]]
            
            # Agent 스펙 생성
            agent_spec = AgentSpec(
                name=agent_data["name"],
                role=agent_data["role"],
                responsibilities=agent_data["responsibilities"],
                tools=tools,
                proposal_prompt = agent_data['proposal_prompt']
            )
            
            # 검증
            self._validate_agent_spec(agent_spec)
            agent_specs.append(agent_spec)

        return agent_specs

    def _validate_agent_spec(self, agent_spec: AgentSpec):
        """Agent 스펙 검증"""
        # 1. Tool 이름 중복 검사
        tool_names = [tool.name for tool in agent_spec.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError(f"Duplicate tool names found in agent {agent_spec.name}")

        # 2. 각 Tool의 구현 예시가 유효한지 검사
        for tool in agent_spec.tools:
            if "return" not in tool.example_implementation:
                raise ValueError(f"Tool {tool.name} implementation must have a return statement")



import nest_asyncio
nest_asyncio.apply()

async def main():
    # 초기화
    llm_client = LLMClient(
        provider=ModelProvider.ANTHROPIC,
        api_key="sk-ant-...",
        fallback_provider=ModelProvider.OPENAI,
        fallback_api_key="sk-proj-..."
    )
    useranalyzer = userqueryAnalyzer(llm_client)
    analyzer = MultiAgentAnalyzer(llm_client)
    implementor = ToolImplementor(llm_client)
    
    # 요구사항 분석 및 Multi-Agent/Tool 도출
    requirements = """
    다음과 같은 조건을 만족하는 Closed QA 생성 시스템이 필요합니다.

    1. 수량
       필요 수량: 56,000건

    2. QA 난이도
       일반인이 물어볼 법한 질의 (전문적인 내용, 혹은 용어는 사용 안함)

    3. 형식적 사항
       input/output 글자수 길이: 각각 max 2,048 (요약 태스크의 경우 max 4,096)
       동일한 context에 여러 유형의 QA 생성 가능
       각기 다른 카테고리에서 다양한 QA 생성 가능
       동일한 context에 여러 유형의 QA 생성 가능 -> **하나의 context 당 각기 다른 카테고리의 QA 생성 가능. 동일한 카테고리에 동일한 context 기반의 QA를 여러 개 생성하면 안됨

    4. 결과 데이터 형식
       열 목록: domain, category, context_type, response_type, instruction, context, response

    5. Context 유형과 답변 유형
       -1. Context 유형:
           문장: 문자열로 이루어진 단일 혹은 다수의 문장
           table: context의 내용이 표로만 이루어진 경우
           table이 포함된 문장: context의 내용으로 표와 문장이 함께 나타난 경우
           table 위주 문장: context의 내용이 주로 표로 이루어져 있으며 간단한 설명이 포함된 경우

    6. 답변 유형:
       -1. table: 표 형태의 답변
       -2. 단답형: 단일어, 다단어의 명사구, 한 문장 형태로 이루어진 답변
       -3. 서술형: 한 문장 이상으로 이루어진 답변

    7. 카테고리 유형
       -1. information Extract: 
             - 답변 유형이 table인 경우 table의 일부 내용 그대로 추출
             - 그 외 답변 유형: context 내에서 정보 추출
       -2. Rewrite:
             - 답변 유형이 table인 경우 table의 형식적인 구조 변경, 답변의 형태가 구조가 변경된 테이블임
             - 그 외 답변 유형: context 일부 혹은 전체의 형식적인 구조 변경
       -3. Summarization: 
             - 답변 유형이 table인 경우 table의 일부 내용 요약
             - 그 외 답변 유형: context 일부 혹은 전체의 내용 요약
    """

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
    Task = await useranalyzer.analyze_user_requirements(requirements, context)
    agent_specs = await analyzer.analyze_requirements(Task)
    
    simple_pipeline = []
    for agent in agent_specs:
        agent_info = {}
        agent_info["Agent"] = agent.name
        agent_info["Role"] = agent.role
        agent_info["prompt"] =  agent.proposal_prompt
        agent_info["responsibilities"] = agent.responsibilities
            
        tools = []
        for tool in agent.tools:
            tool_info = {}
            tool_info['tool_name'] = tool.name
            tool_info['Purpose'] = tool.purpose
            tool_info["Parameters"] = json.dumps(tool.parameters, indent=2)
            tool_info["Return Type"] = json.dumps(tool.return_type, indent=2)
            
            # Tool의 실제 구현 코드 생성
            implementation = await implementor.implement_tool(tool)
            tool_info["Generated Implementation"] = implementation
            tools.append(tool_info)
        agent_info['tools']= tools
        simple_pipeline.append(agent_info)
###############################################################
    #complex pipeline
    complex_pipeline = []
    for task in Task:
        # Multi-Agent 시스템 설계 
        agent_specs_detail = await analyzer.analyze_requirements(task)
    
        # 결과 출력
        sub_pipeline={}
        sub_agent = []
        for agent in agent_specs_detail:
            agent_info = {}
            agent_info["Agent"] = agent.name
            agent_info["Role"] = agent.role
            agent_info["prompt"] =  agent.proposal_prompt
            agent_info["responsibilities"] = agent.responsibilities
            
            tools = []
            for tool in agent.tools:
                tool_info = {}
                tool_info['tool_name'] = tool.name
                tool_info['Purpose'] = tool.purpose
                tool_info["Parameters"] = json.dumps(tool.parameters, indent=2)
                tool_info["Return Type"] = json.dumps(tool.return_type, indent=2)
            
                # Tool의 실제 구현 코드 생성
                implementation = await implementor.implement_tool(tool)
                tool_info["Generated Implementation"] = implementation
                tools.append(tool_info)
            agent_info['tools']= tools
            sub_agent.append(agent_info)
            
        sub_pipeline['task_name'] = task['task_name']
        sub_pipeline['task_purpose'] = task['purpose']
        sub_pipeline['detail'] = task['detail']
        sub_pipeline['task_limitation'] = task['limitation']
        sub_pipeline['task_success_criteria'] = task['success_criteria']
        sub_pipeline['sub_agents'] = sub_agent
        
        complex_pipeline.append(sub_pipeline)

    with open('./task.json', 'w', encoding='utf-8') as f:
        json.dump(Task, f, ensure_ascii=False, indent=4)
    
    with open('./simple_pipeline.json', 'w', encoding='utf-8') as f:
        json.dump(simple_pipeline, f, ensure_ascii=False, indent=4)

    with open('./complex_pipeline.json', 'w', encoding='utf-8') as f:
        json.dump(complex_pipeline, f, ensure_ascii=False, indent=4)
# Jupyter/IPython 환경에서는 아래와 같이 직접 실행
await main()
