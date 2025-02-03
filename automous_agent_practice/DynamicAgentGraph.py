## Have to test it

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage, HumanMessage

class DynamicStateGraph:
    def __init__(self, state_class: Type[BaseModel]):
        self.graph = StateGraph(state_class)
        self.agents = {}  # 생성된 Agent 저장
        self._initialize_base_graph()
        
    def _initialize_base_graph(self):
        """기본 그래프 구조 초기화"""
        self.graph.add_node("orchestrator", orchestrator)
        self.graph.add_node("router", self.dynamic_router)
        self.graph.add_node("agent_completed", agent_completed)
        
        # 기본 엣지 설정
        self.graph.set_entry_point("orchestrator")
        self.graph.add_conditional_edges(
            "orchestrator",
            lambda state: "router" if not state.task_complete else END
        )
        self.graph.add_edge("router", "agent_completed")
        self.graph.add_conditional_edges(
            "agent_completed",
            lambda state: "router" if not state.subtask_complete else "orchestrator"
        )
    
    async def create_agent_node(self, agent_info: Dict):
        """동적으로 Agent 노드 생성"""
        agent_name = agent_info['Agent']
        
        if agent_name not in self.agents:
            llm_agent = LLMAgent(
                agent_info['Agent'],
                agent_info['prompt'],
                agent_info['Role'],
                agent_info['tools']
            )
            
            # Agent 노드 함수 생성
            async def agent_node(state):
                return await DynamicAgent_response(llm_agent, state.llm, state)
            
            # 그래프에 노드 추가
            self.graph.add_node(agent_name, agent_node)
            self.agents[agent_name] = llm_agent
            
            # 라우터와 agent_completed 노드 사이에 새 엣지 추가
            self.graph.add_edge(agent_name, "agent_completed")
    
    async def dynamic_router(self, state):
        """동적 라우팅 로직"""
        agent_type = state.next_agent
        
        if agent_type not in self.agents:
            # Agent 정보 찾기
            agent_info = next(
                (agent for agent in state.available_agents 
                 if agent['Agent'] == agent_type),
                None
            )
            
            if agent_info:
                # 새 Agent 노드 생성
                await self.create_agent_node(agent_info)
        
        # Agent 노드로 라우팅
        if agent_type in self.graph.nodes:
            return agent_type
            
        return "agent_completed"
    
    def compile(self):
        """그래프 컴파일"""
        return self.graph.compile()

# State 클래스 확장
class EnhancedState(State):
    available_agents: List[Dict] = Field(default_factory=list)
    llm: Any = None  # LLM 인스턴스 저장

# 사용 예시
async def main():
    # 동적 그래프 초기화
    dynamic_graph = DynamicStateGraph(EnhancedState)
    
    # 초기 상태 설정
    initial_state = EnhancedState(
        subtasks=[
            Subtask(
                question="What is the capital of France?",
                agents=["Researcher", "Writer"],
                context="Need information about France"
            )
        ],
        available_agents=flattened_sys_msg_list,  # Agent 정보 목록
        llm=llm  # LLM 인스턴스
    )
    
    # 그래프 컴파일 및 실행
    chain = dynamic_graph.compile()
    result = await chain.ainvoke(initial_state)
    return result

if __name__ == "__main__":
    asyncio.run(main())
