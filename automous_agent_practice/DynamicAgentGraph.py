from typing import Dict, List
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
from queue import Queue
import asyncio

class DynamicAgentGraph:
    def __init__(self):
        self.agent_queue = Queue()
        self.graph = StateGraph()
        self.agents = {}
        
    async def create_agent_node(self, agent_info):
        """새로운 Agent 노드를 생성하고 그래프에 추가"""
        agent_id = agent_info['id']
        
        # Agent 노드 생성
        async def agent_node(state):
            # Agent의 실제 처리 로직
            return state
            
        # 그래프에 새 노드 추가
        self.graph.add_node(agent_id, agent_node)
        self.agents[agent_id] = agent_info
        
        # 이전 노드와 연결 (필요한 경우)
        if agent_info.get('previous_agent'):
            self.graph.add_edge(agent_info['previous_agent'], agent_id)
            
        return agent_id

    async def process_queue(self):
        """큐에서 Agent 정보를 가져와 노드 생성"""
        while not self.agent_queue.empty():
            agent_info = self.agent_queue.get()
            await self.create_agent_node(agent_info)
            
    def add_agent_to_queue(self, agent_info: Dict):
        """새로운 Agent를 큐에 추가"""
        self.agent_queue.put(agent_info)
        
    async def run_graph(self, initial_state: Dict):
        """그래프 실행"""
        # 큐에 있는 모든 Agent 처리
        await self.process_queue()
        
        # 그래프 컴파일 및 실행
        app = self.graph.compile()
        return await app.ainvoke(initial_state)

# 사용 예시
async def main():
    dynamic_graph = DynamicAgentGraph()
    
    # Agent 정보를 큐에 추가
    dynamic_graph.add_agent_to_queue({
        'id': 'agent1',
        'previous_agent': None,
        'config': {'type': 'processor'}
    })
    
    dynamic_graph.add_agent_to_queue({
        'id': 'agent2',
        'previous_agent': 'agent1',
        'config': {'type': 'analyzer'}
    })
    
    # 초기 상태 설정
    initial_state = {"message": "시작 메시지"}
    
    # 그래프 실행
    result = await dynamic_graph.run_graph(initial_state)
    return result

# 실행
if __name__ == "__main__":
    asyncio.run(main())
