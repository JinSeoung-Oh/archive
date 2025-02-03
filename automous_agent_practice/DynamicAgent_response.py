# 기존 코드 유지
class LLMAgent:
    def __init__(self, name, prompt, Role, tool):
        self.name = name
        self.prompt = prompt
        self.Role = Role
        self.tool = tool
    
    def __str__(self):
        return f"Agent: {self.name}\nDescription: {self.Role}"

def DynamicAgent_response(llm_agent, llm, state):
    query = state.current_task
    context = state.subtasks[0].context if state.subtasks else ''
    requirement = state.subtasks[0].additioanl_requirement if state.subtasks else ''
    print(llm_agent.tool)
       
    message_content = f"Context: {context}\nQuestion: {query}\nGuidelines:{requirement}"
    response = llm.generate(
        [[
            SystemMessage(content=llm_agent.prompt),
            HumanMessage(content=message_content)
        ]]
    )
    
    state.agent_outputs[llm_agent.name.lower()] = response.generations[0][0].text
    state.last_agent = llm_agent.name
    return state

def create_agent_response(agent_info, llm, state):
    llm_agent = LLMAgent(agent_info['Agent'], agent_info['prompt'], agent_info['Role'], agent_info['tools'])
    new_state = DynamicAgent_response(llm_agent, llm, state)
    return new_state

# 시각화 클래스 추가
class GraphVisualizer:
    def __init__(self):
        self.G = nx.DiGraph()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.pos = None
        
    def update(self, graph, active_node=None):
        """그래프 상태 업데이트"""
        self.G = nx.DiGraph()
        self.G.add_nodes_from(graph.
