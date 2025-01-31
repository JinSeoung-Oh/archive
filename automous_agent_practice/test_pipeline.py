from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import time

import json
import os

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

class Subtask(BaseModel):
    question: str
    agents: List[str]
    context: Optional[str] = None
    additioanl_requirement: Optional[str] = None

class State(BaseModel):
    subtasks: List[Subtask] = Field(default_factory=list)
    current_task: Optional[str] = None
    current_agents: List[str] = Field(default_factory=list)
    next_agent: Optional[str] = None
    agent_outputs: dict = Field(default_factory=dict)
    last_agent: Optional[str] = None
    task_complete: bool = False
    subtask_complete: bool = False

def orchestrator(state):
    if state.subtasks:
        current_subtask = state.subtasks[0]
        state.current_task = current_subtask.question
        state.current_agents = current_subtask.agents.copy()
        state.next_agent = state.current_agents.pop(0)
        state.subtask_complete = False
    else:
        state.task_complete = True
    return state

def route_to_agent(state):
    agent_type = state.next_agent
    print(state)
    print(agent_type)
    if agent_type not in graph.nodes:
        agent_info = next((agent for agent in flattened_sys_msg_list if agent['Agent'] == agent_type), None)
        if agent_info:
            state = create_agent_response(agent_info, llm, state)
    return state

def agent_completed(state):
    if state.current_agents:
        state.next_agent = state.current_agents.pop(0)
    else:
        state.subtasks.pop(0)
        state.subtask_complete = True
    return state

# Initialize the graph
graph = StateGraph(State)

# Add nodes
graph.add_node("orchestrator", orchestrator)
graph.add_node("router", route_to_agent)
graph.add_node("agent_completed", agent_completed)

# Set entry point
graph.set_entry_point("orchestrator")

# Add conditional edges
graph.add_conditional_edges(
    "orchestrator",
    lambda state: "router" if not state.task_complete else END)
graph.add_edge("router", "agent_completed")
graph.add_conditional_edges(
    "agent_completed",
    lambda state: "router" if not state.subtask_complete else "orchestrator")

# Compile the graph
chain = graph.compile()

with open('./complex_pipeline.json', 'r', encoding='utf-8') as f:
    agent_info = json.load(f)

with open('./workflow.json', 'r', encoding='utf-8') as f:
    workflow_info = json.load(f)

# Flatten sys_msg_list
flattened_sys_msg_list = [agent for sublist in agent_info for agent in sublist['sub_agents']]

context = workflow_info[-1]
workflow_info.pop(-1)

analyzed_query = []
for agent, workflow in zip(agent_info, workflow_info):
    data_col = {}
    data_col['question'] = agent['task_purpose']
    agent_list = []
    for info in workflow:
        agents = info['Agent']
        agent_list.append(agents)
    data_col['agents'] = agent_list
    data_col['context'] = context
    requirements = f"Detail: {agent['detail']}'\n'Limitation: {agent['task_limitation']}'\n'Success_condition: {agent['task_success_criteria']}"
    data_col['additioanl_requirement'] = requirements + "한국어로 요구하는 답변만 리턴해주세요. 불필요한 정보는 추가시키지 마세요. Success_condition은 반드시 만족해야만 합니다"
    analyzed_query.append(data_col)

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
llm = ChatAnthropic(model='claude-3-5-sonnet-latest')

# Select Agents
#selected_agents = set()
#for query in analyzed_query:
#    selected_agents.update(query['agents'])

# Create LLM Agents
#llm_agents = []
#for agent_info in flattened_sys_msg_list:
#    if agent_info['name'] in selected_agents:
#        agent = LLMAgent(agent_info['name'], agent_info['system_message'], agent_info['description'])
#        llm_agents.append(agent)

# Run the chain with the analyzed query
results = []
for query in analyzed_query:
    initial_state = State(subtasks=[Subtask(**query)])
    final_state = chain.invoke(initial_state)
    results.append({
        "question": query["question"],
        "output": final_state["agent_outputs"]
    })
    time.sleep(5)

# Create the final response as JSON
final_response = json.dumps(results, ensure_ascii=False, indent=2)

# Print the final response
print(final_response)
