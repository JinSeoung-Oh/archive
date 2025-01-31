!pip install langchain_openai
!pip install langchain_core
!pip install langchain.agents
!pip install langchain.tools
!pip install numexpr

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import load_tools, Tool
from langchain.agents import AgentExecutor, create_react_agent
from datetime import datetime, timedelta
from langchain.tools import BaseTool
import requests
from typing import Optional, Type,Dict 
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from playwright.async_api import async_playwright
import asyncio
import time
import anthropic

import random
from tenacity import retry, stop_after_attempt, wait_exponential
import json
from json_repair import repair_json

current_date = datetime.now().strftime('%Y년 %m월 %d일')

os.environ['OPENAI_API_KEY'] = "sk-..."
openai_llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
an_cluade = "sk-..."
client = anthropic.Anthropic(api_key=an_cluade)

class AISearchPrompts:
    """A class to manage various AI-related search prompts."""
    
    def __init__(self, current_date: str = None):
        """
        Initialize the AISearchPrompts class.
        
        Args:
            current_date (str, optional): Current date to use in prompts. 
                                        If None, uses today's date.
        """
        self.current_date = current_date or datetime.now().strftime("%Y-%m-%d")
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize all search prompts with current date."""
        self.prompts: Dict[str, str] = {
            "all_sources": f"Find out latest AI technology news or research paper or AI blog guid in {current_date}",
            
            "ai_times": f"{self.current_date}에 발행된 AI 기술 관련 뉴스 찾기",
            
            "medium": f"Find Technical AI blog posts or guides about implementing/using AI technologies from verified tech blogs published on {self.current_date}",
            
            "arxiv": f"Find AI research papers about AI/ML algorithms, architectures, or technical innovations published on {self.current_date}",
            
            "validation": """You are a filter that identifies and removes content not related to AI technology from the input document.

                Review the document using these criteria:
                1. Keep AI technology-related content:
                   - AI technology development/research news
                   - AI applications and use cases
                   - AI policy/regulation news
                   - AI company/industry trends

                2. Remove content:
                   - Cryptocurrency/token/airdrop related
                   - General tech news without AI focus
                   - Marketing content that only mentions AI as a keyword
                   - Research/papers from non-AI fields

                Remove all non-AI related items from the input document and return only content that is genuinely related to AI technology.
                
                Return the filtered results in the following like:
                       **News:**
                       1. [Source Name](url): Korean translation of the article summary/description
                       2. [Source Name](url): Korean translation of the article summary/description
                       ...

                       **Research:**
                       1. [Source Name](url): Korean translation of the research summary/description
                       2. [Source Name](url): Korean translation of the research summary/description
                       ...

                       **Blogs:**
                       1. [Source Name](url): Korean translation of the blog post summary/description
                       2. [Source Name](url): Korean translation of the blog post summary/description
     
                       Important rules:
                       1. Only translate the description field into Korean
                       2. Keep all other fields (source, title, link) in their original English
                       3. Ensure the Korean translation is natural and clear
                       4. Keep source URLs and technical terms in their original form""",
            
            "summary": """Please summarize the given text in four lines or less. All important points discussed in the given text must not be omitted. Just focus on main text. You can ignore extra information. Return Korean"""
        }
    
    def get_prompt(self, prompt_type: str) -> str:
        """
        Get a specific prompt by type.
        
        Args:
            prompt_type (str): Type of prompt to retrieve 
                             ('all_sources', 'ai_times', 'medium', 'arxiv', 'validation', 'summary')
        
        Returns:
            str: The requested prompt text
            
        Raises:
            KeyError: If prompt_type is not found
        """
        return self.prompts[prompt_type]
    
    def update_date(self, new_date: str):
        """
        Update the current date and reinitialize all prompts.
        
        Args:
            new_date (str): New date to use in prompts
        """
        self.current_date = new_date
        self._initialize_prompts()
    
    def list_available_prompts(self) -> list:
        """
        List all available prompt types.
        
        Returns:
            list: List of available prompt types
        """
        return list(self.prompts.keys())


class GoogleCustomSearchTool(BaseTool):
    """Tool that uses Google Custom Search API."""
    
    name: str = "Search"
    description: str = "Useful for searching the internet for specific AI news, research papers, and blog guides. Use this for finding current information."
    api_key: str
    cx: str
    
    def search_by_type(self, base_query: str, content_type: str, params: dict) -> list:
        try:
            if content_type == "news":
                type_query = f"{base_query} AI artificial intelligence"
            elif content_type == "research":
                type_query = f"{base_query} AI research paper site:arxiv.org OR site:papers.ssrn.com OR site:springer.com OR site:ieee.org"
            else:  # blog guides
                type_query = f"{base_query} AI tutorial guide blog site:medium.com OR site:towardsdatascience.com OR site:ai.google.blog OR site:openai.com/blog"
            
            params['q'] = type_query
            response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
            results = response.json()
            
            if 'items' not in results:
                return []
                
            return [{
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'snippet': item.get('snippet', ''),
                'type': content_type
            } for item in results['items']]
            
        except Exception as e:
            print(f"Error in {content_type} search: {str(e)}")
            return []

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            params = {
                'key': self.api_key,
                'cx': self.cx,
                'q': query.strip(),  # 불필요한 개행 제거,
                'num': 10,
                'sort': 'date',
                'dateRestrict': 'd1'}
            
            #print("Request params:", params)
            
            response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)

            #print("Response status:", response.status_code)
            #print("Response content:", response.text)
            
            results = response.json()
            
            if 'items' not in results:
                return "No results found."
            
            formatted_results = []
            for item in results['items']:
                result = {
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'date': item.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time', '')}
                formatted_results.append(f"Title: {result['title']}\nLink: {result['link']}\nSummary: {result['snippet']}\n")
            
            return "\n".join(formatted_results)
        
        except Exception as e:
            return f"Error performing search: {str(e)}"

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run Google Custom Search async."""
        raise NotImplementedError("GoogleCustomSearchTool does not support async")

  class SearchAgent:
    def __init__(self, openai_llm, prompts):
        """
        SearchAgent 클래스 초기화
        
        Args:
            openai_llm: OpenAI LLM 인스턴스
            prompts: AISearchPrompts에서 가져온 프롬프트 딕셔너리
        """
        self.openai_llm = openai_llm
        self.prompts = prompts
        self.api_key = "...."
        
        # ReAct 템플릿 설정
        self.react_template = """You are a search agent specialized in finding AI technology information. Your ONLY task is to search and collect information. DO NOT think about dates or data availability - JUST SEARCH.

                                 You have access to the following tools:
                                 {tools}

                                 You MUST follow this EXACT format:
                                 Question: The input question you must answer
                                 Thought: Using Search tool to find the exact information requested
                                 Action: Search - must be one of [{tool_names}]
                                 Action Input: <enter the complete search query with all specified criteria>
                                 Observation: <search results will appear here>
                                 Thought: I have the search results
                                 Final Answer: Here are the findings with direct links like below

                                 **News:**
                                 1. [Source Name](url): Description
                                 2. [Source Name](url): Description
                                 ...

                                 **Research:**
                                 1. [Source Name](url): Description
                                 2. [Source Name](url): Description
                                 ...

                                 **Blogs:**
                                 1. [Source Name](url): Description
                                 2. [Source Name](url): Description
                                 ...

                                 Begin!

                                 Question: {input}
                                 Thought: {agent_scratchpad}

                                 CRITICAL RULES:
                                 - NEVER discuss data availability or dates
                                 - NEVER give general advice 
                                 - ONLY execute searches and return results
                                 - Include 'site:' operators to ensure direct article/paper/post URLs
                                 - NEVER return homepage or section URLs
                                 - Each result MUST have direct URL to specific content
                                 - If results are fewer than desired, still provide what was found
                                 - DO NOT repeat searches or try additional queries
                                 """

        # PromptTemplate 생성
        self.prompt = PromptTemplate(
            template=self.react_template,
            input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
        )
        
        # 각 사이트별 Search Engine ID (cx) 설정
        self.cx_configs = {
            'all': '...',
            'aitimes': '...',
            'medium': '...',
            'arxiv': '...'
        }

    def _execute_search(self, cx, search_prompt):
        """
        실제 검색을 수행하는 내부 메서드
        """
        #print(search_prompt)
        try:
            # Search 도구만 사용
            tools = [
                GoogleCustomSearchTool(
                    api_key=self.api_key,
                    cx=cx
                )
            ]
            #callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            # 에이전트 생성
            agent = create_react_agent(self.openai_llm, tools, self.prompt)
            
            # 실행기 생성
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                #callback_manager=callback_manager,  # 콜백 추가
                handle_parsing_errors=True
                )

            
            # 검색 실행
            response = agent_executor.invoke({
                "input": search_prompt
            })

            return response['output']
            
        except Exception as e:
            print(f"검색 중 오류 발생: {str(e)}")
            return None

    def get_search_all_web_site(self):
        """전체 웹사이트 검색"""
        return self._execute_search(self.cx_configs['all'], self.prompts['all_sources'])

    def get_search_aitimes_site(self):
        """AI Times 사이트 검색"""
        return self._execute_search(self.cx_configs['aitimes'], self.prompts['ai_times'])

    def get_search_medium_site(self):
        """Medium 사이트 검색"""
        return self._execute_search(self.cx_configs['medium'], self.prompts['medium'])

    def get_search_arxiv_site(self):
        """arXiv 사이트 검색"""
        return self._execute_search(self.cx_configs['arxiv'], self.prompts['arxiv'])

def get_response_from_claude(context,sys_prompt):
    result_text = ""
    
    # Claude에 메시지 생성 요청을 보냅니다.
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
        temperature=0.0,
        system= sys_prompt,
        messages=[{"role": "user", "content": f"다음 '{context}'를 분석해줘"}]
    )
    
    # 응답 객체에서 텍스트 내용만 추출합니다.
    if not response.content or not isinstance(response.content, list):
        result_text = "No response or unexpected response format."
    else:
        response_texts = [block.text for block in response.content if hasattr(block, 'text')]
        result_text = " ".join(response_texts)
 
    return result_text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_response_with_retry(file_, template):
    try:
        return get_response_from_claude(file_, template)
    except Exception as e:
        print(f"Error occurred: {str(e)}. Retrying...")
        time.sleep(random.uniform(1, 3))  # Random delay before retry
        raise  # Re-raise the exception to trigger a retry

def send_message_to_eval_chat(message_text: str):
    # Google Chat에 전송할 JSON 데이터 (카드 메시지, 텍스트 메시지 등 다양하게 가능)
    WEBHOOK_URL= "https://chat.googleapis.com/v1/spaces/..."  # 예: 실제 발급받은 URL로 교체하세요.

    data = {
        "text": message_text
    }

    headers = {
        "Content-Type": "application/json; charset=UTF-8"
    }

    response = requests.post(WEBHOOK_URL, data=json.dumps(data), headers=headers)

## 멤버쉽 요구 등으로 접근이 제한되는 링크가 있어 해당 함수는 사용하지 않음

async def scrape_websites(link):
    result = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        try:
            await page.goto(link)
            await page.wait_for_load_state('networkidle')
            
            # 페이지의 모든 텍스트 콘텐츠 가져오기
            content = await page.evaluate('''() => {
                return document.body.innerText;
            }''')

            result = {
                'url': link,
                'content': content
            }
                
        except Exception as e:
            print(f"Error scraping {link}: {str(e)}")
        await browser.close()
    return result

# AI Search Prompts 초기화
ai_prompts = AISearchPrompts(current_date)
prompts = {
    'all_sources': ai_prompts.get_prompt("all_sources")
}

# 하나의 SearchAgent 인스턴스만 생성
search_agent = SearchAgent(
    openai_llm=openai_llm,
    prompts=prompts,
)

all_result = search_agent.get_search_all_web_site()
