! pip install feedparser

import feedparser
import time
from datetime import datetime, timedelta
import anthropic

import random
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
import json
import schedule

from dateutil import parser
from datetime import datetime, timedelta

an_cluade = "sk-..."
client = anthropic.Anthropic(api_key=an_cluade)
Today = datetime.now().strftime('%B %d, %Y')

This_year = datetime.now().strftime('%Y')

today = datetime.now()
yesterday = today - timedelta(days=1)
yesterday = yesterday.strftime('%B %d, %Y')

validation_prompt = f"""당신은 주어진 뉴스들을 한국어로 번역하는 번역가입니다.

                        주의사항:
                        1. 반드시 {This_year}년 뉴스만 포함하세요.
                        2. {This_year}년에 발행된 것들 중에서도 다음 시간 범위에 해당하는 것만 포함하세요::
                           - UTC/GMT 기준: {yesterday} 15:00 ~ {Today} 14:59
                           - 한국 시간 기준: {Today} 00:00 ~ 23:59
                        3. 다양한 날짜 형식이 있으므로 각 뉴스의 발행일을 주의깊게 확인해주세요
                        4. 뉴스가 누락되지 않도록 모든 날짜 형식을 고려해주세요
                        5. 출판 날짜를 반드시 엄수해주세요. 

                        출력 형식:
                        주어진 내용을 분석한 결과, 다음과 같은 주요 뉴스들이 있습니다:

                        순번. 제목: [한국어 제목]
                        링크: [원문 링크 URL]
                        요약: [한국어로 상세 요약]"""

def filter_news_by_date(news_items, start_date, end_date):
    filtered_news = []
    
    for item in news_items:
        try:
            # 어떤 형식이든 날짜로 파싱
            pub_date = parser.parse(item['published'])
            
            # 날짜 범위 체크
            if start_date <= pub_date <= end_date:
                filtered_news.append(item)
        except (ValueError, TypeError):
            # 날짜 파싱 실패시 스킵
            continue
            
    return filtered_news

def read_rss(url):
    # RSS 피드 파싱
    
    feed = feedparser.parse(url)

    # 결과를 담을 리스트
    result = []

    # 피드의 엔트리(기사) 정보를 순회
    for entry in feed.entries:
        # 사용할 정보 추출(필요한 항목만 골라서 사용 가능)
        title = entry.title if 'title' in entry else None
        link = entry.link if 'link' in entry else None
        published = entry.published if 'published' in entry else None
        summary = entry.summary if 'summary' in entry else None
        
        # 한 기사의 정보를 딕셔너리로 구성
        item = {
            "title": title,
            "link": link,
            "published": published,
            "summary": summary
        }
        
        # 결과 리스트에 추가
        result.append(item)

    return result

def get_all_news(feed_list):
    all_news = []
    for feed in feed_list:
        try:
            feed_info = read_rss(feed)
            # 리스트를 확장(extend)하여 단일 리스트로 만듦
            all_news.extend(feed_info)
        except Exception as e:
            print(f"피드 {feed} 처리 중 오류 발생: {str(e)}")
            continue
    
    # 날짜순으로 정렬
    all_news.sort(key=lambda x: parser.parse(x['published']) if x['published'] else datetime.min, reverse=True)
    
    # 디버깅을 위한 출력
    print(f"총 수집된 뉴스 개수: {len(all_news)}")
    
    return all_news

def format_for_claude(news_items):
    formatted_text = ""
    for i, item in enumerate(news_items, 1):
        formatted_text += f"\n#{i}\n"
        formatted_text += f"Title: {item['title']}\n"
        formatted_text += f"Link: {item['link']}\n"
        formatted_text += f"Published: {item['published']}\n"
        formatted_text += f"Summary: {item['summary']}\n"
        formatted_text += "-" * 50 + "\n"
    return formatted_text

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
def get_response_with_retry(content_, template):
    try:
        return get_response_from_claude(content_, template)
    except Exception as e:
        print(f"Error occurred: {str(e)}. Retrying...")
        time.sleep(random.uniform(1, 3))  # Random delay before retry
        raise  # Re-raise the exception to trigger a retry

def send_message_to_google_chat(message_text: str):
    # Google Chat에 전송할 JSON 데이터 (카드 메시지, 텍스트 메시지 등 다양하게 가능)
    WEBHOOK_URL= "https://chat.googleapis.com/v1/spaces/..."  

    data = {
        "text": message_text
    }

    headers = {
        "Content-Type": "application/json; charset=UTF-8"
    }

    response = requests.post(WEBHOOK_URL, data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        print("메시지 전송 성공!")
    else:
        print(f"메시지 전송 실패! 상태 코드: {response.status_code}, 응답 내용: {response.text}")

feed_list = ["https://www.marktechpost.com/feed/","https://aicorr.com/feed/","https://www.marketingaiinstitute.com/blog/rss.xml", "https://blog.chatbotslife.com/feed", "https://nanonets.com/blog/rss/", "https://www.technologyreview.com/topic/artificial-intelligence/feed", "https://www.shaip.com/feed/", "https://aiparabellum.com/feed/", "https://news.mit.edu/rss/topic/artificial-intelligence2", "https://deepmind.google/blog/rss.xml", "https://www.unite.ai/feed/", "https://dailyai.com/feed/"]

all_news = get_all_news(feed_list)

# Claude에 전달할 형식으로 변환
formatted_new = format_for_claude(all_news[:20])
filter_new = get_response_with_retry(formatted_new, validation_prompt)

send_message_to_eval_chat(filter_new)
