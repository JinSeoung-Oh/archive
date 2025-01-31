import schedule
import time
from datetime import datetime, timedelta
import feedparser
import anthropic
import random
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
import json

an_cluade = "sk-ant-..."
client = anthropic.Anthropic(api_key=an_cluade)

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

def daily_news_job():
    feed_list = ["https://www.marktechpost.com/feed/","https://aicorr.com/feed/","https://www.marketingaiinstitute.com/blog/rss.xml", "https://blog.chatbotslife.com/feed", "https://nanonets.com/blog/rss/", "https://www.technologyreview.com/topic/artificial-intelligence/feed", "https://www.shaip.com/feed/", "https://aiparabellum.com/feed/", "https://news.mit.edu/rss/topic/artificial-intelligence2", "https://deepmind.google/blog/rss.xml", "https://www.unite.ai/feed/", "https://dailyai.com/feed/"]
    geek_feed = "https://feeds.feedburner.com/geeknews-feed"
    try:
        # 오늘/어제 날짜 설정
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        Today = today.strftime('%Y년 %m월 %d일')
        yesterday = yesterday.strftime('%Y년 %m월 %d일')
        during = yesterday + ' and ' + Today
        This_year = datetime.now().strftime('%Y')
        
        # RSS 피드에서 데이터 가져오기
        result = []
        for feed in feed_list:
            feed_info = read_rss(feed)
            result.append(feed_info)
            
        # 프롬프트 업데이트
        validation_prompt = f"""당신은 주어진 뉴스들을 한국어로 번역하는 번역가입니다.

                            주의사항:
                            1. 반드시 {This_year}년 뉴스만 포함하세요.
                            2. {This_year}년에 발행된 것들 중에서도 다음 시간 범위에 해당하는 것만 포함하세요::
                               - UTC/GMT 기준: {yesterday} 15:00 ~ {Today} 14:59
                               - 한국 시간 기준: {Today} 00:00 ~ 23:59
                            3. 다양한 날짜 형식이 있으므로 각 뉴스의 발행일을 주의깊게 확인해주세요
                            4. 뉴스가 누락되지 않도록 모든 날짜 형식을 고려해주세요

                            출력 형식:
                            주어진 내용을 분석한 결과, 다음과 같은 주요 뉴스들이 있습니다:

                            순번. 제목: [한국어 제목]
                            링크: [원문 링크 URL]
                            요약: [한국어로 상세 요약]"""

        # Claude를 통한 필터링
        all_news = get_all_news(feed_list)
        formatted_news = format_for_claude(all_news)

        geek_new = get_all_news(geek_feed)
        formatted_geek_news = format_for_claude(geek_new)
        
        filter_new = get_response_with_retry(formatted_news, validation_prompt)
        filter_ = get_response_with_retry(formatted_geek_news, validation_prompt)

        filter_new = 'From international news' + '\n' + filter_new
        filter_  = 'From Geek News' + '\n' + filter_

        result = filter_new + '\n\n' + '-------------------------------------------------------------' + '\n' + filter_ 
        
        # 결과를 Google Chat으로 전송
        send_message_to_eval_chat(result)
        
        send_message_to_google_chat(result)
        
        print(f"작업이 성공적으로 완료되었습니다. 현재 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        error_message = f"작업 실행 중 오류가 발생했습니다: {str(e)}"
        print(error_message)
        # 에러 발생시 Google Chat으로 알림
        send_message_to_eval_chat(error_message)

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
    WEBHOOK_URL= "https://chat.googleapis.com/v1/spaces/..."  # 예: 실제 발급받은 URL로 교체하세요.

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

def show_next_runs(days=5):
    job = schedule.jobs[0]  # 현재 등록된 첫 번째 job
    next_dates = []
    
    # 현재 예정된 다음 실행 시간
    next_run = job.next_run
    
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext 5 scheduled runs:")
    
    for i in range(days):
        print(f"Run {i+1}: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        # 다음 날 같은 시간으로 계산
        next_run = next_run + timedelta(days=1)

  # 기존 스케줄 모두 제거
schedule.clear()

# 스케줄러 설정
print(f"Scheduler setup at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
schedule.every().day.at("11:30").do(daily_news_job)

print(f"Number of jobs scheduled: {len(schedule.jobs)}")
for job in schedule.jobs:
    print(f"Next run time: {job.next_run}")

print("스케줄러가 시작되었습니다.")
print("매일 아침 11:30에 작업이 실행됩니다.")
print("커널이 실행중인 동안 계속 동작합니다.")

# 스케줄러 실행
while True:
    schedule.run_pending()
    time.sleep(60)
