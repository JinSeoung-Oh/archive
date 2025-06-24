! pip install feedparser

import feedparser
import time
import random
import requests
import json
import schedule
import os

from datetime import datetime, timedelta, timezone
from dateutil import parser
from dateutil.tz import gettz
from tenacity import retry, stop_after_attempt, wait_exponential

from openai import OpenAI

# ─── OpenAI Client 초기화 ─────────────────────────────────────────────────────
os.environ["OPENAI_API_KEY"]="sk-proj-......"
client = OpenAI()

# ─── tzinfos 매핑 (예: EDT 같은 약어들) ─────────────────────────────────────────
Today     = datetime.now().strftime('%B %d, %Y')
This_year = datetime.now().strftime('%Y')
today     = datetime.now()
yesterday = (today - timedelta(days=1)).strftime('%B %d, %Y')

validation_prompt = f"""당신은 주어진 뉴스들을 한국어로 번역하는 번역가입니다.

주의사항:
1. 반드시 {This_year}년 뉴스만 포함하세요.
2. {This_year}년에 발행된 것들 중에서도 다음 시간 범위에 해당하는 것만 포함하세요:
   - UTC/GMT 기준: {yesterday} 15:00 ~ {Today} 14:59
   - 한국 시간 기준: {Today} 00:00 ~ 23:59
3. 다양한 날짜 형식이 있으므로 각 뉴스의 발행일을 주의깊게 확인해주세요
4. 뉴스가 누락되지 않도록 모든 날짜 형식을 고려해주세요
5. 출판 날짜를 반드시 엄수해주세요.
6. 뉴스 내용이 로봇, 로봇 AI와 관련 없는 것들은 리턴하지 마세요.

출력 형식:
주어진 내용을 분석한 결과, 다음과 같은 주요 뉴스들이 있습니다:

순번. 제목: [한국어 제목]
링크: [원문 링크 URL]
요약: [한국어로 상세 요약]"""

# ─── tzinfos 매핑 ──────────────────────────────────────────────────────────────
TZINFOS = {
    "EDT": gettz("US/Eastern"),
    "EST": gettz("US/Eastern"),
    "PDT": gettz("US/Pacific"),
    "PST": gettz("US/Pacific"),
}

def parse_utc(dt_str: str) -> datetime:
    dt = parser.parse(dt_str, tzinfos=TZINFOS)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

# ─── RSS 파싱 및 필터링 ────────────────────────────────────────────────────────
def read_rss(url: str):
    feed = feedparser.parse(url)
    result = []
    for entry in feed.entries:
        published = getattr(entry, 'published', None)
        pub_dt     = parse_utc(published) if published else None
        result.append({
            "title":     getattr(entry, 'title',   None),
            "link":      getattr(entry, 'link',    None),
            "published": published,
            "pub_dt":    pub_dt,
            "summary":   getattr(entry, 'summary', None)
        })
    return result

def get_all_news(feed_list):
    all_news = []
    for url in feed_list:
        try:
            all_news.extend(read_rss(url))
        except Exception as e:
            print(f"피드 {url} 오류: {e}")
    all_news.sort(
        key=lambda x: x['pub_dt'] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True
    )
    print(f"총 수집된 뉴스: {len(all_news)}")
    return all_news

def filter_news_by_date(news_items, start_date: datetime, end_date: datetime):
    return [
        item for item in news_items
        if item.get('pub_dt') and start_date <= item['pub_dt'] <= end_date
    ]

def format_for_openai(items):
    out = ""
    for i, it in enumerate(items, 1):
        out += f"\n#{i}\n"
        out += f"Title: {it['title']}\nLink: {it['link']}\n"
        out += f"Published (UTC): {it['pub_dt'].isoformat()}\n"
        out += f"Summary: {it['summary']}\n" + "-"*50 + "\n"
    return out

# ─── OpenAI 호출부 ─────────────────────────────────────────────────────────────
def get_response_from_openai(context: str, sys_prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": f"다음 내용을 분석해줘:\n{context}"}
        ],
        max_tokens=4096,
        temperature=0.0,
    )
    return resp.choices[0].message.content

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_response_with_retry(context: str, sys_prompt: str) -> str:
    try:
        return get_response_from_openai(context, sys_prompt)
    except Exception as e:
        print(f"오류 발생: {e}. 재시도 중...")
        time.sleep(random.uniform(1, 3))
        raise

feed_list =["https://robohub.org/feed/", "http://export.arxiv.org/rss/cs/RO", "https://www.science.org/action/showFeed?type=etoc&feed=rss&jc=scirobotics"
           ,"https://www.nature.com/npjrobot.rss","https://www.therobotreport.com/feed/", "https://www.roboticstomorrow.com/rss/news","https://www.roboticsandautomationnews.com/feed"
           ,"https://spectrum.ieee.org/feeds/topic/robotics.rss", "https://www.theguardian.com/technology/robots/rss", "https://news.mit.edu/topic/mitrobotics-rss.xml"
           ,"https://blog.robotiq.com/rss.xml", "https://unite.ai/category/robotics/feed", "https://techxplore.com/rss-feed/robotics-news/", "https://clearpathrobotics.com/feed"
           ,"https://makezine.com/category/technology/robotics/feed/"]

all_news = get_all_news(feed_list)
utc_start = parse_utc(f"{(datetime.utcnow() - timedelta(days=1)).strftime('%B %d, %Y')} 15:00 UTC")
utc_end   = parse_utc(f"{datetime.utcnow().strftime('%B %d, %Y')} 14:59 UTC")
to_translate = filter_news_by_date(all_news, utc_start, utc_end)

context = format_for_openai(to_translate)
result  = get_response_with_retry(context, validation_prompt)

print(result)
