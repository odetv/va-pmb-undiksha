import feedparser
import json
import re
from bs4 import BeautifulSoup


def scrap_news():
    rss_url = "https://undiksha.ac.id/feed/"
    feed = feedparser.parse(rss_url)
    data = []

    def clean_text(text):
        text = text.replace('\u2013', '-')
        text = re.sub(r'\[\u2026\]', '', text)
        text = re.sub(r'\n', '', text)
        text = text.replace('[]', '')
        return text

    for entry in feed.entries:
        pubDate_cleaned = entry.published.replace(" +0000", "")
        soup = BeautifulSoup(entry.description, "html.parser")
        description_cleaned = soup.get_text()
        description_cleaned = clean_text(description_cleaned)

        item = {
            "title": entry.title,
            "link": entry.link,
            "pubDate": pubDate_cleaned,
            "description": description_cleaned.strip()
        }
        data.append(item)

    news_scrapper = json.dumps(data, indent=4)
    return news_scrapper

# result = scrap_news()
# print(result)
# print("Berita Undiksha Berhasil di Scrapping.")