import argparse
import aiohttp
import asyncio
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

semaphore = asyncio.Semaphore(5)  

def parse_feed(feed_url):
    try:
        feed = feedparser.parse(feed_url)
        return [entry.link for entry in feed.entries if hasattr(entry, 'link')]
    except Exception as e:
        logger.error(f"Error parsing feed {feed_url}: {e}")
        return []

async def fetch_content(session, url, retries=3):
    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                return await response.text()
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(2)  # Retry delay
                continue
            logger.error(f"Failed to fetch content from {url}: {e}")
            return ""

async def process_feed(feed_url, session):
    try:
        post_urls = parse_feed(feed_url)
        if not post_urls:
            logger.warning(f"No valid URLs found in feed {feed_url}")
            return []

        tasks = [fetch_with_semaphore(semaphore, session, post_url) for post_url in post_urls]
        post_contents = await asyncio.gather(*tasks)
        cleaned_contents = [clean_content(content) for content in post_contents]
        return list(zip(post_urls, cleaned_contents))
    except Exception as e:
        logger.error(f"Error processing feed {feed_url}: {e}")
        return []

async def fetch_with_semaphore(semaphore, session, url):
    async with semaphore:
        return await fetch_content(session, url)

def clean_content(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return " ".join(chunk for chunk in chunks if chunk)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed-path", required=True, help="Path to file containing feed URLs")
    return parser.parse_args()

async def main(feed_file):
    async with aiohttp.ClientSession() as session:
        with open(feed_file, "r") as file:
            feed_urls = [line.strip() for line in file]

        tasks = [process_feed(feed_url, session) for feed_url in feed_urls]
        results = await asyncio.gather(*tasks)

    flattened_results = [item for sublist in results for item in sublist]
    df = pd.DataFrame(flattened_results, columns=["URL", "content"])
    df.to_parquet("output.parquet", index=False)


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.feed_path))
