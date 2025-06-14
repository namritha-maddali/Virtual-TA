import os
import json
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

BASE_URL = "https://tds.s-anand.net/#/2025-01/"
MAIN_SITE = "https://tds.s-anand.net/"
OUTPUT_JSON = "data/raw/tds_site_text.json"
OUTPUT_TXT = "data/raw/tds_site_text.txt"

def scrape_tds_site():
    os.makedirs("data/raw", exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print(f"Visiting {BASE_URL}")
        page.goto(BASE_URL, wait_until="networkidle")

        time.sleep(3)

        soup = BeautifulSoup(page.content(), "html.parser")
        all_links = set()

        for a in soup.find_all("a", href=True):
            href = a['href']
            if href.startswith("#/") and "http" not in href:
                full_url = urljoin(MAIN_SITE, href)
                all_links.add(full_url)

        print(f"Found {len(all_links)} internal pages")

        results = []
        plain_text = []

        for i, url in enumerate(sorted(all_links)):
            try:
                print(f"[{i+1}/{len(all_links)}] Loading {url}")
                page.goto(url, wait_until="networkidle")
                time.sleep(1.5)

                content_html = page.content()
                content_soup = BeautifulSoup(content_html, "html.parser")

                title = content_soup.title.string.strip() if content_soup.title else "Untitled"
                main_text = content_soup.get_text(separator="\n", strip=True)

                item = {
                    "url": url,
                    "title": title,
                    "content": main_text
                }

                results.append(item)
                plain_text.append(f"{title}\n{url}\n{main_text}\n\n")

            except Exception as e:
                print(f"Failed to load {url}: {e}")

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
            f.writelines(plain_text)

        print(f"Saved {len(results)} pages to {OUTPUT_JSON}")
        browser.close()

if __name__ == "__main__":
    scrape_tds_site()
