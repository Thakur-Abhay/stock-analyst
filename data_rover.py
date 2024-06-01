import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search


class ContextEnricher():
    def __init__(self) -> None:
        pass

    def google_search(self, query, num_results=3):
        # Perform the search and fetch the top results
        results = search(query, num_results=num_results)

        # Function to fetch and parse webpage content
        def fetch_and_parse(url):
            try:
                response = requests.get(url)
                response.raise_for_status()  # Check for request errors
                soup = BeautifulSoup(response.content, 'html.parser')
                return soup.get_text()  # Extract text content from HTML
            except requests.exceptions.RequestException as e:
                print(f"Error fetching {url}: {e}")
                return ""

        # Fetch and print content from each webpage
        all_content = {}
        for url in results:
            content = fetch_and_parse(url)
            all_content[url] = content

        return all_content
