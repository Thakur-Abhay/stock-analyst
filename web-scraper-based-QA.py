import openai
import asyncio
import requests
from googlesearch import search
from newspaper import Article  # Using newspaper3k for content extraction
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
openai.api_key = ''  # Your OpenAI API Key

# Function to interpret user query using the latest OpenAI API
async def interpret_query(query: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant that refines user queries for optimal web search."},
                  {"role": "user", "content": query}]
    )
    refined_query = response['choices'][0]['message']['content']
    return refined_query

# Function to perform a web search using Googlesearch-python
async def web_search(refined_query: str):
    search_results = []
    for url in search(refined_query, num_results=5):  # Set num_results for top results
        search_results.append(url)
    return search_results

# Function to extract content from a webpage using newspaper3k
def extract_content_from_url(url: str) -> str:
    # Using the library newspaper3k to retrieve data from the news articles in the url
    try:
        article = Article(url)
        article.download()
        article.parse()

        # Get the article's main content
        content = article.text
        return content
        # return content[:1000]  # Limit to first 1000 characters for brevity
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return None

# Function to get embeddings of a text using OpenAI's embedding model
async def get_embeddings(text: str) -> np.ndarray:
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Use the embedding model
        input=text
    )
    return np.array(response['data'][0]['embedding'])

# Function to check relevance of the content using embeddings
async def check_relevance(content: str, query: str) -> bool:
    # Get embeddings for the content and the query
    content_embedding_task = get_embeddings(content)
    query_embedding_task = get_embeddings(query)
    content_embedding, query_embedding = await asyncio.gather(content_embedding_task, query_embedding_task)
    # Compute cosine similarity
    similarity_score = cosine_similarity([content_embedding], [query_embedding])[0][0]

    # Set a threshold for relevance (e.g., 0.7 - you can adjust this based on testing)
    if similarity_score > 0.7:
        return True
    return False

# Function to extract the answer from the content using OpenAI's GPT-4
async def extract_answer_from_context(context: str, query: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a helpful assistant that extracts answers from the given context."},
                  {"role": "user", "content": f"Given the following context: {context}\nAnswer the following question: {query}"}]
    )
    answer = response['choices'][0]['message']['content']
    return answer

# Main function to handle user query and search
async def get_real_time_info(query: str):
    print(f"User Query: {query}")
    
    # Step 1: Interpret the query
    refined_query = await interpret_query(query)
    print(f"Refined Query: {refined_query}")
    
    # Step 2: Search the web
    search_results = await web_search(refined_query)
    
    # Step 3: Extract content and check relevance
    relevant_content = []  # Aggregated content from all relevant pages
    for idx, url in enumerate(search_results, 1):
        print(f"Processing URL {idx}: {url}")
        
        # Extract content from the page using newspaper3k
        content = extract_content_from_url(url)
        
        if content:
            # Check if content is relevant to the query using embeddings
            if await check_relevance(content, refined_query):
                relevant_content.append(content)  # Add relevant content to the list
    
    # Step 4: Use aggregated content as context for the query
    if relevant_content:
        # Combine all relevant content into a single context
        combined_context = "\n".join(relevant_content)
        print(f"\nUsing the following aggregated content as context for the query:\n{combined_context[:1000]}...")  # Display a snippet for debugging
        
        # Step 5: Extract the answer using the aggregated context
        answer = await extract_answer_from_context(combined_context, query)
        print(f"\n\n\nFinal Answer: {answer}")
    else:
        print("No relevant results found.")

# Run the script with user input
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    asyncio.run(get_real_time_info(user_query))
