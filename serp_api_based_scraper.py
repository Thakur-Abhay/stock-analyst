import openai
import asyncio
import requests
from newspaper import Article
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
openai.api_key = ''
SERPAPI_KEY = '76c5172b07ee2ddf20fdc493f3d0ce1e720c2161e1490ea289dc820c95c8c78d'  # Add your SerpAPI key here
# Function to interpret user query using OpenAI
async def interpret_query(query: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that refines user queries for optimal web search."},
            {"role": "user", "content": query}
        ]
    )
    refined_query = response['choices'][0]['message']['content']
    return refined_query

# Function to perform a web search using SerpAPI
async def serpapi_search(refined_query: str):
    search_url = f"https://serpapi.com/search.json?engine=google&q={refined_query}&api_key={SERPAPI_KEY}"
    response = requests.get(search_url)
    data = response.json()
    
    search_results = []
    for result in data.get('organic_results', []):
        url = result.get('link')
        snippet = result.get('snippet')
        if url and snippet:
            search_results.append((url, snippet))
    
    return search_results

# Function to extract main content from a URL using newspaper3k
def extract_content_from_url(url: str) -> str:
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return None

# Function to check relevance using embeddings and cosine similarity
async def check_relevance(content: str, query: str) -> bool:
    content_embedding = get_embeddings(content)
    query_embedding = get_embeddings(query)
    similarity_score = cosine_similarity([content_embedding], [query_embedding])[0][0]
    return similarity_score > 0.7

# Function to get embeddings of a text using OpenAI's embedding model
def get_embeddings(text: str) -> np.ndarray:
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response['data'][0]['embedding'])

# Function to extract answer from context using OpenAI's GPT-4
async def extract_answer_from_context(context: str, query: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts answers from the given context."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {query}"}
        ]
    )
    return response['choices'][0]['message']['content']

# Main function to handle user query and search
async def get_real_time_info(query: str):
    print(f"User Query: {query}")
    
    # Step 1: Interpret the query
    refined_query = await interpret_query(query)
    print(f"Refined Query: {refined_query}")
    
    # Step 2: Search using SerpAPI
    search_results = await serpapi_search(refined_query)
    
    # Step 3: Extract content and check relevance
    relevant_content = []
    for idx, (url, snippet) in enumerate(search_results, 1):
        print(f"Processing URL {idx}: {url}")
        
        # Initial relevance check using snippet
        if await check_relevance(snippet, refined_query):
            content = extract_content_from_url(url)
            if content and await check_relevance(content, refined_query):
                relevant_content.append(content)
    
    # Step 4: Use aggregated content as context for the query
    if relevant_content:
        combined_context = "\n".join(relevant_content)
        print(f"\nUsing the following aggregated content as context:\n{combined_context[:1000]}...")
        
        # Step 5: Extract the answer using the aggregated context
        answer = await extract_answer_from_context(combined_context, query)
        print(f"\n\n\nFinal Answer: {answer}")
    else:
        print("No relevant results found.")

# Run the script with user input
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    asyncio.run(get_real_time_info(user_query))
