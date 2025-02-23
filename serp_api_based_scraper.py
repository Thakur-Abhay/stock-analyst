import openai
import asyncio
import requests
from newspaper import Article
import numpy as np
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
openai.api_key = ''
SERPAPI_KEY = '76c5172b07ee2ddf20fdc493f3d0ce1e720c2161e1490ea289dc820c95c8c78d'  # Add your SerpAPI key here
# Function to interpret user query using OpenAI
async def interpret_query(query: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that refines user queries for optimal web search. Make a precise and concise query that can be used for an effective web search"},
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
def get_embeddings(text: str, model_name: str = "text-embedding-ada-002") -> np.ndarray:
    """
    Returns an average embedding (as a NumPy array) for the given text using
    OpenAI's text-embedding-ada-002 model, automatically handling large inputs
    by splitting them into batches.
    """
    # 1. Encode text into tokens
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    
    # 2. Define a safe chunk size under the model's limit (some overhead recommended)
    chunk_size = 7000
    
    # 3. If text is small enough, just embed it directly
    if len(tokens) <= chunk_size:
        response = openai.Embedding.create(
            model=model_name,
            input=text
        )
        return np.array(response['data'][0]['embedding'])
    
    # 4. Otherwise, split tokens into multiple chunks
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunks.append(tokens[start:end])
        start = end
    
    # 5. Embed each chunk separately
    embeddings_list = []
    for chunk_tokens in chunks:
        chunk_text = enc.decode(chunk_tokens)
        response = openai.Embedding.create(
            model=model_name,
            input=chunk_text
        )
        emb = np.array(response['data'][0]['embedding'])
        embeddings_list.append(emb)
    
    # 6. Average all chunk embeddings to get a single embedding vector
    stacked = np.vstack(embeddings_list)
    final_embedding = np.mean(stacked, axis=0)
    
    return final_embedding

# Function to extract answer from context using OpenAI's GPT-4
async def extract_answer_from_context(context: str, query: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are provided with a context and a question. Your task is to extract the answer to the question from the context."},
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
        answer = await extract_answer_from_context(combined_context, refined_query)
        print(f"\n\n\nFinal Answer: {answer}")
    else:
        print("No relevant results found.")

# Run the script with user input
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    asyncio.run(get_real_time_info(user_query))
