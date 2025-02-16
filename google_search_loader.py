from googlesearch import search

query = "current prime minister of india?"
results = []

# Note that we use num_results=, not num=
for url in search(query, num_results=5):
    results.append(url)

if not results:
    print("No results found.")
else:
    print("Found results:")
    for r in results:
        print(r)
