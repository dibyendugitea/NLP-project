from sentence_transformers import SentenceTransformer
import numpy as np

# Load pre-trained S-BERT model
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Example query and candidate documents
query = 'How to make pizza at home'
docs = ['Easy homemade pizza recipe', 'Making pizza from scratch', 'Pizza recipes for beginners', 'How to make perfect pizza dough', '10 tips for great homemade pizza']

# Generate query embedding
query_embedding = model.encode(query)

# Generate document embeddings
doc_embeddings = model.encode(docs)

# Calculate similarity scores
similarity_scores = np.dot(query_embedding, doc_embeddings.T)

print(similarity_scores)
# Sort documents by similarity scores
sorted_docs = [[i,doc] for i, doc in sorted(zip(similarity_scores, docs), reverse=True)]

# Retrieve top five documents
top_five_docs = sorted_docs[:5]

print(top_five_docs)
