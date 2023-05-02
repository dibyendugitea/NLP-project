from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
import json


def s_bert(query,docs):
    # Load pre-trained S-BERT model
    model_name = "sentence-transformers/all-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    sbert_model = SentenceTransformer(model_name)

    query_embeddings = sbert_model.encode(query, convert_to_tensor=True)

    
    doc_embeddings = sbert_model.encode(docs, convert_to_tensor=True)
    #print(query_embeddings.T)
    #print(np.shape(query_embeddings))
    #print(np.shape(doc_embeddings))
    # Calculate similarity
    similarity_scores = np.dot(query_embeddings, doc_embeddings.T)
    #json.dump(similarity_scores, open("s_bert.txt", 'w'))
    #print(similarity_scores[:2])
    # Rank documents and present top results
    top_results = np.argsort(similarity_scores)[0][::-1]
    top_results=[(i+1)%len(top_results) for i in top_results]
    ans=[]
    t=[]
    for i, result in enumerate(top_results):
        if docs[result]:
            ans.append(result)
        else:
            t.append(result)
    ans+=t
    return ans