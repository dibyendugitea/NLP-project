from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer
import numpy as np
import pickle

def l_sa(query,docs):

    vectorizer = TfidfVectorizer()

    corpus=[' '.join(np.concatenate(item)) if item else '' for item in docs]

    X = vectorizer.fit_transform(corpus)

    svd = TruncatedSVD(n_components=2)
    lsa = svd.fit_transform(X)
    query_vector = vectorizer.transform([' '.join(query[0])])

    query_reduced = svd.transform(query_vector)
    similarity = cosine_similarity(query_reduced,lsa)
    top_results = np.argsort(similarity[0])[::-1]
    # print(top_results)
    # top_results=[(i)%len(top_results) for i in top_results]
    ans=[]
    t=[]
    for i, result in enumerate(top_results):
        if docs[result]:
            ans.append(result)
        else:
            t.append(result)
    ans+=t
    # print(ans)
    return ans
