from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer
import numpy as np

def l_sa(query,docs):
    # Step 1: Collect and preprocess the text corpus
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(docs)
    # Step 2: Create a term-document matrix
    td_matrix = X.T * X

    # Step 3: Apply singular value decomposition (SVD)
    svd = TruncatedSVD(n_components=1)
    # Step 4: Reduce the dimensionality
    lsa = svd.fit_transform(td_matrix)
    # Normalize the LSA matrix
    lsa = Normalizer(copy=False).fit_transform(lsa)

    # Step 5: Calculate document similarity
    query_vector = vectorizer.transform([query])
    # print(np.shape(query_vector.T),np.shape(query_vector))
    query_reduced = svd.transform(query_vector)
    # print(np.shape(lsa),np.shape(query_reduced))
    similarity = cosine_similarity(lsa, query_reduced)

    # Step 6: Return the results
    results = [(docs[i], similarity[i][0]) for i in range(len(docs))]
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    # print(sorted_results[:5])
    # terms = vectorizer.get_feature_names_out()
    # for i, comp in enumerate(svd.components_):
    #     terms_comp = zip(terms, comp)
    #     sorted_terms = sorted(terms_comp, key=lambda x:x[1], reverse=True)[:10]
    #     print("Topic "+str(i)+": ")
    #     for t in sorted_terms:
    #         print(t)
    #     print(" ")
    # print(np.shape(similarity))
    print(similarity)
    top_results = np.argsort(similarity)[0][::-1]
    top_results=[(i+1)%len(top_results) for i in top_results]
    ans=[]
    t=[]
    for i, result in enumerate(top_results):
        if docs[result]:
            ans.append(result)
        else:
            t.append(result)
    ans+=t
    print(ans)
