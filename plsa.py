from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle


def normalize(vec):
    s = sum(vec)
    assert(abs(s) != 0.0) # the sum must not be 0
    """
    if abs(s) < 1e-6:
        print "Sum of vectors sums almost to 0. Stop here."
        print "Vec: " + str(vec) + " Sum: " + str(s)
        assert(0) # assertion fails
    """

    for i in range(len(vec)):
        assert(vec[i] >= 0) # element must be >= 0
        vec[i] = vec[i] * 1.0 / s

class p_lsa(object):

    '''
    A collection of documents.
    '''

    def __init__(self):
        '''
        Initialize empty document list.
        '''
        self.documents = []
        self.vocab_list = None

    def PLSA(self, number_of_topics, max_iter,query,docs):

        '''
        Model topics.
        '''
        print("EM iteration begins...")
        # Get vocabulary and number of documents.


        try:
            with open("vocab_list.txt", "rb") as fp:  # Unpickling to avoid reruning the code
                vocab_list = pickle.load(fp)
        except IOError:
            vocab_list = []
            print('building vocabulary')
            for doc in tqdm(range(len(docs))):
                for sent in docs[doc]:
                    for word in sent:
                        if word not in self.vocab_list:
                            self.vocab_list.append(word)
            vocab_list.sort()
            with open("vocab_list.txt", "wb") as fp:  # Pickling
                pickle.dump(vocab_list, fp)

        docs=[' '.join(np.concatenate(item)) if item else '' for item in docs]
        print(docs)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(docs)
        # Step 2: Create a term-document matrix
        term_doc_matrix = X.T * X

        # self.build_vocabulary()
        number_of_documents = len(docs)
        vocabulary_size = len(vocab_list)
        #
        # # build term-doc matrix
        # term_doc_matrix = np.zeros([number_of_documents, vocabulary_size], dtype = np.int)
        # for d_index, doc in enumerate(self.documents):
        #     term_count = np.zeros(vocabulary_size, dtype = np.int)
        #     for word in doc.words:
        #         if word in self.vocabulary:
        #             w_index = self.vocabulary.index(word)
        #             term_count[w_index] = term_count[w_index] + 1
        #     term_doc_matrix[d_index] = term_count

        # Create the counter arrays.
        self.document_topic_prob = np.zeros([number_of_documents, number_of_topics], dtype=np.float) # P(z | d)
        self.topic_word_prob = np.zeros([number_of_topics, len(vocab_list)], dtype=np.float) # P(w | z)
        self.topic_prob = np.zeros([number_of_documents, len(vocab_list), number_of_topics], dtype=np.float) # P(z | d, w)

        # Initialize
        print("Initializing...")
        # randomly assign values
        self.document_topic_prob = np.random.random(size = (number_of_documents, number_of_topics))
        for d_index in range(len(docs)):
            normalize(self.document_topic_prob[d_index]) # normalize for each document
        self.topic_word_prob = np.random.random(size = (number_of_topics, len(vocab_list)))
        for z in range(number_of_topics):
            normalize(self.topic_word_prob[z]) # normalize for each topic
        """
        # for test, fixed values are assigned, where number_of_documents = 3, vocabulary_size = 15
        self.document_topic_prob = np.array(
        [[ 0.19893833,  0.09744287,  0.12717068,  0.23964181,  0.33680632],
         [ 0.27681925,  0.22971358,  0.1704416,   0.18248461,  0.14054095],
         [ 0.24768207,  0.25136754,  0.14392363,  0.14573845,  0.21128831]])
        self.topic_word_prob = np.array(
      [[ 0.02963563,  0.11659963,  0.06415405,  0.1291839 ,  0.09377842,
         0.09317023,  0.06140873,  0.023314  ,  0.09486251,  0.01538988,
         0.09189075,  0.06957687,  0.05015957,  0.05281074,  0.0140651 ],
       [ 0.09746902,  0.12212085,  0.07635703,  0.02799546,  0.0282282 ,
         0.03685356,  0.01256655,  0.03931912,  0.09545668,  0.00928434,
         0.11392475,  0.12089124,  0.02674909,  0.07219077,  0.12059333],
       [ 0.02209806,  0.05870101,  0.12101806,  0.03733935,  0.02550749,
         0.09906735,  0.0706651 ,  0.05619682,  0.10672434,  0.12259672,
         0.04218994,  0.10505831,  0.00315489,  0.03286002,  0.09682255],
       [ 0.0428768 ,  0.11598272,  0.08636138,  0.10917224,  0.05061344,
         0.09974595,  0.01647265,  0.06376147,  0.04468468,  0.01986342,
         0.10286377,  0.0117712 ,  0.08350884,  0.049046  ,  0.10327543],
       [ 0.02555784,  0.03718368,  0.10109439,  0.02481489,  0.0208068 ,
         0.03544246,  0.11515259,  0.06506528,  0.12720479,  0.07616499,
         0.11286584,  0.06550869,  0.0653802 ,  0.0157582 ,  0.11199935]])
        """
        # Run the EM algorithm
        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")
            print("E step:")
            for d_index, document in enumerate(docs):
                for w_index in range(vocabulary_size):
                    prob = self.document_topic_prob[d_index, :] * self.topic_word_prob[:, w_index]
                    if sum(prob) == 0.0:
                        print("d_index = " + str(d_index) + ",  w_index = " + str(w_index))
                        print("self.document_topic_prob[d_index, :] = " + str(self.document_topic_prob[d_index, :]))
                        print("self.topic_word_prob[:, w_index] = " + str(self.topic_word_prob[:, w_index]))
                        print("topic_prob[d_index][w_index] = " + str(prob))
                        exit(0)
                    else:
                        normalize(prob)
                    self.topic_prob[d_index][w_index] = prob
            print("M step:")
            # update P(w | z)
            for z in range(number_of_topics):
                for q,w_index in range(vocabulary_size):
                    s = 0
                    for d_index in range(len(docs)):
                        count = term_doc_matrix[d_index][w_index]
                        s = s + count * self.topic_prob[d_index, w_index, z]
                    print(w_index,s)
                    self.topic_word_prob[z][w_index] = s
                normalize(self.topic_word_prob[z])

            # update P(z | d)
            for d_index in range(len(docs)):
                for z in range(number_of_topics):
                    s = 0
                    for w_index in range(vocabulary_size):
                        count = term_doc_matrix[d_index][w_index]
                        s = s + count * self.topic_prob[d_index, w_index, z]
                    self.document_topic_prob[d_index][z] = s
#                print self.document_topic_prob[d_index]
#                assert(sum(self.document_topic_prob[d_index]) != 0)
                normalize(self.document_topic_prob[d_index])

        # twp = self.topic_word_prob.T
        # processed_query = preprocessDocs(query)
        # query_topic_word_prob = np.zeros([len(processed_query),number_of_topics], dtype=np.float)
        # for i,element in enumerate(processed_query):
        #     if element in vocab_list:
        #         w_index = vocab_list.index(element)
        #         query_topic_word_prob[i] = twp[w_index]
        #
        # query_topic_word_prob = query_topic_word_prob.T
        # query_reduced = query_topic_word_prob*self.document_topic_prob
        # similarity = cosine_similarity(lsa, query_reduced)

        # Transform the query into a TF-IDF vector representation
        query_vector = vectorizer.transform([query])

        # Calculate the query's topic distribution
        query_topic_dist = np.dot(query_vector, query_topic_word_prob.T)

        # Calculate the similarity between the query topic distribution and document topic distributions (P matrix)
        similarities = cosine_similarity(query_topic_dist, document_topic_prob)

        # Sort the document similarities in descending order and get the indices
        doc_indices = np.argsort(similarities.ravel())[::-1]

        # Retrieve the top relevant documents
        relevant_docs = [doc[i] for i in doc_indices]
        print(relevant_docs)


        #--------------------------------------------------------------------------------------
        # Step 6: Return the results

        # results = [(docs[i], similarity[i][0]) for i in range(len(docs))]
        # sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        #     # print(sorted_results[:5])
        #     # terms = vectorizer.get_feature_names_out()
        #     # for i, comp in enumerate(svd.components_):
        #     #     terms_comp = zip(terms, comp)
        #     #     sorted_terms = sorted(terms_comp, key=lambda x:x[1], reverse=True)[:10]
        #     #     print("Topic "+str(i)+": ")
        #     #     for t in sorted_terms:
        #     #         print(t)
        #     #     print(" ")
        #     # print(np.shape(similarity))
        # print(similarity)
        # top_results = np.argsort(similarity)[0][::-1]
        # top_results=[(i+1)%len(top_results) for i in top_results]
        # ans=[]
        # t=[]
        # for i, result in enumerate(top_results):
        #     if docs[result]:
        #         ans.append(result)
        #     else:
        #         t.append(result)
        # ans+=t
        # print(ans)
