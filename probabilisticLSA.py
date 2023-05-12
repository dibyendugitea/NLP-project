import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# import plsa.Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import pickle
# from plsa import Corpus
from evaluation import Evaluation
import matplotlib.pyplot as plt
from plsa.corpus import Corpus
from plsa.pipeline import DEFAULT_PIPELINE,Pipeline
from plsa.algorithms import PLSA
# define your data and queries here
data = ["doc1 text", "doc2 text", ...]
queries = ["query1", "query2", ...]

# define your hyperparameters and grid search values here
num_topics_values = [10, 20, 30, 40, 50]
# add other hyperparameters you want to tune here

best_model = None
best_score = 0.0

class Problsa(object):
# perform grid search over hyperparameters

    '''
    A collection of documents.
    '''

    def __init__(self):
        '''
        Initialize empty document list.
        '''
        self.documents = []
        self.vocab_list = None
        self.evaluator = Evaluation()

    def rank_docs(self,result,docs,query):
        tdg=np.array(result.topic_given_doc)
        print(query)
        query_topic_dists,number_of_new_words, new_words = result.predict(query)
        topics = np.array(query_topic_dists)

        # pd_t = np.multiply(topics,tdg)
        # # print(tdg.shape)
        # # print(pd_t.shape)
        # # sum the array column-wise
        # col_sums = np.sum(pd_t.T, axis=0)
        print(query_topic_dists.shape)
        print(tdg.shape)
        print(query_topic_dists)
        query_topic_dists.reshape(1,60)

        query_topic_dists = np.array(query_topic_dists)
        similarity_scores = np.dot(query_topic_dists,tdg.T)
        
        print(similarity_scores.shape)
        top_results = np.argsort(similarity_scores)[::-1]
        top_results=[(i)%len(top_results) for i in top_results]
        ans=[]
        t=[]
        for i, result in enumerate(top_results):
            if docs[result]:
                ans.append(result)
        ans+=t

        return ans

    def problsa(self, number_of_topics,query,docs):

        '''
        Model topics.
        '''
        # try:
        #
        #
        #     # print(result.topic_given_doc)
        #     # print(result.word_given_topic)
        #     # print(result.topic)
        #     # with open("plsacorpus.txt", "wb") as fp1:  # Unpickling to avoid reruning the code
        #     #     corpus= pickle.load(fp1)
        #
        #     plsa = PLSA(corpus, number_of_topics, True)
        #     # result = plsa.fit()
        #
        #     result = plsa.best_of(5)
        #     # with open("plsaresult"+str(number_of_topics)+".txt", "rb") as fp:  # Unpickling to avoid reruning the code
        #     #     plsaresult = pickle.load(fp)
        #
        #     # with open("plsawgt"+str(number_of_topics)+".txt", "rb") as fp2:  # Unpickling to avoid reruning the code
		# 	# 	word_given_topic = pickle.load(fp2)
        #     # with open("plsat"+str(number_of_topics)+".txt", "rb") as fp3:  # Unpickling to avoid reruning the code
		# 	# 	topic = pickle.load(fp3)
        # except FileNotFoundError,UnboundLocalError:
        #     pipeline = Pipeline(*DEFAULT_PIPELINE)
        #     corpus = Corpus(docs,pipeline)
        #     plsa = PLSA(corpus, number_of_topics, True)
        #     # result = plsa.fit()
        #
        #     result = plsa.best_of(5)
        #     # print(result.topic_given_doc)
        #     # print(result.word_given_topic)
        #     # print(result.topic)
        #     with open("plsacorpus.txt", "wb") as fp:  # pickling to avoid reruning the code
        #         pickle.dump(corpus,fp)
        #     with open("plsatgd"+str(number_of_topics)+".txt", "wb") as fp:  # pickling to avoid reruning the code
        #         pickle.dump(result,fp)
        #     with open("plsatgd"+str(number_of_topics)+".txt", "wb") as fp1:  # pickling to avoid reruning the code
        #         pickle.dump(result.topic_given_doc,fp1)
        #     # with open("plsawgt"+str(number_of_topics)+".txt", "wb") as fp2:  # pickling to avoid reruning the code
		# 	# 	pickle.dump(result.word_given_topic,fp2)
        #     # with open("plsat"+str(number_of_topics)+".txt", "wb") as fp3:  # pickling to avoid reruning the code
		# 	# 	pickle.dump(result.topic,fp3)
        #     tdg=np.array(result.topic_given_doc)

        pipeline = Pipeline(*DEFAULT_PIPELINE)
        corpus = Corpus(docs,pipeline)
        plsa = PLSA(corpus, number_of_topics, True)
        # result = plsa.fit()

        result = plsa.fit(5)

        tgd = result.topic_given_doc
        print(tgd.shape)
        print(tgd)
        # print(result.word_given_topic)
        # print(result.topic)
        # print(type(corpus))
        # with open("plsacorpus.txt", "wb") as fp:  # pickling to avoid reruning the code
        #     pickle.dump(corpus,fp)
        # with open("plsatgd"+str(number_of_topics)+".txt", "wb") as fp:  # pickling to avoid reruning the code
        #     pickle.dump(result,fp)
        # with open("plsatgd"+str(number_of_topics)+".txt", "wb") as fp1:  # pickling to avoid reruning the code
        #     pickle.dump(result.topic_given_doc,fp1)
        # with open("plsawgt"+str(number_of_topics)+".txt", "wb") as fp2:  # pickling to avoid reruning the code
        # 	pickle.dump(result.word_given_topic,fp2)
        # with open("plsat"+str(number_of_topics)+".txt", "wb") as fp3:  # pickling to avoid reruning the code
        # 	pickle.dump(result.topic,fp3)
        return self.rank_docs(result,docs,query)


    def calculate_metrics(
        self, ranked, qrels, grid_search=False, print_metrics=False
    ):
        precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
        q_ids = np.arange(225)+1
        for k in range(1, 11):
            precision = self.evaluator.meanPrecision(
                ranked, q_ids, qrels, k)
            precisions.append(precision)
            recall = self.evaluator.meanRecall(
                ranked, q_ids, qrels, k)
            recalls.append(recall)
            fscore = self.evaluator.meanFscore(
                ranked, q_ids, qrels, k)
            fscores.append(fscore)
            if print_metrics:
                print("Precision, Recall and F-score @ " +
                    str(k) + " : " + str(precision) + ", " + str(recall) +
                    ", " + str(fscore))
            MAP = self.evaluator.meanAveragePrecision(
                ranked, q_ids, qrels, k)
            MAPs.append(MAP)
            nDCG = self.evaluator.meanNDCG(
                ranked, q_ids, qrels, k)
            nDCGs.append(nDCG)
            if print_metrics:
                print("MAP, nDCG @ " +
                        str(k) + " : " + str(MAP) + ", " + str(nDCG))

        if grid_search:
            return np.max(nDCG), np.max(fscore)
        else:
            return precisions, recalls, fscores, MAPs, nDCGs

    def plot_metrics(self,precisions, recalls, fscores, MAPs, nDCGs ):

        # Plot the metrics and save plot
        plt.plot(range(1, 11), precisions, label="Precision")
        plt.plot(range(1, 11), recalls, label="Recall")
        plt.plot(range(1, 11), fscores, label="F-Score")
        plt.plot(range(1, 11), MAPs, label="MAP")
        plt.plot(range(1, 11), nDCGs, label="nDCG")
        plt.legend()
        plt.title("Evaluation Metrics - Cranfield Dataset")
        plt.xlabel("k")
        plt.show()

    def Grid_search(self, docs, queries, qrels, plot_search=False):
        """
        Grid search on max nDCG@k and F-score
        """
        nDCGs = []
        fscores = []
        ks = []
        max_k = 120
        min_k = 20
        step = 20
        pipeline = Pipeline(*DEFAULT_PIPELINE)
        corpus = Corpus(docs,pipeline)


        for i,k in enumerate(np.arange(min_k,max_k,step)):
            print(f"Evaluating {i+1} of {(max_k-min_k)//step}", end="\r")

            plsa = PLSA(corpus, k, True)
            # result = plsa.fit()

            result = plsa.fit(eps = 1e-02, max_iter = 5, warmup = 2)
            # eps = 1e-05, max_iter = 200, warmup = 5
            # docs_final, queries_final = self.perform_svd(docs, queries, k)
            # ranked = self.rank_docs(docs_final, queries_final)
            ranked=[]
            for queri in queries:
                ranked.append(self.rank_docs(result,docs,queri))

            nDCG, fscore = self.calculate_metrics(
                ranked, qrels,
                grid_search=True,
                print_metrics=False,
            )
            ks.append(k)
            nDCGs.append(nDCG)
            fscores.append(fscore)

        if plot_search:
            plt.figure(figsize=(10,5))
            plt.plot(ks, nDCGs, label='Max nDCG')
            plt.plot(ks, fscores, label='Max F-score')
            plt.xlabel('Number of components retained')
            plt.ylabel('Metric value')
            plt.legend()
            plt.show()

        return ks, nDCGs, fscores
