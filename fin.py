from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation
from sbert import s_bert
from lsa1	 import LSA
from lsa import l_sa
import pickle
import time
import numpy as np
from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt
from probabilisticLSA import Problsa
# from probabilisticLSA import Grid_search,plot_metrics,calculate_metrics,rank_docs

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


class SearchEngine:

	def __init__(self, args):
		self.args = args

		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()
		self.informationRetriever = InformationRetrieval()
		self.evaluator = Evaluation()

	# def s_bert(self, query,docs):
	# 	return self. sbertapplied(query,docs)
	def segmentSentences(self, text):
		"""
		Call the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Call the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)

	def reduceInflection(self, text):
		"""
		Call the required stemmer/lemmatizer
		"""
		return self.inflectionReducer.reduce(text)

	def removeStopwords(self, text):
		"""
		Call the required stopword remover
		"""
		return self.stopwordRemover.fromList(text)



	def preprocessQueries(self, queries):
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""

		# Segment queries
		segmentedQueries = []
		for query in queries:
			segmentedQuery = self.segmentSentences(query)
			segmentedQueries.append(segmentedQuery)
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
		# Tokenize queries
		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQuery = self.tokenize(query)
			tokenizedQueries.append(tokenizedQuery)
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
		# Stem/Lemmatize queries
		reducedQueries = []
		for query in tokenizedQueries:
			reducedQuery = self.reduceInflection(query)
			reducedQueries.append(reducedQuery)
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
		# Remove stopwords from queries
		stopwordRemovedQueries = []
		for query in reducedQueries:
			stopwordRemovedQuery = self.removeStopwords(query)
			stopwordRemovedQueries.append(stopwordRemovedQuery)
		json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

		preprocessedQueries = stopwordRemovedQueries
		return preprocessedQueries

	def preprocessDocs(self, docs):
		"""
		Preprocess the documents
		"""

		# Segment docs
		segmentedDocs = []
		for doc in docs:
			segmentedDoc = self.segmentSentences(doc)
			segmentedDocs.append(segmentedDoc)
		json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
		# Tokenize docs
		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDoc = self.tokenize(doc)
			tokenizedDocs.append(tokenizedDoc)
		json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
		# Stem/Lemmatize docs
		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDoc = self.reduceInflection(doc)
			reducedDocs.append(reducedDoc)
		json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
		# Remove stopwords from docs
		stopwordRemovedDocs = []
		for doc in reducedDocs:
			stopwordRemovedDoc = self.removeStopwords(doc)
			stopwordRemovedDocs.append(stopwordRemovedDoc)
		json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

		preprocessedDocs = stopwordRemovedDocs
		return preprocessedDocs


	def evaluateDataset(self):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""

		# Read queries
		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries
		processedQueries = self.preprocessQueries(queries)

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
								[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)
		
		doc_IDs_ordered=[]
		if self.args.eval == "lsa":
			for queri in processedQueries:
				doc_IDs_ordered.append(l_sa(queri,processedDocs))
		if self.args.eval == "plsa":
			for queri in processedQueries:
				doc_IDs_ordered.append(P_lsa.problsa(60,queri,processedDocs))
		if self.args.eval == "s_bert":
			doc_IDs_ordered=[]
			for queri in processedQueries:
				doc_IDs_ordered.append(s_bert(queri,processedDocs))

		if self.args.eval =="vsm":
			self.informationRetriever.buildIndex(processedDocs, doc_ids)
			doc_IDs_ordered = self.informationRetriever.rank(processedQueries)


		# Read relevance judements
		qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
		for k in range(1, 11):
			precision = self.evaluator.meanPrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			precisions.append(precision)
			recall = self.evaluator.meanRecall(
				doc_IDs_ordered, query_ids, qrels, k)
			recalls.append(recall)
			fscore = self.evaluator.meanFscore(
				doc_IDs_ordered, query_ids, qrels, k)
			fscores.append(fscore)
			print("Precision, Recall and F-score @ " +
				str(k) + " : " + str(precision) + ", " + str(recall) +
				", " + str(fscore))
			MAP = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			MAPs.append(MAP)
			nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, k)
			nDCGs.append(nDCG)
			print("MAP, nDCG @ " +
				str(k) + " : " + str(MAP) + ", " + str(nDCG))

		# Plot the metrics and save plot
		plt.plot(range(1, 11), precisions, label="Precision")
		plt.plot(range(1, 11), recalls, label="Recall")
		plt.plot(range(1, 11), fscores, label="F-Score")
		plt.plot(range(1, 11), MAPs, label="MAP")
		plt.plot(range(1, 11), nDCGs, label="nDCG")
		plt.legend()
		plt.title("Evaluation Metrics - Cranfield Dataset")
		plt.xlabel("k")
		plt.savefig(args.out_folder + "eval_plot.png")


	def handleCustomQuery(self):
		"""
		Take a custom query as input and return top five relevant documents
		"""
		# print('Enter query id below')
		# q=int(input())
		q=1
		#Get query
		# print("Enter query below")
		# query = input()
		# Process documents


		# print(processedQuery)

		# Read queries
		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		query=queries[q-1]
		# print(query)
		processedQuery = self.preprocessQueries([query])[0]
		print(processedQuery)
		# Process documents
		# processedDocs = self.preprocessDocs(docs)
		try:
			with open("stopword_removed.txt", "rb") as fp:  # Unpickling to avoid reruning the code
				processedDocs = pickle.load(fp)
		except FileNotFoundError:
            # docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
            # doc_ids, docs = [item["id"] for item in docs_json], \[item["body"] for item in docs_json]
            # Read documents


			processedDocs = self.preprocessDocs(docs)
			with open("stopword_removed.txt", "wb") as fp:  # Pickling
				pickle.dump(processedDocs, fp)
		if self.args.custom == "lsa":
			doc_IDs_ordered = l_sa(query,processedDocs)

		if self.args.custom == "plsa":
			P_lsa=Problsa()
			doc_IDs_ordered = P_lsa.problsa(60,query,processedDocs)
		if self.args.custom == "s_bert":
			doc_IDs_ordered = s_bert(query,processedDocs)

		if self.args.custom == "vsm":
			self.informationRetriever.buildIndex(processedDocs, doc_ids)
			doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

		print(doc_IDs_ordered[0])
		# Read relevance judements
		qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
		for k in range(1, 11):
			precision = self.evaluator.meanPrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			precisions.append(precision)
			recall = self.evaluator.meanRecall(
				doc_IDs_ordered, query_ids, qrels, k)
			recalls.append(recall)
			fscore = self.evaluator.meanFscore(
				doc_IDs_ordered, query_ids, qrels, k)
			fscores.append(fscore)
			print("Precision, Recall and F-score @ " +
				str(k) + " : " + str(precision) + ", " + str(recall) +
				", " + str(fscore))
			MAP = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			MAPs.append(MAP)
			nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, k)
			nDCGs.append(nDCG)
			print("MAP, nDCG @ " +
				str(k) + " : " + str(MAP) + ", " + str(nDCG))

		# Plot the metrics and save plot
		plt.plot(range(1, 11), precisions, label="Precision")
		plt.plot(range(1, 11), recalls, label="Recall")
		plt.plot(range(1, 11), fscores, label="F-Score")
		plt.plot(range(1, 11), MAPs, label="MAP")
		plt.plot(range(1, 11), nDCGs, label="nDCG")
		plt.legend()
		plt.title("Evaluation Metrics - Cranfield Dataset")
		plt.xlabel("k")
		plt.savefig(args.out_folder + "eval_plot.png")
		plt.plot()


	def PROBLSA(self):
		"""
		Take a custom query as input and return top five relevant documents
		"""

		q=1

		# Read queries
		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]

        # Read relevance judements
		qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]
		processedQueries = self.preprocessQueries(queries)

		try:
			with open("stopword_removed.txt", "rb") as fp:  # Unpickling to avoid reruning the code
				processedDocs = pickle.load(fp)
		except FileNotFoundError:
			processedDocs = self.preprocessDocs(docs)
			with open("stopword_removed.txt", "wb") as fp:
				pickle.dump(processedDocs, fp)

		P_lsa =Problsa()
		ks, nDCGs, fscores = P_lsa.Grid_search(processedDocs, processedQueries, qrels, plot_search=True)

		print(ks, nDCGs, fscores)

	def L_SA(self):
		lsa = LSA()

		docs_df = pd.read_json('data/cranfield/cran_docs.json')
		docs_df.at[470, 'body'] = "<UNK>"
		docs_df.at[994, 'body'] = "<UNK>"
		queries_df = pd.read_json('data/cranfield/cran_queries.json')
		qrels_df = pd.read_json('data/cranfield/cran_qrels.json')
		qrels = json.load(open("data/cranfield/cran_qrels.json", 'r'))[:]

		tf_idf = lsa.get_tfidf_matrices(docs_df['body'], queries_df['query'],)
		docs = tf_idf['documents']
		queries = tf_idf['queries']

		docs_final, query_final = lsa.perform_svd(docs, queries, 500)

		ranked = lsa.rank_docs(docs_final, query_final)

		precisions, recalls, fscores, MAPs, nDCGs = lsa.calculate_metrics(ranked, qrels)

		lsa.plot_metrics(precisions, recalls, fscores, MAPs, nDCGs)

if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/",help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/",help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-custom',  default = "lsa",help = "Take custom query as input [lsa|plsa|s_bert|vsm]")
	parser.add_argument('-plsa', action = "store_true",help = "grid search on plsa")
	parser.add_argument('-lsa', action = "store_true",help = "grid search on lsa")
	parser.add_argument('-eval',  default = "lsa", action = "store_true",help = "evaluate dataset [lsa|plsa|s_bert|vsm]")

	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	# Either handle query from user or evaluate on the complete dataset
	if args.custom:
		searchEngine.handleCustomQuery()
	elif args.plsa:
		searchEngine.PROBLSA()
	elif args.lsa:
		searchEngine.L_SA()
	elif args.eval:
		searchEngine.evaluateDataset()
	else :
		searchEngine.handleCustomQuery()
