from util import *


# Add your import statements here


class Evaluation:

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		query_doc_IDs_ordered : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		query_id : int
			The ID of the query in question
		true_doc_IDs : list
			The list of IDs of documents relevant to the query (ground truth)
		k : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

        precision = len(list(set(query_doc_IDs_ordered[:k]).intersection(true_doc_IDs))) / k

        # Fill in code here

        return precision

    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		doc_IDs_ordered : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		query_ids : list
			A list of IDs of the queries for which the documents are ordered
		qrels : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		k : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

        sum_precision = 0
        for query_no in range(len(query_ids)):
            sum_precision += self.queryPrecision(doc_IDs_ordered[query_no], query_ids[query_no],
                                                 reslist(query_ids[query_no], qrels), k)
        meanPrecision = sum_precision / len(query_ids)
        # Fill in code here

        return meanPrecision

    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		query_doc_IDs_ordered : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		query_id : int
			The ID of the query in question
		true_doc_IDs : list
			The list of IDs of documents relevant to the query (ground truth)
		k : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""
        try:
            recall = len(list(set(query_doc_IDs_ordered[:k]).intersection(true_doc_IDs))) / len(true_doc_IDs)
        except ZeroDivisionError:
            recall = 0
            print(query_id, true_doc_IDs)
        # Fill in code here

        return recall

    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		doc_IDs_ordered : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		query_ids : list
			A list of IDs of the queries for which the documents are ordered
		qrels : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		k : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""
        sum_recall = 0
        for query_no in range(len(query_ids)):
            sum_recall += self.queryRecall(doc_IDs_ordered[query_no], query_ids[query_no],
                                           reslist(query_ids[query_no], qrels), k)
        meanRecall = sum_recall / len(query_ids)

        # Fill in code here

        return meanRecall

    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		query_doc_IDs_ordered : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		query_id : int
			The ID of the query in question
		true_doc_IDs : list
			The list of IDs of documents relevant to the query (ground truth)
		k : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""
        prec = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        recal = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        if prec == 0 and recal == 0:
            fscore = 0
        else:
            fscore = 2 * ((prec * recal) / (prec + recal))

        # Fill in code here

        return fscore

    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		doc_IDs_ordered : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		query_ids : list
			A list of IDs of the queries for which the documents are ordered
		qrels : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		k : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

        sum_fscore = 0
        for query_no in range(len(query_ids)):
            sum_fscore += self.queryFscore(doc_IDs_ordered[query_no], query_ids[query_no],
                                           reslist(query_ids[query_no], qrels), k)
        meanFscore = sum_fscore / len(query_ids)

        # Fill in code here

        return meanFscore

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		query_doc_IDs_ordered : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		query_id : int
			The ID of the query in question
		true_doc_IDs : list
			The list of dictionaries containing IDs of documents relevant to the query (ground truth)
		k : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""
        query_score_list = scoremake(query_doc_IDs_ordered[:k], true_doc_IDs)
        if sorted(query_score_list, reverse=True)[0] == 0:
            nDCG = 0
        else:
            nDCG = dcg(query_score_list) / dcg(sorted(query_score_list, reverse=True))

        # Fill in code here

        return nDCG

    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		doc_IDs_ordered : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		query_ids : list
			A list of IDs of the queries for which the documents are ordered
		qrels : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		k : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""
        sum_ndcg = 0
        for query_no in range(len(query_ids)):
            sum_ndcg += self.queryNDCG(doc_IDs_ordered[query_no], query_ids[query_no],
                                       rellist(query_ids[query_no], qrels), k)
        meanNDCG = sum_ndcg / len(query_ids)

        # Fill in code here

        return meanNDCG

    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		query_doc_IDs_ordered : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		query_id : int
			The ID of the query in question
		true_doc_IDs : list
			The list of documents relevant to the query (ground truth)
		k : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""
        count = 1
        sum_precision = 0
        for i in range(k):
            if query_doc_IDs_ordered[i] in true_doc_IDs:
                count += 1
                sum_precision += self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i+1)

        avgPrecision = sum_precision / count

        # Fill in code here

        return avgPrecision

    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
        """
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		doc_IDs_ordered : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		query_ids : list
			A list of IDs of the queries
		q_rels : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		k : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

        sum_avergeprecision = 0
        for query_no in range(len(query_ids)):
            sum_avergeprecision += self.queryAveragePrecision(doc_IDs_ordered[query_no], query_ids[query_no],
                                                              reslist(query_ids[query_no], q_rels), k)
        meanAveragePrecision = sum_avergeprecision / len(query_ids)

        # Fill in code here

        return meanAveragePrecision
