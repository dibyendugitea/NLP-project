import json
import re
import os

import pandas as pd
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from evaluation import Evaluation

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk import sent_tokenize
stemmer = SnowballStemmer('english')

sns.set()

class LSA():
    
    def __init__(self, blacklist=None, whitelist=None):
        nltk.download('stopwords',quiet=True)
        self.stop_words = set(stopwords.words('english')) #creating a set with all the stopwords
        self.basic_punct = [
            '\'', '"', '“', '”',    # quotations
            ',', '/', ';', ':', '&', # mid-sentence punctuations
            '!', '.', '?',    # end-sentence punctuations
        ]
        
        # FIX: convert to self
        self.lemmatizer = WordNetLemmatizer()
        self.evaluator = Evaluation()

    def get_tfidf_matrices(
        self,
        documents, queries,
        use_lemmatization=False,
        use_bigram=False,
        blacklist=None,
        whitelist=None,
    ):
        def tokenizer_stemming(input_str):
            # Tokenization    
            tokens = [ word for word in re.split(
                '(['+"".join(punct)+'\s])', input_str
            ) if word not in (' ', '') ]
                
            final_tokens = []
            
            for idx,token in enumerate(tokens): #we need positional info as we're replacing with stemmed token
                final_tokens.append(stemmer.stem(token))

            return final_tokens
        def tokenizer_lemm(input_str):
            final_tokens = []
            sents = sent_tokenize(input_str)
            
            for sent in sents:
                # Tokenization
                tokens = [ word for word in re.split(
                    '(['+"".join(punct)+'\s])', sent
                ) if word not in (' ', '') ]
                
                reduced_sent = []
                
                for idx,token in enumerate(tokens): 
                    reduced_sent.append(
                        self.lemmatizer.lemmatize(token)
                    )
                
                    final_tokens.extend(reduced_sent)
                    
            return final_tokens
        def check_list_str(self, text):
            """
            Helper function to perform type-checking on input.
            Checks that input is a list of strings.
            """
            if not isinstance(text, list):
                raise(TypeError(
                        "Input not of list type."+
                        " If passing single sentence, encapsulate in single element list."
                ))
            for i, sent in enumerate(text):
                if not isinstance(sent, str):
                    raise(TypeError(
                        f"Input {i} not of string type."
                    ))

        if blacklist is not None:
            check_list_str(blacklist)
            punct = set(self.basic_punct)-set(blacklist)
        else:
            punct = set(self.basic_punct)
        if whitelist is not None:
            check_list_str(whitelist)
            punct = punct.union(set(whitelist))
        if use_bigram:
            ngram_range = (2,2)
        else:
            ngram_range = (1,1)
        if use_lemmatization:
            tokenizer=tokenizer_lemm
        else:
            tokenizer=tokenizer_stemming

        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            stop_words=self.stop_words,  
            tokenizer=tokenizer,
        )

        docs_tfidf = vectorizer.fit_transform(documents).toarray()
        queries_tfidf = vectorizer.transform(queries).toarray()
        print("Final vocabulary size: ", docs_tfidf.shape[-1])
        
        return {
            'documents': docs_tfidf,
            'queries': queries_tfidf,
        }

    def perform_svd(
        self,
        docs, queries,
        num_components, 
        print_status=False, plot_S=False
    ):
        if print_status:
            print("Performing SVD...")
        U, S, V_T = svd(docs.T, full_matrices=False)
        if print_status:
            print("Factorization complete.")
        if plot_S:
            plt.figure(figsize=(10,5))
            plt.plot(S)
            plt.ylabel('Component')
            plt.ylabel('Singular value')
            plt.title('Singular value vs component')
            plt.plot()

        Uk = U[:,:num_components]
        Sk = np.diag(S[:num_components])
        Vk_T = V_T[:num_components]

        if print_status:
            print("Generating latent space documents and queries...")
        docs_latent = Vk_T.T@Sk
        # docs_latent = docs@Uk
        queries_latent = queries@Uk

        if print_status:
            print("LSA performed successfully.")
            print(f'docs latent shape : {docs_latent.shape}')
            print(f'queries latent shape: {queries_latent.shape}')
        
        return docs_latent, queries_latent

    def rank_docs(self, docs_final, queries_final, metric='cosine'):
        if metric == 'cosine':
            result = (
                docs_final/np.expand_dims(np.linalg.norm(docs_final, axis=1),-1)
            )@(
                queries_final/np.expand_dims(np.linalg.norm(queries_final, axis=1),-1)
            ).T

        elif metric == 'correlation':
            doc_cent = docs_final - np.mean(docs_final, axis=0)
            queries_cent = queries_final - np.mean(queries_final, axis=0)
            
            result = (
                doc_cent/np.expand_dims(np.linalg.norm(doc_cent, axis=1),-1)
            )@(
                queries_cent/np.expand_dims(np.linalg.norm(queries_cent, axis=1),-1)
            ).T

        else:
            raise ValueError(
                "Unknown metric. Please use one of cosine and correlation."
            )

        # Ranked in list form for compatibility
        return (np.argsort(result, axis=0,)[::-1,:].T+1).tolist()

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
    
    def grid_search(self, docs, queries, qrels, plot_search=False):
        """
        Grid search on max nDCG@k and F-score
        """
        nDCGs = []
        fscores = []
        ks = []
        max_k = 1400
        min_k = 20
        step = 20

        for i,k in enumerate(np.arange(min_k,max_k,step)):
            print(f"Evaluating {i+1} of {(max_k-min_k)//step}", end="\r")
            
            docs_final, queries_final = self.perform_svd(docs, queries, k)
            ranked = self.rank_docs(docs_final, queries_final)
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