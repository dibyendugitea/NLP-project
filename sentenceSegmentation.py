from util import *
import nltk
from nltk.tokenize import PunktSentenceTokenizer
import re
# Add your import statements here




class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""
		#print(text,'\n\n')
		splitter=re.split(r' *[\.\?!\(\)][\'"\)\]]* *', text)

		updated_splitter  = []

		for x in splitter:
			if x:
				updated_splitter.append(x)

		#print(updated_splitter,'\n\n')
		return updated_splitter





		#return segmentedText





	def punkt(self, text):
		sent_splitter = PunktSentenceTokenizer()
		segmentedText = sent_splitter.tokenize(text)
		return segmentedText
