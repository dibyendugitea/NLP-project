from util import *
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Add your import statements here




class StopwordRemoval():

	def fromList(self, text):
		stop_words = set(stopwords.words('english'))
		#words_token = word_tokenize(doc)
		#print(text,'\n\n')
		stopwordRemovedText = [[word for word in words if not word in stop_words and len(word)>2] for words in text]
		return stopwordRemovedText
