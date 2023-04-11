from util import *
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import RegexpTokenizer
# Add your import statements here




class Tokenization():

	def naive(self, text):
		#print(text,'\n\n')

		tokenizer = RegexpTokenizer("[\w']+")
		tokenizedText = [tokenizer.tokenize(item) for item in text]
		#tokenizer.tokenize(text)
		#Fill in code here
		#print(tokenizedText)
		return tokenizedText



	def pennTreeBank(self, text):
		
		
		#print(text)
		#print(text,'\n\n')
		tokenizedText = [TreebankWordTokenizer().tokenize(i) for i in text]
		#print(tokenizedText,'\n\n')
		#print(tokenizedText)

		return tokenizedText