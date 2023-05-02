from util import *
import nltk
nltk.download('punkt',quiet=True)											#update: used quite=True to suppress print'already- updated, remove it in case
nltk.download('wordnet',quiet=True)											#of any error.
nltk.download('omw-1.4',quiet=True)
import nltk.tokenize as token
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# Add your import statements here




class InflectionReduction:

	def reduce(self, text):
		lemmatizer = WordNetLemmatizer()
		#print(text,'\n\n')
		#print(text,'\n\n')
		reducedText = [[lemmatizer.lemmatize(word) for word in words] for words in text]
		#print(reducedText,'\n\n')
		return reducedText