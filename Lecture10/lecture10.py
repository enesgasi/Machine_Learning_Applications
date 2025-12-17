from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

sample="This is a sample sentence. This is second sample sentence."
sentences=sent_tokenize(sample)


print("Sentences:", sentences)

words=word_tokenize(sample)

print("Words:", words)

stemmer=PorterStemmer()

words=["cared","university","fairly","easily","singing","sings","sung","singer","sportingly"]


stemmed_words=[stemmer.stem(word) for word in words]

print("Stemmed Words:", stemmed_words)

from nltk.stem import LancasterStemmer
lancaster_stemmer=LancasterStemmer()

lancaster_stemmed_words=[lancaster_stemmer.stem(word) for word in words]
print("Lancaster Stemmed Words:", lancaster_stemmed_words)

from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer("english")

snowball_stemmed_words=[stemmer.stem(word) for word in words]
print("Snowball Stemmed Words:", snowball_stemmed_words)

lemmatizer=WordNetLemmatizer()
lemmatized_words=[lemmatizer.lemmatize(word) for word in words]
print("Lemmatized Words:", lemmatized_words)


print("better:", lemmatizer.lemmatize("better", pos="a"))
print("rocks:", lemmatizer.lemmatize("rocks", pos="v"))
print("corpora:", lemmatizer.lemmatize("corpora", pos="n"))
print("larger:", lemmatizer.lemmatize("larger", pos="a"))
print("worst:", lemmatizer.lemmatize("worst", pos="a"))


from sklearn.feature_extraction.text import CountVectorizer
Sentences=["We are using the Bag of Words model.",
           "The Bag of Words Model is used for extracting the features."]

vectorizer=CountVectorizer()
X=vectorizer.fit_transform(Sentences)
features_text=vectorizer.fit_transform(Sentences).todense()
print("Features Text:\n", features_text)
print(vectorizer.vocabulary_)
