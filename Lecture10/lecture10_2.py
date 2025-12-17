import random
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names



def extract_features(word, N=2):
    last_n_letters = word[-N:]
    return{'feature': last_n_letters.lower()}

male_list = [(name,'male') for name in names.words('male.txt')]
female_list = [(name, 'female') for name in names.words('female.txt')]

data= (male_list+female_list)

random.seed(5)

random.shuffle(data)

namesInput=['David','Jakob','Swati','Shubha']

train_sample=int(0.8*len(data))

for i in  range(1,6):
    print("\n Number of letters ",i)
