import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import string


f=open('corpus.txt','r')
rawData=f.read()
rawData=rawData.lower()# converts to lowercase
sentences = nltk.sent_tokenize(rawData, language='english')# converts to list of sentences
words = nltk.word_tokenize(rawData, language='english')# converts to list of words

#initialise lemmatizer
lemmatizer = WordNetLemmatizer()

#used for removing all punctuation
puncRem = dict((ord(punct), None) for punct in string.punctuation)

print("DerbyBot: Hi im DerbyBot, when finished please say bye")
print("\nWhy dont you start by telling me your name?")
name = input("User: ")
print("\nHi " + name+ "\nPlease ask me questions about Derby University!")

#for each token convert to base form
def lemmatizeTokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

#pass data to lematizeTokens function
def LemNormalize(text):
    return lemmatizeTokens(nltk.word_tokenize(text.lower().translate(puncRem)))

# Generating response
def response(user_response, id):
    #vectoriser used to create a matrix of tf-idf features
    tfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    #convert data into matrix
    tfidf = tfidfVec.fit_transform(sentences)
    #get input sentence
    input = tfidf[-1]
    #calculate similarity -1 used so input is used in comparison to the whole matrix
    vals = cosine_similarity(input, tfidf)
    #sort similarity rating in increasing order, taking highest similarity -2 highest -3 second highest
    ratings = vals.argsort()[0][id]
    #flatten similarity into vector of rows
    flat = vals.flatten()
    #increasing order
    flat.sort()
    #store highest weighting -2 highest -3 second highest
    match = flat[id]
    #if no match
    if(match==0):
        response=("\nSorry I cannot find data for: " + user_response)
        f1 = open("unknown.txt", "a")
        f1.write("Question: " + user_response +"\n" )
        f1.close()
        return response
    else:
        response = "\n" + sentences[ratings]
        return response


while True:
    #get input
    user_response = input(name +": " )
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("DerbyBot: You are welcome " + name)
        else:
             #add to corpus data
             sentences.append(user_response)
             words=words+nltk.word_tokenize(user_response)
             print("DerbyBot: " + user_response + "\n",end="")
             print( response(user_response, -2))
             if  "Sorry I cannot find data for" in response(user_response, -2):
                 # remove input from corpus data
                 sentences.remove(user_response)
             else:
                 print("\nDerbyBot: Was this the result you was looking for? (yes/no)")
                 ans = input()
                 if ans == 'no':
                     print("\nSorry about that let me try again\n")
                     print("DerbyBot: This is the second best result")
                     print(response(user_response, -3))
                     # remove input from corpus data
                     sentences.remove(user_response)

             print("\nwhat else can I do for you " + name +"?")
    else:
        print("Goodbye " + name)
        break