import re
import numpy as np
import pickle
import random
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ERROR_THRESHOLD=0.3

data = pickle.load(open('data', 'rb'))
questions = data['questions']
labels = data['labels']
CQA = data['CQA']
classes_data=pickle.load(open('classes', 'rb'))
classes=classes_data['classes']

# load stemmer and vectorizer objects
stemmer = pickle.load(open('stemmer.pkl', 'rb'))
model_vectorizer=pickle.load(open('vectorizer.pkl', 'rb'))
# load the saved chatbot model
model = load_model('chatbot_model.h5')

similarity_vectorizer=TfidfVectorizer(max_features=350,ngram_range=(1,3))
def vectorize_input(document):
    # remove punctuation
    document = re.sub('[^a-zA-Z]', ' ', document)
    # lowercase
    document = document.lower()
    # stemming
    stemmed_document = [stemmer.stem(word) for word in document.split()]
    # rejoin
    final_document = ' '.join(stemmed_document)
    # create BOW
    vectorized_doc = model_vectorizer.transform(np.array([final_document]))

    return vectorized_doc.toarray()


def classify(sentence):
    vectorized_inp=vectorize_input(sentence)
    top_class_idx = model.predict_classes(vectorized_inp)
    top_class_name=classes[top_class_idx[0]]
    return top_class_name


def text_processing(document):
    # remove punctuation
    punct_document = re.sub('[^a-zA-Z]', ' ', document)
    # lowercase
    lower_document = punct_document.lower()
    # stemming
    stemmed_document = [stemmer.stem(word) for word in lower_document.split()]
    # rejoin stemmed words into a sentence
    final_document = ' '.join(stemmed_document)
    return final_document


def response():
    sentence = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, "You: " + sentence + '\n\n')
    ChatLog.config(foreground="#442265", font=("Verdana", 12))
    top_class = classify(sentence)
    match_questions=CQA[top_class].keys()
    q_idices=list(enumerate(match_questions))
    documents = [text_processing(doc) for doc in match_questions]
    similarity_vectorizer.fit(documents)
    sparse_data=similarity_vectorizer.transform(documents)
    vectorized_matches=sparse_data.toarray()

    processed_q=text_processing(sentence)
    transformed_q = similarity_vectorizer.transform(np.array([processed_q]))
    vectorized_q = transformed_q.toarray()
    cs = cosine_similarity(vectorized_q, vectorized_matches)[0].tolist()
    idx = cs.index(max(cs))
    chosen_q=q_idices[idx][1]
    bot_response= random.choice(CQA[top_class][chosen_q])
    ChatLog.insert(END, "Bot: " + bot_response  + '\n\n')
    ChatLog.config(state=DISABLED)
    ChatLog.yview(END)


import tkinter
from tkinter import *

window = tkinter.Tk()
window.title('Bot Buddy')
window.geometry("400x500")
window.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(window, bd=0, bg="#d7d8d1", height="7", width="50", font="Arial")
ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(window, command=ChatLog.yview, cursor="arrow")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(window, font=("Verdana", 12, 'bold'), text="Send", width="10", height=3,
                    bd=0, bg="#5d5392", activebackground="#241d18", fg='#ffffff',
                    command=response)

# Create the box to enter message
EntryBox = Text(window, bd=0, bg="white", width="29", height="5", font="Arial")

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=280)
SendButton.place(x=290, y=401, height=90)

window.mainloop()