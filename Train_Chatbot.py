import pickle
import re
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

data = pickle.load(open('data', 'rb'))
questions = data['questions']
labels = data['labels']
CQA = data['CQA']

stemmer = PorterStemmer()


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


def create_labels_encoding(labels):
    unique_labels = list(set(labels))
    encodings = {}
    for i in range(len(unique_labels)):
        temp = [0] * len(unique_labels)
        temp[i] = 1
        encodings[unique_labels[i]] = temp
    return unique_labels, encodings


def encode_label(label, encodings):
    return encodings[label]

corpus=' '.join(questions)
unique_words=set(corpus.split())
MAX_WORDS=len(unique_words)
vectorizer = TfidfVectorizer(max_features=MAX_WORDS)
documents = [text_processing(doc) for doc in questions]
vectorizer.fit(documents)
vect_text = vectorizer.transform(documents)
processed_text = vect_text.toarray()

pickle.dump(stemmer,open('stemmer.pkl','wb'))
pickle.dump(vectorizer,open('vectorizer.pkl','wb'))

unique_labels, encodings = create_labels_encoding(labels)
labels = [encode_label(label, encodings) for label in labels]

pickle.dump({'classes':unique_labels},open('classes','wb'))

complete_data = np.append(arr=processed_text, values=np.array(labels), axis=1)
random.shuffle(complete_data)

X = complete_data[:, :len(processed_text[0])]
y = complete_data[:, len(processed_text[0]):]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create model composed of input layer, 2 hidden layers and an output layer with dropout
model = Sequential()
model.add(Dense(100, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(unique_labels), activation='softmax'))

# compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print('Successfully compiled')

# fitting and saving model
hist = model.fit(x_train, y_train, epochs=1000, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Successfully saved all models and dependencies')

scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

