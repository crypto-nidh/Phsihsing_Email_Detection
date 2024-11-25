from django.shortcuts import render

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer #converting text data into numerical data

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from .forms import MessageForm

dataset = pd.read_csv('C:/Users/VoId/Desktop/emails.csv')

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(dataset['text'])

x_train, x_test, y_train, y_test = train_test_split(X, dataset['spam'], test_size=0.2)

model = MultinomialNB()
model.fit(x_train, y_train)

def predictmessage(message):
    messageVector = vectorizer.transform([message])
    prediction = model.predict(messageVector)
    
    return 'Spam' if prediction[0] == 1 else 'Ham'

def Home(request):
    result = None
    if request.method == "POST":
        form = MessageForm(request.POST)
        if form.is_valid():
            message =  form.cleaned_data['text']
            result = predictmessage(message)

    else:
        form = MessageForm()

    return render(request, 'home.html', {'form': form, 'result': result})