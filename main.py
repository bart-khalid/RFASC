import nltk
from nltk.stem.lancaster import LancasterStemmer
from pyjokes import jokes_de
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf 
from tensorflow.python.framework import ops
import random
import json
import pickle
import speech_recognition as sr
listener = sr.Recognizer()
import pyttsx3
engine = pyttsx3.init()
import pywhatkit
import datetime
import wikipedia
wikipedia.set_lang("fr")
import pyjokes
# translate google
#from google_trans_new import google_translator  
#translator = google_translator() 
from translate import Translator
translator= Translator(to_lang="fr")
translator2= Translator(to_lang="en")



with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

#tf.reset_default_graph()
ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)



# say mesessage
def talk(message):
    voice = engine.getProperty('voices')[2] # the french voice
    newVoiceRate = 145
    engine.setProperty('rate',newVoiceRate)
    engine.setProperty('voice', voice.id)
    engine.say(message)
    engine.runAndWait()
    engine.stop()

# recognise voice
def take_command():
    try:
        with sr.Microphone() as source:
            print('listening...')
            voice = listener.listen(source)
            command = listener.recognize_google(voice, language='fr-FR')
            command = command.lower()
            if 'JAWAB' in command:
                command = command.replace('JAWAB', '')
                print(command)
    except:
        print("JAWAB : merci de verifier votre connexion internet !")
        talk("merci de verifier votre connexion internet")
    return command   

def getResponse(command):
    if 'play' in command:
        song = command.replace('play', '')
        print('JAWAB : playing ' + song)
        talk('playing ' + song)
        pywhatkit.playonyt(song)
        print("JAWAB : profitez-en, je l'ai trouvé pour vous")
        talk("profitez-en, je l'ai trouvé pour vous")
    elif 'heure' in command:
        time = datetime.datetime.now().strftime('%I:%M %p')
        print("JAWAB : L'heure actuelle est " + time)
        talk("L'heure actuelle est " + time)
    elif 'qui est' in command:
        person = command.replace('qui est', '')
        info = wikipedia.summary(person, 1)
        print("JAWAB : "+info)
        talk(info)
    elif 'parle moi' in command:
        person = command.replace('parle moi', '')
        info = wikipedia.summary(person, 1)
        print("JAWAB : "+info)
        talk(info)
    elif 'blague' in command:
        joke = pyjokes.get_joke()
        #joke = translator.translate(joke, lang_tgt="fr")
        joke = translator.translate(joke)
        print("JAWAB : "+joke)
        talk(joke)
    else :
        #command = translator.translate(command, lang_tgt="en")
        command = translator2.translate(command)
        results = model.predict([bag_of_words(command, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        response = random.choice(responses)
        #response = translator.translate(response, lang_tgt="fr")
        response = translator.translate(response)
        print("JAWAB : "+ response)        
        talk(response)

def chat():
    print("Commancer a parler avec votre assistant JAWAB (dit quitter pour arreter de parler)!")
    while True:
        inp = take_command()
        print("You : "+inp)
        if inp.lower() == "quitter" or inp.lower() == "quit":
            break
        getResponse(inp)
        

chat()