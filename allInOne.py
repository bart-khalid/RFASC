import os
import numpy as np
import face_recognition
import pyttsx3
import pywhatkit
import datetime
import wikipedia
wikipedia.set_lang("fr")
import pyjokes
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
import nltk
from nltk.stem.lancaster import LancasterStemmer
from pyjokes import jokes_de
stemmer = LancasterStemmer()
from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import speech_recognition as sr

# hand detection
import mediapipe as mp

# translate google
from google_trans_new import google_translator  
translator = google_translator() 

############################################
## Hand detector script
############################################
mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

tipIds = [4, 8, 12, 16, 20]

hands = mp_hand.Hands(max_num_hands=1)
############################################
## End script hand detector
############################################

############################################
## ChatJAWAB Script
############################################
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


    training = np.array(training)
    output = np.array(output)

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
        talk("merci de verifier votre connexion internet !")
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
    elif 'parle moi de' in command:
        person = command.replace('parle moi de', '')
        info = wikipedia.summary(person, 1)
        print("JAWAB : "+info)
        talk(info)
    elif 'blague' in command:
        joke = pyjokes.get_joke()
        joke = translator.translate(joke, lang_tgt="fr")
        print("JAWAB : "+joke)
        talk(joke)
    else :
        command = translator.translate(command, lang_tgt="en")
        results = model.predict([bag_of_words(command, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        response = random.choice(responses)
        response = translator.translate(response, lang_tgt="fr")
        print("JAWAB : "+ response)        
        talk(response)
############################################################
## End script chat
############################################################



############################################################
## script Emotion detector
############################################################

# global variables
global_feeling = 'test'
global_name = 'test'
# for the emotional analytics
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./Emotion_Detection.h5')
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
# end


path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


encodeListKnown = findEncodings(images)
print('Encoding Complete')
####################################################
## End script Emotion
####################################################

####################################################
## while true (run the emotion and the chat JAWAB)
####################################################

cap = cv2.VideoCapture(0)
init = 0
while True:
    succes, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    labels = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    label = ''

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            # print("\nprediction = ",preds)
            label = class_labels[preds.argmax()]

            
            sentiment = label
            if (sentiment == 'Happy'):
                sentiment = 'Heureux'
            elif sentiment == 'Sad':
                sentiment = 'Triste'
            elif sentiment == 'Neutral':
                sentiment = 'Neutre'
            elif sentiment == 'Surprise':
                sentiment = 'Surpris'
            elif sentiment == 'Angry':
                sentiment = 'En colère'


            # print("\nprediction max = ",preds.argmax())
            # print("\nlabel = ",label)
            label_position = (x, y)
            cv2.putText(img, sentiment, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(img, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    nameToBePrinted = ''

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            nameToBePrinted = name
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            # cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            # cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow('Face Recognition & Emotion Detector', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #global global_feeling
    #global global_name
    if (global_feeling != label):
        global_feeling = label
    if (global_name != nameToBePrinted):
        global_name = nameToBePrinted


    
    # 
    # hand detect
    #



    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    lmList = []
    
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            myHands = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHands.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

    fingers = []
    if len(lmList) != 0:
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)  # 1 means open
        else:
            fingers.append(0)  # 0 means close

        # Other four fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)  # 1 means open
            else:
                fingers.append(0)  # 0 means close

        total = fingers.count(1)

    
        if total == 2:
            if label == 'Happy' :
                print("JAWAB : je suis très heureux de vous voir de bonne humeur, j'espère que vous serez toujours heureux dans votre vie")
                talk("je suis très heureux de vous voir de bonne humeur, j'espère que vous serez toujours heureux dans votre vie")
            elif label == 'Angry' : 
                print("JAWAB : je suis si triste de te voir en colère, tu devrais te détendre et ne pas laisser tes problèmes te mettre dans cette humeur")
                talk("je suis si triste de te voir en colère, tu devrais te détendre et ne pas laisser tes problèmes te mettre dans cette humeur")
            elif label == 'Sad' :
                print("JAWAB : Bien que je ne sois qu'un robot, je peux te sentir, comme tu es un être humain, il est normal d'être triste parfois. N'oublie pas que je suis là pour toi. Puis-je te raconter une blague ?")
                talk("Bien que je ne sois qu'un robot, je peux te sentir, comme tu es un être humain, il est normal d'être triste parfois. N'oublie pas que je suis là pour toi. Puis-je te raconter une blague ?")
            elif label == "Neutral" :
                print("JAWAB : Vous avez l'air neutre  ")
                talk("Vous avez l'air neutre  ")
            elif label == "Surprise" :
                print("JAWAB : Je me demande ce qui vous surprend "+ nameToBePrinted)
                talk("Je me demande ce qui vous surprend "+ nameToBePrinted)
            else :
                print("JAWAB : j'arrive pas a analyser votre sentiment je vais essayer encore ")
                talk("j'arrive pas a analyser votre sentiment je vais essayer encore ")

    
    #if command in runChatJAWAB :
    #    if(runned != 0):
    if (init == 0):
        import subprocess
        fileToRun = "python main.py"
        subprocess.run("start cmd.exe @cmd /k "+fileToRun, shell=True)

    #
    #Greating
    #
    if (init == 0) :
        print('JAWAB : Salut '+global_name+' heureux de vous revoir')
        talk('Salut '+global_name+' heureux de vous revoir')
    init = 2