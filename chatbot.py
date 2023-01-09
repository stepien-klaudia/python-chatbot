#PLIK DZIAŁA BEZ GUI

#import bibliotek
import random
import json #praca na plikach json
import pickle # załadowanie danych z plików pickle
import numpy as np # chyba oczywiste

import nltk
from nltk.stem import WordNetLemmatizer # lematyzacja słów użytych w pytaniach

import tensorflow
from keras.models import load_model # odpowiada za załadowanie modelu

# utworzenie zmiennej klasy WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# wczytanie danych z pliku FAQ.json - encoding = 'utf-8' umożliwia czytanie polskich znaków
faq = json.loads(open('FAQ.json', encoding='utf-8').read())

# załadowanie do zmiennych danych z plików pickle 
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

# załadowanie modelu
model = load_model('chatbotmodel.h5')

#funkcja lematyzująca nowe zdanie
# funkcja najpierw tokenizuje (rozdziela słowa), a następnie je lematyzuje
# funkcja zwraca listę zlematyzowanych słów

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#funkcja tworząca "worek słów" (binarna lista słów, które znajdują się w zdanuiu podanym jako argument)
#na początku zostaje użyta funkcja lematyzująca zdanie podane jako argument
#następnie jest sprawdzane czy dane słowo znajduje się na liście wszystkich słów (po wcześniejszym zamieniemu strumienia binarnego na słowa)
# jeżeli takie słowo się znajduje na liście słów to w jego miejscu na liście bag pojawia się 1, jeżeli nie - 0
# funkcja zwraca tabelę z informacjami o "worku słów"  

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

# funkcja przewiduje, jakiej klasy jest dane pytanie/zdanie/wyrażenie 
# na początku wyznaczany jest "worek słów" ze zdania podanego jako argument
# następnie przewidywany jest rezultat dla tego "worka słów"
# następnie ustalany jest błąd, który wyznacza czy dany potencjalny rezultat liczy się do rezultatów ostatecznych
# jest to po prostu minimalny wskaźnik zgodności
# następnie tworzona jest lista rezultatów i jest ona sortowana malejąco ze względu na poziom zgodności
# następnie do zwracanej listy rezultatów dodawana jest lista z nazwą klasy (tagu) oraz prawdopodobieństwm prawidłowego dopasowania

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key = lambda x: x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list

# funkcja zwracająca odpowiedź chatbota
# na wejściu mamy przewidywany tag dla odpowiedzi oraz wczytany plik json z odpowiedziami
# na początku do osobnych zmiennych przypisujemy tag oraz wszystkie intents z pliku json
# następnie dla każdej intents sprawdzamy, czy tag tej intent jest taki sam jak przewidywana klasa pytania
# jeżeli tak to zwracamy jako rezultat losową odpowiedź dla tego tagu - w przypadku pytań jest ona jedna, w przypadku small talk jest ich kilka do wyboru

def get_response(intents_list, faq_json):
    tag = intents_list[0]['intent']
    list_of_intents = faq_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Bot is running!")

# wersja bez GUI
# dopóki program działa wpisujemy odpowiedznie wiadomości do terminala
# następnie program sprawdza jekiej klasy jest dane pytanie i wyświetla najbardziej prawdopodobną odpowiedź

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, faq)
    print(res)
