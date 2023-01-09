# import bibliotek
import random
import json #praca na plikach json
import pickle # przekształcanie obiektów pythona w strumień bajtów i zapisanie ich w pliku
import numpy as np # chyba oczywiste

import nltk
from nltk.stem import WordNetLemmatizer # lematyzacja słów użytych w pytaniach
#nltk.download('omw-1.4')

# stworzenie modelu sieci neuronowej oraz zapisanie go do pliku
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

# utworzenie zmiennej klasy WordNetLemmatizer
lematizer = WordNetLemmatizer()

# wczytanie danych z pliku FAQ.json - encoding = 'utf-8' umożliwia czytanie polskich znaków
faq = json.loads(open('FAQ.json',encoding='utf-8').read())

# stworzenie list słów, klas, dokumentów i ignorowanych znaków
words = [] #lista ze zlematyzowanymi słowami
classes = [] #lista z klasami (tagami)
documents = [] #lista pomocnicza, do stworzenia tablicy training
ignore_letters = ['?','!','.',','] # lista znaków, które są ignorowane przy lematyzowaniu słów

# ten fragment kodu pokazuje lematyzację wszystkich słów użytych w pliku json oraz posortowanie ich w kolejności alfabetycznej
# na początku z tablicy faq są dla każdego tagu wyciągane słowa "kluczowe" czyli słowa składające się na pytania dla tego tagu (funkcja word_tokenize z biblioteki nltk)
# następnie lista wszystkich słów jest rozszerzana o te słowa (tzn. że dodawane są tylko te słowa, których jeszcze nie ma na liście)
# następnie do listy documents dodawana jest lista zawierająca wszystkie słowa dla danego tagu oraz nazwę tego tagu
# jeżeli tag nie znajduje się na liście classes, to jest on dodawany
# dzieje się tak dla każdego pytania

for intent in faq['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# następnie z listy słów usuwane są znaki znajdujące się na liście ignore_letters oraz wszystkie pozostałe słowa są lematyzowane i sortowane w kolejności alfabetycznej 
words = [lematizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

# nazwy klas również sortowane są alfabetycznie
classes = sorted(set(classes))

# następnie następuje zamiana list words oraz classes na strunienie binarne i zapisanie ich w plikach pickle
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

# poniższy kod tworzy tablicę typu np.array, która ma za zadanie wytrenować sieć neuronową stworzoną później
# na początku definujemy dwie listy - jedna jest pusta, a druga jest długości listy classes i wypełniona samymi zerami (na razie)

training = []
output_empty = [0]*len(classes)

# dla każdego elementu z listy documents (zapisywałyśmy tam słowa dla danego tagu oraz nazwę tego tagu)
# definujemy listę bag, która odpowiada za klasyfikację czy dane słowo należy do jakiegoś tagu (lista będzie binarna)
# następnie lematyzujemy słowa dla danego tagu (pierwszy element elementu na liście documents), zamieniamy wszystkie litery na małe, aby nie było komplikacji
# wyniki zapisujemy na liście
# następnie sprawdzamy dla każdego słowa na liście wszystkich słów w pliku json (words), czy dane słowo występuje na liście słów dla danego tagu
# jeżeli tak - wstawiamy na listę bag wartość 1, jeżeli nie - 0
# następnym krokiem jest zaznaczenie, którego tagu dotyczy dana lista na liście documents (jest to zaznaczone 1 na liscie wszystkich klas (output_row))
# ostatnim krokiem jest dodanie listy składającej się z listy bag oraz output_row
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lematizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words :
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# aby poprawnie nauczyć sieć, należy przemieszać elementy listy w losowej kolejności 
random.shuffle(training)

#następnie tworzona jest tabela typu np.array z danych z listy training 
training = np.array(training)

#podział dabych na x i y

train_x = list(training[:,0])
train_y = list(training[:,1])

#stworzenie modelu sieci neuronowej

#na początku definiujemy typ modelu
#Sequential model jest odpowiedni dla zwykłego stosu warstw (jedno wejście - jedno wyjście)
model = Sequential()

#dodajemy warstwę neuronową, która posiada 128 neuronów, ma na jedno wejście i funkcję aktywacji ReLu
model.add(Dense(128,input_shape = (len(train_x[0]),), activation = 'relu'))

#następnie dodajemy dropout. Warstwa Dropout losowo ustawia jednostki wejściowe na 0 z częstotliwością 0.5 na każdym kroku w czasie treningu, co pomaga zapobiegać nadmiernemu dopasowaniu
model.add(Dropout(0.5))

#następnie dodajemy wrstwę z 64 neuronami i funkcją aktywacji ReLu
model.add(Dense(64,activation='relu'))

#następnie dodajemy kolejny dropout
model.add(Dropout(0.5))

# na końcu dodajemy jeden neuron, który jest outputem z funkcją aktywacji Softmax (znormalizowane funkcja wykładnicza)
model.add(Dense(len(train_y[0]),activation='softmax'))

#definiujemy optymalizator (lr - learning rate, momentum - spadek gradientu, nesterov - wartość logiczna decydująca, czy zastosować pęd Niestierowa)
sgd = SGD(lr = 0.01, decay = 1e-6, momentum=0.9, nesterov= True)

# konfiguracja modelu do treningu (loss - funkcja straty (tutaj crossentrophy / może być również np MSE), optimizer - optymalizator)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# definiujemy trening sieci neuronowej z liczbą epok równą 200, batch_size (liczba próbek do aktualizacji gradientu) równą 5, verbose definiuje jak ma wyglądać pokazywanie postępu 
# (w tym przypdaku jest to pasek postępu)
hist = model.fit(np.array(train_x), np.array(train_y),epochs = 200, batch_size = 5,verbose = 1)

# zapisujemy model do pliku wraz z jego treningiem
model.save('chatbotmodel.h5',hist)

# potwierdzenie wykonania pliku bez błędów
print('Done')