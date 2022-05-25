import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
import os

from google.colab import files
uploaded = files.upload()

file = open("1661-0.txt", "r", encoding = "utf8")

# dosyayı listeye aktarma
lines = []
for i in file:
    lines.append(i)

# listeyi string hale convert etme
data = ""
for i in lines:
  data = ' '. join(lines) 

#gereksiz tanımları boşluklar ile değiştirme
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“','').replace('”','') 

#gereksiz boşlukları yok etme 
data = data.split()
data = ' '.join(data)
data[:500]

len(data)

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# predict func için tokenizer kayıt alma
pickle.dump(tokenizer, open('token.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([data])[0]
sequence_data[:15]

len(sequence_data)

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

sequences = []

for i in range(3, len(sequence_data)):
    words = sequence_data[i-3:i+1]
    sequences.append(words)
    
print("The Length of sequences are: ", len(sequences))
sequences = np.array(sequences)
sequences[:10]

X = []
y = []

for i in sequences:
    X.append(i[0:3])
    y.append(i[3])
    
X = np.array(X)
y = np.array(y)

print("Data: ", X[:10])
print("Response: ", y[:10])

y = to_categorical(y, num_classes=vocab_size)
y[:5]

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=3))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))

model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("next_words.h5", monitor='loss', verbose=1, save_best_only=True)
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001))
model.fit(X, y, epochs=20, batch_size=64, callbacks=[checkpoint])

from tensorflow.keras.models import load_model
import numpy as np
import pickle

# modeli ve tokenizeri kullanmak için yükledim
model = load_model('next_words.h5')
tokenizer = pickle.load(open('token.pkl', 'rb'))

def Predict_Next_Words(model, tokenizer, text):

  sequence = tokenizer.texts_to_sequences([text])
  sequence = np.array(sequence)
  preds = np.argmax(model.predict(sequence))
  predicted_word = ""
  
  for key, value in tokenizer.word_index.items():
      if value == preds:
          predicted_word = key
          break
  
  print(predicted_word)
  return predicted_word

while(True):
  text = input("Enter your line: ")
  
  if text == "0":
      print("Execution completed.....")
      break
  
  else:
      try:
          text = text.split(" ")
          text = text[-3:]
          print(text)
        
          Predict_Next_Words(model, tokenizer, text)
          
      except Exception as e:
        print("Error occurred: ",e)
        continue