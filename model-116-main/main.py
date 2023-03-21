import tensorflow.keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
sentence = ["I am happy to meet my friends. We are planning to go a party.",
             "I had a bad day at school. i got hurt while playing football"]
tokanizer=Tokenizer(num_words=10000,oov_token='<OOV>')
tokanizer.fit_on_texts(sentence)
word_index=tokanizer.word_index
sequance=tokanizer.texts_to_sequences(sentence)
print(sequance[0:2])
paded=pad_sequences(sequance,maxlen=100,padding='post',truncating='post')
print(paded[0:2])
model=tensorflow.keras.models.load_model('Text_Emotion.h5')
result=model.predict(paded)
print(result)
predict_class=np.argmax(result,axis=1)
print(predict_class)
