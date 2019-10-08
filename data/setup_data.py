import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl

with open('data/words.txt', 'rb') as f:
    words = set()
    for i, w in enumerate(tqdm(f.readlines())):
        try:
            w = w.decode().strip('\n')
            if w.isalpha(): 
                words.add(w.lower())
        except:
            pass
        
start_meta = '<s>'
end_meta = '<e>'
mask_meta = '<m>'
letters = 'abcdefghijklmnopqrstuvwxyz '

vocab = [start_meta, end_meta, mask_meta] + list(letters)
words_to_idx = {w:i for i,w in enumerate(vocab)}

pkl.dump(vocab_to_ind, open('word_to_idx.pkl', 'wb'))

lengths = [len(word) for word in words]

def translate(word, with_metas):
    res = [vocab_to_ind[c] for c in word]
    if not with_metas:
        return res
    return [vocab_to_ind[start_meta]] + res + [vocab_to_ind[end_meta]]

X0 = []
X1 = []
maxlen = 12

for word in tqdm(words):
    if len(word) < maxlen:
        X0.append(translate(word, with_metas=False))
        X1.append(translate(word, with_metas=True))


fill_value = vocab_to_ind[mask_meta]
def pad_sequences(sequences, maxlen, fill_value):
    X = fill_value * np.ones((len(sequences), maxlen))
    lengths = [len(z) for z in sequences]
    mask = np.arange(maxlen) < np.array(lengths).reshape(-1,1)
    X[mask] = np.concatenate(sequences)
    return X

X0 = pad_sequences(X0, maxlen, vocab_to_ind[mask_meta])
X1 = pad_sequences(X1, maxlen+2, vocab_to_ind[mask_meta])

Y0 = np.eye(X0.shape[1])
Y0 = np.concatenate([Y0, np.zeros((2, X0.shape[1]))], axis=0)[np.newaxis,]
Y0 = np.tile(Y0, (len(X0),1,1))

l0, l1 = X0.shape[1], X1.shape[1]
B = np.tile(np.arange(l0).reshape((1,-1)), (l1, 1)) 
B = -1e9 * (B < np.arange(l1)[:,np.newaxis]).astype(int)
B = np.tile(B[np.newaxis,], (len(X0),1,1))

pkl.dump([X0, X1, Y, B], open('training_data.pkl', 'wb'))        