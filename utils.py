import numpy as np

def write_out(inds, idx_to_word):
    """
    writes out inds back in vocab
    """
    string = ''
    for n in inds:
        string += idx_to_word[n]
    return string

def beam_search(letter_indeces, next_probit_fn, word_to_idx, beam_length=5, max_print=5):
    """
    beam search
    """
    idx_to_word = {v:k for k,v in word_to_idx.items()}
    start = ['<s>']
    beam = np.array([word_to_idx[c] for c in start]).reshape((1,-1))
    beam_probs = np.ones((1,1))
    K = letter_indeces.shape[-1]
    bias = np.zeros((1,1,K))
    print('')
    for i in range(K):
        print(f'beam-search step {i+1}:')
        # run model
        next_probs = next_probit_fn([beam, np.tile(letter_indeces, (len(beam),1)), bias])[:,-1]
        
        # update beam / beam_probs
        beam_probs = beam_probs * next_probs
        beam_probs = np.reshape(beam_probs, (-1,1))        
        beam = np.tile(beam[...,np.newaxis], (1,1,K))
        beam = np.concatenate([beam, np.tile(letter_indeces[np.newaxis,], (len(beam),1,1))], 1)
        beam = np.transpose(beam, (0,2,1)).reshape((-1, i+2))
        inds = beam_probs.argsort(axis=0)[::-1,0][:beam_length]
        beam_probs = beam_probs[inds]
        beam_probs /= beam_probs.sum()
        beam = beam[inds]

        # update bias
        bias_inds = np.tile(np.arange(len(bias))[:,np.newaxis], (1,K))
        bias_inds = bias_inds.reshape((-1,))[inds]
        next_bias = np.tile(np.arange(K)[np.newaxis,:], (len(bias),1))
        next_bias = next_bias.reshape((-1,))
        next_bias = next_bias[inds]
        next_bias = np.eye(K)[next_bias]
        next_bias = bias[bias_inds,-1,:]-1e9*next_bias
        bias = np.concatenate([bias[bias_inds], next_bias[:,np.newaxis,]],1)
        
        # print out beam
        for prob, chars in zip(beam_probs, beam[:max_print]):
            print('\t'+write_out(chars[1:], idx_to_word), f': p={prob}')
        print('')
    print('top selected anagram:' + write_out(beam[0][1:], idx_to_word)) 