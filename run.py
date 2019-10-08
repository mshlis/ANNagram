import sys
import numpy as np
import pickle as pkl
import utils
import models
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

parser = argparse.ArgumentParser(description='run ANNagram')
parser.add_argument('letters', type=str, help='letters to find anagram from')
parser.add_argument('weights', type=str, help='path to model weights')
parser.add_argument('word_to_idx', type=str, help='path to word to index mapping')
parser.add_argument('--model_dim', type=int, default=64, help='model dim')
parser.add_argument('--heads', type=int, default=4, help='number of heads per block')
parser.add_argument('--blocks', type=int, default=2, help='number of encoder/decoder blocks to use')
parser.add_argument('--beam_size', type=int, default=5, help='size of beam to use')
parser.add_argument('--max_print', type=int, default=3, help='prints only top N of beam at each iter')
parser.add_argument('--gpu_num', type=int, default=0, help='gpu index')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_num)


def main():
    mask_meta = '<m>'
    word_to_idx = pkl.load(open('word_to_idx.pkl', 'rb'))
    model = models.pointer_transformer(alphabet_size=len(word_to_idx),
                                       blocks=args.blocks,
                                       num_heads=args.heads,
                                       dim=args.model_dim,
                                       dropout=None,
                                       mask_meta=word_to_idx[mask_meta])
    model.load_weights(args.weights)
    letters = list(args.letters)
    letter_inds = np.array([word_to_idx[c] for c in letters]).reshape((1,-1))
    utils.beam_search(letter_inds, 
                      model.predict, 
                      word_to_idx, 
                      beam_length=args.beam_size,
                      max_print=args.max_print)

if __name__ == '__main__':
    main()