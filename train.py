import pickle as pkl
import numpy as np
import models
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

parser = argparse.ArgumentParser(description='train ANNagram')
parser.add_argument('--weights_path', type=str, default='weights.h5', help='path to model weights')
parser.add_argument('--model_dim', type=int, default=64, help='model dim')
parser.add_argument('--heads', type=int, default=4, help='number of heads per block')
parser.add_argument('--blocks', type=int, default=2, help='number of encoder/decoder blocks to use')
parser.add_argument('--gpu_num', type=int, default=0, help='gpu index')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_num)

def main():
    X0, X1, Y, B = pkl.load(open('data/training_data.pkl', 'rb'))
    words_to_idx = pkl.load(open('words_to_idx.pkl', 'rb'))
    mask_meta = '<m>'
    model = models.pointer_transformer(alphabet_size=len(words_to_idx),
                                       blocks=args.blocks,
                                       num_heads=args.heads,
                                       dim=args.model_dim,
                                       dropout=None,
                                       mask_meta=words_to_idx[mask_meta])


    def loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true[:, :, words_to_idx[mask_meta]:words_to_idx[mask_meta]+1], 1.), tf.float32)
        logits = tf.log(y_pred + 1e-9)
        return - mask * y_true * logits

    inds = np.arange(len(Y))
    np.random.shuffle(inds)

    Ntr = int(.7*len(Y))
    train_inds = inds[:Ntr]
    test_inds = inds[Ntr:]

    rlop = ReduceLROnPlateau(monitor='loss', verbose=1, factor=.5, patience=5)
    es = EarlyStopping(monitor='loss', verbose=1, patience=12)
    mc = ModelCheckpoint(args.weights_path, verbose=1, monitor='val_loss', save_best_only=True)

    model.compile(keras.optimizers.Adam(1e-3), loss)
    history = model.fit([X1[train_inds], X0[train_inds], B[train_inds]], Y[train_inds],
                        batch_size=1024, 
                        epochs=150,
                        validation_data=([X1[test_inds], X0[test_inds], B[test_inds]], Y[test_inds]),
                        callbacks=[rlop, es, mc],
                        verbose=1)

if __name__ == '__main__':
    main()