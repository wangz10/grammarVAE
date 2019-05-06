from __future__ import print_function, division

import argparse
import os
import threading
import h5py
import numpy as np

from models.model_zinc import MoleculeVAE
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import h5py
import zinc_grammar as G
import pdb


rules = G.gram.split('\n')


MAX_LEN = 277
DIM = len(rules)
LATENT = 56
EPOCHS = 100
BATCH = 500
WORKERS = 1

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self): # Py3
        return next(self.it)

    def next(self):     # Py2
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--load_model', type=str, metavar='N', default="")
    parser.add_argument('--epochs', type=int, metavar='N', default=EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT,
                        help='Dimensionality of the latent representation.')
    parser.add_argument('--workers', type=int, metavar='N', default=WORKERS,
                        help='Number of workers when fitting model.')
    return parser.parse_args()


def main():
    # 0. load dataset
    h5f = h5py.File('data/zinc_grammar_dataset.h5', 'r')
    data = h5f['data']
    
    # 1. split into train/test, we use test set to check reconstruction error and the % of
    # samples from prior p(z) that are valid
    XTE = data[0:5000]
    n_samples = data.shape[0]
    n_batches = int(np.ceil((n_samples - 5000) / BATCH))
    h5f.close()

    @threadsafe_generator
    def data_generator(batch_size):
        h5f = h5py.File('data/zinc_grammar_dataset.h5', 'r')
        data = h5f['data']
        n_samples = data.shape[0]
        while True:
            rand_idx = np.arange(5000, n_samples)
            np.random.shuffle(rand_idx)
            n_batches = int(np.ceil((n_samples - 5000) / batch_size))
            for i in range(n_batches):
                idx_start = i * batch_size
                idx_end = (i+1) * batch_size
                rand_idx_batch = rand_idx[idx_start:idx_end]
                rand_idx_batch.sort()
                yield data[rand_idx_batch, :, :], data[rand_idx_batch, :, :]

    np.random.seed(1)
    # 2. get any arguments and define save file, then create the VAE model
    args = get_arguments()
    print('L='  + str(args.latent_dim) + ' E=' + str(args.epochs))
    model_save = 'results/zinc_vae_grammar_L' + str(args.latent_dim) + '_E' + str(args.epochs) + '_val.hdf5'
    print(model_save)
    model = MoleculeVAE()
    print(args.load_model)

    # 3. if this results file exists already load it
    if os.path.isfile(args.load_model):
        print('loading!')
        model.load(rules, args.load_model, latent_rep_size = args.latent_dim, max_length=MAX_LEN)
    else:
        print('making new model')
        model.create(rules, max_length=MAX_LEN, latent_rep_size = args.latent_dim)

    # 4. only save best model found on a 10% validation set
    checkpointer = ModelCheckpoint(filepath = model_save,
                                   verbose = 1,
                                   save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)
    # 5. fit the vae
    train_data_gen = data_generator(BATCH)

    model.autoencoder.fit_generator(train_data_gen, 
        steps_per_epoch=n_batches,
        epochs=args.epochs,
        callbacks=[checkpointer, reduce_lr],
        validation_data=(XTE, XTE),
        workers=args.workers
    )
    # h5f.close()

if __name__ == '__main__':
    main()
