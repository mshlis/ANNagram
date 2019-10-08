## ANNagram
#### Using NeuralNets to find anagrams

In this project I use a Neural Network to generate possible annagrams given some set of letters.

[![asciicast](https://asciinema.org/a/4ASdeigUxAr8LLHtwO5Mn18pp.svg)](https://asciinema.org/a/4ASdeigUxAr8LLHtwO5Mn18pp?loop=1&autoplay=1)

The output space is not in stone, so the obvious choice is to use a pointer net. For the backbone of the model we use a transformer-like encoder decoder network. For architecture construction, I use the same layout as the initial transformer paper but reduce the dimensionality heavily. After a grid search on hyperparameters I find that model_dim=64 and N=2 where N is the number of encode/decoder blocks the model I use. Also note the slight difference of using model_dim/2 for the intermediate linear projections rather than model_dim/8.

I implemented all this in tf.keras and ran on a single RTX2080Ti. Training time was approximately 20 minutes until convergence. 

To run use:
```
python run.py <letters> <weight_path> <word_to_index pickled dict> --<beam_size :: optional> --<max_print :: optional>
```
tested with python=='3.6.8', tf.keras=='2.2.4-tf', tf=='1.14.0'
textfile with english words taken from https://github.com/dwyl/english-words
