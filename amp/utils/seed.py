import numpy as np
import tensorflow as tf
import random as python_random
import torch

# def set_seed(seed: int):

#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

#     np.random.seed(seed)

#     # The below is necessary for starting core Python generated random numbers
#     # in a well-defined state.
#     python_random.seed(seed)

#     # The below set_seed() will make random number generation
#     # in the TensorFlow backend have a well-defined initial state.
#     # For further details, see:
#     # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
#     tf.random.set_seed(seed)

#     torch.backends.cudnn.deterministic = True

def set_seed(seed: int):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     python_random.seed(seed)
     torch.backends.cudnn.deterministic = True