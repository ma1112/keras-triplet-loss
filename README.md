Up to the date of creating this repository I could not find a decent example of how to use triplet loss and online triplet mining with **Keras**, so I decided to create this repository to inspire fellow deep learners. 

I used the triplet loss / triplet minig related code from omoindrot's [tensorflow-triplet-loss][tensorflow-triplet-loss]  repository (implemented in Tensorflow) and created the necessary Keras code around it.

You can see a minimalistic example of online triplet mining with keras in the uploaded [`notebook`](keras_triplet_loss.ipynb).

The only modification you have to do compared to a regular keras pipeline is to flatten the labels's array, as omoidrot's code expects it in a flattened shape. I.e:

```python3
#importing omoindrot's function
from triplet_loss import batch_hard_triplet_loss

#defining a loss function that works with Keras
def keras_batch_hard_triplet_loss(labels, y_pred):
  labels = K.flatten(labels)
  return batch_hard_triplet_loss(labels, y_pred, margin = 0.5)
```

For all further information about what triplet loss is, and what it is good for, I recommend reading omoindrot's [blog post][blog] and studying his GitHub [repository][tensorflow-triplet-loss].

## Requirements

Tensorflow

Keras (using Tensorflow backend)

Numpy



[blog]: https://omoindrot.github.io/triplet-loss
[tensorflow-triplet-loss]: https://github.com/omoindrot/tensorflow-triplet-loss
