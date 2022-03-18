import tensorflow as tf
import tensorflow_datasets as tfds
import os

data_dir = os.path.dirname(os.path.abspath(__file__))

dataset = tfds.load('minerl_navigate', shuffle_files=True, data_dir=data_dir)

test = dataset['test']

train = dataset['train'].repeat(2)# Repeats for the number of epochs
train = train.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(
    tf.reshape(x['video'], (5, 100, 64, 64, 3))))
train = train.shuffle(5000).batch(50).prefetch(1)

for batch in train:
  assert batch.shape == (50, 100, 64, 64, 3)
  assert batch.dtype == tf.uint8
  break