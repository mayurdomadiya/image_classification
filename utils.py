
import tensorflow as tf

def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255.0
  return tf.image.resize(image,[224,224]), tf.one_hot(label, 3)

def get_dataset(datasets,batch_size=32):
  train_dataset_scaled = datasets[0].map(scale).shuffle(1000).batch(batch_size)
  test_dataset_scaled =  datasets[1].map(scale).batch(batch_size)
  val_dataset_scaled =  datasets[2].map(scale).batch(batch_size)
  return train_dataset_scaled, test_dataset_scaled, val_dataset_scaled