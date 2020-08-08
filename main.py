import os
import warnings
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pylab as plt
import tensorflow_datasets as tfds
from utils import *
warnings.filterwarnings('ignore')

feature_extractor = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
datasets, info = tfds.load(name='beans', with_info=True, as_supervised=True, split=['train','test','validation'])
print("dataset info:",info)
print("train dataset lenths:",len(list(datasets[0])) )

train_dataset, test_dataset, val_dataset = get_dataset(datasets)
train_dataset.cache()
val_dataset.cache()

feature_extractor_layer = hub.KerasLayer(feature_extractor, input_shape=(224,224,3))
feature_extractor_layer.trainable = False

model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(3,activation='softmax')
])

print("--------------------")
print("---model summary---")
print("--------------------")
print(model.summary())


model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['acc'])

history = model.fit(train_dataset, epochs=6, validation_data=val_dataset)   

result=model.evaluate(test_dataset)

for test_sample in datasets[1].take(10):  
  image, label = test_sample[0], test_sample[1]
  image_scaled, label_arr= scale(test_sample[0], test_sample[1])
  image_scaled = np.expand_dims(image_scaled, axis=0)   

  img = tf.keras.preprocessing.image.img_to_array(image)                    
  pred=model.predict(image_scaled)
  print(pred)
  plt.figure()
  plt.imshow(image)
  plt.show()
  print("Actual Label: %s" % info.features["label"].names[label.numpy()])
  print("Predicted Label: %s" % info.features["label"].names[np.argmax(pred)])


for f0,f1 in datasets[1].map(scale).batch(200):
  y=np.argmax(f1, axis=1)
  y_pred=np.argmax(model.predict(f0),axis=1)
  print(tf.math.confusion_matrix(labels=y, predictions=y_pred, num_classes=3))
