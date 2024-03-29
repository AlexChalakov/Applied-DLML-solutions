# test script
# adapted from: https://www.tensorflow.org/tutorials/images/cnn

import tensorflow as tf
from keras import models
from keras.datasets import cifar10
from PIL import Image

## cifar-10 dataset
(_, _), (test_images, test_labels) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

## load the trained model
model = models.load_model('saved_model_tf')

## inference
num_images = 8
# use the trained model to predict the labels of the test images
outputs = model.predict(test_images[:num_images,...]/255.0)
# get the index of the class with the highest probability?
predicted = tf.argmax(outputs, 1)
print('Ground-truth:' + ' '.join('%5s' % class_names[test_labels[j,0]] for j in range(num_images)))
print('Predicted: ', ' '.join('%5s' % class_names[predicted[j]] for j in range(num_images)))

# example images
im = Image.fromarray(tf.concat([test_images[i,...] for i in range(num_images)],1).numpy())
im.save("test_tf_images.jpg")
print('test_tf_images.jpg saved.')
