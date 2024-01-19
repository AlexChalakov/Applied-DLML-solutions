# train script
# adapted from: https://www.tensorflow.org/tutorials/images/cnn

import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.datasets import cifar10
from PIL import Image

# additional
from keras.utils import plot_model
import matplotlib.pyplot as plt

## cifar-10 dataset
# load dataset and split into train set with their appropriate labels
(train_images, train_labels), (_, _) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# example images
# save the first 30 images of the train set (if images are less, less accuracy on training, but faster overall to process)
num_images = 30
im = Image.fromarray(tf.concat([train_images[i,...] for i in range(num_images)],1).numpy())
im.save("train_tf_images.jpg")
print('train_tf_images.jpg saved.')
print('Ground truth labels:' + ' '.join('%5s' % class_names[train_labels[j,0]] for j in range(num_images)))

# normalize to [0,1]
train_images = train_images / 255.0

## cnn
# you build the model by stacking layers
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10))

model.summary()

## compile with loss and optimiser
# you calculate loss and accuracy during training, optimizer is adam, loss function is sparse categorical crossentropy
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

## train 
history = model.fit(train_images, train_labels, epochs=10)
print('Training done.')

# plot model's architecture
plot_model(model, to_file='model_architecture.png')
print('Model plotted.')
Image.open('model_architecture.png').show()

# now plot the loss and accuracy from model
history_dict = history.history
print(history_dict.keys())
# plot loss
plt.figure()
plt.plot(history_dict['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('model_loss.png')
# plot accuracy
plt.figure()
plt.plot(history_dict['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('model_accuracy.png')
plt.show()

# save trained model to put in test_tf.py
model.save('saved_model_tf')
print('Model saved.')
