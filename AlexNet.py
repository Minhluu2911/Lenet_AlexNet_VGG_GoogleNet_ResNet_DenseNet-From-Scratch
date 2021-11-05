from utilities import *

# Build AlexNet (with padding SAME)
class AlexNet:
  def __init__(self, activation="relu"):
    self.model = tf.keras.Sequential()
    self.buildModel(activation)

  def buildModel(self, activation):
    # size image = ((n + 2p - f)/s) + 1
    
    # input shape is image with size (32,32) -> let padding same
    # Conv 1
    self.model.add(tfl.Conv2D(filters=96, kernel_size=(11,11),strides=(4,4), padding='SAME', input_shape=(32,32,3)))
    self.model.add(tfl.BatchNormalization())
    self.model.add(tfl.Activation('relu'))
    self.model.add(tfl.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='SAME'))

    # Conv 2
    self.model.add(tfl.Conv2D(filters=256, kernel_size=(5,5),strides=(1,1), padding='SAME'))
    self.model.add(tfl.BatchNormalization())
    self.model.add(tfl.Activation('relu'))
    self.model.add(tfl.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='SAME'))
    
    # Conv 3
    self.model.add(tfl.Conv2D(filters=384, kernel_size=(3,3),strides=(1,1), padding='SAME'))
    self.model.add(tfl.BatchNormalization())
    self.model.add(tfl.Activation('relu'))

    # Conv 4
    self.model.add(tfl.Conv2D(filters=384, kernel_size=(3,3),strides=(1,1), padding='SAME'))
    self.model.add(tfl.BatchNormalization())
    self.model.add(tfl.Activation('relu'))

    # Conv 5
    self.model.add(tfl.Conv2D(filters=256, kernel_size=(3,3),strides=(1,1), padding='SAME'))
    self.model.add(tfl.BatchNormalization())
    self.model.add(tfl.Activation('relu'))
    self.model.add(tfl.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='SAME'))
    
    # FC
    self.model.add(tfl.Flatten())
    self.model.add(tfl.Dense(units=4096, activation=activation))
    self.model.add(tfl.Dense(units=4096, activation=activation))
    self.model.add(tfl.Dense(units=10, activation="softmax"))
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 

  def fit(self, X_train, Y_train, validation_data, epochs=10, batch_size=16):
    print("Training AlexNet model...")
    self.history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
    print("Training completed.")
    
  def evaluate(self, X_test, Y_test):
    self.model.evaluate(X_test, Y_test)
  
  def predict(self, X_test, Y_test, CLASS_NAMES=None, no_show=9):
    print("Generate predictions for {} samples".format(X_test.shape[0]))
    predictions = self.model.predict(X_test)
    print("predictions shape:", predictions.shape)
    fig = plt.figure(figsize=(6, 6))
    n = math.ceil(math.sqrt(no_show))

    if CLASS_NAMES:
      for i in range(no_show):  
        fig.add_subplot(n, n, i+1)
        plt.imshow(np.squeeze(X_test[i]), cmap=plt.get_cmap('gray'))
        plt.title("Predict: {}".format(CLASS_NAMES[np.argmax(predictions[i])]))
    else:
      for i in range(no_show):  
        fig.add_subplot(n, n, i+1)
        plt.imshow(np.squeeze(X_test[i]), cmap=plt.get_cmap('gray'))
        plt.title("Predict: {}".format(np.argmax(predictions[i])))

    plt.tight_layout(True)
    plt.show()

    return predictions

  def plot_result(self):
    # summarize history for accuracy
    plt.plot(self.history.history['accuracy'])
    plt.plot(self.history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(self.history.history['loss'])
    plt.plot(self.history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
  
  

# Build AlexNet (with padding as paper)
class AlexNet1:
  def __init__(self, activation="relu"):
    self.model = tf.keras.Sequential()
    self.buildModel(activation)

  def buildModel(self, activation):
    # size image = ((n + 2p - f)/s) + 1
    
    # In order to get the same size of the paper mentioned -> add padding layer 
    self.model.add(tfl.ZeroPadding2D(padding=(98, 98)))

    # Conv 1
    self.model.add(tfl.Conv2D(filters=96, kernel_size=(11,11),strides=(4,4), padding='VALID', input_shape=(32,32,3)))
    self.model.add(tfl.BatchNormalization())
    self.model.add(tfl.Activation('relu'))
    self.model.add(tfl.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='VALID'))

    # Conv 2
    self.model.add(tfl.Conv2D(filters=256, kernel_size=(5,5),strides=(1,1), padding='SAME'))
    self.model.add(tfl.BatchNormalization())
    self.model.add(tfl.Activation('relu'))
    self.model.add(tfl.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='VALID'))
    
    # Conv 3
    self.model.add(tfl.Conv2D(filters=384, kernel_size=(3,3),strides=(1,1), padding='SAME'))
    self.model.add(tfl.BatchNormalization())
    self.model.add(tfl.Activation('relu'))

    # Conv 4
    self.model.add(tfl.Conv2D(filters=384, kernel_size=(3,3),strides=(1,1), padding='SAME'))
    self.model.add(tfl.BatchNormalization())
    self.model.add(tfl.Activation('relu'))

    # Conv 5
    self.model.add(tfl.Conv2D(filters=256, kernel_size=(3,3),strides=(1,1), padding='SAME'))
    self.model.add(tfl.BatchNormalization())
    self.model.add(tfl.Activation('relu'))
    self.model.add(tfl.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='VALID'))
    
    # FC
    self.model.add(tfl.Flatten())
    # DENSE 1
    self.model.add(tfl.Dense(units=4096))
    self.model.add(tfl.BatchNormalization())
    self.model.add(tfl.Activation('relu'))

    # DENSE 2
    self.model.add(tfl.Dense(units=4096))
    self.model.add(tfl.BatchNormalization())
    self.model.add(tfl.Activation('relu'))

    # OUTPUT
    self.model.add(tfl.Dense(units=10))
    self.model.add(tfl.BatchNormalization())
    self.model.add(tfl.Activation('softmax'))
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 

  def fit(self, X_train, Y_train, validation_data, epochs=10, batch_size=16):
    print("Training AlexNet model...")
    self.history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
    print("Training completed.")
    
  def evaluate(self, X_test, Y_test):
    self.model.evaluate(X_test, Y_test)
  
  def predict(self, X_test, Y_test, CLASS_NAMES=None, no_show=9):
    print("Generate predictions for {} samples".format(X_test.shape[0]))
    predictions = self.model.predict(X_test)
    print("predictions shape:", predictions.shape)
    fig = plt.figure(figsize=(6, 6))
    n = math.ceil(math.sqrt(no_show))

    if CLASS_NAMES:
      for i in range(no_show):  
        fig.add_subplot(n, n, i+1)
        plt.imshow(np.squeeze(X_test[i]), cmap=plt.get_cmap('gray'))
        plt.title("Predict: {}".format(CLASS_NAMES[np.argmax(predictions[i])]))
    else:
      for i in range(no_show):  
        fig.add_subplot(n, n, i+1)
        plt.imshow(np.squeeze(X_test[i]), cmap=plt.get_cmap('gray'))
        plt.title("Predict: {}".format(np.argmax(predictions[i])))

    plt.tight_layout(True)
    plt.show()

    return predictions

  def plot_result(self):
    # summarize history for accuracy
    plt.plot(self.history.history['accuracy'])
    plt.plot(self.history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(self.history.history['loss'])
    plt.plot(self.history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
  