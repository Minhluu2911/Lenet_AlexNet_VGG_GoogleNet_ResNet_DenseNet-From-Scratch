from utilities import *
# Build LeNet

class LeNet:
  def __init__(self, activation="sigmoid"):
    self.model = tf.keras.Sequential()
    self.buildModel(activation)
    

  def buildModel(self, activation):
    # size image = ((n + 2p - f)/s) + 1
    self.model.add(tfl.Conv2D(filters=6, kernel_size=(5,5),strides=(1,1), padding='SAME'))
    self.model.add(tfl.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='SAME'))
    self.model.add(tfl.Conv2D(filters=16, kernel_size=(5,5),strides=(1,1), padding='VALID'))
    self.model.add(tfl.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='VALID'))
    self.model.add(tfl.Flatten())
    self.model.add(tfl.Dense(units=120, activation=activation))
    self.model.add(tfl.Dense(units=84, activation=activation))
    self.model.add(tfl.Dense(units=10, activation="softmax"))
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy']) 

  def fit(self, X_train, Y_train, validation_data, epochs=10, batch_size=16):
    print("Training Lenet-5 model...")
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