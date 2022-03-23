from utilities import *

# Build ResNet
class ResNet:
  def __init__(self, activation="relu"):
    self.model = tf.keras.Sequential()
    self.buildModel(activation)
    
  def identity_block(self, X, f, filters, training=True, initializer=random_uniform):
    """
    Argument:
    X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f: the shape of the middle CONV's window for the main path
    filters: number of filters in the CONV layers of the main path
    training: True: in training mode  ----  False: in predict mode
    initializer: to set up the initial weights of a layer.
    """

    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Store input value for later to add back to the main path. 
    X_shortcut = X
    
    # First component
    X = tfl.Conv2D(filters = F1, kernel_size = 1, strides = 1, padding = 'VALID', kernel_initializer = initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis = 3)(X, training = training)
    X = tfl.Activation('relu')(X)
    
    # Second component
    X = tfl.Conv2D(filters = F2, kernel_size = f, strides = 1, padding = 'SAME', kernel_initializer = initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis = 3)(X, training = training)
    X = tfl.Activation('relu')(X)

    # Third component
    X = tfl.Conv2D(filters = F3, kernel_size = 1, strides = 1, padding = 'VALID', kernel_initializer = initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis = 3)(X, training = training)
    
    # Add the shortcut and the output of third component together
    X = tfl.Add()([X_shortcut, X])
    X = tfl.Activation('relu')(X)

    return X

  def convolutional_block(self, X, f, filters, s = 2, training=True, initializer=glorot_uniform):
    """
    Arguments:
    X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f: the shape of the middle CONV's window for the main path
    filters: number of filters in the CONV layers of the main path
    s: the stride to be used
    training: True: in training mode  ----  False: in predict mode
    initializer: to set up the initial weights of a layer.
    """
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Store input value for later to add back to the main path. 
    X_shortcut = X
    
    # First component
    X = tfl.Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis = 3)(X, training=training)
    X = tfl.Activation('relu')(X)
    
    ## Second component
    X = tfl.Conv2D(filters = F2, kernel_size = f, strides = (1, 1), padding='same', kernel_initializer = initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis = 3)(X, training=training)
    X = tfl.Activation('relu')(X)

    ## Third component
    X = tfl.Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = tfl.BatchNormalization(axis = 3)(X, training=training)
    

    X_shortcut = tfl.Conv2D(filters = F3, kernel_size = 1, strides = s, padding='valid', kernel_initializer = initializer(seed=0))(X_shortcut)
    X_shortcut = tfl.BatchNormalization(axis = 3)(X_shortcut, training=training)

    # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
    X = tfl.Add()([X, X_shortcut])
    X = tfl.Activation('relu')(X)
    
    return X

  def ResNet50(self, input_shape = (32, 32, 3), classes = 10):
    """
    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = tfl.Input(input_shape)

    
    # Zero-Padding
    X = tfl.ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = tfl.Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    X = tfl.BatchNormalization(axis = 3)(X)
    X = tfl.Activation('relu')(X)
    X = tfl.MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = self.convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = self.identity_block(X, 3, [64, 64, 256])
    X = self.identity_block(X, 3, [64, 64, 256])

    ### START CODE HERE
    
    ## Stage 3 (≈4 lines)
    X = self.convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2)
    X = self.identity_block(X, 3, [128, 128, 512])
    X = self.identity_block(X, 3, [128, 128, 512])
    X = self.identity_block(X, 3, [128, 128, 512])
    
    ## Stage 4 (≈6 lines)
    X = self.convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
    X = self.identity_block(X, 3, [256, 256, 1024])
    X = self.identity_block(X, 3, [256, 256, 1024])
    X = self.identity_block(X, 3, [256, 256, 1024])
    X = self.identity_block(X, 3, [256, 256, 1024])
    X = self.identity_block(X, 3, [256, 256, 1024])

    ## Stage 5 (≈3 lines)
    X = self.convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)
    X = self.identity_block(X, 3, [512, 512, 2048])
    X = self.identity_block(X, 3, [512, 512, 2048])

    ## AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = tfl.AveragePooling2D(pool_size=2)(X)
    
    ### END CODE HERE

    # output layer
    X = tfl.Flatten()(X)
    X = tfl.Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = tfl.Model(inputs = X_input, outputs = X)

    return model
  
  def buildModel(self, activation):
    self.kernel_init = keras.initializers.glorot_uniform()
    self.bias_init = keras.initializers.Constant(value=0.2)
    
    kernel_init = self.kernel_init
    bias_init = self.bias_init
    
    # size image = ((n + 2p - f)/s) + 1
    input_layer = keras.Input(shape=(32, 32, 3)) # the paper train on image shape 224,224,3 but I'm train on cifar10 dataset with image shape 32,32,3
    
    
    
    # Global Average Pooling
    X = tfl.GlobalAveragePooling2D()(X)
    
    # Dropout
    X = tfl.Dropout(rate=0.4)(X2)
    
    # Last output Softmax 2
    X = tfl.Dense(units=10, activation='softmax', name='softmax2')(X)
    
    
    self.model = tf.keras.Model(input_layer, [X, X1, X2], name='ResNet')
    
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 

  def fit(self, X_train, Y_train, validation_data, epochs=10, batch_size=16):
    print("Training ResNet model...")
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
  
