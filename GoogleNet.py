from utilities import *

# Build GoogleNet
class GoogleNet:
  def __init__(self, activation="relu"):
    self.model = tf.keras.Sequential()
    self.buildModel(activation)

  '''
  f_1x1: number of filter for 1x1 conv
  f_3x3_reduce: number of filter for 1x1 conv before 3x3 conv
  f_3x3: number of filter for 3x3 conv
  f_5x5_reduce: number of filter for 1x1 conv before 5x5 conv
  f_5x5: number of filter for 5x5 conv
  f_pp: number of filter for 1x1 conv after max pooling
  '''
  def inception_block(self, input_layer, f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp):
    
    
    kernel_init = self.kernel_init
    bias_init = self.bias_init
    
    # conv 1x1 path
    conv1x1 = tfl.Conv2D(filters=f_1x1, kernel_size=1, padding='SAME', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    
    # conv 3x3 path
    conv3x3 = tfl.Conv2D(filters=f_3x3_reduce, kernel_size=1, padding='SAME', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    conv3x3 = tfl.Conv2D(filters=f_3x3, kernel_size=3, padding='SAME', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv3x3)
    
    # conv 5x5 path
    conv5x5 = tfl.Conv2D(filters=f_5x5_reduce, kernel_size=1, padding='SAME', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    conv5x5 = tfl.Conv2D(filters=f_5x5, kernel_size=3, padding='SAME', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv5x5)
    
    # max pooling path
    max_pool = tfl.MaxPool2D(pool_size=(3,3), strides=1 ,padding='SAME')(input_layer)
    max_pool = tfl.Conv2D(filters=f_pp, kernel_size=1, strides = 1 ,padding='SAME', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(max_pool)
    
    # concatenate
    output = tfl.Concatenate(axis=3)([conv1x1, conv3x3, conv5x5, max_pool])
    
    return output

  def buildModel(self, activation):
    self.kernel_init = keras.initializers.glorot_uniform()
    self.bias_init = keras.initializers.Constant(value=0.2)
    
    kernel_init = self.kernel_init
    bias_init = self.bias_init
    
    # size image = ((n + 2p - f)/s) + 1
    input_layer = tfl.Input(shape=(224, 224, 3))
    
    # Conv 7x7/2
    X = tfl.Conv2D(filters=64, kernel_size=7, strides=2, padding='SAME', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    
    # Max pool 3x3/2
    X = tfl.MaxPool2D(pool_size=3, strides=2, padding='SAME')(X)
    
    # Conv 3x3/1
    X = tfl.Conv2D(filters=64, kernel_size=1, strides=1, padding='SAME', activation='relu')(X)
    X = tfl.Conv2D(filters=192, kernel_size=3, strides=1, padding='SAME', activation='relu')(X)
    
    # Max pool 3x3/2
    X = tfl.MaxPool2D(pool_size=3, strides=2, padding='SAME')(X)
    
    # inception 3a block
    [f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp] = [64, 96, 128, 16, 32, 32]
    X = self.inception_block(X, f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp)
    
    # inception 3b block
    [f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp] = [128, 128, 192, 32, 96, 64]
    X = self.inception_block(X, f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp)

    # Max pool 3x3/2
    X = tfl.MaxPool2D(pool_size=3, strides=2, padding='SAME')
    
    # Inception 4a
    [f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp] = [192, 96, 208, 16, 48, 64]
    X = self.inception_block(X, f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp)
    
    # Softmax 0
    X1 = tfl.AveragePooling2D(pool_size=5, strides=3, padding='VALID')(X)
    X1 = tfl.Conv2D(filters=128, kernel_size=1, strides=1, padding='SAME', activation='relu')(X1)
    X1 = tfl.Flatten()(X1)
    X1 = tfl.Dense(units=1024, activation='relu')(X1)
    X1 = tfl.Dropout(rate=0.7)(X1)
    X1 = tfl.Dense(units=10, activation='softmax', name='softmax0')(X1)
    
    # Inception 4b
    [f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp] = [160, 112, 224, 24, 64, 64]
    X = self.inception_block(X, f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp)
    
    # Inception 4c
    [f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp] = [128, 128, 256, 24, 64, 64]
    X = self.inception_block(X, f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp)
    
    # Inception 4d
    [f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp] = [112, 144, 288, 32, 64, 64]
    X = self.inception_block(X, f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp)
    
    # Softmax 1
    X2 = tfl.AveragePooling2D(pool_size=5, strides=3, padding='VALID')(X)
    X2 = tfl.Conv2D(filters=128, kernel_size=1, strides=1, padding='SAME', activation='relu')(X2)
    X2 = tfl.Flatten()(X2)
    X2 = tfl.Dense(units=1024, activation='relu')(X2)
    X2 = tfl.Dropout(rate=0.7)(X2)
    X2 = tfl.Dense(units=10, activation='softmax', name='softmax1')(X2)
    
    # Inception 4e
    [f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp] = [256, 160, 320, 32, 128, 128]
    X = self.inception_block(X, f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp)
    
    # Max pool 3x3/2
    X = tfl.MaxPool2D(pool_size=3, strides=2, padding='SAME')
    
    # Inception 5a
    [f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp] = [256, 160, 320, 32, 128, 128]
    X = self.inception_block(X, f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp)
    
    # Inception 5b
    [f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp] = [384, 192, 384, 48, 128, 128]
    X = self.inception_block(X, f_1x1, f_3x3_reduce, f_3x3, f_5x5_reduce, f_5x5, f_pp)
    
    # Global Average Pooling
    X = tfl.GlobalAveragePooling2D()(X)
    
    # Dropout
    X = tfl.Dropout(rate=0.4)(X2)
    
    # Last output Softmax 2
    X = tfl.Dense(units=10, activation='softmax', name='softmax2')(X)
    
    self.model = tfl.Model(input_layer, [X, X1, X2], name='GoogleNet')
    
    self.model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 

  def fit(self, X_train, Y_train, validation_data, epochs=10, batch_size=16):
    print("Training GoogleNet model...")
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
  