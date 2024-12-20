# fast-food-classification

![Portada](./img/portada.png)

1. [About Dataset](#schema1)
2. [The necessary libraries](#schema2)
3. [Image Generators](#schema3)
4. [Model](#schema4)
5. [Fit Model](#schema5)

[Resource](#schemaref)


<hr>

<a name="schema1"></a>

# 1. About Dataset



This is Fast Food Classification data set containing images of 10 different types of fast food. Each directory represents a class, and each class represents a food type. The Classes are :

- Burger
- Donut
- Hot Dog
- Pizza
- Sandwich
- Baked Potato
- Crispy Chicken
- Fries
- Taco
- Taquito

The data set is divided into 4 parts, the `Tensorflow Records`, `Training`, `Validation Data` and `Testing Data`.

The tensorflow records directory is further divided into 3 parts, the Train, Valid and Test. These images are resized to 256 by 256 pixels. No other augmentation is applied. While loading the tensorflow records files, you can apply any augmentation you want.

- Train : Contains 15,000 training images, with each class having 1,500 images.

- Valid : Contains 3,500 validation images, with each class having 400 images.

- Test : Contains 1,500 validation images, with each class having 100/200 images.

    - Unlike the Tensorflow records data, the Training data, validation data and testing data contains direct images. These are raw images. So any kind of augmentation, and specially resizing, can be applied on them.

- Training Data : This directory contains 5 subdirectories. Each directory representing a class. Each class have 1,500 training images.

    - Validation Data : This directory also contains 10 subdirectories. Each directory representing a class. Each **class have 400 images for monitoring model's performance.

    - Testing Data : This directory also contains 10 subdirectories. Each directory representing a class. Each **class have 100 /200 images for evaluating model's performance.

<hr>

<a name="schema2"></a>

# 2. The necessary libraries

**1. TensorFlow**
- TensorFlow is an open-source library developed by Google for machine learning and artificial intelligence. It provides tools to build and train deep learning models, including neural networks. It is known for its flexibility, support for distributed computing, and ability to run on GPUs and TPUs.

**2. Keras**
- Keras is a high-level API built into TensorFlow that simplifies the development and training of deep learning models. It is beginner-friendly, designed for simplicity, and great for rapid prototyping. Keras supports various types of networks, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs).

**3. Matplotlib**
- Matplotlib is a library for creating 2D plots and graphs. It is commonly used for data visualization, offering tools to generate line plots, bar charts, histograms, scatter plots, and more. Its most popular module, pyplot, provides a MATLAB-like interface for plotting.

**4. NumPy**
- NumPy is a fundamental library for scientific computing in Python. It provides support for multidimensional arrays and matrices along with a wide collection of mathematical functions to operate on them. It serves as the backbone of many other libraries in Python's ecosystem, such as TensorFlow and SciPy.

<hr>

<a name="schema3"></a>

# 3. Image Generators

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
```

The code you've shared is for the `ImageDataGenerator` class from Keras, which is used for real-time data augmentation in deep learning. It generates batches of image data with real-time data augmentation during training. Here’s a breakdown of what each argument does:

1. `rescale=1.0/255`

    - This rescales the pixel values of the images from a range of 0-255 to a range of 0-1. It divides each pixel value by 255, which is commonly done to normalize image data and make the model training more stable.

2. `rotation_range=20`
    - This randomly rotates the image by up to 20 degrees during training. This helps the model generalize better by seeing images at different angles.

3. `width_shift_range=0.2`
    - This shifts the image horizontally by a random factor up to 20% of the image width, which helps with spatial invariance.

4. `height_shift_range=0.2`
    - This shifts the image vertically by a random factor up to 20% of the image height, again promoting better generalization.

5. `shear_range=0.2`
    - This applies a random shear transformation to the image. It slants the image, which can help the model learn from various perspectives of objects.

6. `zoom_range=0.2`
    - This zooms in or out on the image by up to 20%. This transformation is useful for making the model more robust to variations in object size.

7. `horizontal_flip=True`
    - This randomly flips the image horizontally, which helps the model learn from mirrored versions of objects. It’s useful when objects do not have a fixed orientation.

8. `validation_split=0.2`
    - This splits 20% of the data for validation purposes, meaning it automatically reserves 20% of the data for validating the model’s performance during training. This helps avoid overfitting and provides a measure of how well the model generalizes.


<hr>

<a name="schema4"></a>

# 4. Model

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])
```

This model is a Convolutional Neural Network (CNN) with 3 convolutional layers followed by max-pooling and dropout layers. After that, the output of the convolutional layers is flattened and passed through fully connected layers with ReLU activations. The final layer uses softmax activation to output probabilities for classification. The model is designed to perform image classification tasks, with data augmentation provided by the train_generator.

##  1. Sequential()
- This defines a linear stack of layers. Each layer in the model has exactly one input tensor and one output tensor, making it suitable for simple, stackable architectures like CNNs.
## 2. `Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)`)
- `Conv2D`: This is a 2D convolutional layer. It applies 32 filters (kernels) of size 3x3 to the input image, which helps in detecting patterns like edges, textures, etc.
- `activation='relu'`: ReLU (Rectified Linear Unit) is used as the activation function, which introduces non-linearity, allowing the network to learn more complex features.
- `input_shape`: Defines the shape of the input image. Here, IMAGE_SIZE[0] and IMAGE_SIZE[1] are the height and width of the image, and 3 indicates 3 color channels (RGB).
## 3. `MaxPooling2D(pool_size=(2, 2))`
- `MaxPooling2D`: This layer performs max pooling, which reduces the spatial dimensions of the feature maps by selecting the maximum value in each 2x2 block. This reduces the number of parameters and helps prevent overfitting.
## 4. `Dropout(0.25)`
- `Dropout`: This regularization technique randomly sets a fraction (25% in this case) of the input units to zero during training to prevent overfitting and improve generalization.
## 5. `Conv2D(64, (3, 3), activation='relu')`
- Another `Conv2D` layer, but this time with 64 filters. The rest of the configuration is similar to the first convolutional layer, allowing the model to learn more complex patterns.
## 6. `MaxPooling2D(pool_size=(2, 2))`
- Another `MaxPooling2D` layer, which helps in reducing the spatial dimensions of the feature maps after the second convolution.
## 7. `Dropout(0.25)`
- `Dropout with 25%` chance again to reduce overfitting.
## 8. `Conv2D(128, (3, 3), activation='relu')`
- Another convolutional layer with 128 filters, further increasing the capacity of the network to learn complex features.
## 9. `MaxPooling2D(pool_size=(2, 2))`
- Another `MaxPooling2D layer`.
## 10. `Dropout(0.25)`
- Dropout with 25% chance again.
## 11. Flatten()
- `Flatten`: This layer flattens the multi-dimensional feature maps from the previous layer into a one-dimensional vector. This step is necessary before feeding the data into the fully connected (dense) layers.
## 12. `Dense(512, activation='relu')`
- `Dense`: This is a fully connected layer with 512 neurons. The ReLU activation function introduces non-linearity, enabling the model to learn more complex representations.
## 13. `Dropout(0.5)`
- `Dropout` with a higher dropout rate of 50%. This helps to further regularize the model and prevent overfitting.
## 14. `Dense(train_generator.num_classes, activation='softmax')`
- This is the output layer. It has as many neurons as the number of classes in the dataset (`train_generator.num_classes`), and softmax activation is used to output a probability distribution over the classes. The neuron with the highest probability will be considered as the model's prediction.



<hr>

<a name="schema4"></a>

# 5. Fit Model

```python
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)
```

The code trains the model for a specified number of epochs (`EPOCHS`), using the training data provided by train_generator and evaluating on test_generator after each epoch.
The `EarlyStopping` callback is used to stop training early if the validation performance doesn't improve after 5 epochs, and it ensures that the model keeps the best weights during training.


## history = model.fit(...)

- `fit()`: This method trains the model for a fixed number of epochs (iterations over the entire dataset). It uses the data provided in the train_generator and evaluates the model using the test_generator. The method returns a history object that contains the training and validation metrics for each epoch.
## train_generator
- `train_generator`: This is the data generator that provides batches of training data. In this case, it's likely an instance of ImageDataGenerator or another generator that loads and preprocesses image data for training. It provides the input images and labels to the model in real time during training.
## epochs=EPOCHS
- `epochs`: This parameter defines how many times the model will iterate over the entire training dataset. EPOCHS is typically a variable defined elsewhere, and it specifies the number of epochs the model will train for.
## validation_data=test_generator
- `validation_data`: This parameter specifies the data that the model will use for validation after each epoch. The model evaluates its performance on this data at the end of each epoch to check if it's overfitting or improving. Here, test_generator is used to generate the validation data. Like train_generator, it's typically an instance of ImageDataGenerator (or another data generator) for the test set.
## callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
- `callbacks`: This parameter accepts a list of callback functions that are applied during training. In this case, the model is using EarlyStopping.

    - `EarlyStopping`: This callback monitors the training process and can stop training early if the model's performance on the validation set stops improving. It helps to prevent overfitting by stopping the training before the model starts overfitting on the training data.
        - `patience=5`: This means that training will stop if the validation performance (e.g., validation loss or accuracy) does not improve for 5 consecutive epochs. The patience allows some flexibility, so the model doesn’t stop prematurely.
        - `restore_best_weights=True`: This ensures that after early stopping, the model will revert to the weights from the best epoch (the epoch with the lowest validation loss or highest validation accuracy). This helps avoid keeping weights from a later epoch where overfitting might have started.
## What Happens During Training:
1. The model will start training using the batches of data from train_generator.
2. After each epoch, the model will evaluate itself on the validation data from test_generator.
3. The EarlyStopping callback will monitor the validation performance, and if it doesn’t improve for 5 epochs, it will stop training early.
4. The history object will capture the training and validation metrics (like loss and accuracy) for each epoch, which can be used for analysis or plotting.














<hr>

<a name="schemaref"></a>

## Resource 

[Fast Food Kaggle dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset/data)

https://www.kaggle.com/code/daffamaulana42/foodclassification