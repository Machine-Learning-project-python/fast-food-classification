# fast-food-classification

![Portada](./img/portada.png)

1. [About Dataset](#schema1)

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

<a name="schemaref"></a>

## Resource 

[Fast Food Kaggle dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset/data)