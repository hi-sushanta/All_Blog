---
title: "Complete Neural Network Regression In Tensorflow — Guide 2023"
datePublished: Fri Jan 27 2023 03:09:56 GMT+0000 (Coordinated Universal Time)
cuid: clj80mnve000g0al28ox82dzb
slug: neural-network-regression-in-tensorflow-guide
canonical: https://hiwhy.io/neural-network-regression-in-tensorflow-deep-learning/
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1687826974257/f21cb5ce-8821-4d2d-9f7c-6a433326f689.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1687827018624/e193ab53-d4ec-4e9a-8bf0-72a76343aaee.png
tags: data-science, tensorflow, deep-learning, regression

---

Welcome to the complete Neural Network Regression in TensorFlow guide. This is an end-to-end guide you don’t need any experience in this field. We will explain everything. After reading this article you are able to understand the deep learning field and most of all solve any regression problem.

**Keep reading**

## What Is A Regression Problem In Machine Learning

For example, if you build a model and that model predicts your friend’s age. Or if you are building a model predicting ( **Apple stock price** ) this type of problem is called a regression problem. A regression problem is a problem where you can predict continuous values. Some of the examples are a person’s age, weight, earnings, and price.

## What Is Meant By Deep learning

In this topic, I wrote before in depth in this [**article**](https://hiwhy.io/complete-machine-learning-guide). But for now, before learning deep learning, first understand where deep learning comes from.

Deep learning is a subfield of machine learning, and machine learning itself is a subfield of Artificial intelligence ( **AI** ).

![Artificial Intelligence Diagram](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402399912/de415843-f44e-4345-9fd9-24c7a4b2a0e8.png align="center")

## Simple Definition What Is Artificial Intelligence

In general speaking, **Artificial Intelligence ( AI )** is the process we make intelligent machines. Every AI developer’s goal is to use data and build intelligent machines.

### Application Of Artificial Intelligence

Artificial intelligence is like a Sci-Fi movie type. Today’s news, research paper, website articles, and podcasts everyone talking about this field, it’s like a booming topic at this time. You know AI and you think where it’s used. I don’t see AI everywhere — Is this your question a good question? Actually, it’s widely used today, in every product you use, and every company in other terms [**AI is everywhere**](https://www.oreilly.com/radar/ai-adoption-in-the-enterprise-2022/) at this time. You watch a **Youtube** video and one thing notice Youtube recommended what video to watch next that’s AI. According to new research, AI is used today in 80% of companies that grow their [**company**](https://builtin.com/artificial-intelligence/examples-ai-in-industry).

![Woman sees video screen](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402401195/db91bd3e-0d01-47b8-a25a-97be91ef1516.png align="center")

## Machine learning Introduction

Now you have a basic understanding of the field of AI — and it’s time to know the subfield of Machine learning (ML). Machine learning is the process of building better AI. Machine learning is the field we are using algorithms to analyze the data, learn from that data, improve, and then the last step is to make a prediction about new data.

Understand this example — 🔻

We have lots of data, on heart disease patient records. So you built a machine learning model to predict heart disease patients. So our job is to train using ML algorithms and then check the new patient record and see how our model predicts.

There are many types of machine-learning algorithms that have different types of problems. Here is a list of common machine learning algorithms.

* Regression
    
* Decision Trees
    
* Support Vector Machines (SVM)
    
* Naive Bayes Classifiers
    
* K-Nearest Neighbor
    
* Artificial Neural Networks
    

![Artificial Intelligence Diagram](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402403411/0d79ec4d-2727-4dc5-82e5-2b6cf7f6e8f5.png align="center")

## Deep Learning Introduction

Now you understand the basics of Machine Learning (ML) and next is to learn the subfield of Deep Learning ( DL ). Yes, deep learning is a subfield of machine learning. Deep learning is an approach to how to improve machine learning so you can build better AI models. The goal is simply to improve AI intelligence.

Machine learning uses algorithms, and the same deep learning use algorithms are called Artificial Neural Networks (ANNs ). These algorithms are inspired by our human brain neurons. In the upcoming article, we discuss in-depth artificial neural networks . So sign up for our newsletter, so never miss an article the next time we publish.

![ Artificial Neural Networks input and output and hidden layer](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402404890/3baf43bb-fe2e-46e8-85fd-946e1565ca25.png align="center")

Now it’s time to learn the regression problem. First, we created a small example dataset in ourselves and later we used the existing dataset.

```python
import tensorflow as tf
print (tf.__version__) # Check the version tensorflow 

# Output 2.9.2
```

Now it’s time to create some regression data. 🔻

```python
# Import the important libraries we need.
import numpy as np 
import matplotlib.pyplot as plt

# X is a our features - 
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# y is a our labels 
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# let's visulaize our data 
plt.scatter(X, y);
plt.title ("This is our demo data X and y ");
plt.xlabel("This is our X axis ");
plt.ylabel("This is our y axis ");
```

![the regression line in matplotlib plot](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402407040/53f0c143-35ca-4367-b4ba-8a3976595cec.png align="center")

Understand the above code.⬆

* We first import NumPy, in short form np. This library helps me calculate mathematics computation fast. If you learn more about NumPy step by step guide, [**read this article**](https://hiwhy.io/numpy-complete-guide)
    
* Then we import the Matplotlib library, this library helps me visualize our data and model. If you want to learn more, [**read this article**](https://hiwhy.io/matplotlib-complete-guide).
    
* After we create our data, to work on regression problems. In this case, our data. ⬇
    
    ```python
    # X is a features and y is a label 
    X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
    y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])
    ```
    

What problems are we solving using this data? Our goal is to input X value \[10 or any number\] so what is the value y \[?\].

Meaning your input value `X [10]` so `y is [20]`. Our formula is `(y = X + 10)` . Now it’s time to solve this problem `X[ 27 ]` so what is `y [? ]`.

Note 🔥: This is the most important thing in working neural network input and output shapes. Don’t worry too much this time, I will write in full depth in my next article. But for now, a little bit about know this.

**Input shape —** This layer takes input based on existing data.

**Output shape —** Output for predictions based on what data input shape or hidden layer.

In general, neural networks accept numbers and output numbers. These numbers typically act as tensors ( arrays ).

```python
# Now it's time import important libray we need 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf # tf is short form tensorflow

# X is a our features 
X = tf.constant([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])

# y is a our labels (y = X + 10)
y = tf.constant([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# let's visulaize our data 
plt.scatter(X, y);
plt.title ("This is our data X and y ");
plt.xlabel(" X axis ");
plt.ylabel(" Y axis ");
```

![the regression line in matplotlib plot](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402411732/8f8acfa8-a568-4d2f-9bb3-6233af401f20.png align="center")

## Neural Network Architecture Regression Model

![ architecture in neural network regression ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402413075/6b0b39d2-e492-4ec9-9b87-98c613464c67.png align="left")

This is the most common architecture in a neural network, but this is not the only way to write and solve regression problems. In deep learning there are many ways to solve problems. Now is the time to build our first deep learning model.

But before building a model, let’s understand what the steps are.

## How To Build A Deep Learning Model In TensorFlow

Here are the steps you need to understand to build any deep learning model in Tensorflow. In TensorFlow three fundamental steps in creating and training the model. Creating a model, Compiling a model, and Fitting a model.

See the code below so you can understand better.⬇

```python
# Set random seed using Tensorflow
tf.random.set_seed(42)

# Create a model using the Keras Sequential API
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
              optimizer=tf.keras.optimizers.SGD(), # SGD is short for stochastic gradient descent
              metrics=["mae"])

# Fit the model
model.fit(tf.expand_dims(X, axis=-1), y, epochs=5)
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1687445036118/02fdf379-add9-41b8-9463-0242aef61600.png align="center")

Now it’s boom 💥we create our first deep learning model and the next step is to make predictions.

```python
# First see our X and y data.. 
print ("X is -- ", X)
print ("y is -- ", y)
```

![Google colab output NumPy arrays](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402417446/9bd91016-7779-4f41-992c-59f25ff9978e.png align="left")

```python
model.predict([20.0])
```

![Google colab output](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402418410/387b79e6-3cda-4306-bf9b-3d6a0a942e39.png align="left")

OOO no 😞 our model prediction is not good, you accepted. Our model predicted ( y = 14) but the correct result is ( y = 30 ). Meaning we are building a not-a-good deep learning model. So it’s time to improve our model.

### Here Is The Step To Improve Deep Learning Model

* Adding layers
    
* Increase the number of hidden layers
    
* Change the activation function
    
* Change the learning rate
    
* Fitting more data
    
* Fitting for longer ( Epochs 1 to 100 or so on )
    

![The deep learning model improves step](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402420783/6fa7c1a3-e725-45f5-a519-0bc71842c008.png align="left")

Now it’s time to improve our deep learning model step by step. We can change things one at a time. See code below 🔻

```python
# Set random seed using Tensorflow
tf.random.set_seed(42)

# Create a model using the Keras Sequential API
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
              optimizer=tf.keras.optimizers.SGD(), # SGD is short for stochastic gradient descent
              metrics=["mae"])

# Fit the model (This time we train for longer 100 epochs)
model.fit(tf.expand_dims(X, axis=-1), y, epochs=100)
```

![Google colab output](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402422959/f84b185b-866a-4c1b-820f-d97894e23109.png align="left")

```python
# See our data 
print ("X is -- ", X)
print ("y is -- ", y)
```

![Google colab output](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402424299/89ec4bf6-e304-43fa-aeb4-c6f3b02fc1cb.png align="left")

```python
# Try to predict what is y The right answer is 30 (y = X + 10)
model.predict([20.0])
```

![Numpy array google colab](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402425320/7e703bc5-6032-4b87-bdf9-e64ca2d9c8d4.png align="left")

It’s better than before. This time our model predicts 35 — we are very close this time to the right answer. But for now, we can improve better so our mode can predict the right answer.

Now it’s time to create a bigger dataset same as above, but this time is big. Meaning fitting more data.

```python
# X is a our Feature (but this time large dataset)
X = np.arange(-100, 100, 4)
X
```

![NumPy array output google colab](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402426417/ca2a0022-2a76-4686-93d0-83caf32c556f.png align="left")

```python
# y is our labels  
y = np.arange(-90, 110, 4)
y
```

$$This\ is\ our\ formula\ —\ y = X +10$$

![NumPy array output google colab](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402427462/45032c07-e3b1-487f-8ba7-af0700067f57.png align="center")

```python
# Check how many samples we have in both datasets
print("X dataset total sample have -- ", len(X))
print("y dataset total sample have -- ", len(y))


# Output 
# X dataset total sample has --  50
# y dataset total sample has --  50

# Split data into train and test sets

X_train = X[:40] # first 40 examples (80% of data)
y_train = y[:40]

X_test = X[40:] # last 10 examples (20% of data)
y_test = y[40:]


# Check the length
print("X_train",len(X_train))
print("y_train is",len(y_train))
print("X_test is",len(X_test))
print("y_test is",len(y_test))

# Output 
# X_train 40
# y_train is 40
# X_test is 10
# y_test is 10
```

**Now is the time to visualize this data 🔻**

```python
plt.figure(figsize=(10, 7))
# Training data is red
plt.scatter(X_train, y_train, c='r', label='Training data')

# Testing data is black
plt.scatter(X_test, y_test, c='k', label='Testing data')

# Show the legend
plt.legend();
```

![matplotlib visualization model predict](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402429340/200e1da9-0a9a-452d-940f-e0451f428152.png align="left")

**Build a model.**

$$🤖$$

```python
# Set random seed
tf.random.set_seed(42)

# Create a model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(1, input_shape=[1]) # define the input_shape to our model
])

# Compile model 
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# Fiting model 
model.fit(tf.expand_dims(X_train, axis=-1),y_train, epochs=100, verbose=0 )
```

Make predictions.

$$🤖 \ = \ 💯$$

```python
# Make predictions (X_test) data model never seen before. 
y_preds = model.predict(X_test)

# Output 
# 1/1 [==============================] - 0s 174ms/step
```

```python
y_preds
```

![NumPy array ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402433443/93bbde1e-f582-4850-aaff-29a144472a3f.png align="center")

I know you don’t understand, but unfortunately, I don’t understand what’s going on with this number. So I always try to visualize predictions. Now it’s time to visualize our predictions.

**Note 💡:** Always remember when you need something again and again to create a function.

```python
def model_predictions(training_data=X_train, 
                     training_labels=y_train, 
                     testing_data=X_test, 
                     testing_labels=y_test, 
                     predictions=y_preds):
  
  plt.figure(figsize=(10, 7))
  # Plot training data in blue
  plt.scatter(training_data, training_labels, c="r", label="Training data")
  # Plot test data in green
  plt.scatter(testing_data, testing_labels, c="k", label="Testing data")
  # Plot the predictions in red (predictions were made on the test data)
  plt.scatter(testing_data, predictions, c="b", label="Predictions")
  # Show the legend
  plt.legend();
```

```python
model_predictions(training_data=X_train,
                 training_labels=y_train,
                 testing_data=X_test,
                 testing_labels=y_test,
                 predictions=y_preds)
```

![Matplotlib visulaztion ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402435631/e74eb8ce-a94f-47db-8b1c-bf53557f4483.png align="center")

The prediction we see in our model does not predict well. Our goal is to build a model, learn from the `red dots (X_train)`, and predicted `black dots (X_test)`. But in this case, our prediction is outside of testing data, meaning there is so much gap in testing data and prediction. It’s really clear that we are not building a good deep learning model. So it is time to improve the model.

One thing to remember is always to visualize your data and model, this is good for working on any deep learning problem. To learn more about Matplotlib, read our this [**end-to-end guide**](https://hiwhy.io/matplotlib-complete-guide).

Now is the time to improve our deep learning model 🤖

```python
# Set random seed
tf.random.set_seed(42)

#
model_1 = tf.keras.Sequential([
  tf.keras.layers.Dense(1)
])

# Compile the model
model_1.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

# Fiting the model
model_1.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100, verbose=0)

# Make a predictions for model_1

y_test_1 = model_1.predict(X_test)
model_predictions(predictions=y_test_1);
```

![Matplotlib visulazation](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402437578/0a0e879a-fea5-432e-8c72-b76606cfa333.png align="center")

Not’a good model, let’s try another one. 🔻

```python
# Set random seed
tf.random.set_seed(42)

# Create a model and this time we are add extra secound layer
model_2 = tf.keras.Sequential([
  tf.keras.layers.Dense(1),
  tf.keras.layers.Dense(1) # add a second layer
])

# Compile the model
model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

# Fiting the model
model_2.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100, verbose=0)

# Make a predictions for model_2

y_test_2 = model_2.predict(X_test)
model_predictions(predictions=y_test_2);
```

![Matplotlib visualization](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402439929/2bfaf29b-3176-4692-bc0c-3458eee132ef.png align="center")

It’s a good model, now try another one. 🔻

```python
# Set random seed
tf.random.set_seed(42)

# Same model_2
model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(1),
  tf.keras.layers.Dense(1)
])

# Compile the model
model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=['mae'])

# Fit the model 
model_3.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=500, verbose=0) # set verbose to 0 for you don't get any output

# Make a predictions for model_3
y_preds_3 = model_3.predict(X_test)
model_predictions(predictions=y_preds_3);
```

![Deep learning model predictions ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402441858/9204b68c-0686-4619-8d06-d4dbc8eb43be.png align="center")

Now tell me which deep learning model is a good performance at this time. See above.

![👆](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402442829/04f1166f-f395-4cef-bcc3-7289377cd6e5.png align="center")

.

I think `model_2` is good for this time. let’s see this visualization below.

![Matplotlib visualization](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402443985/26b10914-453e-4b18-93dc-0a23154c4f1e.png align="center")

When you build your deep learning model, the next step is to save the model. Here is a code example on how you can save your model.

```python
# Save a model 
model_2.save('my_best_model')
```

To learn more about Tensorflow’s savings model, [**read this article**](https://keras.io/api/models/model_saving_apis/).

I will discuss this library in the article below if you are interested to learn more in-depth read this article below. 🔻

* [**NumPy**](https://hiwhy.io/numpy-complete-guide)
    
* [**Pandas**](https://hiwhy.io/pandas-in-python-complete-guide)
    
* [**Matplotlib**](https://hiwhy.io/matplotlib-complete-guide)
    

All the code in this article is here — [**download link**](https://github.com/hi-sushanta/Blog_Post/blob/master/regression_airtcle.ipynb)**.**

<div data-node-type="callout">
<div data-node-type="callout-emoji">❓</div>
<div data-node-type="callout-text">Thanks for reading. I hope you learn something new from this article. If you have any questions or something don’t understand comment now below. I try my best to answer your all question.</div>
</div>