---
title: "Complete Deep Learning Guide"
datePublished: Thu Feb 16 2023 00:48:56 GMT+0000 (Coordinated Universal Time)
cuid: clj80v2z2000j0al24erx7eoz
slug: complete-deep-learning-guide
canonical: https://hiwhy.io/deep-learning-guide/
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1693616479461/20c8b46a-2fbd-496e-b9b0-756687b8ed43.png
tags: python, hashnode, deep-learning

---

Welcome to the end-to-end deep learning guide. In this article, I explain every step you need to train your neural network. After finishing this article you build your first deep learning model from <s>scratch</s> In PyTorch This is my promise to you. Now let’s begin.

Note 🔥— This is not a blog post that you read. It is a note that I took several years ago when I first came to this field. I am sharing it with everyone today in the hope that it will help your journey.

**Who Is This Article 📌**

* People who are curious about how deep learning works and learn how to build models that solve world problems in the real world.
    
* People who are curious about learning the most important skill in 21 st century.
    
* People who don’t fear math notation. Math is the foundation of deep learning knowing helps you talk to other deep learning researchers.
    

**Requirement 📌**

* GPU
    
* Know how to use LLMs model like ( ChatGPT & Bard ), because this type of model is helpful for learning.
    
* Know how to write Python Class, Function *( …/… )*, and some basic of library like ( [*NumPy*](https://hiwhy.io/numpy-complete-guide), [*Matplotlib*](https://hiwhy.io/matplotlib-complete-guide) )
    

**Goal Of This Airtcle🎯**

* Why and where use deep learning.
    
* Write your own code not copy others. Understand every step of what you write and why you write.
    

I know this article covers a lot of material, but I promise it's fun to learn. If you need help understanding something in the article, please feel free to [ask](https://www.linkedin.com/in/sushanta-das-/) me in a comment below. I'm always happy to help.

Keep reading 📌

## What Is Deep Learning

> \*“A field of study that gives computers the ability to learn without being explicitly programmed.” —\****Arthur Lee Samuel***

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693617104888/196d8521-b6b6-4ac1-9986-782e6ac86db6.png align="center")

When you ask someone or search Google🔍 ( what is deep learning ? ) you find so many different types of answers, sometimes easy or sometimes hard to understand. I know this because I faced the same problem some years ago in my journey.

You already know that [‘machine learning’](https://hiwhy.io/complete-machine-learning-guide) is a field we develop intelligent machines that can learn and interact with humans. The same is deep learning — a subfield of machine learning, but deep learning uses ANN ( artificial neural network ) that inspires our human brain. It takes time confusing but trust me, I explain one by one, Just read!

The Goal of ‘machine learning’ and ‘deep learning’ is to build powerful intelligent ‘artificial intelligence ( AI ) ’. At this time many people think, is human replaced by a computer? The short answer is ‘not’ because you don't know how powerful our brain is.

Ignore 😡 — When someone says you ‘AI replace human’ don’t talk this type of people. Because they don’t know anything about it.

Deep learning is a powerful technique to solve real-world tasks in a variety of fields such as computer vision ( image/ video ), natural language processing ( text ), and automatic speech recognition ( audio ) so many things.

## **The Notation of Neural Network Training**

Before I write code let’s introduce some of the notation you must know meaning. I am showing you the most common notation that is used by almost all deep learning researchers in this field. This thing helps you understand the research paper easily.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694045769713/cf07710c-011f-4835-9129-5c28a7d9ced1.png align="left")

$$ŷ = f ( x )$$

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694046342560/8b6ade4a-430e-4a23-8045-f2c492c11c0e.png align="left")

$$ŷ = fΘ ( x )$$

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693700228292/9733a3d2-9dc3-4180-b4d1-a705cda59372.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694046440712/efac1917-4780-43c6-a9ff-0582d97a72d8.png align="left")

$$p( ŷ|y) = ?$$

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694046521416/135a45f0-36ea-4f77-b807-d1fbe9ae2b8c.png align="left")

Now you understand how to define a model ( ***neural network*** ), but we need one more tool to check how badly perform our model prediction truth value ***y***. Remember Our goal is to label *y* and predication ***ŷ*** how much distance\*. Let’s see one more example.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694047330510/c960236b-5861-4072-87d2-a51d8d5783c2.png align="left")

$$loss\ function {\ (\ ŷ\ 🤖model\ predication - y\ label\ (Truth) }\ -🔍\ ℓ (ŷ , y)$$

At this moment you understand how to define a model and write a loss function! It’s time to train the model and minimize loss ( function ).

Note: Sometimes you hear about neural networks as a function. I know it’s confusing the first time, but trust me it’s a complex function you build as a deep learning practitioner. Our goal is to make this complex function that acts like our <s>human brain.</s>

Now come to our topic, suppose you have training set ***N*** example, use optimizer train model and minimize the loss. So we write this full equation below.

$$\min_\theta \sum_{i = 1}^N \ell (f_\theta (x_i), y_i),$$

Meaning this equation ⁉️

Modify the parameters to minimize the error/loss of the neural network’s prediction against the correct predictions over the complete dataset.

```python
# Equation convert to code 🧑🏻‍💻
def F(X, y, f, theta): 
       total_loss = 0 
       for i in range(N): 
             total_loss += loss(f(X[i,:], theta), y[i])
       return total_loss
```

One thing you notice math describe long sentence, in short, that’s why deep learning researcher uses math to describe their paper. Don’t be confuse see this type of long equation paper, ***{ write down each equation small peace by peace and understand meaning }*.**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694048185347/943098fa-7bc1-42d2-9ef7-4899243666d4.png align="left")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694048267779/43fc77cf-e04e-41f4-a35d-990442378461.png align="left")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694048334156/da756059-7cb5-4ad6-af0a-69f2f2e89f96.png align="left")

$$\begin{equation} \Theta_{k+1} = \Theta_k - \eta \cdot \frac{1}{N} \sum_{i=1}^N \nabla_{\Theta_k} f_{\Theta_k}(x_i, y_i). \end{equation}$$

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694048464734/d78ee679-8c7b-43f4-a126-621a3e668cbb.png align="left")

### PyTorch Training Process

In this article, I am using a deep-learning framework called PyTorch. If you are not familiar don’t worry [***PyTorch documentation***](https://pytorch.org/tutorials/beginner/basics/intro.html) helpful resource for you. In this article, I covet most of the important things you need to know about how to use this framework!

***Question***: Why PyTorch, not TensorFlow or others?

***Answer:*** Because it’s easy to use compared to other frameworks, and most of all big companies like ***Apple, Meta ( Facebook ), OpenAI,*** and ***Google*** used this framework for their research. Also, it’s a popular framework for researchers, because it’s easy and user-friendly, if you learn more about which framework is right for you, [*read this awesome guide written by* ***AssemblyAI***](https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2023/).

If you know one framework ( ***TensorFlow, PyTorch*** ), then switching to another framework takes hours.

Note: I am using PyTorch which means is not TensorFlow is bad. Both frameworks are awesome and provide lots of features you need.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693700748148/67e7104e-4940-4d18-8da5-49a737ff3552.png align="center")

Everything in deep learning comes from big data. Without data, you can't build any model. So your first job as a deep learning partitioner is to collect data and prepare it in a format that can be used to train a neural network…

Here is a framework you remember all the time when you build any model deep learning!

* Load data { data can be any format - ***text, images, videos, etc.***} and turn tensor format.
    
* Build your model or use a pre-trained model. You can find pre-trained models in this place { [***PyTorch Hub***](https://pytorch.org/hub/)***,*** [***TensorFlow Hub***](https://www.tensorflow.org/hub)***,*** [***Hugging Face***](https://huggingface.co/models) } or you search your own. The benefit of using a pre-trained model is that this model builds researchers in this field.
    
* ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694049076083/de162c0d-99a1-4f85-868d-7ee4c7012362.png align="left")
    
* Change model Hyperparameter or other ways say that improves model performance. Deep learning is all about experiment…experiment…experiment.., so never skip this step.
    
* Once you build a model so next step is to show the world what you create, which means save the model and send the world 🌎!
    

These are the foundation steps 👣 you must follow as a deep learning practitioner. ✨

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693700811071/64866bb8-aef1-4a1a-ba31-a6c8eb0014d4.png align="center")

I understand that you are interested in writing code and training neural networks. I am too! However, there is some essential theory that you must understand in order to build and train neural networks effectively. I hope you will not find this boring.🥱

## Building A Linear Regression Model

I know you are bored! But trust me it’s a field that’s more theory and math that needs to be understood otherwise you suffer long run!

In this section, I build a [linear regression model](https://hiwhy.io/neural-network-regression-in-tensorflow-guide), My goal is to build a model and introduce which thing what do, this way you understand more clearly.

Note 🔥: When don’t understand something **‘read again’** or it’s like confusing read other people's work, watch other people's videos then come back **{ Never skip… }** It’s not watching a movie it’s a learning time. If you don’t enjoy learning that means you are not like in this field.

Let’s import required library need solve the regression problem!

Note: I am using Google Colab so PyTorch is already installed , when someone uses a local computer that time requires installing this library on your computer! Here is a step you follow to install PyTorch.

```python
# This is optional for anyone use PyTorch 🔥 local computer 
!pip install torch
```

If you learn more installation process read this [***complete guide in PyTorch***](https://pytorch.org/get-started/locally/)***.***

Note: Sometimes explaining every code is not useful, so when something you not understand. Copy code paste Google and learn, this is my trick learning I think this is helpful for you. The best way to learn anything understand other people's code and write lots of code!

```python
import torch 
print('PyTorch version i use -', torch.__version__)
# Output - PyTorch version i use - 2.0.0+cu118
```

## The Training Loop for a Neural Network

In this section, I write a training loop. This is an important concept in PyTorch, because not have any ***built-in module*** so important to understand every line as necessary.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693700930534/2453ce13-b87c-42ec-bcf8-487a105bfd41.png align="center")

Here is a simple code in PyTorch, on how to implement a training loop. I understand first time it’s very confusing, but after writing lots of code you are very familiar with each of the steps.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693700978681/fb079cdd-65c1-40ce-9ad0-1580aef9a574.png align="center")

Now it’s time let’s go deeper, into the training loop and understand more in detail.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693701045242/89e08e80-f6d5-4c1c-908b-9fc716e7f0c3.png align="center")

Now that we have imported the necessary libraries, let's begin writing a training loop. We will assume that we have a loss function, `loss_func`, that takes a prediction (`ˆy`) and a target (`y`), and returns a single score for how well a model (`f(·)`) has performed. We will also need an iterator that loads the training data for us to train on. This iterator, `training_loader`, will give us pairs of inputs and their associated labels for training.

The above image shows the steps involved in a training loop. The yellow "Prep" section shows the object creation that needs to be done before training can start. This includes selecting the device that will do all the computations (typically a GPU), defining the model `f(·)`, and creating an optimizer for the model's parameters `θ`.

The red regions indicate the start/repetition of the loop, which provides new data for us to train on. The blue region computes the prediction `ˆy` and loss `(ˆy, y)` for the model with its current parameters `Θ`. The green section takes the loss and computes the gradients and updates to alter the parameters `Θ`.

Using PyTorch, we can write a minimal amount of code that is enough to train many different kinds of neural networks. The `simple_network` function in the next code block follows all the parts of the Figure 2.2 process. First, we create an optimizer that takes in the model's parameters `Θ` that will be altered. We then move the model to the correct compute device and repeat the optimization process for some number of epochs. Each epoch means we used every data point `xi` once. Each epoch involves putting the model into training mode with `model.train()`. The `training_loader` gives us our data in groups of tuples (`x, y`), which we move to the same computing device.

The inner loop over these tuples cleans up the optimizer state with `zero_grad()` and then passes the inputs to the model to get a prediction `y_hat`. Our `loss_fun` takes in the prediction `y_hat` and the true labels to calculate a loss, describing how badly our network has done. Then we compute the gradients with `loss.backward()` and take a `step( )` with the optimizer.

I hope you understand, if not **read again**. Now it’s time, to write code in pure PyTorch implement training loop.

```python
# This is a training loop code...

def simple_network(model, loss_func, training_loader, epochs=20, device="cpu"):

    # Create an optimizer object.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # Move the model to the specified device.
    model.to(device) # Chose correct compute resources (CPU or GPU)

    # This two for loops iterating through all the data (batches) multiple times (epochs)
    for epoch in tqdm(range(epochs), desc="Epoch"): #1st
    
        model = model.train() # Set the model to training mode.

        running_loss = 0.0 # Initialize the running loss to 0.

        for inputs, labels in tqdm(training_loader, desc="Batch", leave=False): #2nd
            inputs = moveTo(inputs, device)
            labels = moveTo(labels, device)


            optimizer.zero_grad() # Zero the gradients.

            y_hat = model(inputs) # Forward pass.
            loss = loss_func(y_hat, labels)  # Calculate the loss.

            loss.backward() # Backpropagate the loss.

            optimizer.step() # Update the parameters.
            running_loss += loss.item() # Update the running loss.
```

In short summary — ( Training loop ) ✨

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694133067647/23e82f9d-168d-4498-a59b-8ea4e50ab425.png align="center")

The training loop is where the model learns the relationship between the **features and labels** in the training data. The model does this by repeatedly going through the training data, predicting the labels for each data point, and then comparing its predictions to the actual labels. The model then uses the loss function to calculate how far off its predictions were, and uses the optimizer to update its weights in order to make its predictions closer to the actual labels.

## Defining a Toy Dataset for Model Training

> The world is changed but, one thing never changed numbers — Chi

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693701230252/0c21e27f-6ca7-40a4-a6ab-36ae604cdfa7.png align="center")

The training loop is complete, all you need is some data, neural network and loss function work our linear regression model. Enough time waste let’s make some data!

```python
import seaborn as sns
import numpy as np

X = np.linspace(0, 20, num=300) # 1-dimension input

y = X + np.sin(X)*2 + np.random.normal(size=X.shape) # Create an output 
sns.scatterplot(x=X, y=y);
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693701300526/71f10ae5-b8bb-4f7d-b536-83d7db759a11.png align="center")

This is a simple toy problem I create with strong regression trends up and down, but in the real world, you have not seen this type of simple dataset. I use this because it’s an experiment within a short period of time and you understand what happening model. Basic is the same no matter small or large dataset, the only difference is **{ Write more code }**. In latter, I show you how to work on real-world regression datasets that are publicly available.

```python
"""
**Function Overview**
!. This is a simple Dataset class that takes in X, y as input. 
!!. We need to define a getitem method, which will return the data and label it as a tuple(inputs, outputs).
!!!. len function that returns how large the dataset is.
💩
"""
class SimpleRegressionDataset(Dataset):
        
    def __init__(self, X, y):
        super(SimpleRegressionDataset, self).__init__()
        self.X = X.reshape(-1,1)
        self.y = y.reshape(-1,1)
        
    
    def __getitem__(self, index):
        return torch.tensor(self.X[index,:], dtype=torch.float32), torch.tensor(self.y[index], dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]
    
training_loader = DataLoader(SimpleRegressionDataset(X, y), shuffle=True)
```

## Defining The Model

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693701389626/67c4212d-af0d-4657-a2b3-d59b372b8d37.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694134211584/8c854cfb-096b-4e83-9327-7b55df00dc47.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694134296152/af7aa72d-f21e-4f52-a816-97f660ae463b.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693701632155/c9db2a88-4d6b-4ca4-b4b2-ae70c7473a31.png align="center")

## Defining The Loss Function

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694134397702/0ed452cb-b06c-4c85-bdc1-4d3a6a2ef4b1.png align="center")

$$\ |y - ŷ| = 9 \downarrow$$

$$\ |y - ŷ| = 3 \downarrow$$

$$\ |y - ŷ| = 1 \downarrow$$

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694134523957/ace211a5-a94c-4458-b992-4ba8451031a5.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694134661635/aec8829f-79d3-4270-8647-cc1d1018a00b.png align="center")

**Note ⭐**—To learn more about loss function read this article [***officially written by PyTorch***](https://pytorch.org/docs/stable/nn.html#loss-functions)***.***

### **Putting It Together — Training a Linear Regression Model**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694134835325/db2f4ff1-173d-4066-9af5-2b91c711d69e.png align="center")

```python
input_features = 1 
output_features = 1 
model = nn.Linear(input_features, output_features)
loss_func = nn.MSELoss()

device = torch.device("cuda")

simple_network(model, loss_func,training_loader, device=device)
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693702104940/96f29af2-45ce-4f2b-a56b-e9a8f698f6c4.png align="center")

Note 🔥: ~ If you don’t change runtime in **GPU** it shows an error, see below.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693702190861/fe1be92e-ce6f-42b6-aeca-0e3ff5c29c55.png align="center")

Now our model training is complete! It’s time to see our model prediction, in a visual way. At this time I am using the [Python visualization library Matplotlib.](https://hiwhy.io/matplotlib-complete-guide)

```python
  
# Note -  Predication time doesn't need to calculate Gradient!  
# Gradient only needs training time!

with torch.no_grad():
  Y_pred = model(torch.tensor(X.reshape(-1, 1), device=device,
                              dtype=torch.float32)).cpu().numpy()


# Make a visulize !!!
sns.scatterplot(x=X, y=y, color="black", label="Data");
sns.lineplot(x=X, y=Y_pred.ravel(), color='red', label="Model");
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693702339087/9c965feb-9072-4ba4-8473-735899fa5ac2.png align="center")

Note : You can see the model I build it’s a really good prediction, but in the real world you don’t have this type of small dataset and easy problem. So below I show you how to handle large datasets in real-world problems so you can understand problems fast and solve them quickly.

Hint : Above you see I use [`torch.no`](http://torch.no)`_grad( )` predication time, you can also use `torch.inference_mode( )` . `torch.inference_mode( )` context manager is a newer, potentially faster alternative to [`torch.no`](http://torch.no)`_grad( )`. It is recommended to use `torch.inference_mode( )` when you are sure that your code will not need to interact with autograd. If you learn more see this [***Tweet from PyTorch***](https://twitter.com/PyTorch/status/1437838231505096708?s=20).

Here is an example of how you use `inference_mode( )`

```python
with torch.inference_mode():
    y_preds = model(torch.tensor(X.reshape(-1, 1), # <-- Shape of (N, 1)
                              dtype=torch.float32)).cpu().numpy()

sns.scatterplot(x=X, y=y, color='black', label='Data') # <-- Data
sns.lineplot(x=X, y=Y_pred.ravel(), color='red', label='Model') # <-- What our model learned
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693702399424/6cdb6729-3fac-40b2-82e3-27f09be0b10a.png align="center")

### **Build Food Delivery Time Prediction Model**

Check this [**food delivery time predication project**](https://hiwhy.io/food-delivery-time-prediction-using-deep-learning)**.**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693702454576/27eed722-6d1b-42e0-8e57-d16f8c62d8cf.png align="center")

## **How to Build a Neural Network from <s>Scratch</s>**

At this moment you know how to build a **regression model** in PyTorch and what is behind the scenes in the **training loop**, now it’s time to build our fast fully connected neural network. Basic is the same all the time , no matter how complex the model is.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694135558430/e8ee1f80-bfda-4249-84eb-b78efab5c085.png align="center")

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694135754283/4557f0ed-8941-4812-951a-5a625dce005b.png align="center")

Myth — Don’t need a math background to understand deep learning is wrong. Math is the foundation building block in deep learning research. No research comes in any field without using Math because it’s a 'Universal🪄' tool. So when someone says you don’t need math to understand this field, asked this type of person what is meaning 'Neural Network'?

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693702980249/8ef6e729-6fd1-452e-be5c-ddc9edd6b68a.png align="center")

Learning framework like ( PyTorch, TensorFlow, MxNeT, Jax etc.) or learning Neural Network is two different things. So focus on what is the true meaning of neural network 🔎 and solve our real-world problem. Don't focus on trending things 👻.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693703025253/e8d274b0-dab5-4024-a504-2707b23bd368.png align="center")

### A Deep Dive into the Notation of Fully Connected Neural Networks

Now you solid understand how to build **linear regression** model in PyTorch if not read again and download some data **Kaggle and build a 2-3 regression model** , this is my personal advice for learning anything new. Everything first time hard but after is easy!

> *The difference between a master and a beginner is that the master has failed more times than the beginner has even tried  — Stephen McCranie*

In this section, I build a **feed-forward fully connected** neural network!

$$\begin{equation} y = f(x) = \sigma \left( \sum_{i=1}^n w_i x_i + b \right) \end{equation}$$

<aside>
🤖 Above Notation — Meaning
<ul>
<li><em><strong>y</strong></em> is the output of the neural network</li>
<li><em><strong>f</strong></em>  is the activation function</li>
<li><em><strong>x</strong></em> is the input to the neural network</li>
<li><em><strong>w_i</strong></em> is the weight of the <em><strong>i</strong></em> th connection</li>
<li><em><strong>b</strong></em> is the bias term</li>
<li><em><strong>σ</strong></em> is the sigmoid function</li>
</ul>
</aside>

Note : The name feed-forward because every output from one layer is connected to the other layer which means each layer has one input and one output and progresses sequentially. Fully Connected because each layer input has connected the previous layer. See the below image so you understand better.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693703177133/d9732a02-7925-4366-a9db-cf989d6b264c.png align="center")

**Q.** What is a Hidden layer?

Let's begin with a discussion of hidden layers. The input layer is where the input data is received, and the output layer is where the predictions are made. In a linear regression model, there is only one input layer and one output layer. Hidden layers are any layers that are sandwiched between the input and output layers.

$$X \to \hat{Y}$$

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693703238688/3e98782a-5cf7-4d0c-b146-c9c8a452d3c6.png align="center")

To add a hidden layer to a model, we simply add another matrix between the input and output matrices. This matrix represents the weights of the connections between the neurons in the hidden layer and the neurons in the output layer.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1694136230685/32b2d6f7-87f6-433e-8c21-eb391350046c.png align="center")

### Progress

$$f ( x ) = \mathbf{ x \ }^T W^d,^ C\$$

* `x` is a vector of length `d`
    
* `w` is a matrix of size `d x C`
    
* `C` is the number of output classes
    
* `f(x)` is a vector of length `C`
    

When adding a second layer with a new matrix this notation is like this!👇🏻

$$f(\boldsymbol{x}) = \boldsymbol{x}^\top \boldsymbol{W}^{d \times n}{(h_1)} \boldsymbol{W}^{n \times C}{(\text{out})}$$

### **Simple Definition of Neural Networks**

There are so many definitions have **‘Internet’** neural networks but some of them are useful. Neural networks are a collection of neurons that are connected by layer. Each neuron is performed some ‘calculation’ to solve problem. You can think each neuron is an **‘Employee’** working for a company. And each neuron is an organized layer. There are 3 types of layers input, hidden, and output. Neural networks inspire how the human brain works but how the human brain work is a long-term process of discovery {**running**}.

$$x = \sum{(weights * inputs) + bias}$$

### **Components Of A Neural Network**

There are 3 components of a neural network **Activation function, Weights, Bias**.

$$nn = (i)\ A \ (ii) \ W \ (ii) \ B$$

***Activation Function —*** The activation function determines whether a neuron should be activated or not. It’s an important role play in our neural network it adds **non-linearity** to the model. If neurons are activated that means input is important. There are lots of activation functions, it’s currently active research in the deep learning field. But only some of them are popular and need to know.

* **Sigmoid** is used to predict the probability of an output node between 0 and 1.
    

$$f(x) = {\large \frac{1}{1+e^{-x}}}$$

* Tanh used to predict output node is between 1 and -1.
    

$$f(x) = {\large \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}}$$

* ReLU set our output node 0 if function result is negative
    

$$f(x)= {\small \begin{cases} 0, & \text{if } x < 0\\ x, & \text{if } x\geq 0\\ \end{cases}}$$

* Weights ( W ) and Bias ( b )
    

$$Wij = weight(i, j)$$

Note :~ Wij denotes the weight connecting the i th neuron in the preceding layer to the j th neuron in the current layer. weight is a function that takes two inputs, i and j, and returns the weight between the two neurons.

$$Weight: W or\ W_{ij} \quad Bias: b\ or\ b_j$$

$$bj = \text{bias term associated with the jth neuron in a layer}$$

### Building A Fully Connected Network In PyTorch

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693703783025/5d8ba869-0a00-4c65-b201-303a7a17ff53.png align="center")

```python
model = nn.Sequential(
    
    nn.Linear(1, 10), # hidden layer ? 1 input unit and 10 output
    nn.Linear(10, 1), # output layer ? 10 input units and 1 output
)

simple_network(model, loss_func, training_loader)
```

I use \*[\***nn. Sequential**](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) method defines the model, it’s a quick method to define any model in PyTorch.

**Note** — The model is a sequential model, which means that the layers are arranged in a sequence, and the output of one layer is passed as input to the next layer.

You can also use PyTorch [***nn. Module***](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) style creates a model. I show you two way define model because the real world is not a straight line so what tool where need you don’t know 🔨🔪🪛

Here's an example 👇🏻

```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden_layer = nn.Linear(1, 10)
        self.output_layer = nn.Linear(10, 1)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x

model = MyModel()
```

In this example, I've created a custom class called `MyModel` that inherits from `nn.Module`. Inside the `**__init__**` method, I've defined the hidden layer and output layer as instance variables. The `forward` method takes an input tensor `**x**` and passes it through the hidden and output layers sequentially. Finally, I instantiated the model by creating an object of the `MyModel`class.

In this example, I've created a custom class called `MyModel` that inherits from `nn.Module`. Inside the `**__init__**` method, I've defined the hidden layer and output layer as instance variables. The `forward` method takes an input tensor `**x**` and passes it through the hidden and output layers sequentially. Finally, I instantiated the model by creating an object of the `MyModel`class.

Don't be confused. If you know [Python classes,](https://docs.python.org/3/tutorial/classes.html) this will be easy to understand. I use this method most of the time to define models because it's easy and user-friendly once you get the hang of it.

Remember \*\*\*\*— This model has one input **( X )** and one output **( Y )**, and a single hidden layer with 10 neurons.

Check Model Predication📊 🤖

```python
with torch.no_grad():
  Y_pred = model(torch.tensor(X.reshape(-1, 1), # <-- Shape of (N, 1)
                              dtype=torch.float32)).cpu().numpy()

sns.scatterplot(x=X, y=y, color='black', label='Data') # <-- Data
sns.lineplot(x=X, y=Y_pred.ravel(), color='red', label='Model') # <-- What our model learned
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693704039187/be97ed4c-5a66-4938-bc94-7416b08c2b58.png align="center")

Note ✨ — Sometime i don’t explain what is meaning this line of code , becuse explain line by line of code is takes lot’s of time and sometime it’s boring. So whenever you don’t understand some line of code , use tool Large Language Models (LLMs) like { **ChatGPT , Bard** }.

Here is the step you follow understand any code —

* Copy the code paste — ChatGPT and asked what is meaning this line of code.
    

Use AI tool correct way, and improve your learning progress.

### Adding Nonlinearities

In real-world datasets, not one **straight line** improves our model performance, we need some **non linearities** between every step of training time so our model makes a more **complex function**. At this time we need activation functions like **( Sigmoid, tanh, ReLU, etc..)**

We can think activation function is a teacher guiding our model to improve this way prediction performance 🧪.

Now it’s time to add some Non-linearities in the same model we write before.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693704090982/f03b9b95-172c-452c-bb5b-709e0e5e5a88.png align="center")

```python
model = nn.Sequential(
    nn.Linear(1,  10),#hidden layer
    nn.Tanh(),#activation
    nn.Linear(10, 1),#output layer
)

simple_network(model, loss_func, training_loader, epochs=200)
```

See prediction our model! 🤖

```python
with torch.no_grad():
    Y_pred = model(torch.tensor(X.reshape(-1,1), dtype=torch.float32)).cpu().numpy()
    
sns.scatterplot(x=X, y=y, color='black', label='Data') #The data
sns.lineplot(x=X, y=Y_pred.ravel(), color='red', label='Model') #What our model learned
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693704168678/bac0ddf6-6134-403c-b5b5-4c0e9083fbbb.png align="center")

Now you know basic how to write linear layers and $add^+$ Non-linearities in your model. Now it’s time to learn classification problems in deep learning. Once you understand Regression or Classification you understand every type of problem. But need a little bit of practice and are curious about learning new things!

> Learning is the only thing the mind never exhausts, never fears, and never regrets.
> 
> — Leonardo da Vinci!👋🏻

Note 🔥— In this section, I start with a simple dataset and explain one by one. After the basic is finished then, I download the real-world datasets and make a model. Because in the real world, you don’t work with simple datasets.

## Solving Classification Problems with Deep Learning

Before we get into writing code, let's look at the general architecture of a classification neural network.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693704241174/0dbfffad-faae-47b9-92e8-3d6e97fd4d15.png align="center")

> Recall what is classfication?

### What is a classficiation problem?

A **classfication problem** involves prediction whether something is one thing or another.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693704377688/d6af6fdb-d763-4b4f-84f8-3e7d1cec4d8e.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693704430761/44289cb0-a53a-4325-aad8-f91f1e5156c8.png align="center")

It’s time to generate toy dataset for Classfication problem, so this time i am using Scikit-learn library for python . If you learn more this dataset i am using [read here !](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)

```python

# Generate some toy dataset 
# learn more ->  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html

from sklearn.datasets import make_moons 
X, y = make_moons(n_samples=200, noise=0.05)
sns.scatterplot(x=X[:, 0], y=X[:,1], hue=y, style=y)
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693704601056/f60c7261-61d0-4a15-8098-0e60f6565d9e.png align="center")

```python

classfication_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X, dtype=torch.float32), 
    torch.tensor(y, dtype=torch.long))

training_loader = DataLoader(classfication_dataset)
```

This small line of code converts our classfication\_dataset PyTorch DataLoader object. This is very helpful because when training time large amounts of data need batches, this time PyTorch DataLoader is a very useful class . If you learn more about this class [**read here DataLoader**](https://pytorch.org/docs/stable/data.html) **!**

The dataset is ready, and I convert *the* ***classfication\_dataset*** PyTorch data loader object. So my next step is to define our linear classification model.

Note 🔥— We have two features and two output $(2x \* 2y)$

```python
input_features = 2 
output_features = 2 
model = nn.Linear(input_features, output_features)
```

### Training A Classification Network

Now train the model and see our model performance 🤖

```python
loss_func = nn.CrossEntropyLoss()
simple_network(model, loss_func, training_loader, epochs=50)

# See our model performacne in visual way!
def visualize2DSoftmax(X, y, model, title=None):
    x_min = np.min(X[:,0])-0.5
    x_max = np.max(X[:,0])+0.5
    y_min = np.min(X[:,1])-0.5
    y_max = np.max(X[:,1])+0.5
    xv, yv = np.meshgrid(np.linspace(x_min, x_max, num=20), np.linspace(y_min, y_max, num=20), indexing='ij')
    xy_v = np.hstack((xv.reshape(-1,1), yv.reshape(-1,1)))
    with torch.no_grad():
        logits = model(torch.tensor(xy_v, dtype=torch.float32))
        y_hat = F.softmax(logits, dim=1).numpy()

    cs = plt.contourf(xv, yv, y_hat[:,0].reshape(20,20), levels=np.linspace(0,1,num=20), cmap=plt.cm.RdYlBu)
    ax = plt.gca()
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, style=y, ax=ax)
    if title is not None:
        ax.set_title(title)

visualize2DSoftmax(X, y, model)
```

Note 🔥 — Now our model training is complete, but we need some visual way to see the model performance. I use a contour plot to show the decision texture of our algorithm “Blue represent first class, red mean second class”

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693704765893/e964a856-fa6b-4c0a-a784-d022ccbfa289.png align="center")

We can now see the model I build not solved the problem. Some blue dots have a red place and red dots have a blue place — 🚩

We need to improve our model performance, so in this case, I add two hidden layers and n = 30 hidden units both hidden layers.

```python
model = nn.Sequential(
    # 🤖
    nn.Linear(2, 30),
    nn.Tanh(),
    nn.Linear(30, 30),
    nn.Tanh(),
    nn.Linear(30, 2),
)

simple_network(model, loss_func, training_loader, epochs=300)

visualize2DSoftmax(X, y, model)
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693704881314/b6cce69b-ce39-4afb-ba26-b5e9cc57aa14.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693704909729/98329dce-775f-43a1-91ef-5cf5da7d9e52.png align="center")

One thing to notice this time takes time to model train ( in my case 40 seconds 300 epochs ), you can see our model this time is more confident than the previous one I build.

Let’s see side by side the previous model and the current model I built and see the difference!

**Article Progress**