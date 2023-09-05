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

**Keep reading 📌**

## What Is Deep Learning

> *A field of study that gives computers the ability to learn without being explicitly programmed. -* ***Arthur Lee Samuel***

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693617104888/196d8521-b6b6-4ac1-9986-782e6ac86db6.png align="center")

When you ask someone or search Google🔍 ( what is deep learning ? ) you find so many different types of answers, sometimes easy or sometimes hard to understand. I know this because I faced the same problem some years ago in my journey.

You already know that [‘machine learning’](https://hiwhy.io/complete-machine-learning-guide) is a field we develop intelligent machines that can learn and interact with humans. The same is deep learning — a subfield of machine learning, but deep learning uses ANN ( **artificial neural network** ) that inspires our human brain. It takes time confusing but trust me, I explain one by one, Just read!

The Goal of ‘machine learning’ and ‘deep learning’ is to build powerful intelligent artificial intelligence. At this time many people think, is human replaced by a computer? The short answer is ‘not’ because you don't know how powerful our brain is...

**Ignore 😡 —** When someone says you **‘AI replace human’** don’t talk this type of person. Because they don’t know anything about it.

Deep learning is a powerful technique to solve real-world tasks in a variety of fields such as computer vision ( image/ video ), natural language processing ( text ), and automatic speech recognition ( audio ) so many things.

## **The Notation of Neural Network Training**

Before I write code let’s introduce some of the notation you must know meaning. I am showing you the most common notation that is used by almost all deep learning researchers in this field. This thing helps you understand the research paper easily.

I use **x** to denote input features *and* ***f( )*** *to denote a neural network model. The label associated with* ***x*** *is denoted* ***y****. The model you build in x and produce predication ŷ. full write become a*

$$ŷ = f ( x )$$

I use **Θ** (Greek letter theta ) to denote the parameter. In neural networks lots of parameters **Θ** we need to adjust this parameter and try to reduce the loss function.

$$ŷ = fΘ ( x )$$

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693700228292/9733a3d2-9dc3-4180-b4d1-a705cda59372.png align="center")

**Note** — Loss function used in deep learning to measure the difference between the predicted **ŷ** and actual value **y**.

$$p( ŷ|y) = ?$$

Suppose predicated **ŷ** value is **7** and the actual value **y** is **10.** At this time our loss function tells the gap between the actual value **y 10 - ŷ 7 = ℓ 3**. The loss function is a very useful thing in deep learning, it guides you on how to imporve model accuracy.

Now you understand how to define a model ( ***neural network*** ), but we need one more tool to check how badly perform our model prediction truth value ***y*.** Remember Our goal is to label ***y*** and predication ***ŷ*** how much distance. Let’s see one more example 👇🏻

**Example 🔥 :** Actual truth **y is ( 0.00039 )** and our model **ŷ** predication is **( 0.00021 )** which means **ℓ(ŷ, y)** *—* **$0.00018**‬ distance. What do you think it’s a good model?

**Note:** In deep learning, loss function is written **ℓ (ŷ, y)***.*

$$loss\ function {\ (\ ŷ\ 🤖model\ predication - y\ label\ (Truth) }\ -🔍\ ℓ (ŷ , y)$$

At this moment you understand how to define a model and write a loss function! It’s time to train the model and minimize loss ( function ).

Note :~ Sometime you hear about neural networks as a function. I know it’s confusing the first time, but trust me it’s a complex function you build as a deep learning practitioner. Our goal is to make this complex function that acts like our <s>human brain. ?</s>

Now come to our topic, suppose you have training set $***N$*** example, use optimizer train model and minimize the loss . So we write this full equation below.

$$\min_\theta \sum_{i = 1}^N \ell (f_\theta (x_i), y_i),$$

Meaning this equation ⁉️

Modify the parameters to minimize the error/loss of the neural network’s prediction against the correct predictions over the complete dataset.

Equation convert to code 🧑🏻‍💻

```python
def F(X, y, f, theta): 
       total_loss = 0 
       for i in range(N): 
             total_loss += loss(f(X[i,:], theta), y[i])
       return total_loss
```

One thing you notice math describe long sentence, in short, that’s why deep learning researcher use math to describe their paper ***{ after all math is the universal language communicate },*** Don’t be confuse see this type of long equation paper, ***{ write down each equation small peace by peace and understand meaning }***.

Let’s write this equation piece by piece — $\\sum\_{i = 1}^N $ The summation goes over all $***N***$ pairs of input $x\_i$ and output $\\ y\_i $ and determines how badly the $(\\ell (. , .))$ model doing. This equation you see is not minimized loss $\\ ℓ (ŷ , y)$ only computes how far true label $y$ and predication $ŷ$ is.

So next question in your mind? when this equation $(\\ell (. , .))$ don’t minimize loss , so how we adjust $\\theta$ ( paramater ) minimize loss function $(\\ell (. , .))$. Minimize the loss important step building model so we use $Gradient\\ descent$ as a tool to minimize the $ℓ (ŷ , y)$. The best thing about PyTorch its provide [**automatic differentiation**](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) means you don’t need to worry about tracking everything $\\theta$ because it’s an automatic track for us, that’s why people use this awesome framework, it provides lots of features and flexibility for deep learning researcher.

Suppose our model $current \\ state$ is $\\Theta\_k$ we want to improve and find next $state$ $\\Theta\_{k+1}$, which our hope is to reduce our model loss $(\\ell (. , .))$? So this time equation we want to solve is

$$\begin{equation} \Theta_{k+1} = \Theta_k - \eta \cdot \frac{1}{N} \sum_{i=1}^N \nabla_{\Theta_k} f_{\Theta_k}(x_i, y_i). \end{equation}$$

Here is a breakdown of the equation 😓

* $Θk$ is the model parameter at iteration $k$.
    
* $*η*$ is the learning rate.
    
* $*N*$ is the number of training examples.
    
* $x\_i$ is the **$i$** training example.
    
* $y\_i$ is the label for the $i$ training example.
    
* $*fΘk(x\_i,y\_i)*$ is the loss function for the $i$ training example, evaluated at the model parameter $Θk$.
    
* $\\nabla\_{\\Theta\_k} f\_{\\Theta\_k}(x\_i, y\_i)$ is the gradient of the loss function $*f\_{\\Theta\_k}(x\_i, y\_i)$* with respect to the model parameter $Θ\_k$.
    

The equation says that the model parameter at iteration $*k+1*$ is updated by taking the current model parameter $Θ\_k$, subtracting the learning rate $*η*$ times the average of the gradients of the loss function with respect to the model parameter, evaluated at each training example.

### PyTorch Training Process

In this article, I am using a deep-learning framework called PyTorch. If you are not familiar don’t worry [***PyTorch documentation***](https://pytorch.org/tutorials/beginner/basics/intro.html) helpful resource for you. In this article, I covet most of the important things you need to know about how to use this framework!

***Question*** 🔥:~ Why PyTorch, not TensorFlow or others?

***Answer -*** Because it’s easy to use compared to other frameworks, and most of all big companies like ***Apple, Meta ( Facebook ), OpenAI,*** ***Google*** etc used this framework for their research.

Also, it’s a popular framework for researchers, because it’s easy and user-friendly, if you learn more about which framework is right for you, [*read this awesome guide written by* ***AssemblyAI***](https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2023/).

If you know one framework ( ***TensorFlow, PyTorch*** ), then switching to another framework takes hours .

Note 🔥:~ I am using PyTorch which mean is not TensorFlow is bad. Both framework it’s awesome and provide lots of features you need.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693700748148/67e7104e-4940-4d18-8da5-49a737ff3552.png align="center")

Everything in deep learning comes from big data. Without data, you can't build any model. So your first job as a deep learning partitioner is to collect data and prepare it in a format that can be used to train a neural network…

Here is a framework you remember all the time when you build any model deep learning!

* Load data { data can be any format - ***text, images, videos, etc..*** } and turn tensor format.
    
* Build your own model or use a pre-trained model. You can find pre-trained models in this place { [***PyTorch Hub***](https://pytorch.org/hub/)***,*** [***TensorFlow Hub***](https://www.tensorflow.org/hub)***,*** [***Hugging Face***](https://huggingface.co/models) } or you search your own. The benefit of using a pre-trained model is, this model builds researchers in this field.
    
* Once a model is ready now it’s time to check accuracy $ℓ (ŷ , y) = ?$ of the model, which means evaluating model performance. Suppose your truth is $y = 27.36$ and model prediction is $\\hat{y} = 15.27.$ You can see the model performance is not good, so this time our extra work comes, and that is!
    
* Change model Hyperparameter or other ways say that improves model performance. Deep learning is all about experiment…experiment…experiment.., so never skip this step.
    
* Once you build a model so next step is to show the world what you create, which means save the model and send the world 🌎!
    

These are the foundation step 👣 you must follow as a deep learning practitioner. ✨

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693700811071/64866bb8-aef1-4a1a-ba31-a6c8eb0014d4.png align="center")

I understand that you are interested in writing code and training neural networks. I am too! However, there is some essential theory that you must understand in order to build and train neural networks effectively. I hope you will not find this boring.🥱

## Building A Linear Regression Model

I know you are bored! but trust me it’s a field it’s more theory and math that need to understand otherwise you suffer long run!

In this section, I build a [linear regression model](https://hiwhy.io/neural-network-regression-in-tensorflow-deep-learning/), my goal is not to build a model, but also to introduce which thing what do, this way you understand more clearly.

Note 🔥: When don’t understand something **‘read again’** or it’s like confusing read other people's work, watch other people's videos then come back **{ Never skip… }** It’s not watching a movie it’s a learning time. If you don’t enjoy learning that means you are not like in this field.

Let’s import require library need solve the regression problem!

Note 🔥: I am using Google Colab so PyTorch is already installed , when someone uses a local computer that time requires installing this library on your computer! Here is a step you follow to install PyTorch.

```python
# This is optional for anyone use PyTorch 🔥 local computer 
!pip install torch
```

If you learn more installation process read this [***complete guide in PyTorch***](https://pytorch.org/get-started/locally/)***.***

Note 🔥 : Sometimes explaining every code is not useful, so when something you do not understand. Copy code paste Google and learn, this is my trick learning I think this is helpful for you. The best way to learn anything understand other people's code and write lots of code!

```python
import torch 
print('PyTorch version i use -', torch.__version__)
# Output - PyTorch version i use - 2.0.0+cu118
```

## The Training Loop for a Neural Network

In this section, I write a training loop. This is an important concept in PyTorch🔥 because not have any ***built-in module*** so important to understand every line as necessary!

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693700930534/2453ce13-b87c-42ec-bcf8-487a105bfd41.png align="center")

Here is a simple code in PyTorch, on how to implement a training loop➰. I understand first time it’s very confusing, but after writing lots of code you are very familiar with each of the steps.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693700978681/fb079cdd-65c1-40ce-9ad0-1580aef9a574.png align="center")

Now it’s time let’s go deeper, into the training loop and understand more in detail.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693701045242/89e08e80-f6d5-4c1c-908b-9fc716e7f0c3.png align="center")

Now that we have imported the necessary libraries, let's begin writing a training loop. We will assume that we have a loss function, `loss_func`, that takes a prediction (`ˆy`) and a target (`y`), and returns a single score for how well a model (`f(·)`) has performed. We will also need an iterator that loads the training data for us to train on. This iterator, `training_loader`, will give us pairs of inputs and their associated labels for training.

The above image shows the steps involved in a training loop. The yellow "Prep" section shows the object creation that needs to be done before training can start. This includes selecting the device that will do all the computations (typically a GPU), defining the model `f(·)`, and creating an optimizer for the model's parameters `θ`.

The red regions indicate the start/repetition of the loop, which provides new data for us to train on. The blue region computes the prediction `ˆy` and loss `(ˆy, y)` for the model with its current parameters `Θ`. The green section takes the loss and computes the gradients and updates to alter the parameters `Θ`.

Using PyTorch, we can write a minimal amount of code that is enough to train many different kinds of neural networks. The `simple_network` function in the next code block follows all the parts of the Figure 2.2 process. First, we create an optimizer that takes in the model's parameters `Θ` that will be altered. We then move the model to the correct compute device and repeat the optimization process for some number of epochs. Each epoch means we used every data point `xi` once. Each epoch involves putting the model into training mode with `model.train()`. The `training_loader` gives us our data in groups of tuples (`x, y`), which we move to the same computing device.

The inner loop over these tuples cleans up the optimizer state with `zero_grad()` and then passes the inputs to the model to get a prediction `y_hat`. Our `loss_fun` takes in the prediction `y_hat` and the true labels to calculate a loss, describing how badly our network has done. Then we compute the gradients with `loss.backward()` and take a `step( )` with the optimizer.

I hope you understand, if not **read again** . Now it’s time, to write code in pure PyTorch implement training loop. 🧑🏻‍💻

> The only thing that stands between you and your dream is the will to try and the belief that it is actually possible. - Joel Brown

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

We have a loss function *loss\_func $(\\ell (. , .))$* \*\*\*\*that takes two thing *predication* $( ŷ )$ and a target $(y)$and return single score how well our model is doing $(f( \\ .\\ ))$. Before training starts let’s pick a device that computes, normally in deep learning using $‘GPU’$.

The training loop is where the model learns the relationship between the **features and labels** in the training data. The model does this by repeatedly going through the training data, predicting the labels for each data point, and then comparing its predictions to the actual labels. The model then uses the loss function to calculate how far off its predictions were, and uses the optimizer to update its weights in order to make its predictions closer to the actual labels.

## Defining a Toy Dataset for Model Training

> *The world is changed but, one thing never changed numbers — Chi*

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693701230252/0c21e27f-6ca7-40a4-a6ab-36ae604cdfa7.png align="center")

The training loop is complete, all you need is some *data*, *neural network* and *loss function* $$(\\ell (. , .))$ work our linear regression model. Enough time waste let’s make some data!

```python
import seaborn as sns
import numpy as np

X = np.linspace(0, 20, num=300) # 1-dimension input

y = X + np.sin(X)*2 + np.random.normal(size=X.shape) # Create an output 
sns.scatterplot(x=X, y=y);
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693701300526/71f10ae5-b8bb-4f7d-b536-83d7db759a11.png align="center")

This is a simple toy problem I create with strong regression trends up and down, but in the real world, you have not seen this type of simple dataset. I use this because it’s an experiment within a short period of time and you understand what happening model. Basic is the same no matter small or large dataset, the only difference is { Write more code }. In latter, I show you how to work on real-world regression datasets that are publicly available.

> *The more you know, the more you can do. - Helen Keller*

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

At this point, we have successfully understood how to create a training loop and load dataset using the PyTorch dataset object. last thing we need implementing linear regression network $ f\_\\theta(\\ . \\ ) $In this case we write simple linear function weight matrix$\\ W $take inputs vector product$ \\ x \\ $so in this case model $ f\_\\ (\\ . \\ )$ that look like this 👇🏻

$$f ( x ) = \mathbf{ x \ }^T \ \ \ W^d,^ C$$

x is a vector of all our $d$ features ( in this case, $d = 1$ ) and the matrix $W$ has a row for every feature and a column for every output. I use $W^d,^ C$ to be additionally detailed in that it is a tensor with $\\ d \\ rows$ and $\\ C \\ columns$, this is a common $\\ notation \\ ^\*$ use deep learning paper. In this case, we are predicting a single value, this means $C = 1$

Note : ~ One thing you notice this linear function is not complete. If $\\ x = 0 $ , then $\\ f(x)=0$. One more thing add this function $bias \\ term \\ b$ that has no interaction with *$x$ .* Luckily for us, PyTorch has module $\\ nn. Linear (d,\\ C)$ This module creates linear layer $\\ d \\ is \\ input \\ C \\ output$ exactly what I want.

$$f ( x ) = \mathbf{ x \ }^T \ \ \ W^d,^ C +\ b$$

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693701632155/c9db2a88-4d6b-4ca4-b4b2-ae70c7473a31.png align="center")

## Defining The Loss Function

$nn.Linear( \\ )$ gives us our model $f( \\ x \\ )$ but we still need $\\ loss \\ function\\ \\ell (. , .)$ to measure how well our model performs.

Note : ~ $Loss\\ function/ Cost\\ Function$ or sometimes called $error\\ function$ all have the same meaning, or sometimes you hear the word PyTorch library $Criterion$**.**

So the question is how we measure $\\ loss\\ function \\ ?$ In PyTorch, it’s made easy for us because so many great people write for us awsome $loss\\ function$ our job is just to use this tool and improve our $\\ model\\ accuracy \\ f(x)$. In this case, our label is $\\ y$ and our prediction is $\\ ŷ = f ( x )$ so our job is as deep learning practitioners to find the absolute difference between.

$$\ ℓ (ŷ , y)\ = \ |y - ŷ| = ?$$

Note ⭐ — One of the key points is to remember always try to decrease loss $\\ ℓ (ŷ , y)\\ = \\ \\downarrow\\$ not increase $ℓ (ŷ , y)\\ =\\ \\uparrow \\$ that means see below example.

Wright way decrease loss $\\ ℓ (ŷ , y)\\ = \\ \\downarrow\\$ in model $\\downarrow$

$$\ |y - ŷ| = 21 \downarrow$$

$$\ |y - ŷ| = 9 \downarrow$$

$$\ |y - ŷ| = 3 \downarrow$$

$$\ |y - ŷ| = 1 \downarrow$$

$$[]$$

See below one more example,

This is not good for our model because it increases $\\ ℓ (ŷ , y)\\ =\\ \\uparrow $ our loss after some epoch. $\\uparrow$

$$\ |y - ŷ| = 21 \downarrow$$

$$\ |y - ŷ| = 9 \downarrow$$

$$\ |y - ŷ| = 13 \uparrow$$

$$\ |y - ŷ| = 27 \uparrow$$

$$[]$$

Less is better — for our model $f( \\ x \\ )$, Here is the most commonly used loss function in regression problems!

* $|y - ŷ|$ — ***torch.nn.L1Loss —*** *( mean absolute error (MAE) )*
    
* $(y - ŷ)^2$ — ***torch.nn.MSELoss — (*** *mean squared error* ***)***
    

$Note ⭐—$To learn more about loss function read this article [***officially written by PyTorch***](https://pytorch.org/docs/stable/nn.html#loss-functions)***.***

## **Putting It Together — Training a Linear Regression Model**

At this moment all thing we have created a linear regression dataset ( *SimpleRegressionDataset* ), training loop ( $simplenetwork (\\ )$ , loss function $ℓ$ ( $lossfunc$ ) and $\\ nn.Linear$ model. Now it's time to quickly train our model!🤖

```python
input_features = 1 
output_features = 1 
model = nn.Linear(input_features, output_features)
loss_func = nn.MSELoss()

device = torch.device("cuda")

simple_network(model, loss_func,training_loader, device=device)
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693702104940/96f29af2-45ce-4f2b-a56b-e9a8f698f6c4.png align="center")

Note 🔥: ~ If you don’t change runtime in $\\ GPU\\ $it shows an error, see below.

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

Note :~ You can see the model I build it’s a really good prediction, but in the real world you don’t have this type of small dataset and easy problem. So below I show you how to handle large datasets in real-world problems so you can understand problems fast and solve them quickly!⚡

Hint ✨:~ Above you see I use [`torch.no`](http://torch.no)`_grad( )` predication time, you can also use `torch.inference_mode( )` . `torch.inference_mode( )` context manager is a newer, potentially faster alternative to [`torch.no`](http://torch.no)`_grad( )`. It is recommended to use `torch.inference_mode( )` when you are sure that your code will not need to interact with autograd. If you learn more see this [***Tweet from PyTorch***](https://twitter.com/PyTorch/status/1437838231505096708?s=20).

Here is an example of how you use `inference_mode( )`

```python
with torch.inference_mode():
    y_preds = model(torch.tensor(X.reshape(-1, 1), # <-- Shape of (N, 1)
                              dtype=torch.float32)).cpu().numpy()

sns.scatterplot(x=X, y=y, color='black', label='Data') # <-- Data
sns.lineplot(x=X, y=Y_pred.ravel(), color='red', label='Model') # <-- What our model learned
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693702399424/6cdb6729-3fac-40b2-82e3-27f09be0b10a.png align="center")

### Build Food Delivery Time Prediction Model

Check this [**food delivery time predication project**](https://hiwhy.io/food-delivery-time-prediction-using-deep-learning)

**Exercise :** Download dataset and make some model share others.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693702454576/27eed722-6d1b-42e0-8e57-d16f8c62d8cf.png align="center")

## **How to Build a Neural Network from <s>Scratch</s>**

At this moment you know how to build a **regression model** \*\*in PyTorch and what is behind the scenes in the **training loop**, now it’s time to build our fast fully connected neural network. Basic is the same all the time , no matter how complex the model is!

Note💥 — When I say neural network or deep learning is mean $\\ layers\\ (... )$, Layers are building blocks we define our model $f( \\ x \\ )$. A modern framework like ***( PyTorch )*** is already implemented different layers for different purposes, which means you don’t need to worry about making layers, use your creativity and design your model . In PyTorch, most of the class layers have $( \\ torch. nn\\ )$ module **!.** When you use any layer in PyTorch , use this command to begin your notebook!

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

$$Myth \ 🔥 \downarrow 😭$$

$$\frac{\partial J}{\partial \mathbf{W}^{(1)}}= \text{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)} \$$

Myth 🔥 — Don’t need a math background to understand deep learning is wrong. Math is the foundation building block in deep learning research. No research comes in any field without using Math because it’s a 'Universal🪄' tool. So when someone says you don’t need math to understand this field, asked this type of person what is meaning 'Neural Network'?

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693702980249/8ef6e729-6fd1-452e-be5c-ddc9edd6b68a.png align="center")

Learning framework like ( ***PyTorch, TensorFlow, MxNeT, Jax etc***.) or learning ***Neural Network*** is two different things. So focus on what is the true meaning of neural network 🔎 and solve our real-world problem. Don't focus on trending things 👻.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693703025253/e8d274b0-dab5-4024-a504-2707b23bd368.png align="center")

## A Deep Dive into the Notation of Fully Connected Neural Networks

Now you solid understand how to build **linear regression** model in PyTorch if not read again and download some data **Kaggle and build a 2-3 regression model** , this is my personal advice for learning anything new. Everything first time hard but after is easy!😰

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

Note :~ The name [feed-forward](https://brilliant.org/wiki/feedforward-neural-networks/) because every output from one layer is connected to the other layer which means each layer has one input and one output and progresses sequentially. **Fully Connected** because each layer input has connected the previous layer. See the below image so you understand better.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693703177133/d9732a02-7925-4366-a9db-cf989d6b264c.png align="center")

**Q.** What is a Hidden layer?

Let's begin with a discussion of hidden layers. The input layer is where the input data is received, and the output layer is where the predictions are made. In a linear regression model, there is only one input layer and one output layer. Hidden layers are any layers that are sandwiched between the input and output layers.

$$X \to \hat{Y}$$

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693703238688/3e98782a-5cf7-4d0c-b146-c9c8a452d3c6.png align="center")

To add a hidden layer to a model, we simply add another matrix between the input and output matrices. This matrix represents the weights of the connections between the neurons in the hidden layer and the neurons in the output layer.

$$f ( x ) = \mathbf{ x \ }^T W^d,^ C\$$

* `x` is a vector of length `d`
    
* `w` is a matrix of size `d x C`
    
* `C` is the number of output classes
    
* `f(x)` is a vector of length `C`
    

When adding a second layer with a new matrix this notation is like this!👇🏻

$$f({x}) = {x}^\top {W}^{d \times n}{(h_1)} {W}^{n \times C}{(\text{out})}$$

### **Simple Definition of Neural Networks**

There are so many definitions have **‘Internet’** neural networks but some of them are useful. Neural networks are a collection of neurons that are connected by layer. Each neuron is performed some ‘calculation’ to solve problem. You can think each neuron is an **‘Employee’** working for a company. And each neuron is an organized layer. There are 3 types of layers input, hidden, and output. Neural networks inspire how the human brain works but how the human brain work is a long-term process of discovery {**running**}.

$$x = \sum{(weights * inputs) + bias}$$

### **Components Of A Neural Network**

There are 3 components of a neural network **Activation function, Weights, Bias**.

$$nn = (i)\ A \ (ii) \ W \ (ii) \ B$$

***Activation Function —*** The activation function determines whether a neuron should be activated or not. It’s an important role play in our neural network it adds **non-linearity** to the model. If neurons are activated that means input is important. There are lots of activation functions, it’s currently active research in the deep learning field. But only some of them are popular and need to know.

* **Sigmoid** is used to predict the probability of an output node between 0 and 1.
    

$$f(x) = {\large \frac{1}{1+e^{-x}}}$$

* **Tanh** used to predict output node is between 1 and -1.
    

$$f(x) = {\large \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}}$$

* **ReLU** set our output node 0 if function result is negative
    

$$f(x)= {\small \begin{cases} 0, & \text{if } x < 0\\ x, & \text{if } x\geq 0\\ \end{cases}}$$

* **Weights** ( W ) and Bias ( b )
    

$$Wij = weight(i, j)$$

**Note** :~ ***Wij*** denotes the weight connecting the ***i*** th neuron in the preceding layer to the ***j*** th neuron in the current layer. `weight` is a function that takes two inputs, `i` and `j`, and returns the weight between the two neurons.

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

I use [**nn. Sequential**](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) method defines the model, it’s a quick method to define any model in PyTorch.

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

In this example, I've created a custom class called `MyModel` that inherits from `nn.Module`. Inside the `__init__` method, I've defined the hidden layer and output layer as instance variables. The `forward` method takes an input tensor `x` and passes it through the hidden and output layers sequentially. Finally, I instantiated the model by creating an object of the `MyModel`class.

Don't be confused. If you know [Python classes](https://docs.python.org/3/tutorial/classes.html), this will be easy to understand. I use this method most of the time to define models because it's easy and user-friendly once you get the hang of it.

Remember — This model has one input **( X )** and one output **( Y )**, and a single hidden layer with 10 neurons.

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

<aside>
✨ Recall what is classfication?
</aside>

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

$$Note 🔥— We \space have \space two \space features \space and \space two \space output\space (2x * 2y)$$

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

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693786211093/70248aa2-0e5c-42f5-b580-bdded58aa9cd.png align="center")

You can see the neural network I built ( model 2 ) performance is outstanding compared to model 1 because it’s separate two classes ( blue, and red ).

Note🔥— Deep learning is all about an experiment, experiment, experiment🧑🏻‍🔬 , one solution does not work all the time because every human understands things in different ways 👨🏻‍🎨. So start small changes ( **Hyperparameters** 〰️) and slowly improve when it needs, never change big the first time ( experiment, experiment, experiment🧑🏻‍🔬 ).

## Better Training Code

At this moment we know how to write code regression and classification types of problem in PyTorch! The best thing about this, using this type of framework. If you know some basic things about how deep learning works you can solve problems because lots of functions already have, 'just import and use'. It’s cool ✨

But this time we need to learn how to improve the model training. Because our goal is to improve our model accuracy so we need this tool { **experiment, experiment, experiment**🧑🏻‍🔬 }. Here are the two things you notice when i build a model 🤖

* Problem One 📌 — I am training and evaluating( check model accuracy 💯 ) the same data, this is not the correct way to do that. Because sometimes models memorize training data ( training time ) so we need our dataset 3 parts!🍕. For example, if you have a 1 million image dataset you can split three parts, see below.
    
* **Training set** — 600,000 ( 60% ) of the images.
    
* **Validation set** — 200,000 ( 20% ) of the images.
    
* **Testing set** — 200,000 ( 20% ) of the images
    

Here is an example code below in PyTorch, how you split the dataset.🔥

$$(i)\ Training\ Set \ (ii) \ Validation\ Set \ (iii) \ Testing \ Set$$

```python
import torch
from torch.utils.data import random_split

# Create a dataset object.
dataset = torch.utils.data.ImageFolder("data")

# Get the number of images in the dataset.
num_images = len(dataset)

# Split the dataset into training, validation, and testing sets.
train_set, val_set, test_set = random_split(dataset, [int(0.6 * num_images), int(0.2 * num_images), int(0.2 * num_images)])

# Print the number of images in each set.
print("Number of training images:", len(train_set))
print("Number of validation images:", len(val_set))
print("Number of testing images:", len(test_set))


""" 
** Output ***
Number of training images: 600000
Number of validation images: 200000
Number of testing images: 200000

"""
```

Once you have split the dataset, you can train your model on the training set, evaluate the model's performance on the validation set, and test the model's performance on the testing set.

Once you finish the split the dataset, now you can train the model on the training set.

$$\begin{align*}\text{Training set} &= (X_\text{train}, y_\text{train}) \\\text{Validation set} &= (X_\text{valid}, y_\text{valid}) \\\text{Testing set} &= (X_\text{test}, y_\text{test})\end{align*}$$

## How to Ensure Your Model's Accuracy

We are improving our training technique to better align with pragmatic scenarios. This includes a training stage, where we update the model's parameters, and a testing stage, where we only evaluate the model's performance. It is important to guarantee that the model's parameters are not changed during the testing stage.

Note🔥— If you don’t understand any code meaning: just copy and paste the **‘**[**ChatGPT**](https://chat.openai.com/) **or** [**Bard**](https://bard.google.com/)**’** asked meaning, and they guide you. Use this type of AI tool in your education. Remember everyone doesn’t know anything , first time everything it’s hard and confusing but after it’s easy and fun.

$$Here \ is \ the \ code\\ 👇🏻$$

```python
def run_epoch(model, optimizer, data_loader, loss_func, device, results, score_funcs, prefix="", desc=None):
    running_loss = []
    y_true = []
    y_pred = []
    start = time.time()
    for inputs, labels in tqdm(data_loader, desc=desc, leave=False):
        #Move the batch to the device we are using. 
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        y_hat = model(inputs) #this just computed f_Θ(x(i))
        # Compute loss.
        loss = loss_func(y_hat, labels)

        if model.training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #Now we are just grabbing some information we would like to have
        running_loss.append(loss.item())

        if len(score_funcs) > 0 and isinstance(labels, torch.Tensor):
            #moving labels & predictions back to CPU for computing / storing predictions
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            #add to predictions so far
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())
    #end training epoch
    end = time.time()
    
    y_pred = np.asarray(y_pred)
    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1: #We have a classification problem, convert to labels
        y_pred = np.argmax(y_pred, axis=1)
    #Else, we assume we are working on a regression problem
    
    results[prefix + " loss"].append( np.mean(running_loss) )
    for name, score_func in score_funcs.items():
        try:
            results[prefix + " " + name].append( score_func(y_true, y_pred) )
        except:
            results[prefix + " " + name].append(float("NaN"))
    return end-start #time spent on epoch
```

## **A Better Way to Train Deep Learning Models**

Keypoint remember 👇🏻

* model $f (\\ .\\ )$ — To train our PyTorch model, we need to run it through the dataset one time. This is called an epoch. After one epoch, the model will be a little bit better at making predictions. We can continue to train the model by running more epochs.
    
* loss\_function $\\mathcal{L}(y, \\hat{y})$ — The loss function $\\ell(.\\ , \\ . )$ takes two arguments model output and labels ( **y** ) and return a loss.
    
* train\_loader — This is a DataLoader object that returns tuples of ( **input, label** ) pairs used for training the model.
    
* test\_loader — This is a DataLoader object that returns tuples of (input, label) pairs used for evaluating the model.
    

optimizer $opt(⋅)$— Optimization is an important process in training time because it adjusts model parameters and reduces error. There are many different types of optimization available in PyTorch “***SGD, ADAM & RMSProp”,*** each with its own strengths and weaknesses. The best optimizer depends on what problem you try to solve. Here you can find full [details of optimizer available in PyTorch.\*](https://pytorch.org/docs/stable/optim.html)

```python
'''
A simple example of how you use Optimizer in PyTorch. 
torch.optim is a package in PyTorch that provides various 
optimization algorithms.

'''
import torch.optim as optim
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

$$Here\ is\ the\ full\ code\ I\ implement\\ 👇🏻$$

```python
def simple_network(model, loss_func, train_loader, test_loader=None, score_funcs=None, 
                         epochs=50, device="cpu", checkpoint_file=None):
    to_track = ["epoch", "total time", "train loss"]
    if test_loader is not None:
        to_track.append("test loss")
    for eval_score in score_funcs:
        to_track.append("train " + eval_score )
        if test_loader is not None:
            to_track.append("test " + eval_score )
        
    total_train_time = 0 #How long have we spent in the training loop? 
    results = {}
    #Initialize every item with an empty list
    for item in to_track:
        results[item] = []
        
    #SGD is Stochastic Gradient Decent.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    #Place the model on the correct compute resource (CPU or GPU)
    model.to(device)
    for epoch in tqdm(range(epochs), desc="Epoch"):
        model = model.train()#Put our model in training mode
        
        total_train_time += run_epoch(model, optimizer, train_loader, loss_func, device, results, score_funcs, prefix="train", desc="Training")

        results["total time"].append( total_train_time )
        results["epoch"].append( epoch )
        
        if test_loader is not None:
            model = model.eval()
            with torch.no_grad():
                run_epoch(model, optimizer, test_loader, loss_func, device, results, score_funcs, prefix="test", desc="Testing")
                    
    if checkpoint_file is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results' : results
            }, checkpoint_file)

    return pd.DataFrame.from_dict(results)
```

With the new and improved code, let's retrain our model on the moons dataset. Since accuracy is what we really care about, we import the accuracy metrics from Scikit-learn. Let's also include the F1 score metrics to demonstrate how the code can handle two different metrics at the same time.

```python
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
```

We also want to do a better job of evaluating our model. We can do this by creating a validation set. A validation set is a set of data that is not used for training the model. Instead, it is used to evaluate the model's performance after it has been trained.

Since the moons data is synthetic, we can easily create a new dataset for validation. To do this, we can simply generate more data points. This will give us a more accurate estimate of the model's performance on unseen data.

Rather than performing 200 epochs of training like before, let's generate a larger training set. This will help the model to learn more about the data and improve its performance.

```python
X_train, y_train = make_moons(n_samples=8000, noise=0.4)
X_test, y_test = make_moons(n_samples=200, noise=0.4)
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

###   ...... ###  ...... ###  ...... ### ...... ### 
training_loader = DataLoader(train_dataset, shuffle=True)
testing_loader = DataLoader(test_dataset)
```

We have everything we need to train our model again. We will use the file ***model.pt*** to save the model's results. All we need to do is create a new model object and call the ***train\_simple\_network*** function.

$$Here\ is\ the\ code\\👇🏻$$

```python
model = nn.Sequential(
    nn.Linear(2,  30),
    nn.Tanh(),
    nn.Linear(30,  30),
    nn.Tanh(),
    nn.Linear(30, 2),
)
results_pd = simple_network(model, loss_func, training_loader, epochs=5, 
                            test_loader=testing_loader, checkpoint_file='model.pt', 
                            score_funcs={'Acc':accuracy_score,'F1': f1_score})
```

Now it's time to look at some results. First, let's see if we can load our checkpoint model instead of using the one we already trained. To load a model, we first need to create a new model that has all the same sub-modules as the original model. This is necessary so that the weights in the new model match the weights in the original model. For example, if the original model had 30 neurons in the second hidden layer, then the new model must also have 30 neurons in the second hidden layer. Otherwise, there will be too few or too many neurons, and an error will occur.

One reason I use the ***torch.load*** and ***torch.save*** functions because they provide a ***map\_location*** argument. This argument allows us to specify where the model should be loaded to. For example, if we are using a GPU, we can specify that the model should be loaded to the GPU. This can improve performance because the model will be able to run on the GPU instead of the CPU.

Once we have loaded the dictionary of results, we can use the ***load\_state\_dict*** function to restore the states of our original model into the new object. This will essentially create a copy of the original model. Finally, we can apply the model to the data and see that we get the same results as we did with the original model.

```python
model_new = nn.Sequential(
    nn.Linear(2,  30),
    nn.Tanh(),
    nn.Linear(30,  30),
    nn.Tanh(),
    nn.Linear(30, 2),
)

visualize2DSoftmax(X_test, y_test, model_new, title="Initial Model")
plt.show()

checkpoint_dict = torch.load('model.pt', map_location=device)

model_new.load_state_dict(checkpoint_dict['model_state_dict'])

visualize2DSoftmax(X_test, y_test, model_new, title="Loaded Model")
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693786915008/ba065129-2bef-472f-9f06-9671f7a695eb.png align="center")

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693786959673/b2c394bb-3bb3-45a2-a95a-7f3bb80309dc.png align="center")

The initial model does not give very good predictions because it has not been trained yet. The weights of the model are random values, and they do not yet reflect the relationship between the input and output data. If you run the code several times, you will get slightly different results each time, because the weights of the model are randomly generated each time.

After we load the previous model state into the ***model\_new,*** we get the nice crisp results we expect. This is because the previous model has been trained on a large dataset, and the weights of the model have been optimized to reflect the relationship between the input and output data.

In other words, the initial model is not very good at making predictions because it does not have enough information. It does not know how the input data relates to the output data. The previous model, on the other hand, has been trained on a large dataset, and it has learned how to make predictions based on the input data.

Our new training function returns a panda's DataFrame object with information about the model after every epoch. This information can be easily visualized, for example, by plotting the training and validation accuracy as a function of the epoch.

Here is an example of how to plot the training and validation accuracy:

```python
sns.lineplot(x='epoch', y='train Acc', data=results_pd, label='Train')
sns.lineplot(x='epoch', y='test Acc', data=results_pd, label='Validation')
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693787046439/e0d16bd6-432e-4913-a133-a4a59da363c6.png align="center")

As you can see, the training accuracy is increasing over time, and the validation accuracy is also increasing, but at a slower rate. This is a good sign, as it means that the model is not overfitting to the training data.

You can also use the pandas DataFrame to get other information about the model, such as the loss and the number of epochs it took to train.

### Predict Whether This Person Covid or Not

Learning time simple datasets are useful but in the real world you don’t use this type of simple dataset train model. You need a large amount of datasets and powerful GPU to train the model. Check this [**Covid19 predication model**](https://hiwhy.io/covid-19-deep-learning-model) so you understand better.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693787110459/ef83ac70-6a74-4597-b4f5-8781ec0cec96.png align="center")

### Here Are Some Model Improvement Techniques That You Can Use

Note 🔥 — This is all hyperparameters you can change to improve model performance. Don’t change so many things same time. Start one thing, see the result , then change another one.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1693787160020/c28d13e4-b3d5-40c5-8f61-28422bbbac27.png align="center")

### Learn Right Way

I see many people just read and watch videos and train models using the hello world dataset MNIST, I don’t agree with this type of learning. Because in the real world, this type of dataset you not work and the simple problem does not require deep learning. Below I share some of the techniques you can use to learn, this is my personal advice it’s your choice to follow or not.

* Learn the basic thing you need ( Suppose what is classification and how to build a classification model in PyTorch ). Don't focus on memorizing syntax.
    
* If the basic is complete now it’s time to Download a real-world dataset from [Kaggle](https://www.kaggle.com/) or another [competition website](https://www.drivendata.org/) and try to implement what you learn. I think this way you learn so much.
    
* Learn and implement —&gt; Suppose you read ( Activation function is important ) implement this and see what result you get. The more you practice this way more you understand better.
    
* Connect other great AI researchers, I found [Twitter,](https://www.threads.net/@hi_chiai) and [Linkedin](https://www.linkedin.com/in/sushanta-das-/) is the best place to connect with great people.
    
* Asked question Q , read { English | Math } convert code. **Q = E.M →C**
    

---

Thanks for reading! I hope you found this article helpful. If you have any questions or suggestions, please don't hesitate to comment below. Your comment is important to me, I'm always happy to help people learn about deep learning.