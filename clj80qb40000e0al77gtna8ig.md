---
title: "Food Delivery Time Prediction Using Deep Learning"
datePublished: Tue Mar 07 2023 05:39:50 GMT+0000 (Coordinated Universal Time)
cuid: clj80qb40000e0al77gtna8ig
slug: food-delivery-time-prediction-using-deep-learning
canonical: https://hiwhy.io/food-delivery-time-prediction-using-deep-learning/
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1687654200379/1ef570d5-ae89-4ff5-a0bc-0dcfba8b5e5c.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1687655114956/f3f9f0e2-e534-40ba-baba-95ba495752c0.png
tags: python, machine-learning, hashnode, computer-vision, deep-learning

---

In this article, I build end-to-end deep neural networks to predict Food delivery time prediction using deep learning.⏱️

This is a very helpful article. After reading this article you solve different types of regression problems such as stock price prediction📊, house price prediction🏠, and many more. Because the same concept applies to all deep-learning problems.

Keep reading 🔥

## What Real-World Problem We Are Solve Using Deep Learning

When you order food these types of companies such as **JustEat, DoorDash, Zomato, Swiggy** and etc. This company provides you delivery person’s name, mobile number, and how much time it takes for delivery successfully. Most people already guess which thing to solve in this article.

Our job is to build a deep learning model to predict how much time it takes for delivery successful based on past data.

<div data-node-type="callout">
<div data-node-type="callout-emoji">🔥</div>
<div data-node-type="callout-text">All the code is available on my <a target="_blank" rel="noopener noreferrer nofollow" href="https://github.com/hi-sushanta/Blog_Post/blob/master/Real-Time-Food-Delivery-Time-Prediction/Food_Delivery_Time_Prediction.ipynb" style="pointer-events: none"><strong>GitHub</strong></a>!🙈 .</div>
</div><div data-node-type="callout">
<div data-node-type="callout-emoji">🔥</div>
<div data-node-type="callout-text">This dataset is available on my <a target="_blank" rel="noopener noreferrer nofollow" href="https://mega.nz/folder/is8khQbJ#q25lrBIFiNRUp-ECD9lEvw" style="pointer-events: none"><strong>Mega drive</strong></a>.</div>
</div>

In this dataset under 11 columns and each of these columns is under **45593** value store.

This column represents as a delivery partner how much time to take for delivery successful in the past days.

Now look closely at each of these columns ⤵️

* `ID` — It’s unique and the order Id number.
    
* `Delivery_person_ID` — It is the unique ID number of the Delivery partner.
    
* `Delivery_person_Age` — This is the actual age of the delivery partner
    
* `Delivery_person_Rating` — It’s a rating of the delivery partner based on past deliveries.
    
* `Resturant_latitude` — It’s the latitude of the Restaurant.
    
* `Resturant_longitude` — It’s longitude of Resturant.
    
* `Delivery_location_latitude` — It’s the latitude of actual delivery locations
    
* `Delivery_location_longetude` — It’s longetude of actual delivery locations.
    
* `Type_of_order` — This is a type of meal ordered by a customer.
    
* `Type_of_vehicle` — This is a type of vehicle delivery partner ride on.
    
* `Time_taken(min)` — It’s the actual time taken by a delivery partner to complete this order.
    

Import Some **Require Libary** we need solve this problem.

![👨🏻‍💻](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402143683/301d42c0-6e97-4d12-a04c-f8a68fd5f0ba.png align="center")

```python
import pandas as pd 
import numpy as np
import plotly.express as px
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout,GRU
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
```

## Load Data Into Panda DataFrame

Now time to move on and load data into [**pandas DataFrame**](https://hiwhy.io/complete-pandas-guide/). Use the `read_csv ( )` function to read the `txt file` as well.

```python
dataframe = pd.read_csv("/content/drive/MyDrive/new_article/FoodDeleveryTimePrediction/deliverytimehistory.txt")
dataframe
```

![Google Colab output](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402144796/6105266d-1d90-4ade-9389-caa9221e5acc.png align="left")

Use the `info( )` method in pandas to display dataFrame information. ⤵️

```python
dataframe.info()
```

![Google Colab output in pandas dataframe](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402147123/7a273151-8829-4a05-929a-ff00a42db97a.png align="center")

Make sure our dataset does not have any null value. ⤵️

```python
dataframe.isnull().sum()
```

![Google colab output in pandas dataframe ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402149179/2a11be3d-90a6-428d-966e-67573f87cd47.png align="center")

If you see our dataset doesn’t have any null value. Now move next section!

### **Insert New Distance Column in DataFrame 🗺️**

In this section, I create a small function to calculate the distance between the restaurant and the delivery location.

I am also adding the `distance column` in my pandas dataFrame! 🐼

```python
# Set the earth's radius value (in kilometers)
R = 6371

# Convert degrees to radians
def degree_to_radians(degrees):
    return degrees * (np.pi/180)

# Using herversine formula to calculate the distance between to points
def distcalculate(lat1, lon1, lat2, lon2):
    d_lat = degree_to_radians(lat2-lat1)
    d_lon = degree_to_radians(lon2-lon1)
    a = np.sin(d_lat/2)**2 + np.cos(degree_to_radians(lat1)) * np.cos(degree_to_radians(lat2)) * np.sin(d_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c
  
# Calculate the distance between each pair of points
dataframe['distance'] = np.nan

for i in range(len(dataframe)):
    dataframe.loc[i, 'distance'] = distcalculate(dataframe.loc[i, 'Restaurant_latitude'], 
                                        dataframe.loc[i, 'Restaurant_longitude'], 
                                        dataframe.loc[i, 'Delivery_location_latitude'], 
                                        dataframe.loc[i, 'Delivery_location_longitude'])
```

The above step is complete now display the dataFrame and see what looks like the distance column.

```python
dataframe
```

![Google colab output in pandas dataframe](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402154542/d4eb711d-9898-461e-94d8-40322bad13dd.png align="center")

### Display Relationship Between Distance And Time

At this time I am using the [**matplotlib library**](https://hiwhy.io/matplotlib-complete-guide) to display the relationship between two columns such as `distance and time`. See the code below!

```python
figure = px.scatter(data_frame = dataframe, 
                    x="distance",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    trendline="expanding" ,
                    title = "Relationship Between Distance and Time")
figure.show()
```

![Matplotlib visulaztion output in google colab](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402155757/0e717c28-8a6c-417b-9aa1-dc239d495578.png align="center")

### Display Relationship Between Time And Age

Once again see the relationship between the time and age column.

```python
figure = px.scatter(data_frame = dataframe, 
                    x="Delivery_person_Age",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    color = "distance",
                    trendline="ols", 
                    title = "Relationship Between Time and Age")
figure.show()
```

![Matplotlib data visulaztion output in Google colab](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402157232/d483d420-b672-408e-a516-c0f9e6dd7745.png align="left")

### Display Relationship Between Time and Ratings

Again to see the relationship between time and rating. 🌟🌟⏱️

```python
figure = px.scatter(data_frame = dataframe, 
                    x="Delivery_person_Ratings",
                    y="Time_taken(min)", 
                    size="Time_taken(min)", 
                    color = "distance",
                    trendline="ols", 
                    title = "Relationship Between Time and Ratings")
figure.show()
```

![Matplotlib libraray output in Google colab](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402161911/a258e86a-8de0-4da4-82e5-2b0ca9b7c1eb.png align="center")

### Display Relationship Between Time and Type of Vehicle

The final step is to examine the relationship between time and the type of vehicle.

```python
fig = px.box(dataframe, 
             x="Type_of_vehicle",
             y="Time_taken(min)", 
             color="Type_of_order",
             labels={"Type_of_vehicle": "Vehicle","Time_taken(min)":"Time"})
fig.show()
```

![Data visualization In matplotlib](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402165468/48466e86-a83f-4d07-afc6-61e8e2157255.png align="left")

## Encoding Some Label Value

In this section, I am working with label encoding. This means any string object represents a [**numerical value**](https://hiwhy.io/numpy-complete-guide).

As you also know, my data set is under a `vehicle` and the `type of order` is a string object that is not supported when I train a model. So the first step is to convert all string objects to the **numerical value**.

```python
le = LabelEncoder()

dataframe['Type_of_vehicle'] = le.fit_transform(dataframe['Type_of_vehicle'])
vehicle_label = le.classes_ 
dataframe['Type_of_order'] = le.fit_transform(dataframe['Type_of_order'])
order_label = le.classes_
dataframe
```

![Pandas dataframe output in Google colab](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402166637/5e4fba63-e4ea-4783-8cf6-41cf20ee1d41.png align="left")

If you see that my two columns under all values successfully label encoding. You can also see that the `Delivery person id column` has a string value. But that column is not important in this project so, I drop this column in the next section.

## Split Dataset Into Training And Testing Categories

This is a very important topic in deep learning and machine learning project that split datasets into training and testing sets. 🤖 At this time I divided my dataset into train and test sets. Because when my deep learning model training is complete than next step is to check model prediction accuracy. This time I am using a testing dataset. 💯

Always remember 💥: When you build any deep learning model, first split dataset training and testing set!

```python

# x variable store features column with value.  
# Also note that not store restaurant and delivery location longitude and latitude
# Becuse these two column combine of distance column.

x = np.array(dataframe[["Delivery_person_Age", 
                   "Delivery_person_Ratings", 
                   "distance",
                   'Type_of_vehicle',
                   'Type_of_order']])

# y variable store target column 
y = np.array(dataframe[["Time_taken(min)"]])


# If the  above step is complete now split them train and test set 
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.10, 
                                                random_state=42)

xtrain
```

![Numpy array output in Google Colab](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402173261/1e3ee0b0-c0e6-4fac-8d14-a90bd96cba41.png align="left")

## Build Neural Network And See Summary

In this section, I build a simple deep neural network for predicting food delivery time.

![🚴🏻](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402174748/dea1e253-5e35-47d5-84c6-7f0208346949.png align="center")

![⏱](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402175663/d640aaaa-d24f-4f4b-9496-c43732ef6829.png align="center")

At that time I am using **TensorFlow** for creating deep neural networks. You can also use **PyTorch** and other frameworks which suitable for you.

```python
model = Sequential()
model.add(LSTM(128,return_sequences=True,input_shape=(xtrain.shape[1],1)))
model.add(GRU(128,return_sequences=False))
model.add(Dense(64))
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()
```

![Deep leanring model summary google colab output](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402176736/003f1699-ff29-4d8d-aa1a-85a214f688ec.png align="center")

### Model Compile And Train

When the model architecture is ready now move on compile and train them with the training dataset.

$$🤖$$

```python
model.compile(optimizer=tf.keras.optimizers.Adam(),
               loss='mean_squared_error')
model.fit(xtrain,ytrain,batch_size=1,epochs=9)
```

![Deep learning model output](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402179142/7292d74d-f4c2-4a14-98ea-67a3db5e9407.png align="center")

### Model Evaluate

Once the model trains successfully now time to evaluate the model with a testing dataset. It checks for model performance on unseen data.

$$💯$$

```python
model.evaluate(xtest,ytest)

# OUTPUT 🔻
# 143/143 [==============================] - 2s 4ms/step - loss: 54.6020
# 54.60203552246094
```

### Predict Some Demo Examples

At this time I input some required values and the model predicts the actual time to take for delivery successful. ⏱️🤖

```python
print("Real Time Food Delivery Time Prediction")
age = int(input("Age of Delivery Partner: "))
rating = float(input("Ratings of Previous Deliveries: "))
total_distance = float(input("Total Distance: "))
type_vehicle = int(input('''Type Of Vehicle Options \n(0) bicycle \n(1) electric scooter \n(2) motorcycle \n(3) scooter
             Choose One of these: ''')) 
type_order = int(input('''\nType Of Order Options \n(0) Buffet \n(1) Drink \n(2) Meal \n(3) Snack 
              Choose One Of these: ''')) 
features = np.array([[age, rating, total_distance,type_vehicle,type_order]])
print("\nPredicted Delivery Time in Minutes = ", round(float(model.predict(features))))
```

![Food delivery time predicton output in Google colab in my deep learning model](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402184153/1999a8bd-63b4-437c-80bb-c92a3bb58902.png align="center")

<div data-node-type="callout">
<div data-node-type="callout-emoji">❓</div>
<div data-node-type="callout-text">Thanks for reading. I hope you learn something new from this article. If you have any questions, suggestions, or something doesn’t understand comment below. I try my best to answer your all question. Asking questions is not a bad thing!</div>
</div>