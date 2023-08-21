---
title: "Pandas In Python — Complete Guide 2023"
datePublished: Thu Jan 26 2023 03:58:07 GMT+0000 (Coordinated Universal Time)
cuid: clj80kpad000509lc7o4cbyo7
slug: pandas-in-python-complete-guide
canonical: https://hiwhy.io/complete-pandas-guide/
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1687824720116/0c49b4fb-2c6e-4066-82d5-b4aabf330d36.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1687824757626/86e9fc1a-5f1c-4956-8583-d2474b82514c.png
tags: python, data-science, data-analysis, hashnode, deep-learning

---

Welcome to the complete beginner to the advanced Pandas library guide. In this article, we explain everything about Pandas library, so you don’t need any previous knowledge of this library. After reading this article you understand everything about this library, this is our promise.

Keep reading,

## Introduction To Pandas

Pandas are the most popular DataScience library in Python. This library is used by **Data Scientists, Machine learning Engineers, Artificial Intelligence Developers** and others. Why use pandas for this type of person? Because pandas are **Open Source** libraries. This library helped easily maintain and analyze data.

![2 woman , one man working laptop. ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402452359/3ef6e771-09ac-4d2c-950f-8762e4b17bf2.png align="center")

## How To Install Pandas

Pandas Installation process is very easy. First, install [`pip`](https://pip.pypa.io/en/stable/installation/) `or` [`conda`](https://www.anaconda.com/), then one line of code Pandas install in your system. See below the code 👇🏻

```python
pip install pandas  # This pip version of install
# OR
conda install pandas   # This is conda version of install
```

### Import Pandas In Your Notebook

The next step is to import the Pandas library into your notebook. See code below.

$$⬇️$$

```python
# When you use any library, the first step is to import
import pandas as pd  # pd is short form pandas but you can use anything
```

## Pandas Data Type

![Pandas library dataframe and series ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402455791/d39b8016-edd1-4dad-8088-002e99306d2c.png align="left")

Pandas library has two main datatype **Series** and **DataFrame**.

* **Series** – One-dimension array holding any type. It’s like a Column in a Table.
    
* **DataFrame** – This is a two-dimensional array of data structures. It is like a table with columns and rows.
    

### How To Create A Series In Pandas

You can create a Series datatype, that time uses `pd.Series( )` function and pass a Python list. See the code below. 🔻

```python
# Create a Series of names of the person 
name = pd.Series(["zen","chi"])
name

'''
Output >>> 0    zen
               1    chi
               dtype: object> 
'''

# One more time, this time store passion in your life. 

person_passion = pd.Series(["Deep-Learning-Engineer","NLP Specialist"])
person_passion

''' 
Output >>>  0     Deep-Learning-Engineer
                1     NLP Specialist
                dtype: object
'''
```

### How To Create Data Frame Pandas?

Creating dataframe in pandas is very easy, the same is Python dictionary. When you create pandas DataFrame that time pass Python dictionary. See code below

```python
# create DataFrame to store a person's name and which thing passionate
person_passion_df = pd.DataFrame({"NAME":name,
                                   "PASSION":persion_passion})
person_passion_df

""" Output >>>
      NAME	      PASSION

0    	zen	        Deep-Learning-Engineer
1	    chi	        NLP Spacalist

"""
```

<div data-node-type="callout">
<div data-node-type="callout-emoji">❓</div>
<div data-node-type="callout-text"><strong>Exercise for you</strong></div>
</div>

I highly recommend you do this — don’t see the answer first. Practice, after you are stuck, come back, and see the answer. This is the only way you can learn fast and understand more.👇🏻

* Creates a **Series** of different shoes 👟
    
* Creates a **Series** of different prices 💲
    
* Combine the **Series** of Shoes and Prices in **DataFrame**
    

```python
#  Solution ✔️

# 1. Create a Series of different shoes.
shoes = pd.Series(["Derby Shoes", "Oxford Shoes", "Monk Shoes"])

# 2. Create a Series of different price
price = pd.Series([300,500,600])

# 3. Combine them two series of data
shoes_record = pd.DataFrame({"NAME":shoes,
                             "PRICE":price})

shoes_record

""" 
Output>>>
         NAME	           PRICE

0	       Derby Shoes	    300
1	       Oxford Shoes   	500
2	       Monk Shoes     	600

"""
```

## Read And Write External Files Using Pandas

If you can manually create **Series X** and **DataFrame** from scratch, that would be excellent.

But in the real world, you already have data. And this data is some sort of format you can find in your workspace. There are two most common types of data formats you see again and again.

* **CSV** (Comma Separated Value)
    
* **XLSX** ( Microsoft Excel file )
    

The cool thing about Pandas library, you don’t need the write function to open this type of file, Pandas already have.

* `pd.read_csv ( )` — **.csv file** read function
    
* `pd.read_excel( )` — **.xlsx file** read function
    

Let’s now download the data.

Download this dataset 👉🏻— [**House Price Dataset**](https://www.kaggle.com/datasets/shantanudhakadd/house-prediction-dataset) **🏡**

![Pandas, House, and spreadsheet ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402463954/15d802ce-21c3-4eb8-b3df-2beec8919570.png align="left")

```python
# imported house_price.csv file
house_sales = pd.read_csv("house_price.csv") 
house_sales
```

![Pandas data frame ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402465361/154f0820-5d8c-47b5-9674-17f7926c3783.jpeg align="left")

At this time, your dataset is available in Pandas DataFrame. This takes advantage of use all of Pandas functionality to analyze data.

```python
# pandas DataFrame short for called df
# import house price data to save it df 
df = pd.read_csv("house_price.csv")
df
```

![pandas data frame](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402466587/ad5d88fa-133f-4a57-9d24-93f4eb61ddf0.jpeg align="left")

### Structure Of Pandas DataFrame

The image below shows the main components of DataFrame and their different names. 🔻

![pandas data frame](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402468886/15411da3-4448-48ea-8858-f5ad530a50a8.jpeg align="left")

## Export Data From Pandas DataFrame

Exporting dataframe in Pandas library is very easy, because many functions have. But in this article I am writing about two main functions I think are very useful and most of the time you need.

* `.to_csv( )` — CSV format save function.
    
* `.to_excel( )` **—** Excel format save function.
    

Let’s now export this file **.csv file format**. Our file name is `house_price.csv`

```python
# Export house_price DataFrame to .csv file format
house_sales.to_csv("exported_house_price.csv") # input file path what are you save.
```

![Google colab using pandas library](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402470092/6aadb123-e818-42a6-a5ad-3fafa6fc2646.jpeg align="left")

## First Understand Pandas DataFrame

When you import some of the data into Pandas DataFrame your first job is to analyze. Because knowing your data is the key 🔑 data science.

In this section we are using many pandas function to analyze our ( `house_price` **🏠**) data.

**Keep reading,**

```python
house_sales
```

![pandas data frame](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402472998/b70407f5-041b-4fcc-be4f-a35341dac32f.jpeg align="center")

```python
# .dtypes  — Show each column what data type is stored.
house_sales.dtypes

''' Output >>>
area_type        object
availability     object
location         object
size             object
society          object
total_sqft       object
bath            float64
balcony         float64
price           float64
dtype: object
'''
```

```python
# describe() —Gives a statistical overview of the numerical column such as percentile, mean, std, etc.

house_sales.describe()
```

![pandas data frame](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402474399/3f61f038-6985-4b11-af72-47edb7057aaa.jpeg align="center")

Pandas have  `.info( )` methods to show information about a DataFrame such as.

* How many entries ( **row** ) have DataFrame?
    
* Show up missing values (if columns non-null values are sorter than the number of entries, it has missing values)
    
* Show the data type for each column.
    

```python
house_sales.info()

''' Output >>> 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 13320 entries, 0 to 13319
Data columns (total 9 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   area_type     13320 non-null  object 
 1   availability  13320 non-null  object 
 2   location      13319 non-null  object 
 3   size          13304 non-null  object 
 4   society       7818 non-null   object 
 5   total_sqft    13320 non-null  object 
 6   bath          13247 non-null  float64
 7   balcony       12711 non-null  float64
 8   price         13320 non-null  float64
dtypes: float64(3), object(6)
memory usage: 936.7+ KB
'''
```

You can also use mathematical methods such as `mean( )` or `sum( )` directly **DataFrame** or **Series**

```python
## using mean our DataFrame
house_sales.mean()

'''output >>> 
bath         2.692610
balcony      1.584376
price      112.565627
dtype: float64
'''
```

```python
# Calling mean on Series
house_price = pd.Series([10000,50000,30000])
house_price.mean()

# Output >>> 30000.0
```

If you are interested in adding all the values in each column, then use the pandas `sum( )` method. See the code below.

$$🔻$$

```python
house_sales.sum()

''' Output >>>
area_type       Super built-up  AreaPlot  AreaBuilt-up  AreaSu...
availability    19-DecReady To MoveReady To MoveReady To MoveR...
total_sqft      1056260014401521120011702732330013101020180027...
bath                                                      35669.0
balcony                                                   20139.0
price                                                 1499374.145
dtype: object
'''
```

The same method also uses the Series Data type.

```python
house_price.sum()

# Output >>> 90000
```

**Note** 🔥: `sum ( )` method does not work for the whole pandas DataFrame ❎. It works nicely when you choose to target each column.

### How To Find Column Names In Pandas DataFrame

```python
# Display all columns in DataFrame
house_sales.columns

''' 
Output >>> 
Index(['area_type', 'availability', 'location', 'size', 'society',
       'total_sqft', 'bath', 'balcony', 'price'],
      dtype='object')
'''

# Save all columns in different variables 
house_columns = house_sales.columns
house_columns[0] # get first columns

# Output >>> area_type
```

```python
# Get start and end index.
house_sales.index

# Output >>> RangeIndex(start=0, stop=13320, step=1)
# .index attribute used to show how many indexes we have in DataFrame.
```

Pandas **DataFrame** is like a **Python list**; the index starts with 0. 🔻

![pandas data frame ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402479388/1cbab159-abbc-45bd-bde9-cfe0f8fd9ecb.jpeg align="left")

```python
# show the length of the DataFrame
len(house_sales)
# Output >>> 13320

# Our DataSet Length is 13320, this means the index is 0-13319
```

## Display And Select Data In Pandas DataFrame

In this section, you will learn how to select and display pandas dataframe. This time we are using the most common and important method in the Pandas library.

Keep reading …

* [`head( )`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html)
    
* [`tail( )`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.tail.html)
    
* [`loc`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html)
    
* [`iloc`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html)
    
* [`columns`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html) `– df[‘A’]`
    
* `boolean indexing – df[df[‘A’] > 5]`
    
* [`crosstab( )`](https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html)
    
* [`plot( )`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html)
    
* [`hist( )`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.hist.html)
    

It’s coding time 👨🏻‍💻

```python
#.head( ) method shows 5 rows in DataFrame.
house_sales.head()
```

![pandas head function](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402481565/ab9b7f5a-eaee-47d9-9e4f-41f993a77296.jpeg align="center")

`head( )` function default **5 rows** shown. But if you get more such as 10 or 100 rows. Just pass through integer value in the `head( )` method. See code below for an example.

```python
# Display only 10 rows.
house_sales.head(10) # whatever number you put it in this method.
```

![pandas head function](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402482688/3b8cbb80-df85-40c0-bbbe-00e046c92734.jpeg align="left")

`tail( )` method is similar to `head( )` but this method works from bottom to top.

```python
house_sales.tail()
```

![pandas tail function](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402483873/676ebd5a-a305-46aa-9f40-243faa0b4a1a.jpeg align="left")

You can use `loc [ ]` and `iloc[ ]` property selecting data in your DataFrame and Series. See code below for an example.

$$⬇️$$

```python
# create simple pandas Series
job_role = pd.Series(["Data Scientist","Machine Learning Engineer","NLP Specialist",
                      "Trader","Investor"],index=[0,5,9,4,6])
job_role

''' Output >>> 0              Data Scientist
               5    Machine Learning Engineer
               9                 NLP Specialist
               4                       Trader
               6                     Investor
               dtype: object
'''
```

`loc[ ]` property input as an integer number. This property targets to get all data specific-index points from DataFrame or Series. Which number you put this property under to make sure this index is present in DataFrame or Series, if doesn’t present this property error is thrown.

```python
job_role.loc[4] # I choose 4 index to show value
# Output >>> 'Trader'
```

**Try our home sales dataset.**

$$🏡$$

```python
house_sales.loc[9] # I am Select Row at Index 9

''' Output >>>
rea_type          Plot  Area
availability    Ready To Move
location         Gandhi Bazar
size                6 Bedroom
society                   NaN
total_sqft               1020
bath                      6.0
balcony                   NaN
price                   370.0
Name: 9, dtype: object
'''
```

`iloc[ ]` is similar to `loc[ ]` but works with the exact position of this DataFrame.

```python
# In this code, we are using iloc[]

job_role

''' Output >>> 0              Data Scientist
               5    Machine Learning Engineer
               9                 NLP Specialist
               4                       Trader
               6                     Investor
               dtype: object
'''

job_role.iloc[1] # Select row at a position
# Output >>> 'Machine Learning Engineer'
```

**Note** 💡: That **Machine Learning Engineer** appears in index 5 in the Series. But `iloc[1]` shows it’s because Position is **1st**. **Series** and **DataFrame** starting 0 positions.

```python
# In this code, we are using iloc[] in our housing dataset🏡

house_sales.iloc[3] # Select position 3rd
''' Output >>>
area_type       Super built-up  Area
availability           Ready To Move
location          Lingadheeranahalli
size                           3 BHK
society                      Soiewre
total_sqft                      1521
bath                             3.0
balcony                          1.0
price                           95.0
Name: 3, dtype: object
'''
```

### How To Select Individual Columns In Pandas DataFrame

If you also select individual columns — the Syntax is, `DataFrame [“Columns Name”]`

```python
# Select Society column

house_sales["society"]
''' Output >>>
0        Coomee 
1        Theanmp
2            NaN
3        Soiewre
4            NaN
          ...   
13315    ArsiaEx
13316        NaN
13317    Mahla T
13318    SollyCl
13319        NaN
Name: society, Length: 13320, dtype: object
'''

# Select condition columns
house_sales["area_type"]

''' Output >>> 
0        Super built-up  Area
1                  Plot  Area
2              Built-up  Area
3        Super built-up  Area
4        Super built-up  Area
                 ...         
13315          Built-up  Area
13316    Super built-up  Area
13317          Built-up  Area
13318    Super built-up  Area
13319    Super built-up  Area
Name: area_type, Length: 13320, dtype: object

'''
```

If you set one or many conditions and you accept only get value when this condition is true ✔️. Same as the if and else statement. See code below on how you can do it.

```python
# if the price is greater than 3000 then show it.
house_sales[house_sales["price"] > 3000]
```

![pandas data frame ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402489839/20a44b09-ee75-4eeb-9407-753fa86d0493.jpeg align="center")

```python
# If house_sales bedroom is equal to 16 then show it.
house_sales[house_sales["bath"] == 16]
```

![pandas data frame ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402490939/1e22b564-2e38-404b-a2c0-3f1362439330.jpeg align="left")

`pd.crosstab( )` is the best way to visualize two different columns and compare them with each other.

```python
pd.crosstab(house_sales["price"] , house_sales["bath"])
```

![pandas crosstab function](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402492140/90d51458-53b6-46c0-8dfc-f03b0cd97125.jpeg align="left")

If you are interested in more columns to compare each other, use pandas this method `groupby( )`

```python
# Grouped by bath columns and found the meaning of other columns.

house_sales.groupby(["bath"]).mean()
```

![pandas groupby function](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402493522/0ccb6552-4e57-444b-a009-0db6656eaa2d.jpeg align="left")

Pandas library supports quick plotting in columns, meaning you see data in a visual way. To plot your dataset, first import the **matplotlib** library into your notebook.

This is another useful library in data science, if you want to learn more end to end guide [**read this article**](https://hiwhy.io/matplotlib-complete-guide).

See code below on how to plot data using matplotlib.

```python
# import matplotlib and say it Jupiter shows my plot
import matplotlib.pyplot as plt
%matplotlib inline

house_sales["price"].plot()
```

![matplotlib histogram ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402494901/48982bc1-a07a-492a-b6d4-67a707485de3.png align="center")

```python
house_sales["price"].hist()
```

![matplotlib histogram ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402495918/d9b0eaf8-0a5c-42c9-90fb-46d156f75f48.png align="center")

## Useful Data Manipulation Techniques In Pandas

At this time you know one way to manipulate data in pandas, but there are many ways you can manipulate data in Pandas library. So in this section we explain more!

Keep reading,

Get all the string values using for `str( )` method and convert them to lowercase also don’t forget to resign.

```python
house_sales['society'] = house_sales['society'].str.lower()
house_sales.head()
```

![house sale dataframe pandas ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402497017/6e5fa965-b3e3-4dfd-a624-fc8271c048b3.jpeg align="left")

If you look above DataFrame under the society column all values are lowercase. One question for my won I don’t use resign it but the data is updated. How can I do it?

The answer is that some functions have a parameter called `inplace` which means DataFrame update is in place without using resigning.

See an example of my DataFrame under the society column has a missing value

```python
house_sales.head()
```

![House sales data frame pandas ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402498163/c2c1ec0b-7ffa-4904-a6b8-b095dbfeaa2e.jpeg align="left")

**Note 💡:** Missing value shows `NaN` in pandas library. This is equal to `None in Python`.

Let’s use `fillna( )` method to act on filling missing data. Now fill out the society column under all missing values to mark with Unknown text.

`Inplace parameter` default value is `False` that mean doesn’t change the original DataFrame. If you change the original DataFrame it must be value set to `True`.

```python
house_sales["society"].fillna("unknown",
                              inplace=True) # inplace set True
```

**Check out our original DataFrame.**

$$🔻$$

```python
house_sales.head()
```

![pandas dataframe](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402501510/ba7cb0ab-c580-4325-a6b3-1261138cc9c2.png align="left")

We’ve filled just now a single column under all `NaN` values. But you will notice that the other column has to be the NaN value in `house_sales` DataFrame. Now it’s time for our next job is — remove all rows that have missing values and only work with data with no missing values in our data frame.

At that time we are using pandas useful method `dropna( )`. This method worked to remove all missing values from our dataframe.

**Note 💡**: `dropna( )` has inplace parameters and the default value is `( False )`. When you drop all the **nan values**, make sure the value is set to **True** or **reassign** to DataFrame.

```python
# Two lines of code, same thing do 👇

# Drop all missing and update original DataFrame in place.
house_sales.dropna(inplace=True)

# OR

# Drop all missing value and update original data frame using for equal sign.
house_sales = house_sales.dropna()

# See the Result
house_sales
```

![pandas data frame ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402503890/a3f37e4d-0f13-4ee2-990c-f36ad4989847.jpeg align="left")

### Create New Coloum Existing DataFrame Pandas

In this section you will learn how to create a new coloum with your existing dataframe. For example, create a column named **Bedroom** for the store number of bedrooms.

Syntax looks like —`DataFrame[new_column_name] = data`.

Note 💡: Make sure the column name must be in string format.

When you create a new column, these are the three common ways to add data – **(1)** Pandas Series, **(2)** Python list, or **(3)** existing columns value.

Let’s see example in the code below.

$$🔻$$

```python
# Re-import our data set.

house_sales = pd.read_csv("drive/MyDrive/pandas_article/house_price.csv")
# First, create a small DataSet to understand more easily.

house_sales_small = house_sales[:10]
house_sales_small
```

![House sales data frame ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402507148/09cd8c09-931a-42c0-8858-3895ce9a0438.jpeg align="left")

Let’s create an extra one more column name to specify `bed_rooms,` and using pandas `Series( )` data.

```python
# Create columns using pandas Series 

bed_rooms = pd.Series([3,3,3,3,3,3,3,3,3,3])
house_sales_small["bed_rooms"] = bed_rooms
house_sales_small
```

![House sales dataframe ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402508413/ac5b21ad-bff6-4e1e-b22e-2249911e54b7.jpeg align="left")

If you look at the output above, it shows one extra column `( bed_rooms )` for our DataFrame. Let’s move on once more to add a new column name as a **window,** and this time I am using python list data.

```python
# create a python list to specify how many windows each house
window = [6,5,3,5,3,3,2,9,10,2]
house_sales_small["window"] = window
house_sales_small
```

![house sales pandas data frame ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402509726/0639eaca-a66b-469a-bbe6-fd8b6b38558c.jpeg align="left")

You can also create a new column with two combining values from other columns, such as `price per sqft on the total sqft column.`

```python
# create price per sqft column to formula is price/total_sqft = price_per_sqft
house_sales_small["price per sqft"] = house_sales_small["price"]/house_sales_small["total_sqft"].astype(int)
house_sales_small
```

![House sales dataframe ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402511013/a0d920ff-140c-44c4-99d0-dd702c6c7efe.jpeg align="left")

**Note 💡**: `astype ( )` method convert datatype in pandas.

You can also create a column set to **all values the same**. For example, all houses 🏡 have the same number of doors 🚪. See code below. 👇🏻

```python
house_sales_small["doors"] = 4
house_sales_small
```

![House sales data frame ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402516122/7913b65d-6bb3-4701-85a1-888747f29661.jpeg align="left")

Let’s do one more again. 👇🏻

```python
# create register house column True or False
house_sales_small["register"] = True
house_sales_small
```

![House sales dataframe ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402519299/88c4fef4-eba0-4118-837e-33b8213356e8.jpeg align="left")

You know how to create a column, but sometime you don’t need all the columns in your dataset. So the question is ❓how to remove columns in pandas dataframe.

Removing columns in pandas is very easy because one method has achieved this goal. See code below. 👇🏻

```python
# Method syntax -- drop( COLUMN_NAME, axis=1) 

# Drop the price per sqft column
house_sales_small = house_sales_small.drop("price per sqft",axis=1)
house_sales_small
```

![House sales data frame ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402522437/c642b50d-cddb-42b8-a5e7-b057310da5fa.jpeg align="left")

Pandas Series only has a **0 axis,** but DataFrame is a **two-dimensional** data structure like a spreadsheet, it’s under the **0,** and **1 axis**. See the image below so you understand better.

![pandas library axis ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402523900/b23613b0-3695-4981-a71a-7a6b90e04989.jpeg align="center")

### Shuffle DataFrame In Pandas

Let’s say you interested to shuffle your dataFrame. That time use `sample(frac=1)` method in pandas. `sample( )` method randomly gets a sample of the row from DataFrame. The frac parameter set to 1 because \[frac=1 means 100%, frac=0.5 means 50%, and frac=0.01 means 1% of rows\].

```python
# sample house sales small DataFrame
sample_house_sales = house_sales_small.sample(frac=1)
sample_house_sales
```

![pandas data frame](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402526003/93e9b5d3-e014-4d03-89bf-ed2130a64864.jpeg align="left")

One thing notice 💡: our dataframe row is similar but the order is mixed. 👆

### Apply Pandas Function In Column

One more important thing pandas is apply functions directly in the column. Such as column **baths** add an extra 2 **baths.**

You can use this time `apply( )` function by passing the lambda function. Adding 2 to your bath column means an extra 2 bath add. But make sure to re-assign it, if you don’t re-assign means your column didn’t change the original DataFrame.

```python
house_sales["bath"] = house_sales["bath"].apply(lambda x: x+2)
house_sales
```

![pandas dataframe in housning sales dataset](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402529134/613f6a86-fc3c-41e2-a43b-087dd581edfc.jpeg align="left")

<div data-node-type="callout">
<div data-node-type="callout-emoji">❓</div>
<div data-node-type="callout-text">Thanks for reading. I hope you learn something new from this article. If you have any questions or suggestions comment now below. I try my best to answer your all question.</div>
</div>