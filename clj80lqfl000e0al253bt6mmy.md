---
title: "How To Handle Missing Data Pandas DataFrame"
datePublished: Sat Jan 28 2023 03:15:00 GMT+0000 (Coordinated Universal Time)
cuid: clj80lqfl000e0al253bt6mmy
slug: how-to-handle-missing-data-pandas-dataframe
canonical: https://hiwhy.io/how-to-handle-missing-data-pandas-dataframe/
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1687742498249/ab0efd55-7148-49e3-bd1c-d66f63a33196.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1687742537917/424b298d-4abf-43a2-abc9-18fdb14c8f5f.png
tags: data-science, machine-learning, data-analysis, hashnode, deep-learning

---

In the real world, a dataset comes with some sort of missing value and this missing value is not good for our dataset. So in this article, we learn how to remove missing data from our pandas dataframe. Keep reading💥

## Load Dataset Pandas DataFrame

<div data-node-type="callout">
<div data-node-type="callout-emoji">🔥</div>
<div data-node-type="callout-text">Download this dataset –&nbsp;<a target="_blank" rel="noopener noreferrer nofollow" href="https://www.kaggle.com/datasets/anushabellam/cars-cars-2" style="pointer-events: none"><strong>Car Dataset</strong></a> <strong>🚗</strong></div>
</div>

Our dataset is CSV format, so i am using pandas **read\_csv** method read this dataset. If you learn more about the in-depth Pandas library read our this [**end-to-end guide**](https://hiwhy.io/pandas-in-python-complete-guide).

```python
import pandas as pd
car_india_dataset = pd.read_csv("drive/MyDrive/Dataset/Cars_India_dataset.csv")
car_india_dataset.head()
```

![Google colab output pandas dataframe](https://cdn.hashnode.com/res/hashnode/image/upload/v1687742775972/aab25dfb-4d57-4ebd-af42-e97e0d7ea0ef.png align="center")

## Check Missing Value Pandas DataFrame

Checking the missing values in pandas is very easy because one method has achieved this goal.

```python
car_india_dataset.isnull() # Detect missing values dataframe.
```

You will notice 💡 that our dataset shows **False and True** because when you apply [`isnull ( )`](https://pandas.pydata.org/docs/reference/api/pandas.isnull.html) method this method returns the Boolean value. In simple terms True means that it is a null value False mean doesn’t have a null value.

![Google colab output pandas dataframe ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687742860391/a557961f-b4ac-4809-bf9b-e445b049befc.png align="center")

### Drop All Missing Values DataFrame

```python
car_india_dataset = car_india_dataset.dropna()
car_india_dataset
```

Note 🔥 : Our dataFrame actual 156 rows but when we apply `dropna( )` method that time we see 127 rows. Because the `dropna( )` method removes all null values.

![pandas data frame](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402383826/70bd4e4d-0113-405a-adcf-859ff1a69ced.png align="left")

### **<s>Second Method</s>**

$$👇$$

## Replace Null Value Pandas DataFrame

```python
# Read car dataset 👇

import pandas as pd
car_india_dataset = pd.read_csv("drive/MyDrive/Dataset/Cars_India_dataset.csv")
car_india_dataset.head()
```

![pandas data frame](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402386237/d8ec65eb-23bd-488a-bcd0-61c8c12b35d9.png align="left")

Now it’s time to fill in all the missing values in our dataFrame and this time we are using pandas [**fillna ( )**](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html) method

```python
car_india_dataset = car_india_dataset.fillna("unknown")
car_india_dataset
```

![pandas data frame ](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402387597/f845ad5a-a8d4-4782-ae0f-e56526f046dd.png align="left")

You can see our dataset under have an `unknown` word everywhere.

If you use the `fillna( )` method, make sure to fill in the expected value. But my suggestion is that when you use this method that time only fills each column not the other.

Because different columns have different values and not the same dataType **\[ int, string, float** **\]** . See the code below on how you can use this method correctly.

$$👇$$

```python
# read the dataset file 
import pandas as pd
car_india_dataset = pd.read_csv("drive/MyDrive/Dataset/Cars_India_dataset.csv")
car_india_dataset
```

![pandas data frame](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402390158/e06531d7-a369-4ecb-8df4-d1bac004b9c2.png align="left")

```python
car_india_dataset["Displacement"] = car_india_dataset['Displacement'].fillna(3000)
car_india_dataset
```

![pandas data frame missing value](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402391642/de74bb53-b330-4efa-bbbf-c70dabda3593.png align="left")

You will notice in the `Displacement` column that all null values are replaced with 3000.

<div data-node-type="callout">
<div data-node-type="callout-emoji">🔥</div>
<div data-node-type="callout-text">Thanks for reading! I hope you found this helpful article... If you have any questions, please ask me in the comments below. I will do my best to answer all of your questions. You can also write any suggestions for me. To learn more about the <a target="_blank" rel="noopener noreferrer nofollow" href="https://hiwhy.io/pandas-in-python-complete-guide" style="pointer-events: none"><strong>Pandas Library end-to-end, read this article</strong></a>.</div>
</div>