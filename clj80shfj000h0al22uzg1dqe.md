---
title: "Bird Species Detection Using Deep Learning And YOLOv8"
seoTitle: "Bird Species Detection Using Deep Learning And YOLOv8"
seoDescription: "Learn how to identify birds using YOLOv8 in just 10 minutes! This guide is perfect for beginners."
datePublished: Thu Mar 23 2023 03:37:08 GMT+0000 (Coordinated Universal Time)
cuid: clj80shfj000h0al22uzg1dqe
slug: bird-species-detection-using-deep-learning-and-yolov8
canonical: https://hiwhy.io/bird-species-detection-using-deep-learning-and-yolov8/
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1687651568039/8d13bd21-cfaf-4611-a23a-b3697dfb2eaa.png
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1687651603307/e08f844d-defa-4e32-835b-a7d588b4954e.png
tags: python, hashnode, computer-vision, deep-learning

---

In this article, you learn two things — first, how to detect bird species and how to train using [**computer vision**](https://hiwhy.io/computer-vision-systems-work) pre-trained model YOLOv8 in a custom dataset.

Keep reading 🔥

Note : If you follow this article so make sure you using **GPU**.

All the code in this article you find on [**GitHub**](https://github.com/hi-sushanta/Blog_Post/tree/master/Bird-Species) 👨‍💻

Dataset — In this article, I am using the [**cup 200 species dataset available on Kaggle**](https://www.kaggle.com/datasets/sovitrath/cub-200-bird-species-xml-detection-dataset).

This dataset has under 200 different types of categories of images and labels such as Acadian Flycatcher, American Crow, American Goldfinch, and many more.

## Unzip Dataset

Once you successfully download it. Then see if the dataset format is a zip file, so your first step is to unzip the file. First, create a dataset folder then unzip 📁the actual dataset file and move to this folder.

```python
!mkdir Dataset # Create new Dataset Folder
# Uzip the actual dataset file and move to Dataset folder
!unzip '/content/drive/MyDrive/new_article/Bird-species/cub_200_2011_xml.zip' -d '/content/Dataset/'
```

You notice when running above the code see a huge of output come from. That time don’t confuse this is an image and label file.

### Install The Ultralytics Package In Your Notebook

We first import the Ultralytics package in our notebook, because I am using the YoloV8 model. But some other libraries need complete our detection project complete.

Let’s write some code and import other libraries!

```python
import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
```

## Extract XML File

In this section, I create one function to extract XML files and return a Python dictionary. See the below code.

```python
# Function to get the data from XML Annotation
def extract_xml_file(xml_file):
    xml_root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    img_info_dict = {}
    img_info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in xml_root:
        # Get the file name 
        if elem.tag == "filename":
            img_info_dict['filename'] = elem.text
            
        # Get size of the image
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            img_info_dict['image_size'] = tuple(image_size)
        
        # Get bounding box of the image
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            img_info_dict['bboxes'].append(bbox)
    
    return img_info_dict
```

**Note** 🔥\*\*:\*\* Check the above function and see works properly or not.

```python
print(extract_xml_file('/content/Dataset/train_labels/Acadian_Flycatcher_0003_29094.xml'))
# OUTPUT 🔻
# {'bboxes': [{'class': 'Acadian Flycatcher', 'xmin': 216, 'ymin': 68, 'xmax': 403, 'ymax': 344}], 'filename': 'Acadian_Flycatcher_0003_29094', 'image_size': (500, 400, 3)}
```

Note 🔥: It’s my result and your result is the same meaning your function works properly.

### Get Class Names And Mapping Them To Unique Numbers

In this section, I get all the actual labels and store them on a list. When all labels are complete store and then mapping to a unique number.

```python
class_names = [] # This list container store on all label
class_name_to_id_mapping = {} # This dictionary container mapping all label to unique number.

# get all class names and store on class_name list 
def get_class_names(info_dict):
  for b in info_dict['bboxes']:
      class_names.append(b['class'])
  
def mapping_to_class_name_to_id(class_names):
  unique_class_names = np.unique(class_names)
  for i, unique_label in enumerate(unique_class_names):
    class_name_to_id_mapping[unique_label] = i
```

My above two functions are ready. Now it’s time to use these two functions.

* `get_class_names( )`
    
* `extract_xml_file( )`
    

```python
# Get the all train and validation xml annotations file path
train_annotations_labels = [os.path.join('/content/Dataset/train_labels/', x) for x in os.listdir('/content/Dataset/train_labels/') if x[-3:] == "xml"]
train_annotations_labels.sort()
# test
test_annotations_labels = [os.path.join('/content/Dataset/valid_labels/', x) for x in os.listdir('/content/Dataset/valid_labels/') if x[-3:] == "xml"] 
test_annotations_labels.sort()

# extract xml file and append label into class_names list container
for i,ann in enumerate(tqdm(train_annotations_labels)):
    info_dict = extract_xml_file(ann)
    get_class_names(info_dict)

# If all label store on list container than mapping them unique number 
mapping_to_class_name_to_id(class_names)

# OUTPUT 🔻
# 100%|██████████| 5993/5993 [00:00<00:00, 13355.24it/s]
```

Check the length of `train annotations, test annotations`**,** and `class_name_to_id_mapping`.

Note🔥: If 200 shows that means successful work

```python
len(train_annotations_labels),len(test_annotations_labels),len(class_name_to_id_mapping)

# OUTPUT 🔻
# (5993, 5794, 200)
```

## Create A Label File For Suitable YOLOv8

In this section, I create a new function to convert the `info_dict` value to **Yolov8 txt** file format.

```python
#Convert the info dict to the required yolo txt file format and write it to disk
def convert_to_yolov8(info_dict,path):
    print_buffer = []
    
    # For each bounding box
    for bbox in info_dict["bboxes"]:
     

        try:
            # get class id for each label
            class_id = class_name_to_id_mapping[bbox["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v8
        b_center_x = (bbox["xmin"] + bbox["xmax"]) / 2 
        b_center_y = (bbox["ymin"] + bbox["ymax"]) / 2
        b_width    = (bbox["xmax"] - bbox["xmin"])
        b_height   = (bbox["ymax"] - bbox["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bounding box details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Name of the file which we have to save same as image file name.
    save_file_name = os.path.join(path, info_dict["filename"].replace("jpg", ""))
    save_file_name += '.txt'
    print(save_file_name)
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))
```

My function is complete now it’s time to use this function.

$$⤵️$$

```python
# Convert and save the train annotations
for i,ann in enumerate(tqdm(train_annotations_labels)):
    info_dict = extract_xml_file(ann)
    convert_to_yolov8(info_dict,'/content/Dataset/train_images/')
    
annotations_labels = [os.path.join('/content/Dataset/train_images/', x) for x in os.listdir('/content/Dataset/train_images/') if x[-3:] == "txt"]

# Convert and save the test annotations
for i,ann in enumerate(tqdm(test_annotations_labels)):
    info_dict = extract_xml_file(ann)
    convert_to_yolov8(info_dict,'/content/Dataset/valid_images/')

test_annotations_labels = [os.path.join('/content/Dataset/valid_images/', x) for x in os.listdir('/content/Dataset/valid_images/') if x[-3:] == "txt"]
```

**Note💥 —** I don’t show any <s>output</s> of the above code, because is very long and not a good look in your web browser.

Check the length of the two lists.

```python
len(train_annotations_labels),len(test_annotations_labels)

# OUTPUT 🔻
(5993, 5794)
```

### Display Image With Bounding Box

Let’s write one more function on plot images with a bounding box

```python
random.seed(0)

# Reverse order by class names. example is: 0 : bird_name. 

class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))

def plot_image_with_bounding_box(image, annotation_list):
  '''
     image : It's actual numpy formatted image you input.
     annotation_list : It's give as label with bounding box.

  '''
  # conver numpy array
  annotations = np.array(annotation_list)
  # get image width and height and store them different variable
  w, h = image.size
  
  plotted_image = ImageDraw.Draw(image)

  t_annotations = np.copy(annotations)
  t_annotations[:,[1,3]] = annotations[:,[1,3]] * w
  t_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
  t_annotations[:,1] = t_annotations[:,1] - (t_annotations[:,3] / 2)
  t_annotations[:,2] = t_annotations[:,2] - (t_annotations[:,4] / 2)
  t_annotations[:,3] = t_annotations[:,1] + t_annotations[:,3]
  t_annotations[:,4] = t_annotations[:,2] + t_annotations[:,4]
    
  for ann in t_annotations:
      obj_cls, x0, y0, x1, y1 = ann
      plotted_image.rectangle(((x0,y0), (x1,y1)))
        
      plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
  plt.imshow(np.array(image))
  plt.show()
```

If the above step is complete then go to the next part. Use this function!

```python
# Get any random label file 
label_file = random.choice(train_annotations_labels)
with open(label_file, "r") as file:
    label_with_bounding_box = file.read().split("\n")[:-1]
    label_with_bounding_box = [x.split(" ") for x in label_with_bounding_box]
    label_with_bounding_box = [[float(y) for y in x ] for x in label_with_bounding_box]

# Get the equal image file
image_file = label_file.replace("annotations", "images").replace("txt", "jpg")

assert os.path.exists(image_file)

# Load the image
image = Image.open(image_file)


# Plot the Bounding Box
plot_image_with_bounding_box(image, label_with_bounding_box)
```

![One bird siting one tree, and my yolov8 mode detect](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402028821/56428cd7-d5d1-48c5-b17f-ea2bfbc80a5b.png align="center")

## Read Images And Labels

In this section, I get all train and test images with labels. And also I see a split dataset into valid-test dataset categories.

```python
# Read images and labels
train_images = [os.path.join('/content/Dataset/train_images/', x) for x in os.listdir("/content/Dataset/train_images/") if x[-3:] == 'jpg']
train_labels = [os.path.join('/content/Dataset/train_images/', x) for x in os.listdir('/content/Dataset/train_images/') if x[-3:] == "txt"]

test_images = [os.path.join('/content/Dataset/valid_images/', x) for x in os.listdir("/content/Dataset/valid_images/") if x[-3:] == 'jpg']
test_labels = [os.path.join('/content/Dataset/valid_images/', x) for x in os.listdir('/content/Dataset/valid_images/') if x[-3:] == "txt"]

train_images.sort()
train_labels.sort()

test_images.sort()
test_labels.sort()

# Split the test dataset into valid-test dataset.
val_images, test_images, val_label, test_label = train_test_split(test_images, test_labels, test_size = 0.5, random_state = 1)

# check how many image have each categories
len(train_images),len(train_labels),len(val_images),len(val_label),len(test_images),len(test_label)

# OUTPUT 🔻
#  (5993, 5993, 2897, 2897, 2897, 2897)
```

### Create Some Folder

I create some folders to store images and labels separately.

```python
!mkdir bird_species  bird_species/train bird_species/train/images bird_species/train/labels 
!mkdir bird_species/val bird_species/val/images bird_species/val/labels
!mkdir bird_species/test bird_species/test/images bird_species/test/labels
```

### Move File To Their Specific Folder

If the folder is ready now move all images and labels into the recently created folder.

```python
#Utility function to move images 
def move_files(list_of_files, dst_folder):
    for f in list_of_files:
        try:
            shutil.move(f, dst_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders
move_files(train_images, 'bird_species/train/images/')
print('train')
move_files(val_images, 'bird_species/val/images/')
print('val')
move_files(test_images, 'bird_species/test/images/')
print('test')
move_files(train_labels, 'bird_species/train/labels/')
print('train label')
move_files(val_label, 'bird_species/val/labels/')
print('val_label')
move_files(test_label, 'bird_species/test/labels/')
print('test_label')
```

Now the next interesting part is to start!

Create a YAML file because when I use the YOLOv8 model this time need.

### Create YAML File

It’s an actual file for passing through when the model train. In this file, you can specify all class and length names, and also specify the image path of the train, val, and test image. It’s called an actual label file for the YOLO model.

I am creating a YAML file and the name is **bird\_spacies.yaml.**

```python
train: /content/bird_species/train/images
val: /content/bird_species/val/images
test: /content/bird_species/test/images

nc: 200

# Classes
names: ['Acadian Flycatcher', 'American Crow', 'American Goldfinch',
       'American Pipit', 'American Redstart',
       'American Three Toed Woodpecker', 'Anna Hummingbird', 'Artic Tern',
       'Baird Sparrow', 'Baltimore Oriole', 'Bank Swallow',
       'Barn Swallow', 'Bay Breasted Warbler', 'Belted Kingfisher',
       'Bewick Wren', 'Black And White Warbler', 'Black Billed Cuckoo',
       'Black Capped Vireo', 'Black Footed Albatross', 'Black Tern',
       'Black Throated Blue Warbler', 'Black Throated Sparrow',
       'Blue Grosbeak', 'Blue Headed Vireo', 'Blue Jay',
       'Blue Winged Warbler', 'Boat Tailed Grackle', 'Bobolink',
       'Bohemian Waxwing', 'Brandt Cormorant', 'Brewer Blackbird',
       'Brewer Sparrow', 'Bronzed Cowbird', 'Brown Creeper',
       'Brown Pelican', 'Brown Thrasher', 'Cactus Wren',
       'California Gull', 'Canada Warbler', 'Cape Glossy Starling',
       'Cape May Warbler', 'Cardinal', 'Carolina Wren', 'Caspian Tern',
       'Cedar Waxwing', 'Cerulean Warbler', 'Chestnut Sided Warbler',
       'Chipping Sparrow', 'Chuck Will Widow', 'Clark Nutcracker',
       'Clay Colored Sparrow', 'Cliff Swallow', 'Common Raven',
       'Common Tern', 'Common Yellowthroat', 'Crested Auklet',
       'Dark Eyed Junco', 'Downy Woodpecker', 'Eared Grebe',
       'Eastern Towhee', 'Elegant Tern', 'European Goldfinch',
       'Evening Grosbeak', 'Field Sparrow', 'Fish Crow', 'Florida Jay',
       'Forsters Tern', 'Fox Sparrow', 'Frigatebird', 'Gadwall',
       'Geococcyx', 'Glaucous Winged Gull', 'Golden Winged Warbler',
       'Grasshopper Sparrow', 'Gray Catbird', 'Gray Crowned Rosy Finch',
       'Gray Kingbird', 'Great Crested Flycatcher', 'Great Grey Shrike',
       'Green Jay', 'Green Kingfisher', 'Green Tailed Towhee',
       'Green Violetear', 'Groove Billed Ani', 'Harris Sparrow',
       'Heermann Gull', 'Henslow Sparrow', 'Herring Gull',
       'Hooded Merganser', 'Hooded Oriole', 'Hooded Warbler',
       'Horned Grebe', 'Horned Lark', 'Horned Puffin', 'House Sparrow',
       'House Wren', 'Indigo Bunting', 'Ivory Gull', 'Kentucky Warbler',
       'Laysan Albatross', 'Lazuli Bunting', 'Le Conte Sparrow',
       'Least Auklet', 'Least Flycatcher', 'Least Tern',
       'Lincoln Sparrow', 'Loggerhead Shrike', 'Long Tailed Jaeger',
       'Louisiana Waterthrush', 'Magnolia Warbler', 'Mallard',
       'Mangrove Cuckoo', 'Marsh Wren', 'Mockingbird', 'Mourning Warbler',
       'Myrtle Warbler', 'Nashville Warbler',
       'Nelson Sharp Tailed Sparrow', 'Nighthawk', 'Northern Flicker',
       'Northern Fulmar', 'Northern Waterthrush',
       'Olive Sided Flycatcher', 'Orange Crowned Warbler',
       'Orchard Oriole', 'Ovenbird', 'Pacific Loon', 'Painted Bunting',
       'Palm Warbler', 'Parakeet Auklet', 'Pelagic Cormorant',
       'Philadelphia Vireo', 'Pied Billed Grebe', 'Pied Kingfisher',
       'Pigeon Guillemot', 'Pileated Woodpecker', 'Pine Grosbeak',
       'Pine Warbler', 'Pomarine Jaeger', 'Prairie Warbler',
       'Prothonotary Warbler', 'Purple Finch', 'Red Bellied Woodpecker',
       'Red Breasted Merganser', 'Red Cockaded Woodpecker',
       'Red Eyed Vireo', 'Red Faced Cormorant', 'Red Headed Woodpecker',
       'Red Legged Kittiwake', 'Red Winged Blackbird',
       'Rhinoceros Auklet', 'Ring Billed Gull', 'Ringed Kingfisher',
       'Rock Wren', 'Rose Breasted Grosbeak', 'Ruby Throated Hummingbird',
       'Rufous Hummingbird', 'Rusty Blackbird', 'Sage Thrasher',
       'Savannah Sparrow', 'Sayornis', 'Scarlet Tanager',
       'Scissor Tailed Flycatcher', 'Scott Oriole', 'Seaside Sparrow',
       'Shiny Cowbird', 'Slaty Backed Gull', 'Song Sparrow',
       'Sooty Albatross', 'Spotted Catbird', 'Summer Tanager',
       'Swainson Warbler', 'Tennessee Warbler', 'Tree Sparrow',
       'Tree Swallow', 'Tropical Kingbird', 'Vermilion Flycatcher',
       'Vesper Sparrow', 'Warbling Vireo', 'Western Grebe',
       'Western Gull', 'Western Meadowlark', 'Western Wood Pewee',
       'Whip Poor Will', 'White Breasted Kingfisher',
       'White Breasted Nuthatch', 'White Crowned Sparrow',
       'White Eyed Vireo', 'White Necked Raven', 'White Pelican',
       'White Throated Sparrow', 'Wilson Warbler', 'Winter Wren',
       'Worm Eating Warbler', 'Yellow Bellied Flycatcher',
       'Yellow Billed Cuckoo', 'Yellow Breasted Chat',
       'Yellow Headed Blackbird', 'Yellow Throated Vireo',
       'Yellow Warbler']
```

### Model Train Them

If the YAML file is ready now move on and train the YOLOv8 model.

```python
# Load the model.
model = YOLO('yolov8n.pt')
 
# Training.
model.train(
   data='/content/drive/MyDrive/new_article/Bird-species/bird_spacies.yaml',
   imgsz=640, 
   epochs=10, # In this time I am only using 10 epoch for training. If you try with a little bit more.
   batch=8, 
   name='yolov8_bird_species'
)
```

### Model Evaluate With Unseen Data

If the model train is complete. And now see what model performance on the unseen dataset.💯

```python
import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding
# Above code required for me. If first try to run without above code
!yolo task=detect mode=val model=/content/runs/detect/yolov8_bird_species3/weights/best.pt name=yolov8s_eval data=/content/drive/MyDrive/new_article/Bird-species/bird_spacies.yaml imgsz=640
```

### Model Predict On Test Image

In this section, I check model prediction testing images. 🍃

```python
from ultralytics import YOLO
import os
import random


file_dir = os.listdir("/content/bird_species/test/images/")

# Load the best model.
model = YOLO("/content/drive/MyDrive/new_article/Bird-species/best.pt")

for i in range(2):
  image = random.choice(file_dir)
  full_path = f"/content/bird_species/test/images/{image}/"
  print(full_path)
  result = model.predict(full_path,save=True)
  print(result)
```

### **Output Our Model Prediction💩**

All the predicted image files are stored in `(runs/detect/predict)` folder under that folder created by YOLO automatically. You can open this folder and see what the predicted image looks like.

![Two bird, One is sitting tree, one is water. And our object detection model detect](https://cdn.hashnode.com/res/hashnode/image/upload/v1687402032922/213ebaad-b5ac-45fc-8b72-aa462e9c07b8.png align="center")

<div data-node-type="callout">
<div data-node-type="callout-emoji">🔥</div>
<div data-node-type="callout-text">Thanks for reading. I hope you found this article helpful. If you have any questions related to this article or suggestions, comment below. I try my best to answer your all question. You can also use my <a target="_blank" rel="noopener noreferrer nofollow" href="https://twitter.com/hi_sushanta_" style="pointer-events: none">@Twitter </a>account to ask questions ( if the question is private ).</div>
</div>