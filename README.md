# Detecting-Child-or-Adult
I have developed an artificial intelligence that detects whether a person is a child or an adult. You can find source codes and datasets. I used Teachable Machine and Google Colaboratory.

## Run (In Colaboratory)
To run this project, you can use Google Colab.

* Firstly, add your files in Google Drive.
* After that, connect your Google Drive:

```
from google.colab import drive
drive.mount('/content/drive')
```

* Then reach the keras models.
```
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
```
* Then define your model like this:
```
model = load_model('YOUR MODEL PATH HERE!')
```
* In the image part, use your image path:
```
image = Image.open('YOUR IMAGE PATH HERE!')
```
* First_elem gives child rate, second elem gives adult rate.
* In the Turkish, "Çocuk" means "child" and "Yetişkin" means "adult".
* The model was trained with 300 photos. Sample photos are available in the data file. For more precise results, work with more than 1000 data.

### Model
* I'm giving you my own model with .h5 extension, but if you want to train your own model, you can use a "teachable machine".
