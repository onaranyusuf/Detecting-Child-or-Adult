from google.colab import drive
drive.mount('/content/drive')

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

model = load_model('/content/drive/MyDrive/COCUK - YETISKIN AYIRT ETME/Cocuk - Yetiskin .h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
image = Image.open('/content/drive/MyDrive/COCUK - YETISKIN AYIRT ETME/Datalarim/Çocuk/1 - TEST/istockphoto-511932886-612x612.jpg')
#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

#turn the image into a numpy array
image_array = np.asarray(image)
# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# Load the image into the array
data[0] = normalized_image_array

# run the inference
prediction = model.predict(data)
print(prediction)

#liste tek elemanlı. ilk eleman bizim 2 outputumuz
res = prediction[0]
# 3 haneye yuvarlama
yuvarla = ["%.3f" % x for x in res]
#virgülle ayırdıktan sonra yeni liste içine atama
new_list = []
for item in yuvarla:
    new_list.append(float(item))
#ilk ve ikinci eleman olarak outputları isimlendirme
first_elem = new_list[0]
second_elem = new_list[1]
#ondalık hali yazdırma
print(first_elem)
print(second_elem)
#yuvarlanmış yüzdelik hali
print("ÇOCUK:    {0:.0%}".format(first_elem))
print("YETİŞKİN: {0:.0%}".format(second_elem))