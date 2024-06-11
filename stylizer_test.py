

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math
import urllib
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import PyQt5
import tensorflow



#model_path = '/home/lemonlime/Downloads/face_stylizer_color_sketch.task'
IMAGE_FILENAMES = ['business-person.png']

for name in IMAGE_FILENAMES:
  url = f'https://storage.googleapis.com/mediapipe-assets/{name}'
  urllib.request.urlretrieve(url, name)

# Height and width that will be used by the model
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

window_name = 'image'

# Performs resizing and showing the image
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

  cv2.imwrite('test.png', img)
  path = '/home/lemonlime/mein_Projekte/charmCam/test.png'
  image = cv2.imread(path)
  #img = Image.open('test.png')
  #display(img)
  #img.show()
  cv2.imshow(window_name, image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()



# Preview the image(s)
#images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
#for name, image in images.items():
  #print(name)
 # resize_and_show(image)

# Create the options that will be used for FaceStylizer
base_options = python.BaseOptions(model_asset_path='face_stylizer.task')
options = vision.FaceStylizerOptions(base_options=base_options)

# Create the face stylizer
with vision.FaceStylizer.create_from_options(options) as stylizer:

  # Loop through demo image(s)
  for image_file_name in IMAGE_FILENAMES:

    # Create the MediaPipe image file that will be stylized
    image = mp.Image.create_from_file(image_file_name)
    # Retrieve the stylized image
    stylized_image = stylizer.stylize(image)

    # Show the stylized image
    rgb_stylized_image = cv2.cvtColor(stylized_image.numpy_view(), cv2.COLOR_BGR2RGB)
    resize_and_show(rgb_stylized_image)
