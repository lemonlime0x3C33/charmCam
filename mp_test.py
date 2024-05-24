#TEST FILE BASED ON https://www.assemblyai.com/blog/mediapipe-for-dummies/
#also used this https://ai.google.dev/edge/mediapipe/solutions/tasks

import cv2
import mediapipe as mp
import urllib.request
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import PyQt5
from PIL import Image
from IPython.display import Video
#from IPython.display import display
import nb_helpers
import sys

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

#Camera Setup (need to make this command line argument)
#This is your camera, for mine the internal bad camera is index 0, the good usb camera is 2, there is no camera at index 1
camera_id = 2
delay = 1
window_name = 'frame'

cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print("ERROR: There is no camera at index %s" % camera_id)
    sys.exit()

#leftover from static image testing
#img = Image.open('/home/lemonlime/Downloads/shutterstock_141020407.jpg')
#display(img)
#img.show()

# Define image filename and drawing specifications -- no image using webcam now
#file = '/home/lemonlime/Downloads/shutterstock_141020407.jpg'
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


while True:

    # Create a face mesh object
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        # Read image file with cv2 and process with face_mesh
        #image = cv2.imread(file)
        ret, frame = cap.read()
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Define boolean corresponding to whether or not a face was detected in the image
    face_found = bool(results.multi_face_landmarks)

    if face_found:
        # Create a copy of the image
        #annotated_image = frame.copy()

        # Draw landmarks on face
        mp_drawing.draw_landmarks(
            #image=annotated_image,
            image=frame,
            landmark_list=results.multi_face_landmarks[0],
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        
        # Save image
        #cv2.imwrite('face_tesselation_only.png', annotated_image)
        cv2.imshow(window_name,frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

cv2.destroyWindow(window_name)


    # Open image
    #img = Image.open('face_tesselation_only.png')
    #display(img)
    #img.show()

if face_found:
    # Create a copy of the image
    annotated_image = image.copy()
    
    # For each face in the image (only one in this case)
    for face_landmarks in results.multi_face_landmarks:
        
        # Draw the facial contours of the face onto the image
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
        
        # Draw the iris location boxes of the face onto the image
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())

	# Save the image
    cv2.imwrite('face_contours_and_irises.png', annotated_image)

img = Image.open('face_contours_and_irises.png')
#display(img)
img.show()
