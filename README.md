# charmCam
An attempt for a linux version of beauty/AR filters similair to the discontinued snapCamera for windows/macOS 
mainly iterested in light beautification filters, why should linux users not look their best on video calls?

current goals:

1) face detection for live video feed (must be able to handle natural head movement and be fast)
looking at mediapipe - DONE used mediapipe

2)figure out how to input the video feed from webcam device and not use video file - DONE (look at skipping some frames, it is a bit jumpy)

3)AR beautification filters, machine learning problem... looking for minimal beautification (maybe this https://ai.google.dev/edge/mediapipe/solutions/vision/face_stylizer/python)

4)put it all together, how to get filter output as webcam feed for video applications (zoom, skype, etc.)

5)make it pretty

6)Easy way to make and apply new filters and user friendly cli or maybe a gui 

# Current Issues

mediapipe FaceStylizer is not working, code keeps seg faulting ... looking at tensorRT or mediapipe issues (segfault.txt has gdb backtrace output)

Need to clean up repo, it is getting very messy :( 
