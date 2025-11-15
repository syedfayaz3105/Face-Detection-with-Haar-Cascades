# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

### Program:
```
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

# ---- Load images ----
withglass = cv2.imread('glassimage.jpg', 0)
group = cv2.imread('group.jpg', 0)   # FIXED the filename

# ---- Check if images loaded ----
if withglass is None:
    raise ValueError("Error: 'glassimage.jpg' not found or cannot be loaded.")

if group is None:
    raise ValueError("Error: 'group.jpg' not found or cannot be loaded. Check filename!")

# ---- Show images ----
plt.imshow(withglass, cmap='gray')
plt.title("With Glasses")
plt.show()

plt.imshow(group, cmap='gray')
plt.title("Group Image")
plt.show()

# ---- Load Cascade Classifiers ----
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty():
    raise IOError("Error loading face cascade XML file")
if eye_cascade.empty():
    raise IOError("Error loading eye cascade XML file")

# ---- Detection Functions ----
def detect_face(img, scaleFactor=1.1, minNeighbors=5):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    return face_img

def detect_eyes(img):
    face_img = img.copy()
    eyes = eye_cascade.detectMultiScale(face_img)
    for (x, y, w, h) in eyes:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    return face_img

# ---- Face detection ----
result_withglass_faces = detect_face(withglass)
plt.imshow(result_withglass_faces, cmap='gray')
plt.title("Faces in With Glasses Image")
plt.show()

result_group_faces = detect_face(group)
plt.imshow(result_group_faces, cmap='gray')
plt.title("Faces in Group Image")
plt.show()

# ---- Eye detection ----
result_withglass_eyes = detect_eyes(withglass)
plt.imshow(result_withglass_eyes, cmap='gray')
plt.title("Eyes in With Glasses Image")
plt.show()

result_group_eyes = detect_eyes(group)
plt.imshow(result_group_eyes, cmap='gray')
plt.title("Eyes in Group Image")
plt.show()

```


## OUTPUT:

<img width="837" height="500" alt="Screenshot 2025-11-15 104418" src="https://github.com/user-attachments/assets/1bcab0a8-80e4-4801-83bc-e2ce20655c4a" />

<img width="795" height="520" alt="Screenshot 2025-11-15 104432" src="https://github.com/user-attachments/assets/610a869e-e8e5-4054-a859-e635f3508efc" />

<img width="781" height="514" alt="Screenshot 2025-11-15 104443" src="https://github.com/user-attachments/assets/6683aafd-c6e4-4dbe-b678-7b8498650e9f" />
<img width="727" height="520" alt="Screenshot 2025-11-15 104601" src="https://github.com/user-attachments/assets/7c983f2e-1068-4bdf-8c6b-cd3a740ee96b" />
<img width="739" height="487" alt="Screenshot 2025-11-15 104458" src="https://github.com/user-attachments/assets/2ef8568b-e73c-42d7-b8fd-bdc98ecc286d" />
<img width="769" height="519" alt="Screenshot 2025-11-15 104637" src="https://github.com/user-attachments/assets/f8c657ab-9b90-44b0-938e-e875d59b99f6" />

## result:
thus the program executed succesfully.
