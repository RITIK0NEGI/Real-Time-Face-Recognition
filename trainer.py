import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
dataset_path = 'C:/Users/ritik/face recognition using open cv/online/dataset'
# Ensure the dataset path exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")

# Create recognizer and detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
cascade_path = "C:/Users/ritik/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascade_path)

# Check if the cascade classifier is loaded
if detector.empty():
    raise IOError(f"Could not load Haar cascade file from '{cascade_path}'.")

# Function to get the images and label data
def getImagesAndLabels(path):
    # Ensure the dataset path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path '{path}' does not exist.")

    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    faceSamples = []
    ids = []
    
    for imagePath in imagePaths:
        # Check if file exists
        if not os.path.isfile(imagePath):
            continue
        
        # Convert image to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        # Extract ID from filename (assuming 'User.ID.Image.jpg')
        id_parts = os.path.basename(imagePath).split(".")
        if len(id_parts) < 3:
            continue
        
        try:
            id = int(id_parts[1])
        except ValueError:
            continue
        
        # Detect faces in the image
        faces = detector.detectMultiScale(img_numpy)
        
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids


print("\n [INFO] Training faces. It will take a few seconds. Wait ...")

faces, ids = getImagesAndLabels(dataset_path)

# Ensure that there are faces and ids to train
if len(faces) == 0 or len(ids) == 0:
    raise ValueError("No faces or IDs found to train.")

recognizer.train(faces, np.array(ids))

# Create the trainer directory if it doesn't exist
trainer_path = 'C:/Users/ritik/face recognition using open cv/online/trainer'
os.makedirs(trainer_path, exist_ok=True)

# Save the model into trainer/trainer.yml
recognizer.write(os.path.join(trainer_path, 'trainer.yml'))

# Print the number of faces trained and end program
print(f"\n [INFO] {len(np.unique(ids))} faces trained. Exiting Program.")
