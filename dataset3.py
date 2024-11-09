import cv2
import os

# Load Haar cascade
haar_cascade_path = 'C:/Users/ritik/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(haar_cascade_path)

# Validate if Haar cascade loaded properly
if faceCascade.empty():
    print("Error: Could not load Haar cascade.")
    exit(1)

# Setup video capture
cap = cv2.VideoCapture(0)  # Default webcam
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

face_id = input('\nEnter user ID and press <return> ==> ')
print("\n[INFO] Initializing face capture. Look at the camera and wait...")

count = 0
while True:
    ret, img = cap.read()  # Capture frame

    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)  # Adjust if needed
    )

    # Draw rectangles and save images
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle
        count += 1
        
        # Ensure the dataset directory exists
        dataset_dir = "C:/Users/ritik/face recognition using open cv/online/dataset"
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        # Save the captured images
        img_path = f"{dataset_dir}/User.{face_id}.{count}.jpg"
        cv2.imwrite(img_path, gray[y:y + h, x:x + w])  # Save face image
        
        cv2.imshow('video', img)  # Display the image

    # Exit on 'ESC' key or when 30 samples are collected
    if cv2.waitKey(30) & 0xFF == 27:
        break
    elif count >= 100:
        break

# Cleanup
cap.release()

cv2.destroyAllWindows()
