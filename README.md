# Real-Time-Face-Recognition
Face Recognition Using OpenCV
This project aims to implement a real-time face recognition system using OpenCV. Face recognition is a crucial technology in various applications, ranging from security systems to personalized user experiences. Leveraging a Raspberry Pi with a PiCam, this project captures images of faces, processes them using the Haar Cascade classifier for detection, and recognizes them using the Local Binary Patterns Histograms (LBPH) algorithm.

The project is divided into three main phases: data gathering, training, and real-time recognition. In the first phase, the system captures images of the user's face and stores them in a dataset. These images are then used to train the recognition model in the second phase. Finally, the trained model is employed to identify faces in real-time from video input, displaying the recognized user's name on the screen.

Using the LBPH algorithm, this project ensures effective recognition even in varied lighting conditions. The choice of OpenCV allows for a powerful and efficient implementation, making it well-suited for embedded systems like the Raspberry Pi. With this setup, the project provides a low-cost yet robust solution for face recognition, ideal for applications in home automation, security, and personalized access systems.
