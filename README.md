# Automated Attendance Tracker Using Face Recognition and GUI

## Project Overview

The Automated Attendance Tracker is a real-time face recognition-based
attendance system designed to eliminate manual attendance processes. The
system detects and recognizes faces using computer vision techniques and
automatically records attendance through a graphical user interface
(GUI).

This project improves accuracy, reduces proxy attendance, and saves time
compared to traditional attendance systems.

## Features

-   Real-time face detection using Haar Cascade
-   Face recognition using LBPH algorithm
-   Automatic attendance marking
-   User-friendly GUI interface
-   Dataset creation and model training
-   Attendance stored in CSV/Excel files

## Technologies Used

-   Python
-   OpenCV
-   Tkinter
-   NumPy
-   Pandas

## Project Structure

Automated-Attendance-Tracker │ ├── dataset/ \# Face images dataset ├──
trainer/ \# Trained face recognition model ├── attendance/ \# Attendance
records ├── main.py \# Main GUI application ├── train.py \# Model
training script ├── haarcascade_frontalface_default.xml └── README.md

## Installation

1.  Clone the repository git clone
    https://github.com/sivanagakarthik/Automated-Attendance-Tracker-Using-Face-Recognition-and-GUI.git

2.  Install required libraries pip install -r requirements.txt

3.  Run the application python main.py

## Working Process

1.  Capture face images and create a dataset.
2.  Train the model using LBPH face recognizer.
3.  Start the real-time recognition system.
4.  Recognized faces are automatically recorded in the attendance file.

## Future Improvements

-   Cloud database integration
-   Mobile app integration
-   Deep learning-based face recognition
-   Multi-camera support

## Author

Movva Siva Naga Karthik\
B.Tech Computer Science and Engineering\
Vel Tech R&D Institute of Science and Technology

GitHub: https://github.com/sivanagakarthik\
LinkedIn: https://www.linkedin.com/in/movva-sivanagakarthik
