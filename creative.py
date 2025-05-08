import tkinter as tk
from tkinter import Label, Button, messagebox, ttk
from PIL import Image, ImageTk
import threading
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer
import time
import random
from ttkthemes import ThemedStyle

# Initialize mixer for alert sound
mixer.init()
mixer.music.load("C:\\Users\\Bhargavi\\Downloads\\music.wav")

# EAR calculation function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Dlib setup
thresh = 0.25
frame_check = 20
flag = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\Bhargavi\\Downloads\\shape_predictor_68_face_landmarks (1).dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Global variables
running = False
alerts_triggered = 0
start_time = None

# Tkinter setup
root = tk.Tk()
root.title("Driver Drowsiness Detection - Enhanced")

# Set window to fullscreen
root.attributes("-fullscreen", True)  # Makes the window full-screen
root.configure(bg="#f0f0f5")  # Light gray background color

# Add background image
bg_image = Image.open("C:\\Users\\Bhargavi\\Downloads\\17.jpg")  # Provide your background image path here
bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.Resampling.LANCZOS)  # Resize to screen size
bg_photo = ImageTk.PhotoImage(bg_image)

# Retain the reference to avoid the image being garbage collected
bg_label = Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)  # Set background to cover the entire window

# Themed style
style = ThemedStyle(root)
style.set_theme("breeze")

# Gauge and EAR status
ear_label = ttk.Label(root, text="Eye-Aspect-Ratio: --", font=("Helvetica", 16), background="#f0f0f5", foreground="#000000")
ear_label.pack(pady=10)

alert_label = ttk.Label(root, text="Alert Status: Normal", font=("Helvetica", 16), foreground="green", background="#f0f0f5")
alert_label.pack(pady=10)

# Dashboard
dashboard_label = ttk.Label(root, text="Sleepiness Dashboard", font=("Helvetica", 18, "bold"), background="#f0f0f5", foreground="#000000")
dashboard_label.pack()

stats_label = ttk.Label(root, text="Alerts: 0 | Duration: 0s | Last Alert: N/A", font=("Helvetica", 14), background="#f0f0f5", foreground="#000000")
stats_label.pack()

# Video feed
video_label = Label(root, bg="#f0f0f5")
video_label.pack()

# Fun fact display
fun_fact_label = ttk.Label(root, text="", font=("Helvetica", 14), wraplength=400, background="#f0f0f5", foreground="#000000")
fun_fact_label.pack(pady=20)

def display_fun_fact():
    facts = [
        "Short naps can increase alertness.",
        "Avoid driving when sleepy for safety.",
        "Stay hydrated to stay awake!",
        "Listen to upbeat music while driving."
    ]
    fun_fact_label.config(text=random.choice(facts))
    root.after(10000, display_fun_fact)

display_fun_fact()

def start_detection():
    global running, start_time
    running = True
    start_time = time.time()
    threading.Thread(target=detect_drowsiness).start()

def stop_detection():
    global running
    running = False

def pop_up_alert():
    messagebox.showwarning("Drowsiness Alert", "You are feeling drowsy! Take a break.")

def detect_drowsiness():
    global flag, alerts_triggered
    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detector(gray, 0)

        for subject in subjects:
            shape = predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Update EAR display
            ear_label.config(text=f"Eye-Aspect-Ratio: {ear:.2f}")

            # Drowsiness logic
            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    alerts_triggered += 1
                    alert_label.config(text="Drowsy!", foreground="red")
                    if not mixer.music.get_busy():
                        mixer.music.play()
                    pop_up_alert()
            else:
                flag = 0
                alert_label.config(text="Normal", foreground="green")

            # Stats update
            elapsed_time = int(time.time() - start_time)
            stats_label.config(text=f"Alerts: {alerts_triggered} | Duration: {elapsed_time}s | Last Alert: {time.strftime('%H:%M:%S')}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        video_label.config(image=img)
        video_label.image = img

    cap.release()

# Buttons
start_button = ttk.Button(root, text="Start Detection", command=start_detection)
start_button.pack(pady=10)

stop_button = ttk.Button(root, text="Stop Detection", command=stop_detection)
stop_button.pack(pady=10)

# Style button with a better contrast
style.configure("TButton",
                font=("Helvetica", 16, "bold"),
                padding=10,
                relief="raised",
                background="#4CAF50",  # Soft green background
                foreground="#000000",  # Black text color
                borderwidth=3)
style.map("TButton",
          background=[('active', '#66BB6A')])  # A lighter green for hover effect

# Run the app
root.mainloop()
