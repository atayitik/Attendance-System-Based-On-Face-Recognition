import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import cv2
import face_recognition
import numpy as np
import os
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("500x400")
root.resizable(False, False)

style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12))
style.configure("TLabel", font=("Helvetica", 12))
style.configure("TFrame", background="#f0f0f0")
style.configure("TLabelFrame", background="#f0f0f0")


if not os.path.exists('student_images'):
    os.makedirs('student_images')

if not os.path.exists('student_encodings.npy'):
    np.save('student_encodings.npy', [])
if not os.path.exists('student_names.npy'):
    np.save('student_names.npy', [])


def register_student():
    name = simpledialog.askstring("Input", "Enter student's name:", parent=root)
    surname = simpledialog.askstring("Input", "Enter student's surname:", parent=root)
    student_id = simpledialog.askstring("Input", "Enter student's ID:", parent=root)

    if not name or not surname or not student_id:
        messagebox.showerror("Error", "All fields are required!")
        return

    full_name = f"{name}_{surname}_{student_id}"


    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Info", "Press Q to exit", parent=root)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(f'student_images/{full_name}.jpg', frame)
            break

    cap.release()
    cv2.destroyAllWindows()


    image = face_recognition.load_image_file(f'student_images/{full_name}.jpg')
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        student_encodings = np.load('student_encodings.npy', allow_pickle=True).tolist()
        student_names = np.load('student_names.npy', allow_pickle=True).tolist()

        student_encodings.append(face_encodings[0])
        student_names.append(full_name)

        np.save('student_encodings.npy', student_encodings)
        np.save('student_names.npy', student_names)

        messagebox.showinfo("Success", "Student registered successfully!", parent=root)
    else:
        messagebox.showerror("Error", "No face found in the image. Try again!", parent=root)


def check_attendance():
    cap = cv2.VideoCapture(0)
    known_face_encodings = np.load('student_encodings.npy', allow_pickle=True)
    known_face_names = np.load('student_names.npy', allow_pickle=True)
    attendance_list = []

    messagebox.showinfo("Info", "Press Q to exit ", parent=root)

    while True:
        ret, frame = cap.read()
        rgb_frame = frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                if name not in attendance_list:
                    attendance_list.append(name)

        for (top, right, bottom, left), name in zip(face_locations, attendance_list):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imshow('Attendance Check', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if attendance_list:
        messagebox.showinfo("Attendance", f"Attendance: {', '.join(attendance_list)}", parent=root)
    else:
        messagebox.showinfo("Attendance", "No one is present.", parent=root)
frame = ttk.Frame(root, padding="10 10 10 10")
frame.pack(expand=True)
logo_image = Image.open("bahcesehir_logo.png")
logo_image = logo_image.resize((100, 100), Image.ANTIALIAS)
logo_photo = ImageTk.PhotoImage(logo_image)
logo_label = ttk.Label(frame, image=logo_photo, background="#f0f0f0")
logo_label.pack(pady=(0, 10))
university_label = ttk.Label(frame, text="Bahcesehir University", font=("Helvetica", 16, "bold"), background="#f0f0f0")
university_label.pack(pady=(0, 20))
title_label = ttk.Label(frame, text="Face Recognition Attendance System", font=("Helvetica", 16, "bold"))
title_label.pack(pady=(0, 20))
register_button = ttk.Button(frame, text="Register New Student", command=register_student)
register_button.pack(pady=10)
attendance_button = ttk.Button(frame, text="Check Attendance", command=check_attendance)
attendance_button.pack(pady=10)
root.mainloop()
