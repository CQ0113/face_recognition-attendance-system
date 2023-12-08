import face_recognition as fr
import cv2
import tkinter as tk
import os
import csv
from datetime import datetime

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            self.root.destroy()
            return

        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

        self.attendance_text = tk.Text(root, height=10, width=50)
        self.attendance_text.pack()

        self.start_button = tk.Button(root, text="Start/Stop Recognition", command=self.toggle_recognition)
        self.start_button.pack()

        self.clear_records_button = tk.Button(root, text="Clear Records", command=self.clear_records)
        self.clear_records_button.pack()

        self.register_face_button = tk.Button(root, text="Register New Face", command=self.register_new_face)
        self.register_face_button.pack()

        self.n = ""
        self.known_faces = []
        self.known_names = []

        self.capture_path = r'C:\FACE RECOGNITION PROJECT\face_recognition-attendance-system\Registered_Images'  # Can edit your path here
        os.makedirs(self.capture_path, exist_ok=True)

        self.attendance_file_path = r'C:\FACE RECOGNITION PROJECT\face_recognition-attendance-system\Attendance.csv'
        self.recognition_active = False
        self.current_user = None

        if not os.path.exists(self.attendance_file_path):
            with open(self.attendance_file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Name', 'Timestamp'])

        self.capture_delay = 5  # Set the delay time in seconds
        self.last_capture_time = datetime.now()

        self.registering_face = False
        self.registration_images = []

    def toggle_recognition(self):
        if not self.recognition_active:
            self.recognition_active = True
            self.start_recognition()
        else:
            self.recognition_active = False
            self.result_label.config(text="Recognition Stopped")

    def start_recognition(self):
        while self.recognition_active:
            ret, frame = self.cap.read()

            if not ret:
                print("Error: Failed to grab frame.")
                break

            cv2.imshow("Webcam", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                self.recognition_active = False
            elif key == ord('s') and (datetime.now() - self.last_capture_time).seconds >= self.capture_delay:
                self.n = os.path.join(self.capture_path, "captured_image.jpg")
                cv2.imwrite(self.n, frame)
                print(f"Image saved to {self.n}")
                self.update_attendance()
                self.last_capture_time = datetime.now()

        cv2.destroyAllWindows()

    def update_attendance(self):
        if self.n:
            self.load_known_faces()

            unknown_image = fr.load_image_file(self.n)

            try:
                unknown_face_encoding = fr.face_encodings(unknown_image)[0]
            except IndexError:
                print("I wasn't able to locate any faces in the image. Check the image file. Aborting...")
                return

            results = fr.compare_faces(self.known_faces, unknown_face_encoding, tolerance=0.4)

            if any(results):
                index = results.index(True)
                recognized_name = self.known_names[index]
                self.current_user = recognized_name
                result_text = f"Recognized: {recognized_name}"
                self.result_label.config(text=result_text)
                self.mark_attendance()
                self.display_attendance()

    def mark_attendance(self):
        if self.current_user:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            with open(self.attendance_file_path, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([self.current_user, timestamp])
                print(f"Attendance marked for {self.current_user} at {timestamp}")

    def load_known_faces(self):
        self.known_faces = []
        self.known_names = []

        for file in os.listdir(self.capture_path):
            if file.endswith("_registered_image.jpg"):
                image_path = os.path.join(self.capture_path, file)
                known_image = fr.load_image_file(image_path)
                try:
                    known_face_encoding = fr.face_encodings(known_image)[0]
                    self.known_faces.append(known_face_encoding)
                    self.known_names.append(os.path.splitext(file)[0])
                except IndexError:
                    print(f"I wasn't able to locate any faces in {file}. Check the image file. Aborting...")
                    return

    def clear_records(self):
        # Clear displayed records
        self.attendance_text.delete(1.0, tk.END)

        # Clear historical records in CSV file
        with open(self.attendance_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Name', 'Timestamp'])

        print("Records cleared.")

    def display_attendance(self):
        with open(self.attendance_file_path, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            attendance_records = [row for row in csv_reader]

        self.attendance_text.delete(1.0, tk.END)
        self.attendance_text.insert(tk.END, "Attendance Records:\n")
        for record in attendance_records[1:]:
            self.attendance_text.insert(tk.END, f"{record[0]} - {record[1]}\n")

    def register_new_face(self):
        self.registering_face = True
        self.registration_images = []
        self.capture_registration_images()

    def capture_registration_images(self):
        while self.registering_face:
            ret, frame = self.cap.read()
            cv2.imshow("Register Face - Adjust Angle", frame)

            key = cv2.waitKey(1)
            if key == ord('s'):
                self.registration_images.append(frame)
                print(f"Image {len(self.registration_images)} captured.")
            elif key == ord('q'):
                self.registering_face = False

        # Save the captured images for registration
        if self.registration_images:
            name = input("Enter the name of the person: ")
            for i, img in enumerate(self.registration_images):
                img_path = os.path.join(self.capture_path, f"{name}_registered_image.jpg")
                cv2.imwrite(img_path, img)
            print(f"Image {i+1} for {name} registration saved to {img_path}")

    def quit_app(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
