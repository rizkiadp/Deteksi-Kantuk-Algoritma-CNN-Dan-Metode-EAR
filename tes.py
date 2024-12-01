import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model
from scipy.spatial import distance as dist
import pygame
import time

# Fungsi untuk menghitung Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # Menghitung jarak vertikal antara titik-titik mata
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Menghitung jarak horizontal antara titik mata
    C = dist.euclidean(eye[0], eye[3])
    # EAR = (A + B) / (2.0 * C)
    ear = (A + B) / (2.0 * C)
    return ear

# Menyeting variabel untuk deteksi mata
EYE_AR_THRESH = 0.2  # Ambang batas EAR untuk mata tertutup
EYE_AR_CONSEC_FRAMES = 30  # Jumlah frame mata tertutup berturut-turut untuk mendeteksi kantuk
COUNTER = 0  # Counter untuk mata tertutup berturut-turut

# Load model CNN yang sudah dilatih
model = load_model('model (1).h5')  # Gantilah dengan path model Anda

# Load detektor wajah dan predictor landmark dari dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Pastikan file predictor ada

# Inisialisasi pygame untuk memainkan alarm
pygame.mixer.init()
sound = pygame.mixer.Sound('MENGANTUK.mp3')  # Gantilah dengan path file audio Anda

# Variabel untuk mengontrol alarm
alarm_played = False  # Alarm hanya dimainkan sekali

# Inisialisasi video capture
cap = cv2.VideoCapture(1)  # Menggunakan kamera default

while True:
    # Membaca frame dari kamera
    ret, frame = cap.read()
    
    # Ubah frame ke grayscale untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Deteksi wajah
    faces = detector(gray)
    
    for face in faces:
        # Mendapatkan landmark wajah
        landmarks = predictor(gray, face)
        
        # Mendapatkan koordinat mata kiri dan kanan
        left_eye = []
        right_eye = []
        
        for i in range(36, 42):  # Mata kiri (landmark 36-41)
            left_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        
        for i in range(42, 48):  # Mata kanan (landmark 42-47)
            right_eye.append((landmarks.part(i).x, landmarks.part(i).y))
        
        # Menghitung EAR untuk mata kiri dan kanan
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0  # Rata-rata EAR
        
        # Menampilkan EAR pada frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Gambar garis yang mengikuti bentuk mata kiri dan kanan
        # Gambar garis untuk mata kiri
        left_eye_points = np.array(left_eye, dtype=np.int32)
        cv2.polylines(frame, [left_eye_points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Gambar garis untuk mata kanan
        right_eye_points = np.array(right_eye, dtype=np.int32)
        cv2.polylines(frame, [right_eye_points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Deteksi mata tertutup
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # Jika mata tertutup lebih dari batas waktu, beri peringatan
                cv2.putText(frame, "ALERT: Drowsy!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Ambil gambar wajah untuk klasifikasi dengan model CNN
                face_region = frame[face.top():face.bottom(), face.left():face.right()]
                face_resized = cv2.resize(face_region, (80, 80))  # Ukuran sesuai dengan input CNN
                face_normalized = face_resized / 255.0  # Normalisasi
                face_input = np.expand_dims(face_normalized, axis=0)  # Tambahkan dimensi batch
                
                # Prediksi menggunakan CNN
                prediction = model.predict(face_input)
                class_idx = np.argmax(prediction)  # Indeks kelas
                class_label = 'Drowsy' if class_idx == 1 else 'Not Drowsy'
                
                # Menampilkan hasil klasifikasi pada frame
                cv2.putText(frame, f"Status: {class_label}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Memainkan alarm hanya sekali jika pengemudi terdeteksi ngantuk
                if class_label == 'Drowsy' and not alarm_played:
                    sound.play()
                    alarm_played = True  # Menandakan alarm sudah diputar
                
        else:
            COUNTER = 0
            alarm_played = False  # Reset alarm jika pengemudi sudah sadar
        
    # Tampilkan frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Keluar jika tekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan objek video capture dan tutup jendela
cap.release()
cv2.destroyAllWindows()
