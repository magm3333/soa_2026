#sudo apt-get update
#sudo apt-get install -y cmake build-essential libboost-all-dev libgraphicsmagick1-dev libjpeg-dev
#pip install face-recognition

import cv2
import face_recognition
cap = cv2.VideoCapture(0)
num = 0
color = (50, 50, 255)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    # Creamos una copia LIMPIA para guardar sin rectángulos
    frame_save = frame.copy()
    # Detectar caras
    face_locs = face_recognition.face_locations(frame)
    # Dibujar rectángulos SOLO en la imagen mostrada
    for fl in face_locs:
        top = fl[0]
        right = fl[1]
        bottom = fl[2]
        left = fl[3]
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            color,
            2
        )
    # Generar embeddings
    face_rec = face_recognition.face_encodings(
        frame,
        known_face_locations=face_locs
    )
    print(face_rec)
    # Mostrar video con rectángulos
    cv2.imshow('c=capturar - q=salir', frame)
    key = cv2.waitKey(1)
    # Salir
    if ord('q') == key:
        break
    # Capturar imagen SIN rectángulos
    if ord('c') == key:
        filename = f'images/image{num}.jpg'
        num += 1
        cv2.imwrite(filename, frame_save)
        print(f'Se capturó {filename}')
cap.release()
cv2.destroyAllWindows()