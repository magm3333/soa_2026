import cv2
import mediapipe as mp
import numpy as np
import math
import sys
from time import sleep

# Importación robusta para tu entorno
try:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_drawing
except ImportError:
    print("Error: No se pudo cargar MediaPipe. Revisa la instalación.")
    sys.exit()

cap = cv2.VideoCapture(0)
pts = []
isClosed = False
color = (255, 0, 0) # Azul
thickness = 2

# Iniciamos el detector
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, # Cambiado a 1 para mayor precisión al dibujar
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:
    
    dibujar = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:    
            for hand_landmarks in results.multi_hand_landmarks:
                # Obtenemos coordenadas de pulgar (4) e índice (8)
                # Landmark 8: Índice
                p8 = hand_landmarks.landmark[8]
                x8, y8 = int(p8.x * width), int(p8.y * height)
                
                # Landmark 4: Pulgar
                p4 = hand_landmarks.landmark[4]
                x4, y4 = int(p4.x * width), int(p4.y * height)

                # Lógica de dibujo
                if dibujar:
                    pts.append([x8, y8])
                    cv2.circle(frame, (x8, y8), 5, (0, 255, 0), -1) # Punto guía verde
                
                # Dibujar la línea acumulada
                if len(pts) > 1:
                    pts_array = np.array(pts, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts_array], isClosed, color, thickness)

                # Dibujar círculos en las puntas para feedback visual
                cv2.circle(frame, (x4, y4), 5, (0, 0, 255), -1) # Pulgar Rojo

                # Cálculo de distancia Euclídea
                distancia = math.hypot(x8 - x4, y8 - y4)
                
                # "Gesto de pinza" para activar/desactivar dibujo
                if distancia < 30: # 30 laxo para facilitar la detección
                    dibujar = not dibujar
                    # Feedback visual de cambio de estado
                    cv2.circle(frame, (int((x8+x4)/2), int((y8+y4)/2)), 15, (0, 255, 255), -1)
                    cv2.waitKey(1)
                    sleep(0.4) 

        cv2.imshow('Manos! (q=salir)', frame)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break

cap.release()
cv2.destroyAllWindows()