#pip install mediapipe==0.10.14 opencv-python numpy
import cv2
import sys

# Importación robusta: Forzamos la ruta que funcionó en tu PC
try:
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except ImportError as e:
    print(f"Error: No se pudo cargar MediaPipe. Detalles: {e}")
    sys.exit()

# Inicializamos la captura de video
cap = cv2.VideoCapture(0)

# Configuramos el detector usando el administrador de contexto 'with'
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Procesamiento de imagen
        frame = cv2.flip(frame, 1) # Espejo
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detección
        results = hands.process(frame_rgb)

        # Dibujo de resultados
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujamos las conexiones (líneas) y los puntos (nodos)
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    # Estilo de los puntos (Cian)
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=5),
                    # Estilo de las líneas (Magenta)
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=5)
                )

        # Mostrar el frame
        cv2.imshow('Deteccion de Manos (q=salir)', frame)

        # Salir con 'Esc' o 'q'
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

"""
0 = MUÑECA

PULGAR
1 = ARTICULACIÓN CMC DEL PULGAR
2 = ARTICULACIÓN MCP DEL PULGAR
3 = ARTICULACIÓN IP DEL PULGAR
4 = PUNTA DEL PULGAR

ÍNDICE
5 = ARTICULACIÓN MCP DEL ÍNDICE
6 = ARTICULACIÓN PIP DEL ÍNDICE
7 = ARTICULACIÓN DIP DEL ÍNDICE
8 = PUNTA DEL ÍNDICE

MEDIO
9  = ARTICULACIÓN MCP DEL DEDO MEDIO
10 = ARTICULACIÓN PIP DEL DEDO MEDIO
11 = ARTICULACIÓN DIP DEL DEDO MEDIO
12 = PUNTA DEL DEDO MEDIO

ANULAR
13 = ARTICULACIÓN MCP DEL ANULAR
14 = ARTICULACIÓN PIP DEL ANULAR
15 = ARTICULACIÓN DIP DEL ANULAR
16 = PUNTA DEL ANULAR

MEÑIQUE
17 = ARTICULACIÓN MCP DEL MEÑIQUE
18 = ARTICULACIÓN PIP DEL MEÑIQUE
19 = ARTICULACIÓN DIP DEL MEÑIQUE
20 = PUNTA DEL MEÑIQUE


        8   12   16   20
        |    |    |    |
        7   11   15   19
        |    |    |    |
        6   10   14   18
         \   |   /    /
          5  9  13   17
           \ | /    /
             0
          /  |  \
         1   2   3 -- 4

"""