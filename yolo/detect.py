import argparse
from ultralytics import YOLO
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", dest="modelPath", help="Path al modelo", default="yolo11n.pt", type=str )
args = parser.parse_args()

confianza_minima = 72
model = YOLO(args.modelPath)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Paso 1: Predicción directa sobre el frame original
    # YOLO se encarga internamente de redimensionar para la inferencia,
    # pero te devuelve los resultados escalados a la imagen que le pasas.
    results = model(frame, verbose=False) 

    for r in results:
        # box.xyxy devuelve: x_min, y_min, x_max, y_max
        for box in r.boxes:
            confidence = box.conf[0] * 100
            
            if confidence >= confianza_minima:
                # Obtenemos coordenadas directas para dibujar
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                clase = model.names.get(cls)

                # Paso 2: Dibujar sobre el frame original (sin desfasar)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{clase} ({confidence:.1f}%)"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Detectando... [q=salir!]', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()