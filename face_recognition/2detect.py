import cv2
import argparse
import os
import json
import face_recognition


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--names", help="Nombre/s perfil/es (separados por coma si hay más de uno",
                    default='names', type=str)
args = parser.parse_args()
PROFILES = args.names

if PROFILES == '':
    print('Argumentos erróneos')
    quit()

PROFILES=PROFILES.split(",")

profs=[] #Todos los perfiles en una lista
for p in PROFILES:
    path = os.path.abspath(f'{p}.json')
    with open(path, 'r') as file:
        jo = json.load(file)
        profs.append(jo)
    print(f"Perfil leído: '{jo['profile']}', imágenes: {len(jo['data'])}")


#Se obtiene perfil y un arreglo con todos los face encodings
def getEncodings(jo):
    enc = []
    for d in jo['data']:
        for det in d['detect']:
            for fe in det['faceEncoding']:
                enc.append(fe)
    return {'profile': jo['profile'], 'encodings': enc}


encodings = [] # todos los perfiles y por cada perfil todos los face encoding
for p in profs:
    encodings.append(getEncodings(p))
# print(encodings)

#Retorna el nombre del perfil en el cual se encuentra una coincidencia comparando face encodings
def findProfile(fr):
    for e in encodings:
        for enc in e['encodings']:
            for f in fr:
                result = face_recognition.compare_faces([enc], f)
                if True in result:
                    return e['profile']
    return 'Desconocid@'


cap = cv2.VideoCapture(0)
color = (50, 50, 255)
while True:
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv2.flip(frame, 1)
    faces_locs = face_recognition.face_locations(frame)
    for fl in faces_locs:
        fr = face_recognition.face_encodings(frame, known_face_locations=[fl])
        text = findProfile(fr)
        cv2.rectangle(frame, (fl[3], fl[2]), (fl[1], fl[2] + 30), color, -1)
        cv2.rectangle(frame, (fl[3], fl[0]), (fl[1], fl[2]), color, 2)
        cv2.putText(frame, text, (fl[3], fl[2] + 20), 2, 0.7, (255, 255, 255), 1)

    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cv2.destroyAllWindows()
