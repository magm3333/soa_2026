import cv2
import argparse
import os
import time
from datetime import datetime
import face_recognition
import json
import numpy as np
#Requires: Click, dlib, face-recognition, face-recognition-models, numpy, Pillow
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", help="Directorio de fotos", default='', type=str)
parser.add_argument("-n", "--name", help="Nombre perfil", default='name', type=str)
args = parser.parse_args()
PATH = args.path
PROFILE = args.name

if PATH=='' or PROFILE=='':
    print('Argumentos erróneos')
    quit()

def get_all_file_paths(directory):
    return sorted( filter(os.path.isfile, [os.path.join(directory, file) for file in os.listdir(directory)]), 
        key=lambda p: os.path.exists(p) and os.stat(p).st_mtime or time.mktime(datetime.now().timetuple()))



path=os.path.abspath(PATH)
files=get_all_file_paths(path)
if len(files)==0:
    print(f'No hay imágenes para leer en "{path}"!')



data=[]
for f in files:
    image=cv2.imread(f)
    faces_locs=face_recognition.face_locations(image) #Cajas donde hay rostros
    #print(f, faces_locs)
    fie=[]
    for fl in faces_locs :
        fr=face_recognition.face_encodings(image, known_face_locations=[fl])
        #            Caja con rotro y codificación
        fie.append( {'faceLoc':fl,    'faceEncoding':fr} )
        #print(fr)
    if len(fie)>0:
        # Nombre de archivo imagen, Caja con rotro y codificación
        data.append({'file':f, 'detect':fie})

if len(data)>0 :
    # Perfil + Nombre de archivo imagen, Caja con rotro y codificación 
    final={'profile':PROFILE, 'data':data}
    #contenido=json.dumps(final, cls=NpEncoder)
    contenido=json.dumps(final, cls=NpEncoder, indent=4, ensure_ascii=False)
with open(f'./{PROFILE}.json','w') as file:
    file.write(contenido)

"""
Un embedding es una forma de convertir algo complejo (texto, una cara, una voz, una imagen, etc.) en una lista de números que representa sus características importantes.

“Transformar algo del mundo real en coordenadas matemáticas”.

En LLMs (ChatGPT, embeddings de texto)

Pasa exactamente lo mismo.

El texto se convierte en números.

Por ejemplo:

"gato"

podría convertirse en:

[0.82, -0.11, 0.55, ...]

y:

"felino"

termina cerca matemáticamente.

Mientras:

"tractor"

queda lejos.

Entonces el modelo “entiende” significado

Porque palabras similares quedan cerca en el espacio vectorial.

Ejemplo conceptual:

gato -------- felino

auto -------- coche

pizza -------- pasta
En RAG / búsquedas inteligentes

Cuando hacés:

"problemas con Docker"

el sistema:

genera embedding de tu pregunta,
busca embeddings cercanos,
encuentra documentos similares aunque no tengan exactamente las mismas palabras.

Por eso puede encontrar:

"errores de contenedores"

aunque no diga “Docker”.

Embedding ≠ significado humano

Importante:
el embedding NO “entiende” como una persona.

Simplemente:

aprendió patrones matemáticos, donde cosas parecidas terminan cerca.
Visualización simple

Un espacio 3D:

        animales
             ^
             |
 gato --- perro
             |
             |
 auto --- camión ------ tractor
             |
             +-----------------> vehículos

Los embeddings reales tienen:

128,
512,
768,
1536 dimensiones,
o más.


Resumen:
Un embedding es: una representación numérica compacta de algo complejo.

Sirve para:
-comparar,
-buscar similitud,
-clasificar,
-reconocer,
-agrupar.

En:

-Face recognition → representa caras.
-LLMs → representa significado del texto.
-Audio AI → representa sonidos/voces.
-Vision AI → representa imágenes/objetos.
"""