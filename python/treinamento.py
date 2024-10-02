import cv2
import os
import numpy as np

lbph = cv2.face.LBPHFaceRecognizer_create()


def pegarimgID():
    caminho = [os.path.join('BD/img-faces', f) for f in os.listdir('BD/img-faces')]
    faces = []
    ids = []

    for caminhoimg in caminho:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoimg), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoimg) [-1].split(".")[1])
        ids.append(id)
        faces.append(imagemFace)

    return np.array(ids), faces

ids, faces = pegarimgID()
print('treinando maquina, aprendendo...')

lbph.train(faces, ids)
lbph.write('treinamento-face_recognition.yml')

print('treinamento concluido')