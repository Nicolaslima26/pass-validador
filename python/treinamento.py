import cv2
import os
import numpy as np

lbph = cv2.face.LBPHFaceRecognizer_create()

def pegarimgID():
    # Obter o caminho de cada imagem na pasta de treinamento
    caminho = [os.path.join('BD/img-faces', f) for f in os.listdir('BD/img-faces')]
    faces = []
    ids = []

    for caminhoimg in caminho:
        # Ler a imagem e converter para escala de cinza
        imagemFace = cv2.cvtColor(cv2.imread(caminhoimg), cv2.COLOR_BGR2GRAY)
        # Extrair o ID do nome do arquivo (assumindo que o formato é face.ID.jpg)
        id = int(os.path.split(caminhoimg)[-1].split(".")[1])
        ids.append(id)
        faces.append(imagemFace)

    return np.array(ids), faces

# Obter IDs e faces
ids, faces = pegarimgID()
print('Treinando a máquina, aprendendo...')

# Treinar o modelo
lbph.train(faces, ids)
# Salvar o modelo treinado
lbph.write('treinamento-face_recognition9.yml')

print('Treinamento concluído')
print(ids, faces)