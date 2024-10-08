import cv2

# Carregar o classificador Haar
classificador_video = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
# Carregar o modelo treinado
reconhecimento = cv2.face.LBPHFaceRecognizer_create()
reconhecimento.read("treinamento-face_recognition7.yml")

# Inicializar a webcam
webCamera = cv2.VideoCapture(0)

while True:
    camera, frame = webCamera.read()
    imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect = classificador_video.detectMultiScale(imagemCinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, l, a) in detect:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (100, 100))
        cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 0, 0), 2)
        id, confianca = reconhecimento.predict(imagemFace)
        
        if confianca < 100:
            print(id)# Ajuste este valor conforme necessário
            if id == 1:
                nome = "nicolas"
                cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)
            elif id == 2:
                nome = "eduardo"
                cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)
            elif id == 3:
                nome = "ryan"
                cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)
            elif id == 4:
                nome = "ksousa"
                cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)
            elif id == 5:
                nome = "alessandro"
                cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
                nome = "Desconhecido"
        else:
            cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 0, 0), 2)
            nome = "em analise"
        
        cv2.putText(frame, f"{nome} - {confianca:.2f}", (x, y + a + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

    cv2.imshow("Face Validador", frame)

    if cv2.waitKey(1) == ord("f"):
        break

webCamera.release()
cv2.destroyAllWindows()
