import cv2

# Carregar o classificador Haar
classificador_video = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
reconhecimento = cv2.face.LBPHFaceRecognizer_create()
reconhecimento.read("treinamento-face_recognition2.yml")

webCamera = cv2.VideoCapture(0)

while True:
    camera, frame = webCamera.read()
    imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect = classificador_video.detectMultiScale(imagemCinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, l, a) in detect:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (100, 100))
        cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 0, 0), 2)
        id, confianca = reconhecimento.predict(imagemFace)

        print(f"ID: {id}, Confiança: {confianca}")  # Debugging

        if confianca < 70:  # Ajuste conforme necessário
            if id == 1:
                nome = "Nicolas"
            elif id == 2:
                nome = "Alessandro"
            elif id == 3:
                nome = "Ryan"
            else:
                nome = "ID não mapeado"
        else:
            nome = "Desconhecido"

        cv2.putText(frame, nome, (x, y + (a + 30)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0))

    cv2.imshow("Face Validador", frame)

    if cv2.waitKey(1) == ord("f"):
        break

webCamera.release()
cv2.destroyAllWindows()
