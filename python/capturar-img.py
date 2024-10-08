import cv2 as cv

classificador_video = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
webCamera = cv.VideoCapture(0)

amostra = 1
numeroAmostras = 100
id = input("digite seu idenficador")

while True:
    camera, frame = webCamera.read()
    imagemCinza = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detect = classificador_video.detectMultiScale(imagemCinza)

    for (x, y, l, a) in detect:
        cv.rectangle(frame, (x, y), (x + l, y + a), (0,255,0), 2 )

        #pegar a imagem/frame do id e salvar na pasta 
        if cv.waitKey(1) == ord("c"):
            imagemFace = cv.resize(imagemCinza[y:y + a, x:x + l], (220, 220))
            cv.imwrite(f"BD/img-faces/aluno. {id}. {amostra}.jpg", imagemFace)
            print(f"foto {amostra}capturada com sucesso!!")
            amostra += 1

    cv.imshow("face validador", frame)
    if amostra > numeroAmostras:
        break
    if cv.waitKey(1) == ord("f"):
        break

webCamera.release()
cv.destroyAllWindows()
