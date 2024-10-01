import cv2
#Lógica:
    #carregar algoritimo 
    #carregar minha imagem ou meu video
    #transformar minha imagem na cor cinza
    #detectar o rosto na imagem ou no video
    #mostrar algum valor
    #desenhar o retangulo na face detectada ou as fazer ou seja precisar ser um loop
        #fazer o molde do desenho
    #abrir minha janela para mostrar
    #comandos finais 
#observações:
#parametros para ajustar imagem e sua parasia, scaleFactor= 1.08, minNeighbors=1, minSize=(30,30):
webCamera = cv2.VideoCapture(0)
classificadorvideo = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
classificadorOlho = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

contador = 0
while True:
    camera, frame = webCamera.read()
    print(camera,frame)
    imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect = classificadorvideo.detectMultiScale(imagemCinza, scaleFactor=1.1, minNeighbors=8, minSize=(25, 25))


    for(x, y, l, a) in detect:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (255,0,0), 2);
        pegaOlho = frame[y:y + a, x:x + l]
        olhoCinza = cv2.cvtColor(pegaOlho, cv2.COLOR_BGR2GRAY)
        localizarOlho = classificadorOlho.detectMultiScale(olhoCinza, scaleFactor=1.1, minNeighbors=8, minSize=(25,25))

        
        for (ox, oy, ol, oa) in localizarOlho:
            cv2.rectangle(pegaOlho, (ox, oy), (ox + ol, oy + oa), (255,0,0), 2);
        #contador de quantas pessoas já foram detectadas
        cont = str(detect.shape[0])
        cv2.putText(frame, "faces detectadas:" + cont, (45, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cont = int(cont)
        cont += 1
        contFinal = cont - 1
        print(contFinal)

    cv2.imshow("face Validador", frame)

    if cv2.waitKey(1) == ord("f"):
        break

    
webCamera.release()
cv2.destroyAllWindows()