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
classificadorvideo = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
reconhecimento = cv2.face.LBPHFaceRecognizer_create()
reconhecimento.read("treinamento-face_recognition3.yml")

webCamera = cv2.VideoCapture(0)
contador_pessoas = 0
# Armazena IDs de pessoas já detectadas
pessoas_detectadas = set()

while True:
    camera, frame = webCamera.read()
    imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect = classificadorvideo.detectMultiScale(imagemCinza, scaleFactor= 1.8, minSize=(130, 130))

    for(x, y, l, a) in detect:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (100,100))
        cv2.rectangle(frame, (x, y), (x + l, y + a), (255,0,0), 2)
        id, confianca = reconhecimento.predict(imagemFace)
        # pegaOlho = frame[y:y + a, x:x + l]
        # olhoCinza = cv2.cvtColor(pegaOlho, cv2.COLOR_BGR2GRAY)
        # localizarOlho = classificadorOlho.detectMultiScale(olhoCinza, )
        # for (ox, oy, ol, oa) in localizarOlho:
        #     cv2.rectangle(pegaOlho, (ox, oy), (ox + ol, oy + oa), (255,0,0), 1);
        
        if confianca < 100:  # Ajuste este valor conforme necessário
            if id == 1:
                nome = "Nicolas"
                cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255,0), 2)
            elif id == 2:
                nome = "ryan"
                cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255,0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
                nome = "Desconhecido"
        else:
            nome = "em analise"
            
        cv2.putText(frame, f"{nome} - {confianca:.2f} ",(x,y +(a +30)), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0))
        #contador de   quantas pessoas já foram detectadas
        if id not in pessoas_detectadas and id != -1:  # Ignorar ID -1 ou IDs desconhecidos
            pessoas_detectadas.add(id)  # Adicionar o novo ID à lista de detectados
            contador_pessoas += 1  # Incrementar o contador
        cont = str(detect.shape[0])
        cv2.putText(frame, "faces detectadas:" + cont, (45, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow("face Validador", frame)
    print(id)
    
    if cv2.waitKey(1) == ord("f"):
        break
    
print(f"dia 09/10/2024 estiveram presente {contador_pessoas} funcionarios sendo eles: {pessoas_detectadas}")  
webCamera.release()
cv2.destroyAllWindows()
