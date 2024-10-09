import cv2
import serial
import time
arduino = serial.Serial('COM5', 9600, timeout=10)
time.sleep(2)  # Tempo para estabilizar a conexão serial
# Carregar o classificador Haar
classificador_video = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
# Carregar o modelo treinado
reconhecimento = cv2.face.LBPHFaceRecognizer_create()
reconhecimento.read("treinamento-face_recognition9.yml")

# Inicializar a webcam
webCamera = cv2.VideoCapture(0)
contadorPessoas = 0
pessoas_detectadas = set()
try:
    while True:
        camera, frame = webCamera.read()
        imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detect = classificador_video.detectMultiScale(imagemCinza, scaleFactor=1.1, minNeighbors=5, minSize=(33, 33))

        for (x, y, l, a) in detect:
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (100, 100))
            cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 0, 0), 2)
            id, confianca = reconhecimento.predict(imagemFace)
            
            if confianca < 100:
                print(id)# Ajuste este valor conforme necessário
                if id == 1:
                    nome = "nicolas"
                    cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)
                        # Se detectar rosto, envia o comando para acender o LED
                    if len(detect) > 0:
                        arduino.write(b'1')  # Envia o caractere '1' para o Arduino (acender LED)
                    else:
                        arduino.write(b'0')  # Envia o caractere '0' para o Arduino (apagar LED)

                elif id == 2:
                    nome = "ryan"
                    cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)
                    if len(detect) > 0:
                        arduino.write(b'1')  # Envia o caractere '1' para o Arduino (acender LED)
                    else:
                        arduino.write(b'0')  # Envia o caractere '0' para o Arduino (apagar LED)

                # elif id == 3:
                #     nome = "eduardo"
                #     cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)
                #     if len(detect) > 0:
                #         arduino.write(b'1')  # Envia o caractere '1' para o Arduino (acender LED)
                #     else:
                #         arduino.write(b'0')  # Envia o caractere '0' para o Arduino (apagar LED)

                # elif id == 4:
                #     nome = "ksousa"
                #     cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)
                #     if len(detect) > 0:
                #         arduino.write(b'1')  # Envia o caractere '1' para o Arduino (acender LED)
                #     else:
                #         arduino.write(b'0')  # Envia o caractere '0' para o Arduino (apagar LED)

                # elif id == 5:
                #     nome = "alessandro" 
                #     cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)
                #     if len(detect) > 0:
                #         arduino.write(b'1')  # Envia o caractere '1' para o Arduino (acender LED)
                #     else:
                #         arduino.write(b'0')  # Envia o caractere '0' para o Arduino (apagar LED)

                else:
                    cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
                    nome = "Desconhecido"
            else:
                cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 0, 0), 2)
                nome = "em analise"
            
            cv2.putText(frame, f"{nome} - {confianca:.2f}", (x, y + a + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            
            if id not in pessoas_detectadas and id != -1:  # Ignorar ID -1 ou IDs desconhecidos
                pessoas_detectadas.add(id)  # Adicionar o novo ID à lista de detectados
                contadorPessoas += 1  # Incrementar o contador
            
            cont = str(len(detect))  # Mostrar o número de faces detectadas no frame atual
            cont = str(detect.shape[0])
            cv2.putText(frame, "faces detectadas:" + cont, (45, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow("Face Validador", frame)

        if cv2.waitKey(1) == ord("f"):
            break

finally:
    print(f"dia 09/10/2024 estiveram presente {contadorPessoas} funcionarios sendo eles: {pessoas_detectadas}" )
    webCamera.release()
    cv2.destroyAllWindows()
