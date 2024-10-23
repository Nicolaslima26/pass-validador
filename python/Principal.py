import cv2
import serial
import time

# Inicializar a comunicação serial
arduino = serial.Serial('COM5', 9600, timeout=3)
time.sleep(2)  # Tempo para estabilizar a conexão serial

# Carregar o classificador Haarcascade
classificador_video = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# Carregar o modelo treinado
reconhecimento = cv2.face.LBPHFaceRecognizer_create()
reconhecimento.read("treinamento-face_recognition12.yml")

# Inicializar a webcam
webCamera = cv2.VideoCapture(0)
contadorPessoas = 0
pessoas_detectadas = set()

# Variável para rastrear o estado anterior
estado_anterior = None

try:
    while True:
        camera, frame = webCamera.read()
        imagemCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detect = classificador_video.detectMultiScale(imagemCinza, scaleFactor=1.1, minNeighbors=5, minSize=(33, 33))
        
        face_reconhecida = False  # Flag para verificar se pelo menos uma face foi reconhecida
        face_desconhecida = False  # Flag para rosto desconhecido
        confianca_baixa = False  # Flag para análise/baixa confiança

        for (x, y, l, a) in detect:
            imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (100, 100))
            cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 0, 0), 2)
            id, confianca = reconhecimento.predict(imagemFace)
            
            if confianca < 100: 
                print(id)
                if id == 1:
                    nome = "nicolas"
                elif id == 2:
                    nome = "lavine"
                elif id == 3:
                    nome = "raimundo"
                elif id == 4:
                    nome = "eduardo"
                # Adicionar outros IDs conforme necessário
                else:
                    nome = "Desconhecido"
                
                if nome != "Desconhecido":
                    face_reconhecida = True
                    cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)  # Verde
                else:
                    face_desconhecida = True
                    cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)  # Vermelho
            else:
                confianca_baixa = True
                nome = "em analise"
                cv2.rectangle(frame, (x, y), (x + l, y + a), (255, 0, 0), 2)  # Amarelo
            
            cv2.putText(frame, f"{nome} - {confianca:.2f}", (x, y + a + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
            
            if id not in pessoas_detectadas and id != -1 and nome != "Desconhecido":
                pessoas_detectadas.add(id)
                contadorPessoas += 1
            
            cont = str(len(detect))
            cv2.putText(frame, "faces detectadas:" + cont, (45, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        # Determinar o comando a ser enviado com base na detecção
        if face_reconhecida:
            comando = '1'  # Rosto conhecido
        elif face_desconhecida:
            comando = '2'  # Rosto desconhecido
        elif confianca_baixa or len(detect) == 0:
            comando = '3'  # Em análise ou sem rosto

        # Enviar o comando apenas se houver uma mudança de estado
        if comando != estado_anterior:
            arduino.write(comando.encode())
            estado_anterior = comando

        cv2.imshow("Face Validador", frame)

        if cv2.waitKey(1) == ord("f"):
            break

finally:
    print(f"dia 09/10/2024 estiveram presente {contadorPessoas} funcionarios sendo eles: {pessoas_detectadas}" )
    webCamera.release()
    cv2.destroyAllWindows()
    # arduino.close()
