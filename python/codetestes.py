import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
 

while True:
    #capturar frame a frame
    ret, frame = cap.read()

    #converte o frame para a escala de cinza (necessário para a detecção)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detectar rostos no frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #Desenhar retângulos ao redor dos rostos detectados

    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w, y+h), (0,0,255), 2)

    # mostrar os frames com as detecções
    cv2.imshow('Face Validator', frame)

    #sair do loopb qunado a tecla 'q' for pressionada 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

