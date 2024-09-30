import cv2

#observações:
    #parametros para ajustar imagem e sua parasia, scaleFactor= 1.08, minNeighbors=1, minSize=(30,30):

#carregar o algoritmo do opencv tipo a função que estou usando
carregar_face = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
carregar_olhos = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

#pegar minha imagem e tranformar em cinza fica melhor para identificar
imagem = cv2.imread('BD/cicera.jpeg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

#detectar a face na imagem
detect_faces = carregar_face.detectMultiScale(imagemCinza)

print(detect_faces)

for (x, y, l, a) in detect_faces:
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
    #desenhar o local onde foi detectado
    desenho_imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (255,0,0), 2)

    #identificar olho quando identificar um rosto
    localolho = desenho_imagem[y:y + a, x:x +l] #tem que entender melhor esse bglh pq é dificil pra prr

    localolho_cinza = cv2.cvtColor(localolho, cv2.COLOR_BGR2GRAY)
    olho_detectado = carregar_olhos.detectMultiScale(localolho_cinza)

    for (ox, oy, ol, oa) in olho_detectado:
        cv2.rectangle(localolho, (ox, oy), (ox + ol, oy + oa), (255,0, 255),2)

#mostrar na tela 
cv2.imshow('Face Validador', imagem )
cv2.waitKey()
cv2.destroyAllWindows()
#git remote add origin