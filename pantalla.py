import cv2 # OpenCV
import pyautogui
import numpy as np
codec = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("Grabacion.avi", codec , 60, (1366, 768)) 
cv2.namedWindow("Grabando", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Grabando", 900, 850) #Los ultimos dos argunmentos son las dimensiones de la ventana de grabación
while True:
    img = pyautogui.screenshot() # tomamos un pantallazo
    frame = np.array(img) # convertimos la imagen a un arreglo de numeros
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convertimos la imagen BGR a RGB
    out.write(frame) # adjuntamos al archivo de video
    frame = frame[:800, :800]
    cv2.imshow('Grabando', frame) # mostramos el cuadro que acabamos de grabar
    if cv2.waitKey(1) == ord('q'): # si el usuario presiona q paramos de grabar.
        break

out.release() # cerrar el archivo de video
cv2.destroyAllWindows() # cerrar la ventana