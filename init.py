import cv2

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar un fotograma
    ret, frame = cap.read()

    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar detección de bordes para encontrar el objeto
    edges = cv2.Canny(gray, 50, 150)

    # Buscar contornos en el fotograma
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Seleccionar el contorno que corresponde al objeto
    cnt = None
    for c in contours:
        if cv2.contourArea(c) > 1000: # ajustar este umbral según la escala del objeto en la imagen
            cnt = c
            break

    # Calcular la longitud y el ancho del objeto
    if cnt is not None:
        x, y, w, h = cv2.boundingRect(cnt)
        length = w # asumir que el objeto es rectangular y está orientado horizontalmente
        width = h # asumir que el objeto es rectangular y está orientado horizontalmente
        print("Length:", length, "pixels")
        print("Width:", width, "pixels")
    else:
        print("No se ha encontrado el objeto en la imagen")

    # Mostrar el fotograma
    cv2.imshow("Frame", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara
cap.release()
cv2.destroyAllWindows()
