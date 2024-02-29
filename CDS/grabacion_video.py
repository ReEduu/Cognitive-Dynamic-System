import os
import cv2
import time
import sys

def grabar_video(tiempo_segundos, nombre_archivo, mostrar_video=False, fps=30.0):
    # Configurar captura de video
    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        print("Error al abrir la cámara")
        return

    # Crear carpetas si no existen
    directorio_base = "videos/records"
    os.makedirs(directorio_base, exist_ok=True)

    # Configurar codificador de video con el códec H.264
    codec = cv2.VideoWriter_fourcc(*'avc1')
    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ruta_video = os.path.join(directorio_base, nombre_archivo)
    salida = cv2.VideoWriter(ruta_video, codec, fps, (ancho, alto))

    # Tiempo de inicio
    tiempo_inicio = time.time()

    # Iniciar grabación
    while True:
        ret, frame = captura.read()
        if not ret:
            break
        salida.write(frame)

        if mostrar_video:
            # Mostrar el video en una ventana
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Detener la grabación después del tiempo especificado
        if time.time() - tiempo_inicio >= tiempo_segundos:
            break

    # Liberar recursos
    captura.release()
    salida.release()
    cv2.destroyAllWindows()

    print("Video grabado exitosamente en:", ruta_video)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python grabacion_video.py <tiempo_segundos> <nombre_archivo> [--mostrar_video]")
        sys.exit(1)
        
    tiempo_segundos = int(sys.argv[1])
    nombre_archivo = sys.argv[2]
    mostrar_video = "--mostrar_video" in sys.argv
    grabar_video(tiempo_segundos, nombre_archivo, mostrar_video)
