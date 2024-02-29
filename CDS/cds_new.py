import os
import cv2
import time
import argparse
import pandas as pd
from playsound import playsound
import subprocess

def leer_narrativa(ruta_narrativa):
    ruta_csv = os.path.join(ruta_narrativa, "csv", "narrativa.csv")
    narrativa_df = pd.read_csv(ruta_csv)
    narrativa_dict = narrativa_df.to_dict(orient="records")
    return narrativa_dict

def mostrar_contenido(narrativa_dict):
    for item in narrativa_dict:
        ruta_imagen = os.path.join("narrativas", item["ruta_imagen"])
        ruta_audio = os.path.join("narrativas", item["ruta_audio"])
        duracion = item["duracion"]
        
        try:
            imagen = cv2.imread(ruta_imagen)
            if imagen is not None:
                cv2.imshow('Imagen', imagen)
                cv2.waitKey(500)  # Esperar el tiempo indicado en segundos
                playsound(ruta_audio)
                cv2.waitKey(1500)
        except Exception as e:
            print(f"Error al abrir la imagen {ruta_imagen}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Script para grabar video mientras se muestra una narrativa.')
    parser.add_argument('narrativa', type=str, help='Carpeta dentro de "narrativas" que contiene la narrativa a mostrar')
    parser.add_argument('--mostrar_video', action='store_true', help='Mostrar una ventana con el video mientras se graba')
    args = parser.parse_args()
    
    ruta_narrativa = os.path.join("narrativas", args.narrativa)
    narrativa_dict = leer_narrativa(ruta_narrativa)

    #print(narrativa_dict)

    
    tiempo_narrativa = sum(item["duracion"] for item in narrativa_dict)
    subprocess.Popen(["python", "grabacion_video.py", str(tiempo_narrativa), "video_grabado_6.mp4", "--mostrar_video" if args.mostrar_video else ""])

    cv2.waitKey(1000)
    #mostrar_contenido(narrativa_dict)

if __name__ == "__main__":
    main()
