import argparse
import os
import csv
import re
import codecs
from TTS.api import TTS


def obtener_parrafos(texto):
    # Dividir el texto en párrafos usando dos saltos de línea como separador
    parrafos = re.split(r'\n\n', texto)
    return parrafos


def crear_texto(folder_name):
    ruta_archivo = os.path.join(folder_name, 'texto.txt')

    # Comprobar si el archivo de texto existe
    if not os.path.isfile(ruta_archivo):
        print(f"No se encontró el archivo de texto en el directorio '{folder_name}'.")
        return

    with open(ruta_archivo, 'r') as archivo:
        contenido = archivo.read()

    parrafos = obtener_parrafos(contenido)

    return parrafos

def generar_audio(texto,folder_name, folder_output, model):
    # Crear directorio si no existe
    if not os.path.exists(os.path.join(folder_name,folder_output)):
        os.makedirs(os.path.join(folder_name,folder_output))

    print("Se estan generando audios")
    # Generar speech y guardar en el directorio de salida
    for i, text in enumerate(texto, start=1):
        file_name = f"{i}.wav"
        file_path = os.path.join(folder_name,folder_output, file_name)
        model.tts_to_file(text,
                file_path=file_path,
                speaker_wav="voces/nueva_grabacion.m4a",
                language="es")
    cantidad_audios = len(texto)
    print(f"Se generaron {cantidad_audios} audios")



def generar_csv(texto, directorio):
    num_audios = len([name for name in os.listdir(os.path.join(directorio, "Audios")) if name.endswith('.wav')])
    num_imagenes = len([name for name in os.listdir(os.path.join(directorio, "Imagenes")) if name.endswith('.jpg') or name.endswith('.png')])
    
    if len(texto) != num_audios or len(texto) != num_imagenes:
        print("La longitud de los archivos de audio, imágenes y la lista de textos no coincide.")
        return
    
    csv_dir = os.path.join(directorio, "csv")
    os.makedirs(csv_dir, exist_ok=True) 
    
    with codecs.open(os.path.join(csv_dir, 'narrativa.csv'), 'w', 'utf-8-sig') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['numero', 'texto', 'ruta_imagen', 'ruta_audio', 'emocion_objetivo', 'duracion'])
        
        for i in range(1, len(texto)+1):
            audio_path = os.path.join(directorio, "Audios", f"{i}.wav")
            imagen_files = [name for name in os.listdir(os.path.join(directorio, "Imagenes")) if name.startswith(f"{i}_")]
            if len(imagen_files) == 1:
                imagen_path = os.path.join(directorio, "Imagenes", imagen_files[0])
                emocion_objetivo = imagen_files[0].split('_')[1].split('.')[0]  # Eliminar la extensión
            else:
                print(f"No se encontró una única imagen para el número {i}.")
                continue
            
            duracion = round(os.path.getsize(audio_path) / (2 * 24000)) + 2


            
            csvwriter.writerow([i, texto[i-1], imagen_path, audio_path, emocion_objetivo, duracion])


    
    print("CSV generado")






def main():
    # Definir y parsear los argumentos de línea de comandos
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='Narrative Folder Name')
    parser.add_argument('--audios', type=str, help='Narrative Folder Name', default="false")
    args = parser.parse_args()
    folder_name = args.name
    audios = args.audios
    # Comprobar si el directorio existe
    if not os.path.exists(folder_name):
        print(f"El directorio '{folder_name}' no existe.")
        return
    


    
    texto = crear_texto(folder_name)
    
    if audios == "true":
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        generar_audio(texto, folder_name,"Audios", tts)
        

    generar_csv(texto, folder_name)
    



if __name__ == '__main__':
    main()
