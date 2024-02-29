import os
import cv2
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from feat import Detector, utils




def limpiar_carpeta(ruta):
    if os.path.isdir(ruta):
        elementos = os.listdir(ruta)
        for elemento in elementos:
            ruta_elemento = os.path.join(ruta, elemento)
            if os.path.isfile(ruta_elemento):
                os.remove(ruta_elemento)
            elif os.path.isdir(ruta_elemento):
                limpiar_carpeta(ruta_elemento)
        print(f"Carpeta {ruta} limpiada exitosamente.")
    else:
        print(f"{ruta} no es una carpeta válida.")


"""

Hacer un video con 20 segundos de cada emocion con imagenes del conjunto de prueba

"""


def leer_narrativa(ruta_narrativa):
    ruta_csv = os.path.join(ruta_narrativa, "csv", "narrativa.csv")
    narrativa_df = pd.read_csv(ruta_csv)
    return narrativa_df

def get_fps(video_path, subsampling):
    vidcap = cv2.VideoCapture(video_path)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    return int (fps/subsampling)


def extract_frames(video_path, output_folder, subsampling=1):
    print("Extrayendo frames")
    
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    seconds = 0
    num_frame = 0

    # Crea la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while success:
        
        second_str = str(seconds).zfill(2)
        
        #print("frame normal", count)
        if(count%subsampling==0):
            #print(count)
        
            frame_path = os.path.join(output_folder, f"{second_str}_{num_frame % int(fps/subsampling)}.jpg")
            cv2.imwrite(frame_path, image)    
            num_frame +=1

        # Incrementa el contador de frames y ajusta los segundos
        count += 1
        if count % fps == 0:
            seconds += 1
        

        
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
        success,image = vidcap.read()

    vidcap.release()
    


def find_face_emotion(frame, detector):
    single_face_prediction = detector.detect_image(frame)
    data = single_face_prediction
    df = single_face_prediction.emotions
    if len(df) == 1 and df.isnull().all().all():
        emotion_list = []
    else:
        dict = df.idxmax(axis=1).to_dict()
        emotion_list = list(dict.values())
    return emotion_list,data


def analyze_images_in_folder(folder_path, detector, csv):
    
    csv_folder_path = os.path.join('csv')
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)

    
    files = os.listdir(folder_path)

    
    results_df = pd.DataFrame(columns=['second', 'num_frame', 'emotion', 'group'])

    
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):  
            
            nombre_archivo, extension = file.split('.')

            second, frame = nombre_archivo.split('_')[:2]
            second = int(second)
            frame = int(frame)

            print(second, frame)

        
            image_path = os.path.join(folder_path, file)

            
        
            emotion_list, data = find_face_emotion(image_path, detector)

            if(len(emotion_list)>0):
                emotion = emotion_list[0]
            else:
                emotion = ''


            if emotion == 'happiness':
                group = 'positive'
            elif emotion in ['disgust', 'anger', 'sadness']:
                group = 'negative'
            elif emotion == 'neutral':
                group = 'neutral'
            elif emotion in ['surprise']:
                group = 'surprise'
            else:
                group = 'unknown'

           
            row_data = {'second': second, 'num_frame': frame, 'emotion': emotion, 'group': group}
            row_data.update(data.iloc[0].to_dict())
            results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)

    
    results_df = results_df.sort_values(by=['second', 'num_frame'], ascending=[True, True])
    results_df.reset_index(drop=True, inplace=True)
    results_df.to_csv(csv, index=False)

    



def plot_group_by_time(csv_path, ventana_de_tiempo, frames_per_second, multiple=False, lista_segundos=[]):
    if multiple:
        archivos_csv = [archivo for archivo in os.listdir(csv_path) if archivo.endswith('.csv')]

        for archivo in archivos_csv:
            ruta_completa = os.path.join(csv_path, archivo)
            plot_single_csv(ruta_completa, ventana_de_tiempo, frames_per_second)
        
        if lista_segundos:
            for segundo in lista_segundos:
                plt.axvline(x=segundo, color='red', linestyle='--', label=f'Segundo {segundo}')
        plt.show()
    else:
        plot_single_csv(csv_path, ventana_de_tiempo, frames_per_second)
        if lista_segundos:
            for segundo in lista_segundos:
                plt.axvline(x=segundo, color='red', linestyle='--', label=f'Segundo {segundo}')
        plt.show()


def plot_single_csv(csv_path, ventana_de_tiempo, frames_per_second):
    
    df = pd.read_csv(csv_path)
    frames_por_ventana = ventana_de_tiempo * frames_per_second    
    total_filas = len(df)

    tiempos = []
    grupos = []

    
    for i in range(total_filas):
        # Calcular el índice final de la ventana actual
        indice_final = min(i + int(frames_por_ventana), total_filas)
        # Seleccionar las filas correspondientes a la ventana actual
        ventana = df.iloc[i:indice_final]
        # Calcular el grupo predominante en la ventana actual
        grupo_predominante = ventana['group'].mode().iloc[0]
        # Calcular el tiempo de la ventana actual
        tiempo_ventana = ventana['second'].iloc[0] + ventana['num_frame'].iloc[0] / frames_per_second
        

        tiempos.append(tiempo_ventana)
        grupos.append(grupo_predominante)

    
    plt.plot(tiempos, grupos, marker='o', label=os.path.basename(csv_path))
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Grupo')
    plt.title('Gráfico de Grupo por Tiempo')
    plt.xticks(rotation=45)
    
    plt.legend()
    plt.grid(True)

    

    





def dividir_csv_subsampling(path_csv, subsampling):
    
    df = pd.read_csv(path_csv)
    df = df.sort_values(by=['second', 'num_frame'])
    df.reset_index(drop=True, inplace=True)
    
    
    df_subsampling = df[df.index % subsampling == 0]

    nombre_archivo, extension = os.path.splitext(os.path.basename(path_csv))
    nuevo_nombre_archivo = f"{nombre_archivo}_subsampling_{subsampling}{extension}"
    # Ruta donde se guardará el nuevo archivo CSV
    ruta_guardado = os.path.join('csv', 'subsampling', nuevo_nombre_archivo)
    
    
    df_subsampling.to_csv(ruta_guardado, index=False)
    print(f"El archivo se ha dividido con un subsampling de {subsampling}, se ha ordenado y se ha guardado como {ruta_guardado}")

    return ruta_guardado


    

def main():

    
    parser = argparse.ArgumentParser(description='Script para analizar el perfil del usuario')
    parser.add_argument('--subsampling', type=int, default=1)
    parser.add_argument('--createcsv', type=str, default="False")
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--multiple', type=bool, default=False)
    parser.add_argument('--lines', type=bool, default=False)
    args = parser.parse_args()

    name = "_" + args.name if args.name != "" else args.name

    if args.lines:
        ruta_narrativa = "narrativas/Nube"
        
        # Leer la narrativa desde el archivo CSV
        narrativa_df = leer_narrativa(ruta_narrativa)
        
        # Crear lista de segundos
        duraciones = narrativa_df["duracion"].tolist()
        lista_segundos = [0] + [sum(duraciones[:i+1]) for i in range(len(duraciones) - 1)]

        print(lista_segundos)
    else:
        lista_segundos = []
    
    
    
    video_path = "videos/records/video_grabado"+name+".mp4"
    output_dir = "frames_output" + name
    csv = 'csv/analyze' + name + ".csv"

    if(args.createcsv == "True"):
        detector = Detector()
        limpiar_carpeta(output_dir)
        extract_frames(video_path , output_dir, 1)
        analyze_images_in_folder(output_dir, detector, csv)

    new_fps = get_fps(video_path, args.subsampling)
    print(new_fps)

    
    if args.subsampling != 1:
        nuevo_csv = dividir_csv_subsampling(csv, 2)
    else:
        nuevo_csv = csv
    
    if args.multiple == True:
        nuevo_csv = "csv/"

    
    #lista_ventana = [0.14,0.20,0.25,0.30,0.40,0.5,0.6,0.7,0.75]

    #for ventana in lista_ventana:
    plot_group_by_time(nuevo_csv, 0.5, new_fps, multiple=args.multiple, lista_segundos=lista_segundos)


    

if __name__ == "__main__":
    main()
