import os
import cv2
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



def leer_narrativa(ruta_narrativa):
    ruta_csv = os.path.join(ruta_narrativa, "csv", "narrativa.csv")
    narrativa_df = pd.read_csv(ruta_csv)
    return narrativa_df

def extract_frames(video_path, lista_segundos, output_dir):
    # Verificar y crear el directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Recorrer la lista de segundos
    for i, segundo in enumerate(lista_segundos, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(segundo * fps))
        for j in range(1, 41):  # 10 frames por segundo durante 4 segundos
            ret, frame = cap.read()
            if ret:
                # Guardar el frame con el formato especificado
                frame_name = f"{i}_{j}.jpg"
                cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            else:
                break
    
    # Liberar el objeto VideoCapture
    cap.release()


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


def analyze_images_in_folder(folder_path, detector):
    # Crear la carpeta 'csv' si no existe
    csv_folder_path = os.path.join('csv')
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)

    # Obtener una lista de los archivos en la carpeta
    files = os.listdir(folder_path)

    # Inicializar un DataFrame para almacenar los resultados
    results_df = pd.DataFrame(columns=['indice', 'num_frame', 'emotion', 'group'])

    # Iterar sobre los archivos en la carpeta
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):  # Asumiendo que son imágenes jpg o png
            # Obtener el índice y el tiempo de la imagen
            nombre_archivo, extension = file.split('.')

            indice, frame = nombre_archivo.split('_')[:2]
            indice = int(indice)
            frame = int(frame)

            print(indice, frame)

            # Cargar la imagen
            image_path = os.path.join(folder_path, file)

            # Detectar emociones en la imagen
            emotion_list, data = find_face_emotion(image_path, detector)

            if(len(emotion_list)>0):
                emotion = emotion_list[0]
            else:
                emotion = ''


            if emotion == 'happiness':
                group = 'positive'
            elif emotion in ['disgust', 'anger']:
                group = 'negative'
            elif emotion == 'neutral':
                group = 'neutral'
            elif emotion in ['surprise', 'sadness']:
                group = 'surprise'
            else:
                group = 'unknown'

            # Agregar los resultados al DataFrame
            row_data = {'indice': indice, 'num_frame': frame, 'emotion': emotion, 'group': group}
            row_data.update(data.iloc[0].to_dict())
            results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)

    # Guardar los resultados en un archivo CSV
    results_df = results_df.sort_values(by=['indice', 'num_frame'], ascending=[True, True])
    csv_path = os.path.join(csv_folder_path, 'analisis.csv')
    results_df.to_csv(csv_path, index=False)

def plot_groups_by_index(csv_path, frames_per_second):
    # Leer el archivo CSV
    df = pd.read_csv(csv_path)

    df = df[df['group'] != 'unknown']

    # Calcular el tiempo total en segundos como un valor de punto flotante
    df['Tiempo'] = df['indice'] + df['num_frame'] / frames_per_second

    # Graficar los datos
    plt.plot(df['Tiempo'], df['group'], marker='o')
    plt.xlabel('Indice (Momento clave)')
    plt.ylabel('Grupo')
    plt.title('Gráfico de Grupo por Tiempo')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()






def main():

    detector = Detector()


    # Carpeta de la narrativa
    ruta_narrativa = "narrativas/Nube"
    
    # Leer la narrativa desde el archivo CSV
    narrativa_df = leer_narrativa(ruta_narrativa)
    
    # Crear lista de segundos
    duraciones = narrativa_df["duracion"].tolist()
    lista_segundos = [0] + [sum(duraciones[:i+1]) for i in range(len(duraciones) - 1)]

    print(lista_segundos)
    
    
    # Ruta del video grabado
    video_path = "videos/records/video_grabado.mp4"
    
    # Extraer los frames según los segmentos de tiempo en la lista de segundos
    output_dir = "frames"

    #limpiar_carpeta(output_dir)

    # Llamar a la función para extraer los frames
    #extract_frames(video_path, lista_segundos, output_dir)


    #analyze_images_in_folder(output_dir, detector)

    plot_groups_by_index('csv/analisis.csv', 40)
    

if __name__ == "__main__":
    main()
