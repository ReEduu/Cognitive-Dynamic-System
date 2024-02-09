import os
import face_recognition
import cv2
import numpy as np
import tkinter as tk
import time
import openai
import torch
import csv
import random
import graphviz
import tempfile
import matplotlib.pyplot as plt
import cv2
import sys
import mediapipe as mp
import PIL
import argparse
import pandas as pd
import subprocess
from collections import deque
from statistics import mode
from matplotlib.animation import FuncAnimation
from datetime import datetime
from feat import Detector, utils
from feat.utils.io import get_test_data_path
from feat.plotting import imshow
from PIL import Image, ImageTk
from audiocraft.models import MusicGen
from audiocraft.utils.notebook import display_audio
from audiocraft.data.audio import audio_write
from diffusers import StableDiffusionInpaintPipeline

utils.set_torch_device(device='mps')
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

emocion_actual = "initial_state"
#emocion_objetivo = ""
state_index = 0
window = tk.Tk()
window.title("Image Viewer")
window.geometry("600x800")
label = None

openai.api_key = "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"
messages = [ {"role": "system", "content":
                "You are a intelligent assistant."} ]

img_path = "imagenes/paint/imagenes/paint.jpg"  
century_mask_path = "imagenes/paint/mascaras/no_skull.png"
narrative_csv = 'csv/narrativa.csv'

last_row_index = 0


KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.6
MODEL = 'cnn'
detector = Detector()


emocion_columna = {
    'sadness': 4,
    'neutral': 5,
    'fear': 6,
    'happiness': 7,
    'surprise': 8,
    'anger': 9,
    'disgust': 10
}



parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, default="datos.csv", help='Nombre del archivo CSV')
parser.add_argument('--video', type=str, help='Nombre del archivo de video')
parser.add_argument('--gpt_prompts', type=str,default="false", help='Forma en que se generan los prompts')
parser.add_argument('--record', type=str, default="false", help='Grabar video: true o false')
parser.add_argument('--century', type=int, default=None, help='Definir siglo a adaptar')
parser.add_argument('--control', type=str, default="true", help='Take control actions')
parser.add_argument('--windows', type=str, default="true", help='Show graphs and photos')
args = parser.parse_args()

nombre_archivo = os.path.join('csv', args.csv)



if args.video:
    nombre_video = args.video 


if os.path.isfile(nombre_archivo):
    with open(nombre_archivo, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)
        if data:
            last_row = data[-1]
            num_test = int(last_row[0]) + 1
        else:
            num_test = 1
else:
    num_test = 1

states = ["initial_state","anger", "fear", "sadness", "surprise", "neutral", "disgust", "happiness"]

# Inicializa la estructura de datos
fsm = {state: {"tiempo_mantenido": 0, "transiciones": []} for state in states}

# Función para agregar una transición con una lista de imágenes y emociones objetivo
def update_fsm(fsm, estado_actual, emocion_transicion, imagenes_y_emociones, state_time, state_index):
    fsm[estado_actual]["tiempo_mantenido"] += state_time
    fsm[estado_actual]["transiciones"].append((list(imagenes_y_emociones), emocion_transicion, state_time, state_index))
    start_state_time = time.time()
    state_index = state_index + 1
    return fsm, start_state_time, state_index

def save_fsm(diccionario, csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['State', 'Total time', 'Transitions'])
        for emocion, info in diccionario.items():
            tiempo_mantenido = info['tiempo_mantenido']
            transiciones = info['transiciones']
            transiciones_str = ', '.join([str(t) for t in transiciones])
            csv_writer.writerow([emocion, tiempo_mantenido, transiciones_str])

# Función para mostrar el grafo como un diagrama
def draw_fsm(maquina_estados, numero_test):
    dot = graphviz.Digraph(comment="Máquina de Estados Finita")

    for emocion, info in maquina_estados.items():
        label = f"State: {emocion}\nTotal time: {info['tiempo_mantenido']} seconds"
        dot.node(emocion, label=label)

        for transicion in info["transiciones"]:
            emocion_transicion = transicion[1]
            state_index = transicion[3]
            state_time = transicion[2]
            label = f"{state_index} - State time: {state_time}\n"

            for imagen, target_emotion in transicion[0]:
                label += f"Imagen: {imagen}\nTarget Emotion: {target_emotion}\n"

            dot.edge(emocion, emocion_transicion, label=label)

    file_path = f"diagramas/fsm_{numero_test}"
    dot.render(file_path, format="png")


def get_current_image(narrative_csv, last_row_index):
    with open(narrative_csv, 'r', newline='', encoding='utf-8-sig') as archivo:
        lector_csv = csv.reader(archivo)
        filas = list(lector_csv)  # Leer todas las filas del archivo CSV
        if last_row_index < len(filas):
            fila = filas[last_row_index]
            nombre_imagen = fila[0]
            duracion = int(fila[1])
            nombre_sin_extension = os.path.splitext(nombre_imagen)[0]
            ruta_imagen = os.path.join('imagenes', nombre_sin_extension, "imagenes", nombre_imagen)
            target_emotion = fila[2]  
            return {
                "image_name": nombre_sin_extension,
                "image_path": ruta_imagen,
                "duration": duracion,
                "target_emotion": target_emotion
            }, last_row_index + 1  # Devuelve el diccionario y el nuevo índice
        else:
            return None, last_row_index  # Si no hay más filas

#Sin uso actual
def close_all_images():
    for img in os.listdir('/tmp'):
        if img.startswith('PIL_'):
            os.remove(os.path.join('/tmp', img))

def resize_image(image, width, height):
    aspect_ratio = float(image.width) / float(image.height)
    if width / aspect_ratio <= height:
        new_width = int(width)
        new_height = int(width / aspect_ratio)
    else:
        new_width = int(height * aspect_ratio)
        new_height = int(height)
    return image.resize((new_width, new_height), Image.ANTIALIAS)


def show_image(image_path, window, label):

    new_img = Image.open(image_path)
    new_img = resize_image(new_img, window.winfo_width(), window.winfo_height())
    new_img_tk = ImageTk.PhotoImage(new_img)

    if label is not None:
        label.config(image=new_img_tk)
        label.image = new_img_tk
    else:
        label = tk.Label(window, image=new_img_tk)
        label.image = new_img_tk
        label.pack()

    return window, label


def crear_carpetas():
    if not os.path.exists('graficas'):
        os.makedirs('graficas')
    if not os.path.exists('videos'):
        os.makedirs('videos')
    if not os.path.exists('videos/records'):
        os.makedirs('videos/records')
    if not os.path.exists('videos/generated'):
        os.makedirs('videos/generated')
    if not os.path.exists('csv'):
        os.makedirs('csv')

if args.record.lower() == "true":
    video_record = True
    nombre_video_grabado = f'videos/records/record_{num_test}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(nombre_video_grabado, fourcc, 30, (640, 480))

else:
    video_record = False

if args.control.lower() == "true":
    control_mode = True

else:
   control_mode = False

if args.windows.lower() == "true":
    windows_mode = True
else:
   windows_mode = False



def generate_inpainting_prompt(description, emotion, element):
    if gpt_mode:
        prompt = f"I am going to give you the description of an image and I need you to generate 10 prompts that are suitable for an inpainting model. These prompts should be generated with the goal of intensifying the following emotion in the image for the person viewing it: {emotion}. I also need you to generate prompts on a scale of 1 to 10, where 1 means to intensify the emotion a little and 10 means to intensify the emotion a lot, I need 1 prompt for each level. The inpainting model allows a maximum of 77 tokens so each of your generated prompts should be around that length or a little less. Try to generate prompts with concrete elements that intensify emotions, the inpainting model does not understand abstract things like emotions very well, so avoid generating prompts with adjectives that include emotions. Description: {description}. It is important that your answer must follow the next text format: emotion-level:prompt."
    else:
        emotion_values = {
        "anger": (-1, -1, -1),
        "fear": (-1, 1, -1),
        "disgust": (None, None, None),
        "happiness": (1, 1, 0),
        "sadness": (-1, -1, -1),
        "surprise": (0, 1, -1),
        "neutral": (0, 0, 0) 
    }

    # Create a dictionary with modifications for each element based on valence, arousal, and dominance values
    element_modifications = {
        "flower": {
            (-1, -1, -1): "Darker colors. The flower withers and shrinks. Petals and leaves harden. More pointed shapes and thorns.",
            (-1, 1, -1): "Darker colors. The flower blooms and grows. Petals and leaves harden. More pointed shapes and thorns.",
            (None, None, None): "Dull colors. The flower remains the same. Petals and leaves wrinkle. Irregular shapes.",
            (1, 1, 0): "Brighter colors. The flower blooms and grows. Petals and leaves soften. Rounded and smooth shapes.",
            (-1, -1, -1): "Darker colors. The flower withers and shrinks. Petals and leaves harden. More pointed shapes and thorns.",
            (0, 1, -1): "Varied colors. The flower blooms and grows. Petals and leaves harden. Unexpected and surprising shapes.",
            (0, 0, 0): "Neutral colors. The flower remains unchanged. No significant changes in shape or size."  # Prompt for "neutral" emotion
        },
        "hourglass": {
            (-1, -1, -1): "Darker colors. The hourglass becomes older, deteriorates. Time passes more slowly.",
            (-1, 1, -1): "Darker colors. The hourglass becomes older, deteriorates. Time passes more quickly.",
            (None, None, None): "Dull colors. The hourglass remains the same. Time passes randomly.",
            (1, 1, 0): "Brighter colors. The hourglass becomes more modern, new, and shiny. Time passes at the desired pace.",
            (-1, -1, -1): "Darker colors. The hourglass becomes older, deteriorates. Time passes more slowly.",
            (0, 1, -1): "Varied colors. The hourglass becomes randomly more modern or older. Time passes unpredictably.",
            (0, 0, 0): "Neutral colors. The hourglass remains unchanged. Time stands still."  # Prompt for "neutral" emotion
        }
    }

    # Get valence, arousal, and dominance values for the given emotion
    valence, arousal, dominance = emotion_values[emotion]

    # Get the modification for the given element based on valence, arousal, and dominance values
    modification = element_modifications[element][(valence, arousal, dominance)]

    # Create the prompt using the modification
    prompt = f"{modification}"

    # Return the prompt
    return prompt





def modify_century_image(img_path,mask_path,century, pipe):
    print(f"Modifying the elements of the image to the {century}th century")
    generated_image_name = f"image_century_{century}.png"
    base_generated_folder = os.path.join("imagenes", "paint", "imagen cambiada")
    generated_image_path = os.path.join(base_generated_folder, generated_image_name)
    init_image = load_image(img_path).resize((512, 512))
    mask_image = load_image(mask_path).resize((512, 512))
    pipe = pipe.to("mps")
    prompt = f"Make the table, the vase and the hourglass look {century}th century"
    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=50).images[0]
    image.save(generated_image_path)
    if args.video:
        image = cv2.imread(generated_image_path)
        for _ in range(30):
            video_writer.write(image)

    else:
        subprocess.Popen(["open", generated_image_path])

    return generated_image_path



#Regresar diccionario con elementos
def generate_elements_dict(archivo_csv,img_name,mode):
    elements = {}
    columns = ['sadness', 'neutral', 'fear', 'happiness', 'surprise', 'anger', 'disgust']
    initial_df = pd.DataFrame(columns=columns, index=range(10))
    with open(archivo_csv, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if len(row) == 0:
                break
            if len(row) >= 3:
                key = row[1]
                name = row[0]
                if name == img_name:
                    value = row[2]
                    if (mode == 1):
                        elements[key] = {'value': value, 'count': 0,'dataframe': initial_df.copy()}
                    elif(mode == 2):
                        elements[key] = {'value': value, 'count': 0,'prompt': ""}
    if not elements:
        return None
    
    return elements


def select_random_element(diccionario,img_path):

    img_directory = os.path.dirname(img_path)
    img_directory = os.path.dirname(img_directory)  
    masks_directory = os.path.join(img_directory, "mascaras")

    if not diccionario:
        return None, None
    
    llave_aleatoria = random.choice(list(diccionario.keys()))
    elemento = diccionario[llave_aleatoria]
    valor = elemento['value']
    mask_path = os.path.join(masks_directory, llave_aleatoria+".png")
    if elemento['count'] <=10:
        elemento['count'] += 1
    
    return llave_aleatoria,valor, mask_path



#Almacenar prompts en DF

def store_prompts_in_dataframe(df, formatted_reply, emotion):
    emotion_column_mapping = {
        'sadness': 'sadness',
        'neutral': 'neutral',
        'fear': 'fear',
        'happiness': 'happiness',
        'surprise': 'surprise',
        'anger': 'anger',
        'disgust': 'disgust'
    }
    emotion_column = emotion_column_mapping.get(emotion)
    if emotion_column is not None:
        df[emotion_column] = formatted_reply
    else:
        print("Emotion not recognized.")

def format_reply(text,  num_elements=10):
    lines = text.strip().split('\n')
    extracted_texts = []

    for line in lines:
        parts = line.split(': ')
        if len(parts) > 1:
            _, extracted_text = parts
            extracted_texts.append(extracted_text)
            
            if len(extracted_texts) >= num_elements:
                break

    return extracted_texts

def select_dataframe_in_dict(dict_key, data_dict):
    if dict_key in data_dict:
        return data_dict[dict_key]['dataframe'] 
    else:
        print(f"Key '{dict_key}' not found in the dictionary.")

def generate_prompts(prompt, df, emotion):
    if len(prompt.split()) <= 250:
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an artist specializing in image inpainting."},
            {"role": "user", "content": prompt}
        ],
        temperature=1,
        max_tokens=77*10,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    reply = response.choices[0].message.content
    formatted_reply = format_reply(reply)
    store_prompts_in_dataframe(df, formatted_reply, emotion)



#generacion de musica

def generate_music(description, img_path, emocion_actual):


    # Obtener el nombre de la imagen original sin la extensión
    image_name = os.path.splitext(os.path.basename(img_path))[0]

    # Construir el nombre de la carpeta para guardar la música generada
    output_folder = os.path.join(".", f"{image_name}_music")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Verificar si el archivo ya existe antes de generar música
    first_music_path = os.path.join(output_folder, f"musica_{emocion_actual}_0.wav")
    if os.path.exists(first_music_path):
        subprocess.Popen(["open", first_music_path])
        return
    

    model = MusicGen.get_pretrained('small', device='cpu')

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=5
    )

    # Generar música solo si no existe el archivo previamente

    output = model.generate(
        descriptions=[
            description
        ],
        progress=True
    )

    # Guardar el audio generado.
    for idx, one_wav in enumerate(output):
        music_name = f"musica_{emocion_actual}_{idx}"
        music_path = os.path.join(output_folder, music_name)

        # Verificar si el archivo ya existe antes de guardarlo
        if not os.path.exists(music_path):
            audio_write(music_path, one_wav.cpu(), model.sample_rate, strategy="loudness")

    # Reproducir solo el primer archivo generado
    subprocess.Popen(["open", first_music_path])

    # Display the generated audio.
    #display_audio(output, sample_rate=32000)


#Generación de imagenes

def load_image(file_path):
    return PIL.Image.open(file_path).convert("RGB")


import os
import subprocess
from PIL import Image

def generate_and_save_inpainting(img_path, mask_path, prompt, emocion_actual,pipe,video_writer=None):


    mask_name = os.path.splitext(os.path.basename(mask_path))[0]
    image_name = os.path.splitext(os.path.basename(img_path))[0]
    base_generated_folder = os.path.join("imagenes", "paint", "imagen cambiada")

    # Crear carpeta si no existe
    os.makedirs(base_generated_folder, exist_ok=True)

    # Construir la ruta de la imagen a guardar
    generated_image_name = f"imagen_resultante.png"
    generated_image_path = os.path.join(base_generated_folder, generated_image_name)

    init_image = load_image(img_path).resize((512, 512))
    mask_image = load_image(mask_path).resize((512, 512))
    pipe = pipe.to("mps")
    image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=50).images[0]
    image.save(generated_image_path)
    #Abrir la imagen generada
    if args.video:
        image = cv2.imread(generated_image_path)
        for _ in range(30):
            video_writer.write(image)

    else:
        subprocess.Popen(["open", generated_image_path])

    return generated_image_path
        



#Manejo de emociones

def procesar_archivo_csv(nombre_archivo, emocion_actual, emocion_objetivo):
    emocion_cambio = False
    ultimas_emociones = deque(maxlen=5)  # Ahora usamos una cola de longitud 5

    with open(nombre_archivo, 'r') as archivo:
        lector_csv = csv.reader(archivo)
        datos = list(lector_csv)

        # Obtener las últimas emociones de las últimas 5 filas
        for fila in datos[-5:]:
            ultimas_emociones.append(fila[3])

        emocion_anterior = emocion_actual

        if len(ultimas_emociones) < 5:
            # Obtener la emoción de la primera fila si hay menos de 5 filas
            emocion_actual = datos[0][3]
        else:
            # Calcular la emoción más repetida de las últimas 5 filas
            emocion_actual = mode(ultimas_emociones)

        if emocion_actual != emocion_anterior:
            emocion_cambio = True                

    return emocion_actual, emocion_anterior, emocion_cambio


def retrieve_prompts(data_dict, key, emotion, inpainting_prompt, current_df=None):
    if gpt_mode:
        image_prompt = None
        if key in data_dict and 'dataframe' in data_dict[key]:
            df = data_dict[key]['dataframe']
            count = data_dict[key]['count']
            if not df.empty and emotion in df.columns:
                value = df.loc[count - 1, emotion]
                if not pd.isna(value):
                    image_prompt = value
                else:
                    print("Prompt no encontrado. Generando Prompts")
                    generate_prompts(inpainting_prompt, current_df, emotion)
                    return retrieve_prompts(data_dict, key, emotion, inpainting_prompt, current_df)
        if image_prompt:
            print("Se encontró el prompt para generar imagen")
            print(data_dict)
        else:
            print("Prompt para generar imagen no encontrado en el archivo CSV.")
    else:
        if key in data_dict and 'prompt' in data_dict[key]:
            data_dict[key]['prompt'] = inpainting_prompt
            image_prompt = inpainting_prompt
            


    # Cambiar el valor de la variable prompt dependiendo de la emoción repetida
    if emocion_actual == 'anger':
        music_prompt = '80s pop track with bassy drums and synth'
    elif emocion_actual == 'disgust':
        music_prompt = '90s pop track with bassy drums and synth'
    elif emocion_actual == 'fear':
        music_prompt = '80s rock track with bassy drums and synth'
    elif emocion_actual == 'happiness':
        music_prompt = '70s pop track with bassy drums and synth'
    elif emocion_actual == 'sadness':
        music_prompt = '60s rock track with bassy drums and synth'
    elif emocion_actual == 'surprise':
        music_prompt = '2000s pop track with bassy drums and synth'
    elif emocion_actual == 'neutral':
        music_prompt = '2010s pop track with bassy drums and synth'

    return music_prompt, image_prompt





    

#Guardar en el csv

def save_data(emociones, personas, movements, dataframe=None):
    # Verificar si las listas están vacías
    if not emociones or not personas or not movements:
        return

    # Obtener la fecha y hora actual
    fecha_actual = datetime.now().strftime("%Y-%m-%d")
    hora_actual = datetime.now().strftime("%H:%M:%S")

    # Abrir el archivo CSV en modo de agregado ('a')
    with open(nombre_archivo, 'a', newline='') as archivo_csv:
        writer = csv.writer(archivo_csv)

        # Escribir cada elemento en una nueva fila
        for emocion, persona, movement in zip(emociones, personas, movements):
            if dataframe is not None:
                data_row = [num_test,fecha_actual, hora_actual, emocion, persona, movement]
                data_row += list(dataframe.iloc[0])  # Agregar datos del DataFrame
                writer.writerow(data_row)
            else:
                writer.writerow([fecha_actual, hora_actual, emocion, persona, movement])

    print("--------Datos guardados correctamente.---------------")

#Movimiento con mediapipe

def add_movement(movement_detected, pose_detected):
    movement_history = []
    if pose_detected:
        if movement_detected:
            movement_label = "Movimiento"
        else:
            movement_label = "Sin movimiento"
        movement_history.append(movement_label)
    else:
        movement_label = ""
    return movement_label, movement_history

def find_face_emotion(frame):
    single_face_prediction = detector.detect_image(frame)
    data = single_face_prediction
    df = single_face_prediction.emotions
    if len(df) == 1 and df.isnull().all().all():
        emotion_list = []
    else:
        dict = df.idxmax(axis=1).to_dict()
        emotion_list = list(dict.values())
    return emotion_list,data


def detect_movement(frame):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Convertir la imagen BGR a RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Realizar la detección de pose
        results = pose.process(image)

        # Determinar si hay movimiento
        movement_detected = False
        if results.pose_landmarks is not None:
            pose_detected = True
            # Comprueba la posición de múltiples articulaciones, como las muñecas, los codos, las rodillas y la cabeza
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
            head = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

            # Obtener las dimensiones de la imagen
            image_height, image_width, _ = image.shape

            # Normalizar las coordenadas de las articulaciones
            left_wrist_normalized = (left_wrist.x * image_width, left_wrist.y * image_height)
            right_wrist_normalized = (right_wrist.x * image_width, right_wrist.y * image_height)
            left_elbow_normalized = (left_elbow.x * image_width, left_elbow.y * image_height)
            right_elbow_normalized = (right_elbow.x * image_width, right_elbow.y * image_height)
            left_knee_normalized = (left_knee.x * image_width, left_knee.y * image_height)
            right_knee_normalized = (right_knee.x * image_width, right_knee.y * image_height)
            head_normalized = (head.x * image_width, head.y * image_height)

            movement_threshold = 50  # Umbral de movimiento en píxeles

            # Verificar si hay movimiento en alguna de las articulaciones seleccionadas
            if abs(left_wrist_normalized[0] - right_wrist_normalized[0]) > movement_threshold or \
                    abs(left_wrist_normalized[1] - right_wrist_normalized[1]) > movement_threshold:
                movement_detected = True
            elif abs(left_elbow_normalized[0] - right_elbow_normalized[0]) > movement_threshold or \
                    abs(left_elbow_normalized[1] - right_elbow_normalized[1]) > movement_threshold:
                movement_detected = True
            elif abs(left_knee_normalized[0] - right_knee_normalized[0]) > movement_threshold or \
                    abs(left_knee_normalized[1] - right_knee_normalized[1]) > movement_threshold:
                movement_detected = True
            elif abs(head_normalized[0] - 0.5 * image_width) > movement_threshold or \
                    abs(head_normalized[1] - 0.5 * image_height) > movement_threshold:
                movement_detected = True
        else:
            pose_detected = False


        return movement_detected,results, pose_detected


#Reconocimiento de rostros con dlib


def is_image_file(filename):
    try:
        img = Image.open(filename)
        img.verify()
        return True
    except:
        return False

def load_known_faces_and_encode(dir_name):
    print('Loading known faces ...')
    known_face_encodings = []
    known_face_names = []
    for name in os.listdir(dir_name):
        subdir = os.path.join(dir_name, name)
        if os.path.isdir(subdir):
            for filename in os.listdir(subdir):
                file_path = os.path.join(subdir, filename)
                if is_image_file(file_path):
                    image = face_recognition.load_image_file(file_path)
                    encoding = face_recognition.face_encodings(image)[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)
    return known_face_encodings, known_face_names



def extract_faces_and_encode(rgb_image):
    # specify the model cnn or hog, the later is the deafault.
    face_locations = face_recognition.face_locations(rgb_image, model = MODEL)
    face_encodings = face_recognition.face_encodings(rgb_image,face_locations)
    return face_locations, face_encodings


def find_face_matches(face_encodings, known_face_encodings, known_face_names):
    face_names = []
    for face_encoding in face_encodings:
        name = "Unknown"
        # TOLERANCE = 0.6
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, TOLERANCE)
        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]
        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
    return face_names




def init_camera():
    video_capture = cv2.VideoCapture(0)
    ret = video_capture.set(3, 640)
    ret = video_capture.set(4, 480)
    return video_capture


    
def acquire_image(video_capture, max_attempts=3):
    attempts = 0

    while attempts < max_attempts:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        
        if ret:
            scaled_rgb_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            scaled_rgb_frame = np.ascontiguousarray(scaled_rgb_frame[:, :, ::-1])
            temp_dir = tempfile.mkdtemp()
            temp_file = os.path.join(temp_dir, "temp_frame.jpg")
            cv2.imwrite(temp_file, scaled_rgb_frame)
            return frame, scaled_rgb_frame, temp_file
        else:
            attempts += 1

    print("--------No se pudo capturar la imagen / Fin del video------")
    return None, None, None

def show_frame(frame):
    # Display the resulting image frame in the PAC
    cv2.imshow('Video',frame)


#Dibujar figuras

def draw_face_info_on_frame(frame, face_names=None, face_emotions=None, results=None, movement_label=None, face_locations=None):
    # Draw Pose 
    if results:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

    # Draw Label Pose 
    if movement_label:
        cv2.putText(frame, movement_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Check if face information is provided
    if face_locations is not None and face_names is not None and face_emotions is not None:
        for (top, right, bottom, left), name, emotion in zip(face_locations, face_names, face_emotions):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with name and emotion below the face
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.8
            font_thickness = 1
            name_label_size, _ = cv2.getTextSize(name, font, font_scale, font_thickness)
            emotion_label_size, _ = cv2.getTextSize(emotion, font, font_scale, font_thickness)
            name_label_width = name_label_size[0]
            emotion_label_width = emotion_label_size[0]
            label_width = max(name_label_width, emotion_label_width)

            # Draw name and emotion labels separately
            cv2.putText(frame, name, (left + 6, bottom - 6 - int(label_width/2)), font, font_scale, (0, 0, 255), font_thickness)
            cv2.putText(frame, emotion, (left + 6, bottom - 6 + int(label_width/2) + name_label_size[1]), font, font_scale, (0, 0, 255), font_thickness)
    elif face_emotions:
        first_emotion = face_emotions[0]
        cv2.putText(frame, first_emotion, (frame.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


def resize_video(input_file, output_file):
    cap = cv2.VideoCapture(input_file)

    # Obtener el número de frames por segundo del video de entrada
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Obtener el tamaño del video de entrada
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Crear el objeto VideoWriter para el archivo de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Usamos el códec 'mp4v' para MP4
    out = cv2.VideoWriter(output_file, fourcc, fps, (640, 480))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar el frame a 640x480
        frame_resized = cv2.resize(frame, (640, 480))

        # Escribir el frame en el archivo de salida cada 1 segundo
        if frame_count % fps == 0:
            out.write(frame_resized)

        frame_count += 1

    # Liberar recursos
    cap.release()
    out.release()





# Crear la figura y los ejes del gráfico una sola vez
fig, ax = plt.subplots()
plt.xlabel('Fecha y Hora')
plt.ylabel('Emoción')
plt.title('Emoción en tiempo real')

# Definir colores predefinidos para cada emoción
colores_emociones = {
    'anger': 'red',
    'disgust': 'green',
    'fear': 'purple',
    'happiness': 'yellow',
    'sadness': 'blue',
    'neutral': 'gray',
    'surprise': 'orange'
}

# Función para leer el CSV y obtener los datos del test actual
def leer_csv(nombre_archivo):
    df = pd.read_csv(nombre_archivo, sep=',', usecols=range(5), header=None, names=['numero_test', 'fecha', 'hora', 'emocion', 'nombre_persona'])

    # Obtener el número del test actual (asumiendo que es el valor más reciente en la columna 'numero_test')
    numero_test_actual = df['numero_test'].iloc[-1]

    # Filtrar el DataFrame para obtener solo los datos del test actual
    df_actual = df[df['numero_test'] == numero_test_actual]

    # Obtener la emoción actual
    emocion_actual = df_actual['emocion'].iloc[-1]

    return df_actual, numero_test_actual, emocion_actual

# Función para graficar los datos del test actual
def graficar_emociones(df_actual, numero_test_actual, emocion_actual):
    # Obtener las fechas y horas en formato datetime
    df_actual['fecha_hora'] = pd.to_datetime(df_actual['fecha'] + ' ' + df_actual['hora'])

    # Limpiar la gráfica antes de mostrar los nuevos datos
    ax.clear()

    # Dibujar la línea de emociones
    ax.plot(df_actual['fecha_hora'], df_actual['emocion'], marker='o', linestyle='-', color=colores_emociones[emocion_actual])
    plt.xticks(rotation=45)

    plt.tight_layout()
    if not args.video and args.windows == "true": 
        plt.show() 
    plt.savefig(f'graficas/grafica_emociones_test_{numero_test_actual}.png')


#Programa principal

state_images = []

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    )

if args.video:
    resize_video(nombre_video,"video_temporal.mp4")

crear_carpetas()

if args.century != None:
    century = args.century
    generated_image_path = modify_century_image(img_path,century_mask_path,century,pipe)
else:
    generated_image_path = img_path


#################################################################
# SENSING - PERCEPTION
# Setup and initialization of perception

if  args.video and os.path.isfile("video_temporal.mp4"):
    video_capture = cv2.VideoCapture("video_temporal.mp4")
    #Iniciamos el writer
    nombre_video = f'videos/generated/video_{num_test}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video_writer = cv2.VideoWriter(nombre_video, fourcc, 30, (512, 512))
else:
    video_capture = init_camera()


# LONG TERM MEMORY
# Load database of known face and encode
known_faces_encodings, known_faces_names = load_known_faces_and_encode(KNOWN_FACES_DIR)
#################################################################

#Obtenemos la primera imagen para mostrar y creamos su diccionario correspondiente
current_image, last_row_index = get_current_image(narrative_csv, last_row_index) 
if current_image is not None:
    img_path = current_image["image_path"]
    img_duration = current_image["duration"]
    target_emotion = current_image["target_emotion"]
    img_name = current_image["image_name"]
    state_images.append((img_name, target_emotion))
else:
    print("Cant read the csv file with narrative")


window, label = show_image(img_path, window, label)
window.update()



if args.gpt_prompts.lower() == "true":
    gpt_mode = True
    elements_dict = generate_elements_dict("csv/descriptions.csv",img_name,1)
else:
     gpt_mode = False
     elements_dict = generate_elements_dict("csv/descriptions.csv",img_name,2)
if not elements_dict:
    print("There are not elements in the folder to create the dict")



#################################################################
# COMMUNICATION - LANGUAGE - PROTOCOLS
# Defining some time scales for communications
lastPublication = 0.0
lastImageChange = time.time()
PUBLISH_TIME   = 10     # seconds
start_state_time = time.time()
#################################################################

plt.ion()  # Activar el modo interactivo de matplotlib

#################################################################
# Perception-action-loop
#################################################################
# While ALIVE DO
try:
    while (True):
        

        #############################################################
        # SENSING LAYER
        rgb_frame, scaled_rgb_frame, temp_file = acquire_image(video_capture)
        if rgb_frame is None:
            break
        # Percepts are obtained
        face_locations, face_encodings = extract_faces_and_encode(scaled_rgb_frame)
        # Recognition is performed - Short-Long term memory
        face_names = find_face_matches(face_encodings, known_faces_encodings, known_faces_names)
        # Emotion recognition
        face_emotions,data = find_face_emotion(temp_file)
        #Pose recognition
        movement_detected,results,pose_detected = detect_movement(rgb_frame)
        #Save movement in array
        movement_label,movements = add_movement(movement_detected,pose_detected)

        if video_record:
            video_writer.write(rgb_frame)

        #Mostramos resultados
        print(movements)
        print(face_names)
        print(face_emotions)

        #############################################################
        
        
        #############################################################
        # COMMUNICATION LAYER: messages to trigger actions
        # on the external world (SPATIAL-temporal SCALES)

        if np.abs(time.time()-lastPublication) > PUBLISH_TIME or args.video:
            try:
                save_data(face_emotions,face_names,movements,data)
                if os.path.isfile(nombre_archivo):
                    emocion_actual,emocion_anterior,emocion_cambio = procesar_archivo_csv(nombre_archivo,emocion_actual,target_emotion)
                    print("Emocion actual: ", emocion_actual)
                    df,numero_test,emocion = leer_csv(nombre_archivo)
                    graficar_emociones(df, numero_test,emocion)
                    if (emocion_cambio or args.video):
                        end_state_time = time.time()
                        state_time = round(end_state_time - start_state_time,2)
                        print(state_images)
                        fsm, start_state_time, state_index = update_fsm(fsm, emocion_anterior, emocion_actual, state_images, state_time, state_index)
                        print(fsm)
                        state_images.clear()
                        state_images.append((img_name, target_emotion))
                        if control_mode:
                            random_element, image_description, mask_path = select_random_element(elements_dict, img_path)
                            print("---------------------Elemento a modificar: ", random_element)
                            inpainting_prompt = generate_inpainting_prompt(image_description,emocion_actual, random_element)
                            if gpt_mode:
                                current_df = select_dataframe_in_dict(random_element, elements_dict) #OJO
                                music_prompt, image_prompt = retrieve_prompts(elements_dict, random_element, emocion_actual, inpainting_prompt, current_df)
                            else:
                                music_prompt, image_prompt = retrieve_prompts(elements_dict, random_element, emocion_actual, inpainting_prompt)
                            print(music_prompt)
                            print(image_prompt)
                                
                            print("Entro al if emocion_cambio")
                            if(emocion_actual != target_emotion):
                                print("Entro al if emocion_distinta")
                                if not args.video:
                                    generate_music(music_prompt, img_path, emocion_actual)
                                    generated_image_path = generate_and_save_inpainting(generated_image_path, mask_path, image_prompt, emocion_actual, pipe)
                                else:
                                    generated_image_path = generate_and_save_inpainting(generated_image_path, mask_path, image_prompt, emocion_actual, pipe,video_writer)
                            else:
                                if args.video:
                                    image = cv2.imread(img_path)
                                    for _ in range(30):
                                        video_writer.write(image)
                                else:
                                    image = Image.open(img_path)
                                    image.show()
                        else:
                            print("No control")
                else:
                    print("Archivo csv no encontrado/generado aun.")    
            except (KeyboardInterrupt):
                break
            except Exception as e:
                print(e)
            lastPublication = time.time()

        #############################################################
        
        
        #############################################################
        # CONTROL LAYER - triggered actions to
        # the local/remote environment
        if(windows_mode):
            draw_face_info_on_frame(rgb_frame, face_names, face_emotions,results,movement_label, face_locations)
            show_frame (rgb_frame)

        if np.abs(time.time()-lastImageChange) > img_duration and not args.video:
            try:
                current_image, last_row_index = get_current_image(narrative_csv, last_row_index) 
                if current_image is not None:
                    img_path = current_image["image_path"]
                    img_duration = current_image["duration"]
                    target_emotion = current_image["target_emotion"]
                    img_name = current_image["image_name"]
                    state_images.append((img_name, target_emotion))
                    if args.gpt_prompts.lower() == "true":
                        gpt_mode = True
                        elements_dict = generate_elements_dict("csv/descriptions.csv",img_name,1)
                    else:
                        gpt_mode = False
                        elements_dict = generate_elements_dict("csv/descriptions.csv",img_name,2)
                    if not elements_dict:
                        print("There are not elements in the folder to create the dict")

                else:
                    end_state_time = time.time()
                    state_time = round(end_state_time - start_state_time,2)
                    window.destroy()
                    fsm[emocion_actual]["tiempo_mantenido"] += state_time
                    save_fsm(fsm,f"csv/fsm_{num_test}.csv")
                    draw_fsm(fsm, numero_test)
                    print("Fin de la narrativa")
                    break
                window, label = show_image(img_path, window, label)
                window.update()
            except (KeyboardInterrupt):
                break
            except Exception as e:
                print(e)
            lastImageChange = time.time()
        
        
        
        
        #############################################################

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # END OF THE GAME/LIFE
            break
except KeyboardInterrupt:
    # Cerrar correctamente el escritor de video antes de finalizar
    if video_record or args.video:
        video_writer.release()

    # Cerrar el objeto de captura de video y las ventanas de OpenCV
    video_capture.release()
    cv2.destroyAllWindows() 

# LAST STUFF BEFORE BEING OFICIALLY DEAD
# Release handle to the webcam
video_capture.release()

if args.video:
    video_writer.release()

cv2.destroyAllWindows()
