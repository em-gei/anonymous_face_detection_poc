import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import os
import glob

# --- CONFIGURAZIONE ---
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, 'face_landmarker.task')
INPUT_FOLDER = os.path.join(script_dir, 'test_images')
EXTENSIONS = ['*.jpg', '*.jpeg', '*.png']
EAR_THRESHOLD = 0.25

# --- INDICI OCCHI ---
LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]

# Setup utilità di disegno (Solo stili e costanti, niente modelli)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh_connections = mp.solutions.face_mesh

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def calculate_ear(landmarks, indices, img_w, img_h):
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))

    p1, p2, p3, p4, p5, p6 = coords
    v1 = euclidean_distance(p2, p6)
    v2 = euclidean_distance(p3, p5)
    h = euclidean_distance(p1, p4)

    if h == 0: return 0.0
    return (v1 + v2) / (2.0 * h)

def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Disegna la mesh anonima costruendo manualmente l'oggetto Protobuf.
    Questo evita di dover chiamare la vecchia API buggata.
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.zeros_like(rgb_image) # Tela Nera

    # Cicla su tutte le facce rilevate
    for face_landmarks in face_landmarks_list:

        # --- FIX DEFINITIVO ---
        # Costruiamo manualmente la lista di landmark nel formato che piace a mp_drawing
        # senza inizializzare nessun modello legacy.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in face_landmarks
        ])

        # Disegna Tassellatura (Pelle)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp_face_mesh_connections.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        # Disegna Contorni (Occhi, Bocca)
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp_face_mesh_connections.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

    return annotated_image

def main():
    print(f"Directory di lavoro: {script_dir}")

    # Caricamento Modello in Memoria (Buffer)
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = f.read()
        print(f"Modello caricato in RAM ({len(model_data)} bytes).")
    except Exception as e:
        print(f"ERRORE: Impossibile leggere il file {MODEL_PATH}")
        print(f"Dettagli: {e}")
        return

    # Inizializzazione Detector
    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # Raccolta Immagini
    image_files = []
    for ext in EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    if not image_files:
        print(f"Nessuna immagine trovata in '{INPUT_FOLDER}'.")
        # Crea la cartella se non esiste
        if not os.path.exists(INPUT_FOLDER):
            os.makedirs(INPUT_FOLDER)
            print(f"Ho creato la cartella '{INPUT_FOLDER}'. Mettici dentro delle foto!")
        return

    print("Avvio elaborazione...")
    print("--- INIZIO DEBUG ---")
    print(f"Cerco immagini in: {INPUT_FOLDER}")
    print(f"File trovati dalla ricerca: {len(image_files)}")

    for file_path in image_files:
        # 1. Leggi il file come stream di byte (Python nativo)
        with open(file_path, 'rb') as f:
            file_bytes = bytearray(f.read())

        # 2. Converti i byte in array numpy
        numpy_array = np.asarray(file_bytes, dtype=np.uint8)

        # 3. Decodifica l'immagine da memoria
        image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        # -----------------------------------------

        if image is None: continue
        h, w, _ = image.shape

        # BGR -> RGB e creazione oggetto mp.Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # RILEVAMENTO
        detection_result = detector.detect(mp_image)

        # Output su Tela Nera
        black_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        status_text = "Nessun volto"
        color = (100, 100, 100)

        if detection_result.face_landmarks:
            # 1. Disegna Anonimato (Funzione corretta)
            black_canvas = draw_landmarks_on_image(image, detection_result)

            # 2. Calcola EAR
            face_landmarks = detection_result.face_landmarks[0]
            left_ear = calculate_ear(face_landmarks, LEFT_EYE_IDXS, w, h)
            right_ear = calculate_ear(face_landmarks, RIGHT_EYE_IDXS, w, h)
            ear_avg = (left_ear + right_ear) / 2.0

            if ear_avg < EAR_THRESHOLD:
                status_text = "SONNOLENZA (Occhi Chiusi)"
                color = (0, 0, 255)
            else:
                status_text = "SVEGLIO"
                color = (0, 255, 0)

            # Testo a schermo
            cv2.putText(black_canvas, f"EAR: {ear_avg:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(black_canvas, status_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Visualizza
        combined = cv2.hconcat([image, black_canvas])
        # Resize dinamico per schermi piccoli
        scale = 50
        if combined.shape[1] > 1500: scale = 40
        dim = (int(combined.shape[1] * scale / 100), int(combined.shape[0] * scale / 100))
        resized = cv2.resize(combined, dim, interpolation = cv2.INTER_AREA)

        cv2.imshow('POC Risultato', resized)
        if cv2.waitKey(0) == 27: break # ESC per uscire

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
