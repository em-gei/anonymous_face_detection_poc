import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# --- CONFIGURAZIONE ---
script_dir = os.path.dirname(os.path.abspath(__file__))
LANDMARKER_MODEL_PATH = os.path.join(script_dir, 'face_landmarker.task')
FACE_DETECTOR_MODEL_PATH = os.path.join(script_dir, 'blaze_face_short_range.tflite')

IMAGE_PATH = os.path.join(script_dir, 'test_images', 'photo1.jpg')

# --- FUNZIONI DI SUPPORTO ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh_connections = mp.solutions.face_mesh

def draw_anon_image(rgb_image, landmark_result):
    """Genera l'immagine anonima (Wireframe)"""
    face_landmarks_list = landmark_result.face_landmarks
    annotated_image = np.zeros_like(rgb_image)

    for face_landmarks in face_landmarks_list:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in face_landmarks
        ])
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp_face_mesh_connections.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    return annotated_image

def detect_face_mediapipe(img, detector, name):
    """Prova a trovare un volto usando il Face Detector di MediaPipe"""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)

    print(f"--- Risultati su {name} ---")

    if detection_result.detections:
        print(f"⚠️  SUCCESSO: Trovati {len(detection_result.detections)} volti! (Privacy NON garantita)")
        # Disegna il rettangolo di rilevamento per visualizzazione
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            start_point = (bbox.origin_x, bbox.origin_y)
            end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
            cv2.rectangle(img, start_point, end_point, (0, 0, 255), 4)
            cv2.putText(img, "VOLTO RILEVATO", (bbox.origin_x, bbox.origin_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    else:
        print(f"✅ FALLIMENTO: Nessun volto trovato. (Privacy GARANTITA)")

    return img

def main():
    # 1. CARICAMENTO DEL FACE LANDMARKER (per generare l'immagine anonima)
    try:
        with open(LANDMARKER_MODEL_PATH, 'rb') as f: landmark_model_data = f.read()
    except Exception:
        print("Manca il modello face_landmarker.task!")
        return

    base_options_lm = python.BaseOptions(model_asset_buffer=landmark_model_data)
    options_lm = vision.FaceLandmarkerOptions(base_options=base_options_lm, num_faces=1)
    landmarker_detector = vision.FaceLandmarker.create_from_options(options_lm)

    # 2. CARICAMENTO DEL FACE DETECTOR (per testare l'anonimato)
    try:
        if not os.path.exists(FACE_DETECTOR_MODEL_PATH):
             print("\nATTENZIONE: Usando FaceLandmarker per il test di Detection.")
             face_test_detector = vision.FaceLandmarker.create_from_options(options_lm)
        else:
             with open(FACE_DETECTOR_MODEL_PATH, 'rb') as f: detector_model_data = f.read()
             base_options_det = python.BaseOptions(model_asset_buffer=detector_model_data)
             options_det = vision.FaceDetectorOptions(base_options=base_options_det, min_detection_confidence=0.5)
             face_test_detector = vision.FaceDetector.create_from_options(options_det)
        print("✅ Detector di test caricato correttamente.")

    except Exception as e:
        print(f"Errore caricamento Face Detector di test: {e}")
        return

    # 3. CARICAMENTO IMMAGINE
    if not os.path.exists(IMAGE_PATH):
        print(f"Errore: Non trovo l'immagine {IMAGE_PATH}")
        return
    with open(IMAGE_PATH, 'rb') as f:
        file_bytes = bytearray(f.read())
        numpy_array = np.asarray(file_bytes, dtype=np.uint8)
        original_img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

    # 4. GENERAZIONE IMMAGINE ANONIMA
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    landmark_result = landmarker_detector.detect(mp_image)

    if not landmark_result.face_landmarks:
        print("MediaPipe non ha trovato il volto nell'originale! Usa una foto migliore.")
        return

    anon_img = draw_anon_image(original_img, landmark_result)

    # 5. ESPERIMENTO DI VALIDAZIONE
    print("\n--- AVVIO ATTACCO PRIVACY CON MEDIAPIPE ---")
    res_orig = detect_face_mediapipe(original_img.copy(), face_test_detector, "FOTO ORIGINALE")
    res_anon = detect_face_mediapipe(anon_img.copy(), face_test_detector, "FOTO ANONIMIZZATA")

    # Mostra confronto
    scale = 0.5
    h, w = res_orig.shape[:2]
    new_dim = (int(w*scale), int(h*scale))

    res_orig_s = cv2.resize(res_orig, new_dim)
    res_anon_s = cv2.resize(res_anon, new_dim)

    comparison = cv2.hconcat([res_orig_s, res_anon_s])

    print("\nMostra risultati... (Premi ESC per chiudere)")
    cv2.imshow('Esperimento Privacy: Sinistra (Detected) vs Destra (Not Detected)', comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
