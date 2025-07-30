import cv2, numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import imagehash
import face_recognition

def comparar_imagenes(img1_path, img2_path):
    detalles = {}

    # Leer imágenes
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # SSIM
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(gray1, gray2, full=True)
    detalles["SSIM"] = round(score * 100, 2)

    # Diferencia de píxeles
    abs_diff = cv2.absdiff(img1, img2)
    non_zero = np.count_nonzero(abs_diff)
    total = abs_diff.size
    detalles["Pixeles"] = round((1 - (non_zero / total)) * 100, 2)

    # ImageHash
    hash1 = imagehash.average_hash(Image.open(img1_path))
    hash2 = imagehash.average_hash(Image.open(img2_path))
    distancia_hash = hash1 - hash2
    detalles["ImageHash"] = round(max(0, (1 - distancia_hash / 64) * 100), 2)

    # Face Recognition (si hay rostros)
    try:
        encodings1 = face_recognition.face_encodings(face_recognition.load_image_file(img1_path))[0]
        encodings2 = face_recognition.face_encodings(face_recognition.load_image_file(img2_path))[0]
        dist = face_recognition.face_distance([encodings1], encodings2)[0]
        detalles["FaceRecognition"] = round((1 - dist) * 100, 2)
    except:
        detalles["FaceRecognition"] = 0.0

    # Porcentaje final (ponderado)
    similitud_total = round(np.mean(list(detalles.values())), 2)

    return {
        "similitud_total": similitud_total,
        "detalles": detalles
    }
