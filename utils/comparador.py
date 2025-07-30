import cv2, numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import imagehash

def comparar_imagenes(img1_path, img2_path):
    detalles = {}

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    detalles["SSIM"] = round(score * 100, 2)

    abs_diff = cv2.absdiff(img1, img2)
    non_zero = np.count_nonzero(abs_diff)
    total = abs_diff.size
    detalles["Pixeles"] = round((1 - (non_zero / total)) * 100, 2)

    hash1 = imagehash.average_hash(Image.open(img1_path))
    hash2 = imagehash.average_hash(Image.open(img2_path))
    distancia_hash = hash1 - hash2
    detalles["ImageHash"] = round(max(0, (1 - distancia_hash / 64) * 100), 2)

    similitud_total = round(np.mean(list(detalles.values())), 2)

    return {
        "similitud_total": similitud_total,
        "detalles": detalles
    }
