from flask import Flask, request, jsonify
import os, requests, tempfile
from utils.comparador import comparar_imagenes
import numpy as np

app = Flask(__name__)

base_url = "http://190.116.178.163/Biblioteca_Grafica/Fotos/"

# Serializador seguro para JSON
def serializar(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: serializar(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [serializar(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

@app.route("/comparar", methods=["POST"])
def comparar():
    print("📩 [INFO] Solicitud recibida en /comparar")

    url_usuario = request.form.get("url")
    if not url_usuario:
        return jsonify({"error": "Debes enviar la URL de la imagen"}), 400

    # Extraer código
    codigo = os.path.splitext(os.path.basename(url_usuario))[0]
    print(f"➡️ [DEBUG] Código extraído: {codigo}")

    try:
        # Descargar imagen del usuario en archivo temporal
        resp = requests.get(url_usuario, timeout=10)
        if resp.status_code != 200:
            return jsonify({"error": "No se pudo descargar la imagen enviada"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            f.write(resp.content)
            img_usuario_path = f.name
    except Exception as e:
        return jsonify({"error": f"Error al descargar imagen: {e}"}), 500

    # Descargar imagen base
    img_base_path = None
    url_imagen_base = None

    for ext in [".jpg", ".jpeg", ".png"]:
        test_url = f"{base_url}{codigo}{ext}"
        try:
            resp = requests.get(test_url, timeout=10)
            if resp.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                    f.write(resp.content)
                    img_base_path = f.name
                url_imagen_base = test_url
                break
        except:
            pass

    if not url_imagen_base:
        os.remove(img_usuario_path)
        return jsonify(serializar({
            "existe": False,
            "error": f"No existe imagen base con código {codigo}"
        })), 404

    # Comparar imágenes
    try:
        resultados = comparar_imagenes(img_usuario_path, img_base_path)
    except Exception as e:
        return jsonify({"error": f"Error al comparar imágenes: {e}"}), 500
    finally:
        # Eliminar archivos temporales
        if os.path.exists(img_usuario_path):
            os.remove(img_usuario_path)
        if os.path.exists(img_base_path):
            os.remove(img_base_path)

    autenticado = resultados["detalles"].get("FaceRecognition", 0) >= 85

    return jsonify(serializar({
        "codigo": codigo,
        "existe": True,
        "url_imagen_base": url_imagen_base,
        "url_imagen_usuario": url_usuario,
        "autenticado": autenticado,
        "estadisticas": resultados
    }))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
