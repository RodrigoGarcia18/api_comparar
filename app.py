from flask import Flask, request, jsonify
import os, requests
from utils.comparador import comparar_imagenes
import numpy as np

app = Flask(__name__)

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# URL base de imágenes registradas en Biblioteca Gráfica
base_url = "http://190.116.178.163/Biblioteca_Grafica/Fotos/"

# Función de serialización segura
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

    url_usuario = request.form.get("url")  # ahora recibimos la URL completa
    print(f"➡️ [DEBUG] URL recibida: {url_usuario}")

    if not url_usuario:
        print("❌ [ERROR] No se envió la URL de la imagen")
        return jsonify({"error": "Debes enviar la URL de la imagen"}), 400

    # Extraer el código del nombre del archivo
    codigo = os.path.splitext(os.path.basename(url_usuario))[0]
    print(f"➡️ [DEBUG] Código extraído de la URL: {codigo}")

    # Descargar la imagen del usuario
    imagen_usuario_path = os.path.join(data_dir, f"{codigo}_usuario.jpg")
    try:
        print(f"🌐 [INFO] Descargando imagen del usuario desde {url_usuario}")
        resp = requests.get(url_usuario, timeout=10)
        if resp.status_code != 200:
            return jsonify({"error": "No se pudo descargar la imagen enviada"}), 400
        with open(imagen_usuario_path, "wb") as f:
            f.write(resp.content)
    except Exception as e:
        print(f"❌ [ERROR] No se pudo descargar la imagen enviada: {e}")
        return jsonify({"error": "Error al descargar la imagen enviada"}), 500

    # Buscar la imagen base en Biblioteca Gráfica
    extensiones = [".jpg", ".JPG", ".jpeg", ".png"]
    imagen_base_path = os.path.join(data_dir, f"{codigo}_base.jpg")

    url_imagen_base = None
    for ext in extensiones:
        test_url = f"{base_url}{codigo}{ext}"
        try:
            print(f"🔎 [INFO] Buscando imagen base en: {test_url}")
            resp = requests.get(test_url, timeout=10)
            if resp.status_code == 200:
                with open(imagen_base_path, "wb") as f:
                    f.write(resp.content)
                url_imagen_base = test_url
                print(f"✅ [INFO] Imagen base encontrada en: {test_url}")
                break
        except Exception as e:
            print(f"⚠️ [WARNING] Error probando {test_url}: {e}")

    if not url_imagen_base:
        print(f"❌ [ERROR] No existe imagen base para el código {codigo}")
        return jsonify(serializar({
            "existe": False,
            "error": f"No existe imagen base con código {codigo}"
        })), 404

    # Comparar imágenes
    try:
        print(f"🔄 [INFO] Comparando imagen recibida con imagen base")
        resultados = comparar_imagenes(imagen_usuario_path, imagen_base_path)
        print(f"📊 [INFO] Resultados: {resultados}")
    except Exception as e:
        print(f"❌ [ERROR] Fallo en la comparación: {e}")
        return jsonify(serializar({"error": "Error al comparar imágenes"})), 500

    autenticado = resultados["detalles"].get("FaceRecognition", 0) >= 85

    respuesta = {
        "codigo": codigo,
        "existe": True,
        "url_imagen_base": url_imagen_base,
        "url_imagen_usuario": url_usuario,
        "autenticado": autenticado,
        "estadisticas": resultados
    }

    # Limpieza
    try:
        if os.path.exists(imagen_usuario_path):
            os.remove(imagen_usuario_path)
        if os.path.exists(imagen_base_path):
            os.remove(imagen_base_path)
    except:
        pass

    return jsonify(serializar(respuesta))

if __name__ == "__main__":
    print("🚀 [INFO] Iniciando servidor Flask en 0.0.0.0:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
