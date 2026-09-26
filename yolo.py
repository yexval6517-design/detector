# -*- coding: utf-8 -*-
"""
DETECTOR DE PERSONAS CON YOLO - Versión sencilla (paso 1)
Sube una imagen y detecta personas usando un modelo YOLO ya entrenado.
Próximo paso: detectar específicamente casco y chaleco de seguridad.
"""

from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Detector de Personas (YOLO)", page_icon="🦺", layout="centered")

st.title("🦺 Detector de Personas en Obra")
st.caption("Sube una foto y el sistema detecta cuántas personas hay usando YOLO.")

# Cargamos el modelo una sola vez y lo dejamos en caché para que no se recargue cada vez
@st.cache_resource
def cargar_modelo():
    # yolov8n = versión "nano": la más liviana y rápida, ideal para empezar
    return YOLO("yolov8n.pt")

modelo = cargar_modelo()

archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen = Image.open(archivo).convert("RGB")

    with st.spinner("Analizando la imagen..."):
        resultados = modelo(imagen)

    resultado = resultados[0]
    imagen_anotada = resultado.plot()  # imagen con los cuadros dibujados (formato numpy, BGR)
    imagen_anotada = imagen_anotada[:, :, ::-1]  # convertir BGR a RGB para mostrarla bien

    # Contamos cuántas detecciones son de la clase "person"
    nombres = resultado.names
    conteo_personas = sum(
        1 for c in resultado.boxes.cls if nombres[int(c)] == "person"
    )

    st.image(imagen_anotada, caption="Resultado de la detección", use_container_width=True)
    st.success(f"👷 Personas detectadas: {conteo_personas}")

    with st.expander("Ver todas las detecciones"):
        for c, conf in zip(resultado.boxes.cls, resultado.boxes.conf):
            st.write(f"- {nombres[int(c)]} (confianza: {float(conf):.2f})")
else:
    st.info("👆 Sube una imagen para comenzar.")
