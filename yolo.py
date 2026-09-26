# -*- coding: utf-8 -*-
"""
DETECTOR DE EPP (Equipo de Protección Personal) - Versión 2
Detecta casco, chaleco, arnés de alturas, guantes y gafas.
Muestra una alerta si detecta a alguien SIN algún elemento de seguridad.
"""

from PIL import Image
import streamlit as st
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Detector de EPP", page_icon="🦺", layout="centered")

st.title("🦺 Detector de Equipo de Protección Personal")
st.caption("Sube una foto de un trabajador y el sistema detecta si tiene puesto el equipo de seguridad completo (casco, chaleco, arnés de alturas, guantes, gafas).")

# Traducción de las clases que detecta el modelo, para mostrarlas en español
TRADUCCION = {
    "Hardhat": "Casco",
    "Safety Vest": "Chaleco de seguridad",
    "Gloves": "Guantes",
    "Goggles": "Gafas de seguridad",
    "Mask": "Tapabocas",
    "NO-Hardhat": "Sin casco",
    "NO-Safety Vest": "Sin chaleco de seguridad",
    "NO-Gloves": "Sin guantes",
    "NO-Goggles": "Sin gafas de seguridad",
    "NO-Mask": "Sin tapabocas",
    "No_Harness": "Sin arnés de alturas",
    "Fall-Detected": "Caída detectada",
    "Person": "Persona",
}

# Clases que representan una VIOLACIÓN de seguridad (faltante)
CLASES_ALERTA = {
    "NO-Hardhat", "NO-Safety Vest", "NO-Gloves",
    "NO-Goggles", "NO-Mask", "No_Harness", "Fall-Detected"
}

@st.cache_resource
def cargar_modelo():
    # Modelo YOLOv8 ya entrenado específicamente para detección de EPP
    ruta_pesos = hf_hub_download(repo_id="ayushgupta7777/safetyvision-yolov8", filename="v2/best.pt")
    return YOLO(ruta_pesos)

with st.spinner("Cargando modelo de detección de EPP (solo la primera vez tarda más)..."):
    modelo = cargar_modelo()

archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if archivo is not None:
    imagen = Image.open(archivo).convert("RGB")

    with st.spinner("Analizando la imagen..."):
        resultados = modelo(imagen)

    resultado = resultados[0]
    imagen_anotada = resultado.plot()[:, :, ::-1]  # BGR -> RGB

    nombres = resultado.names
    detecciones = [nombres[int(c)] for c in resultado.boxes.cls]

    st.image(imagen_anotada, caption="Resultado de la detección", use_container_width=True)

    violaciones = [d for d in detecciones if d in CLASES_ALERTA]

    if violaciones:
        st.error("🚨 ALERTA: Se detectaron incumplimientos de seguridad")
        for v in sorted(set(violaciones)):
            cuenta = violaciones.count(v)
            st.write(f"- {TRADUCCION.get(v, v)} ({cuenta} caso{'s' if cuenta > 1 else ''})")
    else:
        st.success("✅ No se detectaron incumplimientos de seguridad visibles en la imagen.")

    with st.expander("Ver todas las detecciones"):
        for c, conf in zip(resultado.boxes.cls, resultado.boxes.conf):
            nombre = nombres[int(c)]
            st.write(f"- {TRADUCCION.get(nombre, nombre)} (confianza: {float(conf):.2f})")
else:
    st.info("👆 Sube una imagen para comenzar.")
