import joblib
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_mensajes.pkl"

RESPUESTAS = {
    "queja": "Lamentamos lo ocurrido. Tu caso será derivado al área de atención al cliente para revisión prioritaria.",
    "correo": "Tu mensaje fue registrado correctamente y será revisado por el área correspondiente.",
    "venta": "Gracias por tu interés. Un asesor comercial podrá continuar con la atención de tu solicitud.",
}

EJEMPLOS = [
    "Quiero presentar una queja porque mi pedido llegó dañado.",
    "Adjunto los documentos solicitados para continuar con el trámite.",
    "Estoy interesado en comprar 15 unidades y necesito una cotización.",
]

st.set_page_config(page_title="Clasificador de mensajes", page_icon="🤖", layout="centered")

st.title("🤖 Asistente inteligente para clasificar mensajes")
st.write(
    "Este prototipo interpreta mensajes escritos por usuarios y los clasifica en **queja**, **correo** o **venta**."
)

with st.sidebar:
    st.header("Ejemplos de prueba")
    for i, ejemplo in enumerate(EJEMPLOS, start=1):
        st.markdown(f"**{i}.** {ejemplo}")
    st.caption("Si todavía no entrenaste el modelo, ejecuta primero: python train_model.py")

if not MODEL_PATH.exists():
    st.error("No se encontró el modelo entrenado. Ejecuta primero train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

mensaje = st.text_area(
    "Escribe el mensaje del cliente",
    height=180,
    placeholder="Ejemplo: Quiero reclamar porque me cobraron dos veces y nadie responde.",
)

if st.button("Clasificar mensaje", type="primary"):
    if not mensaje.strip():
        st.warning("Por favor, escribe un mensaje antes de clasificar.")
    else:
        categoria = model.predict([mensaje])[0]
        st.success(f"Categoría detectada: {categoria.upper()}")
        st.info(f"Respuesta sugerida: {RESPUESTAS.get(categoria, 'Mensaje procesado correctamente.')}")

        with st.expander("Detalle para la presentación"):
            st.write("- Entrada: mensaje escrito por el usuario")
            st.write("- Proceso: clasificación automática de texto con IA")
            st.write("- Salida: categoría sugerida para atención")
