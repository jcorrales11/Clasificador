from __future__ import annotations

import csv
import os
import smtplib
import uuid
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_mensajes.pkl"
CASES_PATH = BASE_DIR / "historial_casos.csv"

CATEGORY_UI = {
    "queja": {
        "title": "Queja",
        "icon": "🚨",
        "color": "#ef4444",
        "message": "Tu queja se ha registrado con éxito y será revisada por el equipo de atención al cliente.",
        "next_step": "Próximo paso: revisión prioritaria del caso y contacto del área responsable.",
    },
    "correo": {
        "title": "Correo / consulta",
        "icon": "📩",
        "color": "#2563eb",
        "message": "Tu solicitud se ha registrado con éxito y fue enviada al área correspondiente.",
        "next_step": "Próximo paso: validación de la información y seguimiento interno del caso.",
    },
    "venta": {
        "title": "Venta",
        "icon": "💼",
        "color": "#0f766e",
        "message": "Tu solicitud comercial se ha registrado con éxito y será atendida por el equipo de ventas.",
        "next_step": "Próximo paso: contacto de un asesor para cotización o propuesta comercial.",
    },
}

DEFAULT_THRESHOLD = 0.55
MIN_WORDS = 3
EXAMPLES = [
    "Quiero reportar que mi pedido llegó dañado y necesito un reembolso.",
    "Adjunto la documentación solicitada para continuar mi trámite.",
    "Estoy interesado en una cotización para 25 unidades de su producto.",
]


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f5f7fb 0%, #eef2ff 100%);
        }
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        .hero-card {
            background: linear-gradient(135deg, #111827 0%, #1f2937 48%, #312e81 100%);
            color: white;
            border-radius: 24px;
            padding: 28px 32px;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
            margin-bottom: 1rem;
        }
        .brand-chip {
            display: inline-block;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.18);
            color: #ffffff;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
        }
        .hero-title {
            font-size: 2.3rem;
            font-weight: 800;
            line-height: 1.1;
            margin: 0;
        }
        .hero-subtitle {
            color: #dbeafe;
            font-size: 1rem;
            margin-top: 0.7rem;
            margin-bottom: 0;
        }
        .metric-card {
            background: white;
            border-radius: 18px;
            padding: 18px 18px;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            border: 1px solid #e5e7eb;
            margin-bottom: 0.8rem;
        }
        .metric-label {
            color: #6b7280;
            font-size: 0.9rem;
            margin-bottom: 0.2rem;
        }
        .metric-value {
            color: #111827;
            font-size: 1.3rem;
            font-weight: 700;
        }
        .section-card {
            background: white;
            border-radius: 22px;
            padding: 22px;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
            border: 1px solid #e5e7eb;
        }
        .result-box {
            border-radius: 18px;
            padding: 18px 20px;
            margin-top: 0.9rem;
            margin-bottom: 0.9rem;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
        }
        .stTextArea textarea, .stTextInput input {
            border-radius: 14px !important;
        }
        .stButton>button {
            border-radius: 12px;
            font-weight: 700;
            padding: 0.6rem 1.2rem;
            border: none;
            background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
            color: white;
        }
        .sidebar-card {
            background: rgba(255,255,255,0.7);
            border-radius: 18px;
            padding: 14px 16px;
            border: 1px solid #e5e7eb;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifact():
    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact
    return {"model": artifact, "threshold": DEFAULT_THRESHOLD, "version": "1.0"}


def classify_message(model, threshold: float, message: str):
    clean = " ".join(message.strip().split())
    words = clean.split()
    if len(words) < MIN_WORDS:
        return {
            "status": "manual_review",
            "category": None,
            "confidence": 0.0,
            "reason": "El mensaje es demasiado corto. Añade más detalle para clasificarlo mejor.",
        }

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([clean])[0]
        classes = list(model.classes_)
        best_index = max(range(len(probs)), key=lambda i: probs[i])
        confidence = float(probs[best_index])
        category = classes[best_index]
    else:
        category = model.predict([clean])[0]
        confidence = 0.60

    if confidence < threshold:
        return {
            "status": "manual_review",
            "category": None,
            "confidence": confidence,
            "reason": "La confianza del modelo es baja. Conviene pedir más contexto o revisar manualmente.",
        }

    return {
        "status": "classified",
        "category": category,
        "confidence": confidence,
        "reason": "Clasificación realizada con éxito.",
    }


def generate_ticket_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d")
    short = str(uuid.uuid4()).split("-")[0].upper()
    return f"CJ2-{stamp}-{short}"


def save_case(record: dict):
    file_exists = CASES_PATH.exists()
    with CASES_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fecha",
                "ticket_id",
                "cliente",
                "correo_cliente",
                "mensaje",
                "categoria",
                "confianza",
                "estado",
                "notificacion_email",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def send_email_notification(to_email: str, ticket_id: str, category: str, message_text: str):
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = os.getenv("SMTP_PORT", "587")
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from = os.getenv("SMTP_FROM", smtp_user or "")

    if not (smtp_host and smtp_user and smtp_password and smtp_from and to_email):
        return False, "Notificación por correo no configurada todavía."

    ui = CATEGORY_UI[category]
    msg = EmailMessage()
    msg["Subject"] = f"CJ² | Confirmación de solicitud {ticket_id}"
    msg["From"] = smtp_from
    msg["To"] = to_email
    msg.set_content(
        f"""
Hola,

Hemos registrado correctamente tu solicitud en CJ².

Ticket: {ticket_id}
Tipo detectado: {ui['title']}
Mensaje recibido: {message_text}

{ui['next_step']}

Gracias por contactar con CJ².
        """.strip()
    )

    port = int(smtp_port)
    if port == 465:
        with smtplib.SMTP_SSL(smtp_host, port) as server:
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

    return True, "Correo de confirmación enviado correctamente."


st.set_page_config(page_title="CJ² Smart Desk", page_icon="📨", layout="wide")
inject_css()

if not MODEL_PATH.exists():
    st.error("No se encontró el modelo entrenado. Ejecuta primero: python train_model.py")
    st.stop()

artifact = load_artifact()
model = artifact["model"]
threshold = float(artifact.get("threshold", DEFAULT_THRESHOLD))
version = artifact.get("version", "1.0")

with st.sidebar:
    st.markdown("<div class='sidebar-card'><h2 style='margin:0;'>CJ²</h2><p style='margin:0.35rem 0 0 0;'>Centro inteligente de clasificación y atención digital.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-card'><strong>Ejemplos de prueba</strong><br><br>1. Quiero reportar que mi pedido llegó dañado y necesito un reembolso.<br><br>2. Adjunto la documentación solicitada para continuar mi trámite.<br><br>3. Estoy interesado en una cotización para 25 unidades de su producto.</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sidebar-card'><strong>Estado del sistema</strong><br><br>Modelo: v{version}<br>Umbral de confianza: {threshold:.2f}<br>Categorías: queja, correo, venta</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='hero-card'>
        <div class='brand-chip'>CJ² · Smart Desk</div>
        <h1 class='hero-title'>Asistente inteligente de atención y clasificación de mensajes</h1>
        <p class='hero-subtitle'>Plataforma profesional para registrar y clasificar automáticamente solicitudes de tipo <strong>queja</strong>, <strong>correo / consulta</strong> o <strong>venta</strong>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("<div class='metric-card'><div class='metric-label'>Empresa</div><div class='metric-value'>CJ²</div></div>", unsafe_allow_html=True)
with col_b:
    st.markdown("<div class='metric-card'><div class='metric-label'>Motor IA</div><div class='metric-value'>Clasificación NLP</div></div>", unsafe_allow_html=True)
with col_c:
    st.markdown("<div class='metric-card'><div class='metric-label'>Estado</div><div class='metric-value'>Operativo</div></div>", unsafe_allow_html=True)

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("Registro de solicitud")
st.caption("Completa los datos del cliente y escribe el mensaje. El sistema generará un ticket y clasificará automáticamente la solicitud.")

with st.form("classification_form"):
    col1, col2 = st.columns(2)
    with col1:
        cliente = st.text_input("Nombre del cliente", placeholder="Ejemplo: María López")
    with col2:
        correo_cliente = st.text_input("Correo del cliente (opcional)", placeholder="cliente@correo.com")

    mensaje = st.text_area(
        "Mensaje del cliente",
        height=180,
        placeholder="Ejemplo: Quiero reclamar porque el producto llegó en mal estado y necesito una solución.",
    )

    submitted = st.form_submit_button("Registrar y clasificar")

if submitted:
    clean_message = " ".join(mensaje.strip().split())
    if not clean_message:
        st.warning("Escribe un mensaje antes de procesar la solicitud.")
    else:
        result = classify_message(model, threshold, clean_message)
        ticket_id = generate_ticket_id()
        now_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        if result["status"] == "classified":
            category = result["category"]
            ui = CATEGORY_UI[category]
            email_status = "no enviado"
            email_msg = ""

            if correo_cliente.strip():
                ok, email_msg = send_email_notification(correo_cliente.strip(), ticket_id, category, clean_message)
                email_status = "enviado" if ok else "pendiente de configuración"

            save_case(
                {
                    "fecha": now_str,
                    "ticket_id": ticket_id,
                    "cliente": cliente.strip(),
                    "correo_cliente": correo_cliente.strip(),
                    "mensaje": clean_message,
                    "categoria": category,
                    "confianza": f"{result['confidence']:.4f}",
                    "estado": "registrado",
                    "notificacion_email": email_status,
                }
            )

            st.success(f"Solicitud registrada con éxito. Ticket generado: {ticket_id}")
            st.markdown(
                f"<div class='result-box' style='border-left: 8px solid {ui['color']};'>"
                f"<h4 style='margin:0 0 0.4rem 0;'>{ui['icon']} Categoría detectada: {ui['title']}</h4>"
                f"<p style='margin:0 0 0.45rem 0;'>{ui['message']}</p>"
                f"<p style='margin:0;'><strong>Confianza del modelo:</strong> {result['confidence']*100:.1f}%</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.info(ui["next_step"])
            if correo_cliente.strip():
                if email_status == "enviado":
                    st.success("Notificación por correo enviada correctamente al cliente.")
                else:
                    st.warning(
                        "La app ya está preparada para enviar correos, pero faltan credenciales SMTP para activarlo en producción. "
                        "La solicitud sí quedó registrada correctamente."
                    )
                    st.caption(email_msg)
            else:
                st.caption("No se indicó correo del cliente. Solo se registró el ticket en el sistema.")

        else:
            save_case(
                {
                    "fecha": now_str,
                    "ticket_id": ticket_id,
                    "cliente": cliente.strip(),
                    "correo_cliente": correo_cliente.strip(),
                    "mensaje": clean_message,
                    "categoria": "revision_manual",
                    "confianza": f"{result['confidence']:.4f}",
                    "estado": "pendiente",
                    "notificacion_email": "no enviado",
                }
            )
            st.warning(f"No se pudo clasificar con suficiente seguridad. Ticket generado: {ticket_id}")
            st.info(result["reason"])
            st.caption("Sugerencia: añade más detalle al mensaje, por ejemplo si es reclamo, consulta administrativa o intención de compra.")

with st.expander("Cómo entrenar y mejorar la IA"):
    st.markdown(
        """
        1. Añade más ejemplos reales en `mensajes.csv`.
        2. Ejecuta `python train_model.py` para volver a entrenar el modelo.
        3. Prueba mensajes nuevos y corrige los casos donde falle.
        4. Si la app devuelve revisión manual, incorpora ejemplos parecidos al dataset.
        """
    )

st.markdown("</div>", unsafe_allow_html=True)
