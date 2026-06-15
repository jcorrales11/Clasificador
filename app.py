from __future__ import annotations

import csv
import os
import re
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

DEFAULT_THRESHOLD = 0.50
MIN_WORDS = 3

RULE_KEYWORDS = {
    "queja": [
        "reclamo", "queja", "quejar", "reembolso", "cobro mal", "cobro incorrecto", "me cobraron",
        "mala atencion", "mala atención", "defectuoso", "dañado", "danado", "vencido", "error en mi compra",
        "devolucion", "devolución", "no funciona", "mal estado", "cajero", "cargo no autorizado",
    ],
    "correo": [
        "adjunto", "documentos", "archivo", "tramite", "trámite", "formulario", "expediente",
        "constancia", "contrato", "reenvio", "reenvío", "consulta", "seguimiento", "solicitud",
        "datos", "comprobante", "evidencia", "certificado",
    ],
    "venta": [
        "cotizacion", "cotización", "comprar", "compra", "precio", "precios", "promocion", "promoción",
        "cajas", "unidades", "proveedor", "oferta", "asesor comercial", "venta", "catalogo", "catálogo",
        "paquete", "plan", "licencias", "suscripcion", "suscripción",
    ],
}

EXAMPLES = [
    "El cajero me cobró mal y necesito solución.",
    "Adjunto la documentación solicitada para continuar el trámite.",
    "Quiero 50 cajas de leche y necesito una cotización.",
]


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7f8fc 0%, #eef2ff 100%);
        }
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        .hero-card {
            background: linear-gradient(135deg, #111827 0%, #1f2937 45%, #1d4ed8 100%);
            color: white;
            border-radius: 24px;
            padding: 28px 32px;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
            margin-bottom: 1rem;
        }
        .brand-chip {
            display: inline-block;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.20);
            color: #ffffff;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.9rem;
            margin-bottom: 0.8rem;
        }
        .hero-title {
            font-size: 2.35rem;
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
            padding: 18px;
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
            font-size: 1.25rem;
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
        .sidebar-card {
            background: rgba(255,255,255,0.78);
            border-radius: 18px;
            padding: 14px 16px;
            border: 1px solid #e5e7eb;
            margin-bottom: 0.8rem;
        }
        .stTextArea textarea, .stTextInput input {
            border-radius: 14px !important;
        }
        .stButton>button {
            border-radius: 12px;
            font-weight: 700;
            padding: 0.62rem 1.2rem;
            border: none;
            background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
            color: white;
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


def normalize_text(text: str) -> str:
    clean = " ".join(text.strip().split())
    return clean.lower()


def detect_rule_based_intent(message: str):
    message_lower = normalize_text(message)

    for keyword in RULE_KEYWORDS["queja"]:
        if keyword in message_lower:
            return "queja", 0.92

    quantity_match = re.search(r"\b(\d+)\b", message_lower)
    if quantity_match and any(word in message_lower for word in ["caja", "cajas", "unidad", "unidades", "litros", "pedido"]):
        return "venta", 0.93

    for keyword in RULE_KEYWORDS["venta"]:
        if keyword in message_lower:
            return "venta", 0.90

    for keyword in RULE_KEYWORDS["correo"]:
        if keyword in message_lower:
            return "correo", 0.88

    return None, None


def classify_message(model, threshold: float, message: str):
    clean = " ".join(message.strip().split())
    words = clean.split()

    if len(words) < MIN_WORDS:
        return {
            "status": "manual_review",
            "category": None,
            "confidence": 0.0,
            "reason": "Necesitamos un poco más de detalle para clasificar la solicitud correctamente.",
        }

    rule_category, rule_confidence = detect_rule_based_intent(clean)
    if rule_category:
        return {
            "status": "classified",
            "category": rule_category,
            "confidence": rule_confidence,
            "reason": "Clasificación realizada con apoyo de reglas de negocio.",
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
            "reason": "No se pudo determinar la categoría con suficiente seguridad.",
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
        return False

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

    return True


def init_state():
    defaults = {
        "cliente_input": "",
        "correo_input": "",
        "mensaje_input": "",
        "last_feedback": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_form_fields():
    st.session_state["cliente_input"] = ""
    st.session_state["correo_input"] = ""
    st.session_state["mensaje_input"] = ""


def store_feedback(payload: dict):
    st.session_state["last_feedback"] = payload


def render_feedback():
    payload = st.session_state.get("last_feedback")
    if not payload:
        return

    if payload["status"] == "classified":
        ui = CATEGORY_UI[payload["category"]]
        st.success(f"Solicitud registrada con éxito. Ticket generado: {payload['ticket_id']}")
        st.markdown(
            f"<div class='result-box' style='border-left: 8px solid {ui['color']};'>"
            f"<h4 style='margin:0 0 0.4rem 0;'>{ui['icon']} Categoría detectada: {ui['title']}</h4>"
            f"<p style='margin:0 0 0.45rem 0;'>{ui['message']}</p>"
            f"<p style='margin:0;'><strong>Confianza estimada:</strong> {payload['confidence']*100:.1f}%</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.info(ui["next_step"])
        if payload.get("email_sent"):
            st.success("Se ha enviado una confirmación al correo indicado.")
    else:
        st.warning(f"Tu solicitud se registró como pendiente de revisión. Ticket generado: {payload['ticket_id']}")
        st.info("Por favor, añade un poco más de detalle para poder clasificarla mejor.")


st.set_page_config(page_title="CJ² Smart Desk", page_icon="📨", layout="wide")
inject_css()
init_state()

if not MODEL_PATH.exists():
    st.error("No se encontró el modelo entrenado. Ejecuta primero: python train_model.py")
    st.stop()

artifact = load_artifact()
model = artifact["model"]
threshold = float(artifact.get("threshold", DEFAULT_THRESHOLD))

with st.sidebar:
    st.markdown("<div class='sidebar-card'><h2 style='margin:0;'>CJ²</h2><p style='margin:0.35rem 0 0 0;'>Centro inteligente de clasificación y atención digital.</p></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sidebar-card'><strong>Ejemplos de solicitud</strong><br><br>"
        f"1. {EXAMPLES[0]}<br><br>2. {EXAMPLES[1]}<br><br>3. {EXAMPLES[2]}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='sidebar-card'><strong>Atención</strong><br><br>El sistema registra solicitudes de queja, correo / consulta y venta.</div>", unsafe_allow_html=True)

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
    st.markdown("<div class='metric-card'><div class='metric-label'>Servicio</div><div class='metric-value'>Recepción inteligente</div></div>", unsafe_allow_html=True)
with col_c:
    st.markdown("<div class='metric-card'><div class='metric-label'>Estado</div><div class='metric-value'>Disponible</div></div>", unsafe_allow_html=True)

st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.subheader("Registro de solicitud")
st.caption("Completa los datos del cliente y escribe el mensaje. El sistema generará un ticket y clasificará automáticamente la solicitud.")

render_feedback()

with st.form("classification_form"):
    col1, col2 = st.columns(2)
    with col1:
        cliente = st.text_input(
            "Nombre del cliente",
            placeholder="Ejemplo: María López",
            key="cliente_input",
        )
    with col2:
        correo_cliente = st.text_input(
            "Correo del cliente (opcional)",
            placeholder="cliente@correo.com",
            key="correo_input",
        )

    mensaje = st.text_area(
        "Mensaje del cliente",
        height=180,
        placeholder="Ejemplo: Quiero reclamar porque el producto llegó en mal estado y necesito una solución.",
        key="mensaje_input",
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
            email_sent = False
            if correo_cliente.strip():
                try:
                    email_sent = send_email_notification(correo_cliente.strip(), ticket_id, category, clean_message)
                except Exception:
                    email_sent = False

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
                    "notificacion_email": "enviado" if email_sent else "no enviado",
                }
            )
            store_feedback(
                {
                    "status": "classified",
                    "ticket_id": ticket_id,
                    "category": category,
                    "confidence": result["confidence"],
                    "email_sent": email_sent,
                }
            )
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
            store_feedback(
                {
                    "status": "manual_review",
                    "ticket_id": ticket_id,
                }
            )

        clear_form_fields()
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
