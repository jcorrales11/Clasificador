# CJ² Smart Desk

Sistema inteligente para clasificar mensajes de clientes en tres categorías:

- **queja**
- **correo / consulta**
- **venta**

## Funciones incluidas

- Interfaz más profesional con marca **CJ²**
- Registro de ticket automático
- Clasificación automática con IA
- Detección de baja confianza para enviar a **revisión manual**
- Registro local de casos en `historial_casos.csv`
- Confirmación al usuario cuando su solicitud se registra con éxito
- Preparado para enviar correo de confirmación si configuras SMTP

## Archivos principales

```bash
app.py
train_model.py
mensajes.csv
modelo_mensajes.pkl
requirements.txt
README.md
```

## Instalación

```bash
pip install -r requirements.txt
```

## Reentrenar la IA

Cada vez que agregues nuevos ejemplos a `mensajes.csv`, vuelve a entrenar:

```bash
python train_model.py
```

## Ejecutar la aplicación

```bash
python -m streamlit run app.py
```

## Cómo mejorar el modelo

1. Añade más frases reales al archivo `mensajes.csv`.
2. Mantén un buen equilibrio entre categorías.
3. Incluye ejemplos cortos, largos y con errores comunes.
4. Reentrena con `python train_model.py`.
5. Prueba casos nuevos y corrige los que fallen.

## Correos de confirmación (opcional)

La app ya está preparada para enviar correos reales, pero debes configurar estas variables de entorno:

```bash
SMTP_HOST
SMTP_PORT
SMTP_USER
SMTP_PASSWORD
SMTP_FROM
```

### Ejemplo orientativo

```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=tu_correo@gmail.com
SMTP_PASSWORD=tu_clave_o_app_password
SMTP_FROM=tu_correo@gmail.com
```

> Nota: para Gmail normalmente debes usar una **App Password**.

## Qué pasa si el mensaje no se entiende

Si el texto es muy corto o la confianza del modelo es baja, la app no fuerza una categoría. En ese caso lo manda a **revisión manual**.

## Ideas de mejora futura

- Guardar casos en base de datos
- Panel de administración
- Inicio de sesión para operadores
- Notificaciones reales por correo y panel interno
- Más categorías: soporte, facturación, devoluciones, seguimiento
