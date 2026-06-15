# Proyecto IA: Clasificador de mensajes

Este proyecto es un prototipo de asistente inteligente que interpreta mensajes escritos por usuarios y los clasifica en tres categorías:

- **queja**
- **correo**
- **venta**

El sistema usa **Python**, **scikit-learn** y **Streamlit**.

## 1. Estructura del proyecto

```bash
proyecto_ia_clasificador/
│
├── app.py
├── train_model.py
├── mensajes.csv
├── modelo_mensajes.pkl
├── requirements.txt
└── README.md
```

## 2. Cómo abrirlo en PyCharm

1. Descarga y descomprime el proyecto.
2. Abre **PyCharm**.
3. Selecciona **Open** y elige la carpeta `proyecto_ia_clasificador`.
4. Espera a que PyCharm detecte el entorno.

## 3. Instalar dependencias

Abre la terminal dentro de PyCharm y ejecuta:

```bash
pip install -r requirements.txt
```

## 4. Entrenar el modelo

```bash
python train_model.py
```

Esto generará el archivo `modelo_mensajes.pkl`.

## 5. Ejecutar la aplicación

```bash
streamlit run app.py
```

## 6. Ejemplos de prueba

### Ejemplo 1
**Entrada:**

```text
Quiero presentar una queja porque mi pedido llegó dañado
```

**Salida esperada:** `queja`

### Ejemplo 2
**Entrada:**

```text
Adjunto los documentos solicitados para continuar el trámite
```

**Salida esperada:** `correo`

### Ejemplo 3
**Entrada:**

```text
Estoy interesado en comprar 20 unidades y necesito una cotización
```

**Salida esperada:** `venta`

## 7. Idea para sustentar ante el profesor

Puedes explicar el proyecto así:

> Se desarrolló un asistente inteligente basado en clasificación automática de texto. El sistema analiza mensajes escritos por usuarios y, mediante técnicas de procesamiento de lenguaje natural, los clasifica según su intención principal: queja, correo o venta.

## 8. Mejoras futuras

- Agregar más categorías
- Detectar prioridad alta, media o baja
- Guardar historial en base de datos
- Añadir análisis de sentimiento
- Generar respuestas automáticas más avanzadas
