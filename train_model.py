import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "mensajes.csv"
MODEL_PATH = BASE_DIR / "modelo_mensajes.pkl"
CONFIDENCE_THRESHOLD = 0.50


def main():
    df = pd.read_csv(DATA_PATH)

    if "mensaje" not in df.columns or "categoria" not in df.columns:
        raise ValueError("El archivo mensajes.csv debe contener las columnas 'mensaje' y 'categoria'.")

    df = df.dropna(subset=["mensaje", "categoria"]).copy()
    df["mensaje"] = df["mensaje"].astype(str).str.strip()
    df["categoria"] = df["categoria"].astype(str).str.strip().str.lower()
    df = df[df["mensaje"] != ""]

    X = df["mensaje"]
    y = df["categoria"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=" * 70)
    print("RESULTADOS DEL ENTRENAMIENTO - CJ²")
    print("=" * 70)
    print(f"Total de registros: {len(df)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nReporte de clasificación:\n")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:\n")
    print(confusion_matrix(y_test, y_pred))

    artifact = {
        "model": model,
        "threshold": CONFIDENCE_THRESHOLD,
        "classes": sorted(y.unique().tolist()),
        "version": "2.0",
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModelo guardado en: {MODEL_PATH}")
    print(f"Umbral de confianza configurado: {CONFIDENCE_THRESHOLD}")
    print("Si el texto es muy corto o la confianza es baja, la app devolverá 'revisión manual'.")


if __name__ == "__main__":
    main()
