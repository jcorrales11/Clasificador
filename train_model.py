import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "mensajes.csv"
MODEL_PATH = BASE_DIR / "modelo_mensajes.pkl"


def main():
    df = pd.read_csv(DATA_PATH)

    if "mensaje" not in df.columns or "categoria" not in df.columns:
        raise ValueError("El archivo mensajes.csv debe contener las columnas 'mensaje' y 'categoria'.")

    X = df["mensaje"].astype(str)
    y = df["categoria"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
        ("clf", LinearSVC()),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=" * 60)
    print("RESULTADOS DEL ENTRENAMIENTO")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nReporte de clasificación:\n")
    print(classification_report(y_test, y_pred))
    print("Matriz de confusión:\n")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo guardado en: {MODEL_PATH}")


if __name__ == "__main__":
    main()
