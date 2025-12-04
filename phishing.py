
"""
Funcionalidades:
- Genera (o carga) dataset en español orientado a adultos mayores.
- Preprocesa texto, vectoriza con TF-IDF (1-2 ngrams).
- Entrena RandomForest (pipeline), evalúa métricas y guarda artefactos.
- Extrae top-features (explicabilidad aproximada).
- Ofrece función que devuelve una explicación amigable para adultos mayores.
- Exporta artefactos y un JSON resumen reproducible.
- Incluye un endpoint FastAPI mínimo para demostrar despliegue (opcional).
"""

from pathlib import Path
import random
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import datetime
import argparse
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# Configuración / Constants
# -------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# 1) Generador / carga de datos
# -------------------------
def generate_synthetic_dataset(n_per_class: int = 100) -> pd.DataFrame:
    """
    Genera un dataset sintético balanceado (phishing vs legit) en español,
    con plantillas orientadas a adultos mayores (pensiones, soporte, urgencia).
    Reemplaza o amplía con datos reales anotados cuando estén disponibles.
    """
    phishing_templates = [
        "URGENTE: Su cuenta bancaria ha sido bloqueada. Ingrese aquí {url} y verifique su identidad.",
        "Estimado cliente, hemos detectado actividad sospechosa. Confirme sus datos en {url}",
        "Su pensión no ha sido pagada. Actualice sus datos en {url} para recibir el pago.",
        "Llamada de soporte: necesitamos su número de documento para restablecer acceso.",
        "Gane un premio si confirma su cuenta en {url} – oferta limitada.",
        "Hay un problema con su seguridad social, entre a {url} y actualice su información.",
        "Su tarjeta fue utilizada fuera del país. Llame al 123-456 para verificar.",
        "Adjunto encontrará la factura pendiente. Abra el enlace {url} para pagar.",
        "Actualización obligatoria de datos Bancolombia: {url}",
        "Su cuenta será eliminada si no confirma en {url} antes de 24 horas."
    ]
    legit_templates = [
        "Recordatorio: su cita con el médico es el próximo martes a las 9:00.",
        "Boleta de agua del mes adjunta. Verifique el documento en su cuenta.",
        "Comunicado del banco: nuevas opciones de atención disponibles en nuestra web.",
        "Amigos: les comparto las fotos de la reunión familiar.",
        "Factura electrónica emitida correctamente, puede descargarla desde su perfil.",
        "Correo del seguro: su póliza fue renovada correctamente.",
        "Mensaje del vecino: por favor recoja el paquete en la portería.",
        "Notificación: su pedido ha sido enviado y llegará en 3 días.",
        "Aviso de mantenimiento programado en la plataforma municipal.",
        "Información sobre el servicio de atención al adulto mayor el próximo viernes."
    ]
    urls_full = [
        "http://seguridad-banco.example.com/login",
        "http://activar-cuenta.example.org",
        "http://pension.confirmar.example.net",
        "http://pago.example.com/pagar",
        "http://bancolombia.example.fake/login",
        "http://seg-salud.example.com"
    ]
    urls_short = ["http://bit.ly/abc123", "http://tinyurl.com/xyz789", "http://ow.ly/1a2b3"]

    data = []
    for _ in range(n_per_class):
        t = random.choice(phishing_templates)
        url = random.choice(urls_full + urls_short)
        text = t.format(url=url)
        # variaciones leves para parecer más real
        if random.random() < 0.25:
            text = text.replace("Su", "SU")
        data.append({"text": text, "label": "phishing"})

    for _ in range(n_per_class):
        t = random.choice(legit_templates)
        # añadir contexto local ocasional
        if random.random() < 0.2:
            t = t + " -- Oficina central Bogotá."
        data.append({"text": t, "label": "legit"})

    df = pd.DataFrame(data).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


def load_or_generate(path: Optional[str] = None, n_per_class: int = 100) -> pd.DataFrame:
    """
    Si se pasa path con CSV, se carga; si no, se genera sintético.
    CSV esperado: columnas 'text' y 'label' (label: phishing|legit)
    """
    if path:
        df = pd.read_csv(path)
        # validar formato mínimo
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV debe contener columnas 'text' y 'label'")
        return df[["text", "label"]].dropna().reset_index(drop=True)
    else:
        return generate_synthetic_dataset(n_per_class=n_per_class)


# -------------------------
# 2) Entrenamiento del pipeline
# -------------------------
def train_pipeline(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Entrena pipeline TF-IDF (1-2 grams) + RandomForest.
    Devuelve diccionario con pipeline, metrics y artefactos.
    """
    X = df["text"].values
    y = df["label"].values

    # Split 80/20 stratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words=None, max_features=3000)),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, class_weight="balanced"))
    ])

    pipeline.fit(X_train, y_train)

    # Predicciones y métricas
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Extraer top features (approx) desde TF-IDF + feature_importances_
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    importances = clf.feature_importances_
    # safety: si importances length mismatches
    if len(importances) == 0 or len(importances) != len(feature_names):
        top_features = []
    else:
        top_n = min(30, len(importances))
        top_idx = np.argsort(importances)[-top_n:][::-1]
        top_features = [{"feature": feature_names[i], "importance": float(importances[i])} for i in top_idx]

    result = {
        "pipeline": pipeline,
        "report": report,
        "confusion_matrix": cm,
        "top_features": top_features,
        "X_test": X_test.tolist(),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist()
    }
    return result


# -------------------------
# 3) Función de explicación amigable (para UI)
# -------------------------
def explain_for_user(text: str, pipeline: Pipeline, top_features: List[Dict[str, Any]], top_k: int = 5) -> str:
    """
    Genera una explicación en lenguaje simple para adultos mayores.
    Busca cuáles de las 'top_features' aparecen en el texto.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    feature_names = vectorizer.get_feature_names_out()
    vec = vectorizer.transform([text])
    present_idx = vec.nonzero()[1]
    present_features = set(feature_names[i] for i in present_idx)
    important_present = [f["feature"] for f in top_features if f["feature"] in present_features]
    if not important_present:
        return "No se detectaron palabras de alto riesgo. Revise igualmente si el remitente o el enlace son conocidos."
    # construir frase amigable
    detected = ", ".join(important_present[:top_k])
    return f"Palabras de riesgo detectadas: {detected}. Recomendación: no responder ni abrir enlaces. ¿Desea avisar a un contacto de confianza?"


# -------------------------
# 4) Guardado de artefactos y reporte
# -------------------------
def save_artifacts(pipeline: Pipeline, report: Dict, cm: np.ndarray, top_features: List[Dict], timestamp: str):
    # pipeline
    joblib.dump(pipeline, ARTIFACT_DIR / "pipeline_tfidf_rf.pkl")
    # confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="viridis")
    ax.set_title("Matriz de Confusión")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["legit", "phishing"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["legit", "phishing"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    # results json
    results = {
        "timestamp": timestamp,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "top_features": top_features
    }
    with open(ARTIFACT_DIR / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# -------------------------
# 5) Función principal / flujo (CRISP-DM implementado)
# -------------------------
def run_pipeline(data_csv: Optional[str] = None, n_per_class: int = 150, do_api: bool = False):
    """
    Ejecuta toda la cadena: cargar/generar datos, entrenar, evaluar, guardar artefactos.
    Set do_api=True para ejecutar un endpoint FastAPI (requiere uvicorn externo).
    """
    timestamp = datetime.datetime.now().isoformat()

    print("==> 1. Cargar o generar dataset")
    df = load_or_generate(data_csv, n_per_class=n_per_class)
    print(f"    Registros: {len(df)}  (ej. 5 muestras):\n", df.head(5).to_dict(orient="records"))

    print("==> 2. Entrenar pipeline")
    res = train_pipeline(df)
    pipeline = res["pipeline"]

    print("==> 3. Evaluación (reporte resumido)")
    print(json.dumps(res["report"], indent=2, ensure_ascii=False))

    print("==> 4. Top features (muestra)")
    for f in res["top_features"][:10]:
        print(f"    - {f['feature']}: {f['importance']:.6f}")

    print("==> 5. Guardar artefactos")
    save_artifacts(pipeline, res["report"], res["confusion_matrix"], res["top_features"], timestamp)
    print(f"    Artefactos guardados en: {ARTIFACT_DIR.resolve()}")

    # 6. Ejemplos de uso y explicaciones amigables
    examples = [
        "URGENTE: Su cuenta bancaria ha sido bloqueada. Ingrese aquí http://bit.ly/abc123 y verifique su identidad.",
        "Recordatorio: su cita con el médico es el próximo martes a las 9:00.",
        "Su pensión no ha sido pagada. Actualice sus datos en http://pago.example.com/pagar",
    ]
    demo = []
    for ex in examples:
        prob = float(pipeline.predict_proba([ex])[0][1])
        label = "phishing" if prob > 0.5 else "legit"
        explanation = explain_for_user(ex, pipeline, res["top_features"], top_k=5)
        demo.append({"text": ex, "risk_score": round(prob, 3), "label": label, "explanation": explanation})
    # Append demo to results_summary for paper
    with open(ARTIFACT_DIR / "results_summary.json", "r+", encoding="utf-8") as f:
        summary = json.load(f)
        summary["demo_examples"] = demo
        f.seek(0)
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.truncate()

    print("==> 6. Ejemplos (demo):")
    for d in demo:
        print(f" - [{d['label']}] score={d['risk_score']} -> {d['explanation']}")

    if do_api:
        # Nota: para ejecución real de la API correr: uvicorn phishing_proof_program:app --reload
        try:
            from fastapi import FastAPI
            from pydantic import BaseModel

            app = FastAPI()

            class DetectRequest(BaseModel):
                user_id: Optional[str] = None
                source: Optional[str] = None
                subject: Optional[str] = None
                body: str

            @app.post("/detect")
            def detect(payload: DetectRequest):
                text = (payload.subject or "") + " " + payload.body
                prob = float(pipeline.predict_proba([text])[0][1])
                label = "phishing" if prob > 0.5 else "legit"
                explanation = explain_for_user(text, pipeline, res["top_features"], top_k=5)
                return {"risk_score": prob, "label": label, "explanation": explanation}

            print("\nFastAPI app definida en variable 'app'. Para ejecutar la API utilice:")
            print("  uvicorn phishing_proof_program:app --host 0.0.0.0 --port 8000 --reload")
        except Exception as e:
            print("No se pudo levantar API (librería faltante):", e)

    return ARTIFACT_DIR


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta el prototipo que justifica el paper.")
    parser.add_argument("--csv", type=str, default=None, help="Ruta a CSV con columnas 'text' y 'label' (opcional).")
    parser.add_argument("--n", type=int, default=150, help="Número de instancias por clase (si se genera sintético).")
    parser.add_argument("--api", action="store_true", help="Definir endpoint FastAPI en variable 'app' (no lo inicia automáticamente).")
    args = parser.parse_args()
    run_pipeline(data_csv=args.csv, n_per_class=args.n, do_api=args.api)
