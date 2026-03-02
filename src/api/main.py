"""
Point d'entrée de l'application FastAPI — crée et configure le serveur web.

CE FICHIER SERT À QUOI ?
    C'est le fichier qui DÉMARRE l'application. Il :
    1. Crée l'application FastAPI (le serveur web)
    2. Configure les "middlewares" (des couches de traitement communes à toutes les requêtes)
    3. Monte les fichiers statiques (CSS, JS) et les templates HTML
    4. Enregistre les routes (les endpoints de l'API)

    Pour lancer l'application :
        uvicorn src.api.main:app --reload

    Cela démarre un serveur web accessible sur http://localhost:8000
    - http://localhost:8000       → Interface web (page HTML)
    - http://localhost:8000/docs  → Documentation interactive de l'API (Swagger UI)
    - http://localhost:8000/api/v1/health → Vérifier que le serveur fonctionne

QU'EST-CE QU'UN MIDDLEWARE ?
    Un middleware est un "filtre" qui s'exécute AVANT et APRÈS chaque requête.
    Par exemple, le middleware CORS autorise les requêtes depuis d'autres domaines,
    et le middleware Metrics mesure le temps de réponse de chaque requête.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.api.routes import router as api_router
from src.api.ui_routes import router as ui_router
from src.core.config import get_settings
from src.enterprise.logging import setup_logging
from src.enterprise.monitoring import MetricsMiddleware

# Chemin vers la racine du projet (pour trouver les fichiers static/ et templates/)
_BASE_DIR = Path(__file__).resolve().parent.parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de l'application (démarrage / arrêt).

    Le code AVANT le "yield" s'exécute au DÉMARRAGE du serveur.
    Le code APRÈS le "yield" s'exécuterait à l'ARRÊT (ici on n'en a pas besoin).
    On en profite pour configurer le système de logs.
    """
    setup_logging()  # Initialiser les logs structurés
    yield  # L'application tourne... et quand elle s'arrête, on passe la main


def create_app() -> FastAPI:
    """Crée et configure l'application FastAPI.

    C'est une "factory function" : au lieu de créer l'app directement dans une variable globale,
    on la crée dans une fonction. Cela permet de la recréer facilement dans les tests.
    """
    settings = get_settings()

    # Créer l'application FastAPI
    app = FastAPI(
        title="Enterprise RAG Demo",
        description="Retrieval-Augmented Generation API powered by Claude and ChromaDB",
        version="1.0.0",
        lifespan=lifespan,  # Associer le gestionnaire de cycle de vie
    )

    # --- Middleware CORS ---
    # CORS (Cross-Origin Resource Sharing) autorise les requêtes depuis d'autres domaines.
    # Sans ça, un navigateur web bloquerait les appels depuis une page hébergée ailleurs.
    # allow_origins=["*"] = accepter les requêtes de N'IMPORTE quel domaine (OK pour le dev).
    # En production, on limiterait aux domaines autorisés.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Middleware Prometheus ---
    # Mesure automatiquement le nombre de requêtes et le temps de réponse
    # pour chaque endpoint. Les métriques sont exposées sur /api/v1/metrics.
    if settings.monitoring.prometheus_enabled:
        app.add_middleware(MetricsMiddleware)

    # --- Fichiers statiques (CSS, JavaScript, images) ---
    # "Monter" un dossier statique = rendre son contenu accessible via une URL.
    # Exemple : le fichier static/css/style.css sera accessible sur /static/css/style.css
    static_dir = _BASE_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # --- Routes de l'API ---
    # Les routes API sont préfixées par /api/v1 (convention de versioning)
    # Exemple : POST /api/v1/query, GET /api/v1/health
    app.include_router(api_router, prefix="/api/v1")

    # Les routes UI (la page web) sont à la racine (/)
    app.include_router(ui_router)

    return app


# Créer l'instance de l'application (utilisée par uvicorn)
app = create_app()


def run() -> None:
    """Lance le serveur web Uvicorn.

    Uvicorn est un serveur ASGI (Asynchronous Server Gateway Interface)
    compatible avec FastAPI. C'est lui qui écoute les connexions réseau
    et transmet les requêtes à notre application.
    """
    settings = get_settings().server
    uvicorn.run(
        "src.api.main:app",         # Chemin Python vers l'objet app
        host=settings.host,         # Adresse d'écoute (0.0.0.0 = toutes les interfaces)
        port=settings.port,         # Port d'écoute (8000 par défaut)
        workers=settings.workers,   # Nombre de processus parallèles
        reload=settings.reload,     # Rechargement automatique en dev
    )


if __name__ == "__main__":
    # Permet de lancer directement avec : python -m src.api.main
    run()
