"""
Métriques Prometheus et middleware de monitoring.

CE FICHIER SERT À QUOI ?
    Il mesure et expose des STATISTIQUES sur l'utilisation de l'application :
    - Combien de requêtes HTTP ont été reçues ?
    - Combien de temps chaque requête a-t-elle pris ?
    - Combien de questions RAG ont été posées ?
    - Combien de documents ont été indexés (succès vs échecs) ?

POURQUOI C'EST UTILE ?
    En production, on veut savoir si l'application fonctionne bien :
    - Le temps de réponse augmente ? Peut-être qu'on a besoin de plus de mémoire.
    - Le taux d'erreurs monte ? Il y a un bug à corriger.
    - Beaucoup de requêtes ? Il faut peut-être ajouter des serveurs.

QU'EST-CE QUE PROMETHEUS ?
    Prometheus est un outil de surveillance open-source très populaire.
    Il fonctionne en 2 étapes :
    1. Notre application EXPOSE ses métriques sur un endpoint (GET /metrics)
       dans un format texte standardisé
    2. Le serveur Prometheus RÉCUPÈRE ces métriques régulièrement (toutes les 15s par ex.)
       et les stocke pour créer des graphiques et des alertes (souvent via Grafana)

TYPES DE MÉTRIQUES :
    - Counter : un compteur qui ne fait qu'augmenter (ex: nombre total de requêtes)
    - Histogram : mesure la distribution des valeurs (ex: temps de réponse)
      Un histogram crée automatiquement des "buckets" : combien de requêtes
      ont pris < 0.1s, < 0.5s, < 1s, etc.
"""

from __future__ import annotations

import time

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware

# =============================================================================
# DÉFINITION DES MÉTRIQUES
# =============================================================================
# Chaque métrique a un nom unique et une description.
# Les "labels" permettent de filtrer les métriques (ex: par méthode HTTP, par endpoint...).

# Compteur de requêtes HTTP (toutes routes confondues)
# Labels : method (GET/POST/...), endpoint (/query, /documents/...), status (200, 404, 500...)
REQUEST_COUNT = Counter(
    "rag_http_requests_total",        # Nom de la métrique (convention : prefixe_nom_total)
    "Total HTTP requests",             # Description (affichée dans Prometheus/Grafana)
    ["method", "endpoint", "status"],  # Labels pour filtrer
)

# Temps de réponse des requêtes HTTP (en secondes)
# Les buckets définissent les seuils : combien de requêtes < 10ms, < 50ms, < 100ms...
REQUEST_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Compteur de requêtes RAG spécifiquement (questions posées au système)
QUERY_COUNT = Counter(
    "rag_queries_total",
    "Total RAG queries processed",
)

# Temps de réponse des requêtes RAG (embedding + recherche + génération LLM)
# Les buckets sont plus larges car une requête RAG est plus lente (appel API externe)
QUERY_LATENCY = Histogram(
    "rag_query_duration_seconds",
    "RAG query end-to-end latency",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# Compteur de documents ingérés, avec un label "status" (indexed ou failed)
DOCUMENTS_INGESTED = Counter(
    "rag_documents_ingested_total",
    "Total documents ingested",
    ["status"],  # Permet de distinguer les succès des échecs
)

# Compteur de chunks stockés dans la base vectorielle
CHUNKS_STORED = Counter(
    "rag_chunks_stored_total",
    "Total chunks stored in vector store",
)


# =============================================================================
# MIDDLEWARE DE MONITORING
# =============================================================================


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware qui mesure automatiquement chaque requête HTTP.

    Un middleware s'exécute AUTOUR de chaque requête :
    - AVANT : il démarre le chronomètre
    - Le endpoint s'exécute normalement
    - APRÈS : il arrête le chronomètre et enregistre les métriques

    Avantage : on n'a pas besoin d'ajouter du code de mesure dans chaque endpoint.
    Le middleware le fait automatiquement pour TOUTES les routes.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """Intercepte chaque requête, mesure son temps de traitement, et enregistre les métriques."""
        start = time.perf_counter()              # Démarrer le chronomètre
        response = await call_next(request)       # Laisser l'endpoint s'exécuter
        elapsed = time.perf_counter() - start     # Calculer le temps écoulé

        endpoint = request.url.path

        # Incrémenter le compteur de requêtes (avec les labels pour le filtrage)
        REQUEST_COUNT.labels(
            method=request.method,        # GET, POST, DELETE...
            endpoint=endpoint,            # /api/v1/query, /api/v1/health...
            status=response.status_code,  # 200, 404, 500...
        ).inc()  # inc() = incrémenter de 1

        # Enregistrer le temps de réponse dans l'histogramme
        REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(elapsed)

        return response


def metrics_response() -> Response:
    """Génère la réponse pour l'endpoint GET /metrics.

    generate_latest() produit toutes les métriques au format texte Prometheus :
        # HELP rag_http_requests_total Total HTTP requests
        # TYPE rag_http_requests_total counter
        rag_http_requests_total{method="GET",endpoint="/health",status="200"} 42.0
        ...

    CONTENT_TYPE_LATEST est le Content-Type attendu par Prometheus.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
