"""
Routes REST de l'API — les "guichets" du système RAG.

CE FICHIER SERT À QUOI ?
    Il définit tous les endpoints (points d'accès) de l'API REST.
    Chaque endpoint est une URL associée à une action :

    SYSTÈME :
      GET  /health     → Vérifier que le serveur fonctionne
      GET  /metrics    → Obtenir les statistiques d'utilisation (Prometheus)

    AUTHENTIFICATION :
      POST /auth/token → Se connecter et obtenir un token d'accès

    DOCUMENTS :
      POST   /documents/upload    → Uploader un fichier (lance le pipeline d'ingestion)
      GET    /documents           → Lister tous les documents indexés
      GET    /documents/{id}      → Voir le détail d'un document
      DELETE /documents/{id}      → Supprimer un document et ses chunks

    RAG (le cœur du système) :
      POST /query → Poser une question et obtenir une réponse basée sur les documents

    COLLECTIONS :
      GET /collections → Lister les collections de documents

QU'EST-CE QU'UNE ROUTE REST ?
    REST est une convention pour structurer les APIs web :
    - L'URL indique la RESSOURCE (ex: /documents, /query)
    - La méthode HTTP indique l'ACTION (GET = lire, POST = créer, DELETE = supprimer)
    - Les données sont échangées en JSON

    FastAPI gère automatiquement :
    - La validation des données entrantes (grâce aux schémas Pydantic)
    - La sérialisation de la réponse en JSON
    - La génération de la documentation (Swagger UI sur /docs)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
import structlog

from src.enterprise.auth import get_current_user
from src.enterprise.monitoring import (
    DOCUMENTS_INGESTED,
    QUERY_COUNT,
    QUERY_LATENCY,
    metrics_response,
)
from src.ingestion.parser import SUPPORTED_TYPES
from src.ingestion.pipeline import (
    delete_document,
    get_document,
    ingest_document,
    list_documents,
)
from src.models.schemas import (
    CollectionInfo,
    CollectionListResponse,
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    TokenRequest,
    TokenResponse,
)
from src.retrieval.rag_engine import query as rag_query
from src.retrieval.vector_store import get_vector_store, _stores

logger = structlog.get_logger(__name__)

# APIRouter regroupe les routes. Il sera ensuite inclus dans l'app FastAPI
# avec un préfixe (ex: /api/v1) dans main.py.
router = APIRouter()


# =============================================================================
# SYSTÈME — Vérification de l'état du serveur
# =============================================================================


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Vérifie que le serveur et ses composants fonctionnent.

    Typiquement appelé par un outil de monitoring ou un load balancer
    pour savoir si le serveur est "vivant" et prêt à recevoir des requêtes.
    """
    store = get_vector_store()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components={
            "vector_store": "ok",
            "chunks_indexed": str(store.count),  # Nombre de chunks dans ChromaDB
        },
    )


# =============================================================================
# MÉTRIQUES — Statistiques d'utilisation (format Prometheus)
# =============================================================================


@router.get("/metrics", tags=["System"], include_in_schema=False)
async def prometheus_metrics():
    """Expose les métriques au format Prometheus.

    include_in_schema=False → cet endpoint n'apparaît pas dans la doc Swagger
    (c'est un endpoint technique, pas destiné aux utilisateurs).
    """
    return metrics_response()


# =============================================================================
# AUTHENTIFICATION — Connexion et obtention d'un token JWT
# =============================================================================


@router.post("/auth/token", response_model=TokenResponse, tags=["Auth"])
async def login(request: TokenRequest):
    """Authentifie un utilisateur et retourne un token JWT.

    L'utilisateur envoie son username et password.
    Si les identifiants sont valides, le serveur retourne un token JWT
    que le client devra inclure dans l'en-tête de ses requêtes suivantes :
        Authorization: Bearer <token>

    Note : l'authentification est désactivée par défaut (auth.enabled=false dans la config).
    """
    from src.enterprise.auth import authenticate_user, create_access_token

    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    # Créer un token contenant le nom d'utilisateur et son rôle
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return TokenResponse(access_token=token)


# =============================================================================
# DOCUMENTS — Upload, consultation et suppression
# =============================================================================


@router.post(
    "/documents/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,  # 201 = "Created" (convention HTTP pour une création)
    tags=["Documents"],
)
async def upload_document(
    file: UploadFile = File(...),
    # ↑ FastAPI gère automatiquement le multipart/form-data (le format standard pour l'upload de fichiers)
    user: dict | None = Depends(get_current_user),
    # ↑ Depends() injecte automatiquement le résultat de get_current_user.
    #   Si l'auth est activée, vérifie le token. Sinon, retourne None.
):
    """Upload un document et lance le pipeline d'ingestion complet.

    Le fichier est lu, puis passe par : parse → chunk → embed → store.
    Le résultat contient l'ID du document et son statut (indexed ou failed).
    """
    content_type = file.content_type or "application/octet-stream"
    content = await file.read()  # Lire le contenu binaire du fichier uploadé
    filename = file.filename or "untitled"

    logger.info("upload_request", filename=filename, content_type=content_type, size=len(content))

    # Lancer le pipeline d'ingestion complet (parse → chunk → embed → store)
    result = ingest_document(content, filename, content_type)

    # Incrémenter le compteur Prometheus (pour le monitoring)
    DOCUMENTS_INGESTED.labels(status=result.metadata.status.value).inc()

    # Si l'ingestion a échoué, retourner une erreur HTTP 422
    if result.metadata.status == "failed":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Ingestion failed: {result.metadata.error}",
        )

    return result


@router.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def get_documents(user: dict | None = Depends(get_current_user)):
    """Liste tous les documents qui ont été indexés."""
    docs = list_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.get("/documents/{document_id}", response_model=DocumentResponse, tags=["Documents"])
async def get_document_by_id(
    document_id: str,
    # ↑ FastAPI extrait automatiquement {document_id} depuis l'URL
    user: dict | None = Depends(get_current_user),
):
    """Retourne le détail d'un document par son identifiant."""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Documents"])
async def remove_document(
    document_id: str,
    user: dict | None = Depends(get_current_user),
):
    """Supprime un document et tous ses chunks de la base vectorielle.

    Retourne 204 No Content si la suppression a réussi (convention REST :
    une suppression réussie ne retourne pas de corps de réponse).
    """
    if not delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")


# =============================================================================
# RAG — Poser une question (le cœur du système !)
# =============================================================================


@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(
    request: QueryRequest,
    user: dict | None = Depends(get_current_user),
):
    """Pose une question au système RAG et retourne une réponse.

    C'est l'endpoint principal de toute l'application !
    Le processus complet :
    1. La question est vectorisée (embedding)
    2. Les chunks les plus similaires sont retrouvés dans ChromaDB
    3. Les chunks sont injectés dans le prompt envoyé à Claude
    4. Claude génère une réponse basée sur ces documents
    5. La réponse est retournée avec les sources citées
    """
    import time

    # Incrémenter le compteur de requêtes RAG (métrique Prometheus)
    QUERY_COUNT.inc()
    start = time.perf_counter()

    # Appeler le moteur RAG (retrieve → augment → generate)
    result = rag_query(request)

    # Mesurer et enregistrer le temps de réponse (métrique Prometheus)
    elapsed = time.perf_counter() - start
    QUERY_LATENCY.observe(elapsed)

    return result


# =============================================================================
# COLLECTIONS — Gestion des collections de documents
# =============================================================================


@router.get("/collections", response_model=CollectionListResponse, tags=["Collections"])
async def list_collections(user: dict | None = Depends(get_current_user)):
    """Liste toutes les collections de documents disponibles.

    Une collection est un regroupement de chunks dans ChromaDB (comme une "table" en SQL).
    On peut avoir plusieurs collections pour séparer différents types de documents.
    """
    infos = []
    for name, store in _stores.items():
        infos.append(
            CollectionInfo(
                name=name,
                document_count=0,
                chunk_count=store.count,
            )
        )
    return CollectionListResponse(collections=infos)
