"""
Schémas Pydantic — définissent la STRUCTURE des données échangées dans l'application.

CE FICHIER SERT À QUOI ?
    Quand un utilisateur envoie une requête à l'API (ex: poser une question, uploader un document),
    il faut vérifier que les données sont bien formées. Pydantic le fait automatiquement :
    - Vérifie les types (str, int, float...)
    - Applique des contraintes (longueur min/max, valeurs autorisées...)
    - Génère des messages d'erreur clairs si les données sont invalides

    Ces schémas servent aussi à documenter l'API : FastAPI les utilise pour générer
    automatiquement la page de documentation interactive (Swagger UI sur /docs).

ANALOGIE :
    C'est comme un formulaire papier avec des cases à remplir.
    Chaque classe ci-dessous décrit un formulaire différent :
    - QueryRequest = le formulaire "poser une question"
    - DocumentResponse = le reçu qu'on vous donne après avoir uploadé un document
    etc.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# =============================================================================
# MODÈLES LIÉS AUX DOCUMENTS
# =============================================================================
# Ces classes décrivent les données autour des documents uploadés par l'utilisateur.


class DocumentStatus(str, Enum):
    """Les différents états possibles d'un document pendant son traitement.

    Le cycle de vie d'un document est :
    PENDING → PROCESSING → INDEXED (succès) ou FAILED (échec)
    """
    PENDING = "pending"        # En attente de traitement
    PROCESSING = "processing"  # En cours de traitement (parse, chunk, embed...)
    INDEXED = "indexed"        # Traitement terminé, le document est cherchable
    FAILED = "failed"          # Le traitement a échoué (format non supporté, fichier corrompu...)


class DocumentMetadata(BaseModel):
    """Métadonnées d'un document (informations SUR le document, pas son contenu).

    Exemple : pour un fichier "rapport-q3.pdf" de 150 Ko, on stocke :
    - filename = "rapport-q3.pdf"
    - content_type = "application/pdf"
    - size_bytes = 153600
    - chunk_count = 12 (il a été découpé en 12 morceaux)
    - status = "indexed" (traitement terminé avec succès)
    """
    filename: str                                           # Nom du fichier original
    content_type: str                                       # Type MIME (ex: "application/pdf")
    size_bytes: int                                         # Taille en octets
    chunk_count: int = 0                                    # Nombre de chunks créés (0 avant traitement)
    status: DocumentStatus = DocumentStatus.PENDING         # État actuel du traitement
    created_at: datetime = Field(default_factory=datetime.utcnow)  # Date/heure d'upload
    error: str | None = None                                # Message d'erreur si le traitement a échoué


class DocumentResponse(BaseModel):
    """Réponse retournée après l'upload d'un document.

    Contient l'identifiant unique du document et ses métadonnées.
    Le client peut utiliser document_id pour ensuite consulter ou supprimer le document.
    """
    document_id: str            # Identifiant unique (UUID généré automatiquement)
    metadata: DocumentMetadata  # Toutes les infos sur le document


class DocumentListResponse(BaseModel):
    """Réponse pour la route GET /documents — liste tous les documents indexés."""
    documents: list[DocumentResponse]  # La liste des documents
    total: int                         # Nombre total de documents


# =============================================================================
# MODÈLES LIÉS AUX CHUNKS
# =============================================================================
# Un chunk = un morceau de document, prêt à être vectorisé et stocké.


class Chunk(BaseModel):
    """Un morceau de document avec ses métadonnées.

    Quand on découpe un PDF de 10 pages, on obtient par exemple 30 chunks.
    Chaque chunk contient :
    - chunk_id : identifiant unique (format: "document_id::random_12chars")
    - document_id : identifiant du document parent (pour retrouver l'original)
    - text : le texte du morceau (ex: un paragraphe de ~512 caractères)
    - metadata : infos supplémentaires (nom du fichier, position dans le document...)
    """
    chunk_id: str                              # ID unique de ce chunk
    document_id: str                           # ID du document parent
    text: str                                  # Le contenu textuel du chunk
    metadata: dict = Field(default_factory=dict)  # Métadonnées libres (dict vide par défaut)


# =============================================================================
# MODÈLES LIÉS AUX REQUÊTES RAG (question → réponse)
# =============================================================================
# C'est le cœur du système : l'utilisateur pose une question, le RAG répond.


class QueryRequest(BaseModel):
    """Requête envoyée par l'utilisateur pour poser une question.

    Exemple d'utilisation :
        POST /api/v1/query
        {
            "question": "Quelle est la politique de télétravail ?",
            "top_k": 5,
            "score_threshold": 0.3
        }
    """
    question: str = Field(..., min_length=1, max_length=2000)
    # ... = obligatoire (pas de valeur par défaut). min_length/max_length = contraintes de longueur.

    top_k: int = Field(default=5, ge=1, le=20)
    # Nombre maximum de chunks à retrouver. ge=1 (minimum 1), le=20 (maximum 20).

    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    # Score minimum de similarité pour qu'un chunk soit retenu (entre 0 et 1).

    collection: str | None = None
    # Collection cible (optionnel). Si None, utilise la collection par défaut.


class SourceDocument(BaseModel):
    """Un chunk retrouvé par la recherche, avec son score de pertinence.

    Représente une "source" citée dans la réponse du RAG.
    Le score indique à quel point ce chunk est pertinent par rapport à la question
    (1.0 = parfaitement pertinent, 0.0 = aucun rapport).
    """
    document_id: str                           # ID du document parent
    chunk_id: str                              # ID du chunk
    text: str                                  # Contenu textuel du chunk
    score: float                               # Score de similarité (0 à 1)
    metadata: dict = Field(default_factory=dict)  # Métadonnées (filename, etc.)


class QueryResponse(BaseModel):
    """Réponse complète du système RAG.

    Contient la réponse générée par Claude, les sources utilisées pour la produire,
    et des métadonnées sur la requête (modèle utilisé, temps de réponse...).

    Exemple :
        {
            "answer": "D'après le document politique-rh.pdf, le télétravail est...",
            "sources": [{"text": "...", "score": 0.87, ...}],
            "query": "Quelle est la politique de télétravail ?",
            "model": "claude-sonnet-4-20250514",
            "latency_ms": 1234.5
        }
    """
    answer: str                        # La réponse générée par le LLM
    sources: list[SourceDocument]      # Les chunks qui ont servi de contexte
    query: str                         # La question originale (pour référence)
    model: str                         # Le modèle LLM utilisé
    latency_ms: float                  # Temps total de traitement en millisecondes


# =============================================================================
# MODÈLES LIÉS À L'AUTHENTIFICATION
# =============================================================================


class TokenRequest(BaseModel):
    """Requête de connexion : l'utilisateur envoie son nom et mot de passe."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Réponse après connexion réussie : le serveur retourne un token JWT.

    Le client devra inclure ce token dans l'en-tête de ses requêtes suivantes :
    Authorization: Bearer <access_token>
    """
    access_token: str          # Le token JWT à utiliser pour les requêtes suivantes
    token_type: str = "bearer" # Type de token (toujours "bearer" pour JWT)


# =============================================================================
# MODÈLES UTILITAIRES
# =============================================================================


class HealthResponse(BaseModel):
    """Réponse du endpoint /health — permet de vérifier que le serveur fonctionne."""
    status: str = "healthy"
    version: str = "1.0.0"
    components: dict[str, str] = Field(default_factory=dict)
    # Détail de l'état de chaque composant (ex: {"vector_store": "ok", "chunks_indexed": "42"})


class CollectionInfo(BaseModel):
    """Informations sur une collection de documents dans ChromaDB."""
    name: str              # Nom de la collection
    document_count: int    # Nombre de documents
    chunk_count: int       # Nombre total de chunks dans cette collection


class CollectionListResponse(BaseModel):
    """Liste de toutes les collections disponibles."""
    collections: list[CollectionInfo]
