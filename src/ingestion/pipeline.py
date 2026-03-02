"""
Pipeline d'ingestion complet : parse → chunk → embed → store.

CE FICHIER SERT À QUOI ?
    C'est le CHEF D'ORCHESTRE de l'ingestion. Quand un utilisateur uploade un document,
    c'est ce fichier qui enchaîne toutes les étapes dans l'ordre :
      1. PARSE  — extraire le texte brut du fichier (parser.py)
      2. CHUNK  — découper le texte en morceaux (chunker.py)
      3. EMBED & STORE — transformer en vecteurs et sauvegarder dans ChromaDB (vector_store.py)

    Il gère aussi le "registre de documents" : un fichier JSON qui garde la trace
    de tous les documents uploadés (leur nom, état, nombre de chunks...).
    Ce registre est sauvegardé sur disque pour survivre aux redémarrages du serveur.

POURQUOI UN PIPELINE ?
    Séparer les étapes permet de :
    - Tester chaque étape indépendamment
    - Facilement changer une étape (ex: remplacer le parseur PDF)
    - Gérer proprement les erreurs (si le parsing échoue, on ne tente pas le chunking)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import structlog

from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_text
from src.core.config import get_settings
from src.models.schemas import DocumentMetadata, DocumentResponse, DocumentStatus
from src.retrieval.vector_store import get_vector_store

logger = structlog.get_logger(__name__)

# =============================================================================
# REGISTRE PERSISTANT DES DOCUMENTS
# =============================================================================
# Le "registre" est un fichier JSON qui liste tous les documents indexés.
# Il est sauvegardé dans le même dossier que la base ChromaDB.
# Sans ce registre, on perdrait la liste des documents au redémarrage du serveur
# (ChromaDB stocke les chunks, mais pas les métadonnées des documents eux-mêmes).

_REGISTRY_PATH = Path(get_settings().vector_store.persist_directory) / "document_registry.json"

# Dictionnaire en mémoire : { document_id → DocumentResponse }
_document_registry: dict[str, DocumentResponse] = {}


def _load_registry() -> None:
    """Charge le registre de documents depuis le fichier JSON sur disque.

    Appelé au démarrage de l'application pour restaurer la liste des documents
    qui ont été indexés lors des sessions précédentes.
    """
    if not _REGISTRY_PATH.exists():
        return
    try:
        data = json.loads(_REGISTRY_PATH.read_text())
        for doc_id, raw in data.items():
            meta = DocumentMetadata(**raw["metadata"])
            _document_registry[doc_id] = DocumentResponse(document_id=doc_id, metadata=meta)
        logger.info("registry_loaded", count=len(_document_registry))
    except Exception as exc:
        logger.error("registry_load_failed", error=str(exc))


def _save_registry() -> None:
    """Sauvegarde le registre de documents sur disque (fichier JSON).

    Appelé après chaque modification (ajout, suppression de document)
    pour que les données survivent à un redémarrage du serveur.
    """
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for doc_id, doc in _document_registry.items():
        meta = doc.metadata.model_dump()  # Convertit l'objet Pydantic en dict Python
        meta["created_at"] = meta["created_at"].isoformat() if meta.get("created_at") else None
        data[doc_id] = {"metadata": meta}
    _REGISTRY_PATH.write_text(json.dumps(data, indent=2))


# Charger le registre dès l'import du module (= au démarrage de l'application)
_load_registry()


# =============================================================================
# PIPELINE D'INGESTION
# =============================================================================


def ingest_document(
    content: bytes,
    filename: str,
    content_type: str,
    collection: str | None = None,
) -> DocumentResponse:
    """Ingère un document à travers le pipeline complet : parse → chunk → embed → store.

    C'est LA fonction principale de ce module. Elle est appelée par l'endpoint
    POST /documents/upload quand un utilisateur envoie un fichier.

    Args:
        content: Le contenu binaire du fichier uploadé
        filename: Le nom du fichier (ex: "politique-rh.pdf")
        content_type: Le type MIME (ex: "application/pdf")
        collection: La collection ChromaDB cible (optionnel, utilise la collection par défaut sinon)

    Returns:
        Un objet DocumentResponse contenant l'ID du document et ses métadonnées
        (dont le statut : "indexed" si tout s'est bien passé, "failed" sinon)
    """
    # Générer un identifiant unique pour ce document (32 caractères hexadécimaux)
    document_id = uuid.uuid4().hex

    # Créer les métadonnées initiales (statut = "processing", on est en cours de traitement)
    metadata = DocumentMetadata(
        filename=filename,
        content_type=content_type,
        size_bytes=len(content),
        status=DocumentStatus.PROCESSING,
        created_at=datetime.utcnow(),
    )

    # Enregistrer le document dans le registre (même avant la fin du traitement)
    doc_response = DocumentResponse(document_id=document_id, metadata=metadata)
    _document_registry[document_id] = doc_response

    try:
        # ── Étape 1 : PARSE ── Extraire le texte brut du fichier
        logger.info("ingestion_parse", document_id=document_id, filename=filename)
        text = parse_document(content, content_type, filename)

        if not text.strip():
            raise ValueError("Document produced no extractable text")

        # ── Étape 2 : CHUNK ── Découper le texte en morceaux
        logger.info("ingestion_chunk", document_id=document_id)
        chunks = chunk_text(
            text,
            document_id,
            metadata={"filename": filename, "content_type": content_type},
        )

        # ── Étape 3 : EMBED & STORE ── Vectoriser les chunks et les sauvegarder
        # (L'embedding est fait à l'intérieur de add_chunks, via la fonction embed_texts)
        logger.info("ingestion_store", document_id=document_id, chunk_count=len(chunks))
        store = get_vector_store(collection=collection)
        store.add_chunks(chunks)

        # ── Succès : mettre à jour les métadonnées ──
        metadata.chunk_count = len(chunks)
        metadata.status = DocumentStatus.INDEXED  # Le document est maintenant cherchable !
        doc_response.metadata = metadata
        _document_registry[document_id] = doc_response
        _save_registry()

        logger.info("ingestion_complete", document_id=document_id)

    except Exception as exc:
        # ── Échec : enregistrer l'erreur dans les métadonnées ──
        logger.error("ingestion_failed", document_id=document_id, error=str(exc))
        metadata.status = DocumentStatus.FAILED
        metadata.error = str(exc)
        doc_response.metadata = metadata
        _document_registry[document_id] = doc_response
        _save_registry()

    return doc_response


# =============================================================================
# FONCTIONS UTILITAIRES (CRUD sur les documents)
# =============================================================================
# CRUD = Create, Read, Update, Delete — les 4 opérations de base sur les données.


def get_document(document_id: str) -> DocumentResponse | None:
    """Retourne un document par son ID, ou None s'il n'existe pas."""
    return _document_registry.get(document_id)


def list_documents() -> list[DocumentResponse]:
    """Retourne la liste de tous les documents indexés."""
    return list(_document_registry.values())


def delete_document(document_id: str) -> bool:
    """Supprime un document et tous ses chunks de la base vectorielle.

    Returns:
        True si le document existait et a été supprimé, False sinon.
    """
    if document_id in _document_registry:
        # Supprimer les chunks de ChromaDB
        store = get_vector_store()
        store.delete_by_document_id(document_id)
        # Supprimer du registre en mémoire
        del _document_registry[document_id]
        # Sauvegarder sur disque
        _save_registry()
        return True
    return False
