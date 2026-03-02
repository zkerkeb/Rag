"""
Découpage de texte en chunks — divise les documents en morceaux indexables.

CE FICHIER SERT À QUOI ?
    C'est la DEUXIÈME étape du pipeline d'ingestion : le CHUNKING.
    Après avoir extrait le texte brut d'un document (étape 1 : parsing),
    on le découpe en petits morceaux appelés "chunks".

POURQUOI DÉCOUPER ?
    1. Un document entier est souvent trop gros pour le prompt du LLM
    2. On veut retrouver LE passage pertinent, pas tout le document
    3. Le modèle d'embedding fonctionne mieux sur des textes courts

    Analogie : c'est comme découper un livre en fiches de révision.
    Quand on cherche une info, on parcourt les fiches — pas tout le livre.

COMMENT ÇA DÉCOUPE ?
    On utilise le "RecursiveCharacterTextSplitter" de LangChain.
    "Récursif" signifie qu'il essaie de couper intelligemment :
      1. D'abord entre les paragraphes (\\n\\n)
      2. Si le morceau est encore trop gros, entre les lignes (\\n)
      3. Puis entre les phrases (". ")
      4. En dernier recours, entre les mots (" ")
    Ainsi, on évite autant que possible de couper au milieu d'une idée.
"""

from __future__ import annotations

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
import structlog

from src.core.config import get_settings
from src.models.schemas import Chunk

logger = structlog.get_logger(__name__)


def chunk_text(
    text: str,
    document_id: str,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Découpe un texte en une liste de chunks.

    Args:
        text: Le texte brut à découper (résultat du parsing)
        document_id: L'identifiant du document parent (pour garder le lien)
        metadata: Métadonnées à attacher à chaque chunk (filename, content_type...)

    Returns:
        Une liste d'objets Chunk, chacun contenant un morceau de texte
        et ses métadonnées (position dans le document, fichier d'origine...).

    Exemple :
        Un texte de 2000 caractères avec chunk_size=512 et overlap=50
        donnera environ 4-5 chunks de ~512 caractères, avec 50 caractères
        de chevauchement entre chaque chunk consécutif.
    """
    settings = get_settings().chunking
    extra_meta = metadata or {}

    # Création du "splitter" — l'outil qui va découper le texte
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,       # Taille max d'un chunk (ex: 512 caractères)
        chunk_overlap=settings.chunk_overlap,  # Chevauchement entre chunks (ex: 50 caractères)
        separators=settings.separators,        # Ordre de priorité pour le découpage
        length_function=len,                   # Comment mesurer la taille (ici : nombre de caractères)
    )

    # split_text retourne une liste de chaînes de caractères
    raw_chunks = splitter.split_text(text)
    logger.info("text_chunked", document_id=document_id, chunk_count=len(raw_chunks))

    # On transforme chaque morceau de texte brut en un objet Chunk structuré
    chunks: list[Chunk] = []
    for idx, raw in enumerate(raw_chunks):
        # Création d'un identifiant unique pour ce chunk
        # Format : "id_du_document::12_caractères_aléatoires"
        # Permet de retrouver facilement de quel document vient ce chunk
        chunk_id = f"{document_id}::{uuid.uuid4().hex[:12]}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=raw,
                metadata={
                    **extra_meta,          # On recopie les métadonnées du document (filename, etc.)
                    "chunk_index": idx,    # Position de ce chunk dans le document (0, 1, 2...)
                    "total_chunks": len(raw_chunks),  # Nombre total de chunks pour ce document
                },
            )
        )

    return chunks
