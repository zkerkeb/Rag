"""
Modèle d'embedding — transforme du texte en vecteurs (listes de nombres).

CE FICHIER SERT À QUOI ?
    C'est le "traducteur" entre le monde du texte et le monde des maths.
    Il prend une phrase en français (ou autre langue) et la convertit en une liste
    de 384 nombres décimaux. Cette liste de nombres s'appelle un "vecteur" ou "embedding".

POURQUOI ON A BESOIN DE VECTEURS ?
    Un ordinateur ne peut pas mesurer directement si deux phrases "parlent de la même chose".
    Mais il sait très bien comparer des listes de nombres ! L'embedding capture le SENS
    du texte dans ces nombres, de sorte que :
    - "Le chat dort" et "Le félin sommeille" → vecteurs PROCHES (même sens)
    - "Le chat dort" et "La bourse monte" → vecteurs ÉLOIGNÉS (sens différent)

    C'est ce qui permet la recherche par similarité : quand l'utilisateur pose une question,
    on la transforme en vecteur et on cherche les chunks dont les vecteurs sont les plus proches.

LE MODÈLE UTILISÉ :
    all-MiniLM-L6-v2 (via la bibliothèque sentence-transformers)
    - Gratuit, open-source, tourne en local (pas besoin d'Internet)
    - Produit des vecteurs de 384 dimensions
    - Multilingue (comprend le français)
    - ~80 Mo en mémoire — assez léger pour un laptop
"""

from __future__ import annotations

from functools import lru_cache

import structlog
from sentence_transformers import SentenceTransformer

from src.core.config import get_settings

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)  # Cache : le modèle n'est chargé en mémoire qu'UNE seule fois
def _load_model() -> SentenceTransformer:
    """Charge le modèle d'embedding en mémoire.

    Le chargement prend quelques secondes (téléchargement au 1er lancement, puis cache local).
    Grâce à @lru_cache, cette opération n'est faite qu'une fois : les appels suivants
    retournent directement le modèle déjà chargé.
    """
    settings = get_settings().embeddings
    logger.info("loading_embedding_model", model=settings.model)
    model = SentenceTransformer(settings.model)  # Charge "all-MiniLM-L6-v2"
    logger.info("embedding_model_loaded", model=settings.model)
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Transforme une LISTE de textes en une liste de vecteurs.

    Utilisé lors de l'ingestion : on vectorise tous les chunks d'un document d'un coup.
    Le traitement par lot ("batch") est beaucoup plus rapide que de vectoriser un par un.

    Args:
        texts: Liste de chaînes de caractères (ex: les chunks d'un document)

    Returns:
        Liste de vecteurs (chacun = 384 nombres décimaux).
        Le vecteur à l'index i correspond au texte à l'index i dans la liste d'entrée.

    Exemple :
        embed_texts(["Le chat dort", "Le budget Q3"]) →
        [[0.02, -0.15, ..., 0.23],   # vecteur pour "Le chat dort"
         [0.81, 0.04, ..., -0.67]]    # vecteur pour "Le budget Q3"
    """
    model = _load_model()
    embeddings = model.encode(
        texts,
        show_progress_bar=False,       # Pas de barre de progression (on est dans un serveur)
        normalize_embeddings=True,      # Met tous les vecteurs à la même échelle (norme 1)
        # ↑ Normaliser permet d'utiliser la similarité cosinus de manière fiable
        # et rend les calculs de comparaison plus rapides
    )
    return embeddings.tolist()  # Convertit le tableau numpy en liste Python standard


def embed_query(query: str) -> list[float]:
    """Transforme UNE question en un seul vecteur.

    Utilisé au moment de la requête : on vectorise la question de l'utilisateur
    pour la comparer aux vecteurs des chunks stockés dans ChromaDB.

    Args:
        query: La question de l'utilisateur (ex: "Quelle est la politique de télétravail ?")

    Returns:
        Un seul vecteur de 384 nombres décimaux.
    """
    model = _load_model()
    embedding = model.encode(
        query,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return embedding.tolist()
