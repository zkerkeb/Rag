"""
Base vectorielle ChromaDB — stocke les chunks et permet la recherche par similarité.

CE FICHIER SERT À QUOI ?
    C'est le "moteur de recherche" du système RAG. Il fait deux choses :

    1. STOCKER les chunks (lors de l'ingestion)
       Chaque chunk est sauvegardé avec : son texte, son vecteur, et ses métadonnées.

    2. CHERCHER les chunks pertinents (lors d'une question)
       Quand l'utilisateur pose une question, on la transforme en vecteur
       et on cherche les chunks dont les vecteurs sont les plus proches.

POURQUOI CHROMADB ?
    Une base de données classique (SQL) cherche par mots-clés exacts.
    ChromaDB cherche par SIMILARITÉ DE SENS :
    - SQL : "budget" ne trouvera PAS un document qui parle de "dépenses"
    - ChromaDB : "budget" TROUVERA "dépenses" car les vecteurs sont proches

    ChromaDB est gratuit, open-source, et tourne en local (pas besoin de serveur externe).
    Les données sont sauvegardées sur disque et survivent au redémarrage de l'application.

COMMENT FONCTIONNE LA RECHERCHE ?
    ChromaDB utilise l'algorithme HNSW (Hierarchical Navigable Small World) :
    au lieu de comparer la question à CHAQUE chunk (trop lent si on a des millions),
    il navigue intelligemment dans un graphe pour trouver les plus proches rapidement.
    C'est comme un GPS qui zoome progressivement au lieu de scanner toute la carte.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
import structlog

from src.core.config import get_settings
from src.models.schemas import Chunk, SourceDocument
from src.retrieval.embeddings import embed_texts, embed_query

logger = structlog.get_logger(__name__)


class VectorStore:
    """Encapsule une collection ChromaDB avec des méthodes typées et documentées.

    Une "collection" dans ChromaDB est l'équivalent d'une "table" en SQL :
    elle regroupe un ensemble de vecteurs associés à des textes et des métadonnées.
    """

    def __init__(self, collection_name: str | None = None) -> None:
        """Initialise la connexion à ChromaDB et crée/ouvre une collection.

        Args:
            collection_name: Nom de la collection à utiliser.
                            Si None, utilise le nom par défaut dans la config.
        """
        settings = get_settings().vector_store

        # Créer le dossier de stockage s'il n'existe pas encore
        persist_dir = settings.persist_directory
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        # PersistentClient = les données sont sauvegardées sur disque (dans persist_dir)
        # Contrairement à un Client() en mémoire, les données survivent au redémarrage
        self._client = chromadb.PersistentClient(path=persist_dir)

        # get_or_create_collection : ouvre la collection si elle existe, sinon la crée
        name = collection_name or settings.collection_name
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
            # ↑ On utilise la distance cosinus pour comparer les vecteurs.
            # Le cosinus mesure l'angle entre deux vecteurs (= la direction),
            # pas leur taille. C'est plus fiable pour comparer des textes de longueurs différentes.
        )
        logger.info("vector_store_ready", collection=name)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Ajoute une liste de chunks dans la base vectorielle.

        Pour chaque chunk, cette méthode :
        1. Génère son embedding (vecteur de 384 nombres) via embed_texts()
        2. Stocke dans ChromaDB : l'ID, le texte original, le vecteur, et les métadonnées

        Args:
            chunks: Liste d'objets Chunk à indexer (résultat du chunking).
        """
        if not chunks:
            return

        # Préparer les données pour ChromaDB
        texts = [c.text for c in chunks]          # Les textes bruts
        ids = [c.chunk_id for c in chunks]         # Les identifiants uniques
        metadatas = [{**c.metadata, "document_id": c.document_id} for c in chunks]
        # ↑ On ajoute document_id dans les métadonnées pour pouvoir retrouver
        #   tous les chunks d'un document (utile pour la suppression)

        # Générer les vecteurs pour tous les chunks d'un coup (traitement par lot)
        embeddings = embed_texts(texts)

        # Stocker le tout dans ChromaDB
        self._collection.add(
            ids=ids,               # Identifiants uniques (pour éviter les doublons)
            documents=texts,       # Texte original (retourné lors de la recherche)
            embeddings=embeddings, # Vecteurs 384D (utilisés pour la recherche par similarité)
            metadatas=metadatas,   # Métadonnées (filename, document_id, position...)
        )
        logger.info("chunks_stored", count=len(chunks))

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> list[SourceDocument]:
        """Cherche les chunks les plus similaires à une question.

        C'est LE cœur de la recherche RAG. Le processus :
        1. Transformer la question en vecteur (embed_query)
        2. Demander à ChromaDB les top_k vecteurs les plus proches
        3. Filtrer ceux dont le score est trop bas (< score_threshold)
        4. Retourner les résultats triés par pertinence décroissante

        Args:
            query: La question de l'utilisateur en texte
            top_k: Nombre maximum de résultats à retourner
            score_threshold: Score minimum (0-1) pour garder un résultat

        Returns:
            Liste de SourceDocument triée par score décroissant (le plus pertinent en premier).
        """
        # Étape 1 : vectoriser la question
        query_embedding = embed_query(query)

        # Étape 2 : chercher les vecteurs les plus proches dans ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],  # Le vecteur de la question
            n_results=top_k,                      # Nombre max de résultats
            include=["documents", "metadatas", "distances"],
            # ↑ On demande à récupérer le texte, les métadonnées, ET la distance
        )

        sources: list[SourceDocument] = []
        if not results["ids"] or not results["ids"][0]:
            return sources  # Aucun résultat trouvé

        # Étape 3 : convertir les distances en scores et filtrer
        for idx, chunk_id in enumerate(results["ids"][0]):
            # ChromaDB retourne une DISTANCE cosinus (0 = identique, 2 = opposé)
            # On la convertit en SCORE de similarité (1 = identique, 0 = sans rapport)
            # car c'est plus intuitif : un score élevé = un résultat pertinent
            distance = results["distances"][0][idx]
            score = 1.0 - distance

            # Filtrer les résultats dont le score est trop bas
            if score < score_threshold:
                continue

            meta = results["metadatas"][0][idx] if results["metadatas"] else {}
            # On extrait document_id des métadonnées (il a été ajouté dans add_chunks)
            document_id = meta.pop("document_id", "unknown")

            sources.append(
                SourceDocument(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    text=results["documents"][0][idx],  # Le texte original du chunk
                    score=round(score, 4),               # Score arrondi à 4 décimales
                    metadata=meta,
                )
            )

        # Étape 4 : trier par score décroissant (le plus pertinent en premier)
        sources.sort(key=lambda s: s.score, reverse=True)
        return sources

    def delete_by_document_id(self, document_id: str) -> None:
        """Supprime tous les chunks d'un document de la base vectorielle.

        Utilise le champ document_id stocké dans les métadonnées de chaque chunk.
        Appelé quand l'utilisateur supprime un document via DELETE /documents/{id}.
        """
        self._collection.delete(where={"document_id": document_id})
        logger.info("chunks_deleted", document_id=document_id)

    @property
    def count(self) -> int:
        """Retourne le nombre total de chunks dans cette collection."""
        return self._collection.count()

    @property
    def name(self) -> str:
        """Retourne le nom de cette collection."""
        return self._collection.name


# =============================================================================
# CACHE DES INSTANCES (SINGLETON PAR COLLECTION)
# =============================================================================
# On ne veut pas créer un nouvel objet VectorStore à chaque requête :
# ça rouvrirait la connexion ChromaDB à chaque fois (lent et inutile).
# On garde une seule instance par collection dans un dictionnaire.

_stores: dict[str, VectorStore] = {}


def get_vector_store(collection: str | None = None) -> VectorStore:
    """Retourne l'instance VectorStore pour une collection donnée.

    Si l'instance n'existe pas encore, elle est créée et mise en cache.
    Les appels suivants pour la même collection retournent la même instance.

    Args:
        collection: Nom de la collection. Si None, utilise la collection par défaut.
    """
    key = collection or get_settings().vector_store.collection_name
    if key not in _stores:
        _stores[key] = VectorStore(collection_name=key)
    return _stores[key]


def reset_stores() -> None:
    """Vide le cache d'instances (utile pour les tests unitaires)."""
    _stores.clear()
