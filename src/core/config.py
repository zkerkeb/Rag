"""
Configuration de l'application — charge les réglages depuis settings.yaml et les variables d'environnement.

CE FICHIER SERT À QUOI ?
    Centraliser tous les paramètres de l'application dans un seul endroit.
    Plutôt que d'écrire "512" en dur dans le code du chunking, on lit la valeur
    depuis le fichier config/settings.yaml. Si on veut changer la taille des chunks,
    on modifie le YAML et c'est tout — pas besoin de toucher au code Python.

COMMENT ÇA MARCHE ?
    1. Au démarrage, on lit le fichier YAML et on stocke ses valeurs dans un dict Python.
    2. On crée des classes Pydantic qui valident et structurent ces valeurs.
    3. La fonction get_settings() retourne un objet Settings avec tous les paramètres,
       et grâce au cache (@lru_cache), le fichier n'est lu qu'une seule fois.

POURQUOI PYDANTIC ?
    Pydantic vérifie automatiquement les types (ex: chunk_size doit être un int).
    Si le YAML contient une erreur, Pydantic lève une exception claire au démarrage
    plutôt qu'un bug mystérieux plus tard dans l'exécution.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


# Chemin vers le fichier de configuration YAML.
# __file__ = ce fichier Python (config.py)
# .parent.parent.parent = on remonte de src/core/ jusqu'à la racine du projet
_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"


def _load_yaml_config() -> dict:
    """Lit le fichier YAML et retourne son contenu sous forme de dictionnaire Python.

    Si le fichier n'existe pas (ex: en test), retourne un dict vide
    et les valeurs par défaut seront utilisées.
    """
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


# On charge le YAML une seule fois au démarrage du module
_yaml = _load_yaml_config()


# =============================================================================
# CLASSES DE CONFIGURATION
# =============================================================================
# Chaque classe ci-dessous correspond à une section du fichier settings.yaml.
# Les valeurs sont lues depuis le YAML, avec des valeurs par défaut si absentes.
# Pydantic valide automatiquement les types (str, int, float, bool, list...).
# =============================================================================


class ServerSettings(BaseSettings):
    """Réglages du serveur web (host, port, etc.)."""
    host: str = _yaml.get("server", {}).get("host", "0.0.0.0")
    port: int = _yaml.get("server", {}).get("port", 8000)
    workers: int = _yaml.get("server", {}).get("workers", 1)
    reload: bool = _yaml.get("server", {}).get("reload", True)


class LLMSettings(BaseSettings):
    """Réglages du modèle de langage (Claude) qui génère les réponses.

    La clé API est sensible : elle n'est pas dans le YAML mais lue depuis
    la variable d'environnement ANTHROPIC_API_KEY (ou un fichier .env).
    """
    model_config = {"env_file": ".env", "extra": "ignore"}

    provider: str = _yaml.get("llm", {}).get("provider", "anthropic")
    model: str = _yaml.get("llm", {}).get("model", "claude-sonnet-4-20250514")
    max_tokens: int = _yaml.get("llm", {}).get("max_tokens", 4096)
    temperature: float = _yaml.get("llm", {}).get("temperature", 0.1)
    # Field avec validation_alias : Pydantic cherche la variable d'env ANTHROPIC_API_KEY
    api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")


class EmbeddingSettings(BaseSettings):
    """Réglages du modèle d'embedding (all-MiniLM-L6-v2) qui transforme le texte en vecteurs."""
    model: str = _yaml.get("embeddings", {}).get("model", "all-MiniLM-L6-v2")
    dimension: int = _yaml.get("embeddings", {}).get("dimension", 384)


class VectorStoreSettings(BaseSettings):
    """Réglages de la base vectorielle ChromaDB (où les vecteurs sont stockés)."""
    provider: str = _yaml.get("vector_store", {}).get("provider", "chromadb")
    persist_directory: str = _yaml.get("vector_store", {}).get(
        "persist_directory", "./data/chroma_db"
    )
    collection_name: str = _yaml.get("vector_store", {}).get(
        "collection_name", "enterprise_docs"
    )


class ChunkingSettings(BaseSettings):
    """Réglages du découpage des documents en chunks.

    - chunk_size : taille max d'un chunk en caractères (512 = ~1 paragraphe)
    - chunk_overlap : nombre de caractères en commun entre 2 chunks consécutifs
      (pour éviter de couper une phrase en plein milieu)
    - separators : l'algorithme essaie de couper dans cet ordre de priorité
    """
    strategy: str = _yaml.get("chunking", {}).get("strategy", "recursive")
    chunk_size: int = _yaml.get("chunking", {}).get("chunk_size", 512)
    chunk_overlap: int = _yaml.get("chunking", {}).get("chunk_overlap", 50)
    separators: list[str] = _yaml.get("chunking", {}).get(
        "separators", ["\n\n", "\n", ". ", " "]
    )


class RetrievalSettings(BaseSettings):
    """Réglages de la recherche de documents pertinents.

    - top_k : combien de chunks retourner au maximum
    - score_threshold : score minimum (0-1) pour qu'un chunk soit considéré pertinent
    - rerank : si True, un 2e modèle ré-ordonne les résultats pour plus de précision
    """
    top_k: int = _yaml.get("retrieval", {}).get("top_k", 5)
    score_threshold: float = _yaml.get("retrieval", {}).get("score_threshold", 0.3)
    rerank: bool = _yaml.get("retrieval", {}).get("rerank", False)


class AuthSettings(BaseSettings):
    """Réglages de l'authentification JWT (JSON Web Token).

    JWT = un standard pour créer des "jetons" d'accès sécurisés.
    Quand un utilisateur se connecte, le serveur lui donne un token signé.
    À chaque requête suivante, le client envoie ce token pour prouver son identité.
    """
    enabled: bool = _yaml.get("auth", {}).get("enabled", False)
    secret_key: str = _yaml.get("auth", {}).get("secret_key", "change-me-in-production")
    algorithm: str = _yaml.get("auth", {}).get("algorithm", "HS256")
    access_token_expire_minutes: int = _yaml.get("auth", {}).get(
        "access_token_expire_minutes", 60
    )


class LoggingSettings(BaseSettings):
    """Réglages des logs (journaux d'événements de l'application)."""
    level: str = _yaml.get("logging", {}).get("level", "INFO")
    format: str = _yaml.get("logging", {}).get("format", "json")


class MonitoringSettings(BaseSettings):
    """Réglages du monitoring Prometheus (métriques de surveillance)."""
    prometheus_enabled: bool = _yaml.get("monitoring", {}).get("prometheus_enabled", True)
    metrics_path: str = _yaml.get("monitoring", {}).get("metrics_path", "/metrics")


class Settings(BaseSettings):
    """Objet principal qui regroupe TOUTES les catégories de configuration.

    Utilisation dans le code :
        settings = get_settings()
        settings.chunking.chunk_size   → 512
        settings.llm.model             → "claude-sonnet-4-20250514"
        settings.llm.api_key           → la clé API depuis l'environnement
    """
    server: ServerSettings = ServerSettings()
    llm: LLMSettings = LLMSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    auth: AuthSettings = AuthSettings()
    logging: LoggingSettings = LoggingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()


@lru_cache  # Cache : la fonction n'est exécutée qu'une seule fois, ensuite le résultat est réutilisé
def get_settings() -> Settings:
    """Point d'accès unique à la configuration.

    Grâce à @lru_cache, l'objet Settings est créé une seule fois
    puis retourné directement à chaque appel suivant.
    Tout le code de l'application appelle cette fonction pour accéder aux réglages.
    """
    return Settings()
