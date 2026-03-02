"""
Configuration des logs structurés avec structlog.

CE FICHIER SERT À QUOI ?
    Les "logs" sont les messages que l'application écrit pour dire ce qu'elle fait :
    - "Document parsé avec succès"
    - "Recherche effectuée en 234ms, 3 résultats trouvés"
    - "ERREUR : fichier PDF corrompu"

    Sans logs, on navigue à l'aveugle : impossible de savoir ce qui s'est passé
    quand quelque chose ne marche pas.

POURQUOI STRUCTLOG ?
    Les logs classiques (print ou logging) produisent du texte brut :
        INFO 2024-01-15 Document parsed successfully

    structlog produit des logs STRUCTURÉS en JSON :
        {"level": "info", "event": "document_parsed", "document_id": "abc123", "timestamp": "..."}

    L'avantage du JSON : les outils de monitoring (Elasticsearch, Datadog, Grafana...)
    peuvent facilement filtrer et analyser les logs. Par exemple :
    "Montre-moi tous les logs d'erreur du dernier quart d'heure pour les documents PDF".

DEUX MODES :
    - "json" : pour la production (lisible par les machines)
    - "console" : pour le développement (lisible par les humains, avec des couleurs)
"""

from __future__ import annotations

import logging
import sys

import structlog

from src.core.config import get_settings


def setup_logging() -> None:
    """Configure le système de logs structurés.

    Appelé une seule fois au démarrage de l'application (dans main.py → lifespan).
    """
    settings = get_settings().logging

    # Les "processors" sont une chaîne de transformations appliquées à chaque message de log.
    # Chaque processor ajoute ou formate une information avant que le log ne soit écrit.
    processors: list = [
        structlog.contextvars.merge_contextvars,     # Ajoute les variables de contexte (ex: request_id)
        structlog.stdlib.add_log_level,               # Ajoute le niveau (INFO, ERROR, etc.)
        structlog.stdlib.add_logger_name,              # Ajoute le nom du module qui a écrit le log
        structlog.processors.TimeStamper(fmt="iso"),   # Ajoute un timestamp au format ISO 8601
        structlog.processors.StackInfoRenderer(),      # Ajoute la stack trace en cas d'erreur
        structlog.processors.UnicodeDecoder(),         # Gère proprement les caractères Unicode
    ]

    # Le dernier processor décide du FORMAT de sortie
    if settings.format == "json":
        # Mode production : chaque log = une ligne JSON
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Mode développement : logs colorés et lisibles dans le terminal
        processors.append(structlog.dev.ConsoleRenderer())

    # Appliquer la configuration à structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,  # Optimisation : met le logger en cache après 1er usage
    )

    # Configurer aussi le module logging standard de Python
    # (certaines bibliothèques tierces l'utilisent directement)
    level = getattr(logging, settings.level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,   # Écrire les logs sur la sortie standard (le terminal)
        level=level,         # Niveau minimum : INFO par défaut (ignore les DEBUG)
    )
