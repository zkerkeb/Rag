"""
Routes UI — sert la page web de démonstration.

CE FICHIER SERT À QUOI ?
    Il définit un seul endpoint : la page d'accueil (/).
    Quand un utilisateur ouvre http://localhost:8000 dans son navigateur,
    ce code lui envoie la page HTML de l'interface de démonstration.

    Cette page HTML (templates/index.html) est un formulaire qui permet de :
    - Uploader des documents
    - Poser des questions au système RAG
    - Voir les réponses et les sources citées

    Le rendu HTML est fait avec Jinja2, un moteur de templates Python.
    Jinja2 permet d'injecter des données Python dans le HTML avant de l'envoyer
    (même si ici on n'injecte que l'objet "request" requis par FastAPI).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Chemin vers le dossier templates/ à la racine du projet
_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_TEMPLATES = Jinja2Templates(directory=str(_BASE_DIR / "templates"))

# include_in_schema=False → ces routes n'apparaissent pas dans la doc Swagger
# (c'est une page web, pas un endpoint d'API)
router = APIRouter(include_in_schema=False)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Sert la page d'accueil de l'interface web.

    FastAPI exige que l'objet "request" soit passé au template Jinja2
    (c'est une contrainte technique de Starlette, le framework sous-jacent).
    """
    return _TEMPLATES.TemplateResponse("index.html", {"request": request})
