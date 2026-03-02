"""
Parseurs de documents — extraient le texte brut depuis différents formats de fichiers.

CE FICHIER SERT À QUOI ?
    C'est la PREMIÈRE étape du pipeline d'ingestion : le PARSING.
    Quand un utilisateur uploade un fichier (PDF, Word, texte...), on doit en extraire
    le texte brut (une simple chaîne de caractères). Ce texte sera ensuite découpé
    en chunks, puis transformé en vecteurs.

    Chaque format de fichier a son propre "lecteur" :
    - Les fichiers texte (.txt, .md, .csv) sont simplement décodés en UTF-8
    - Les PDF sont lus page par page avec la bibliothèque pypdf
    - Les fichiers Word (.docx) sont lus paragraphe par paragraphe avec python-docx

POURQUOI C'EST NÉCESSAIRE ?
    Un modèle d'embedding ne peut pas lire un fichier binaire PDF directement.
    Il a besoin de texte brut. Le parser est le "traducteur" entre le format
    du fichier et le texte que le reste du pipeline peut traiter.
"""

from __future__ import annotations

import io
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Dictionnaire qui associe chaque type MIME à un type de parseur.
# Le type MIME est un standard du web qui identifie le format d'un fichier.
# Par exemple : "application/pdf" = un fichier PDF, "text/plain" = un fichier texte.
SUPPORTED_TYPES: dict[str, str] = {
    "text/plain": "text",       # Fichier .txt
    "text/markdown": "text",    # Fichier .md (Markdown)
    "text/csv": "text",         # Fichier .csv (tableur)
    "application/pdf": "pdf",   # Fichier .pdf
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    # ↑ Type MIME officiel (et très long !) pour les fichiers Word .docx
}


def parse_document(content: bytes, content_type: str, filename: str) -> str:
    """Fonction principale : extrait le texte brut d'un document.

    Args:
        content: Le contenu binaire du fichier (les octets bruts lus depuis le fichier uploadé)
        content_type: Le type MIME du fichier (ex: "application/pdf")
        filename: Le nom du fichier (utilisé pour les logs)

    Returns:
        Le texte brut extrait du document (une simple chaîne de caractères)

    Raises:
        ValueError: Si le format du fichier n'est pas supporté et ne peut pas être décodé en texte
    """
    # On cherche quel parseur utiliser en fonction du type MIME
    parser_key = SUPPORTED_TYPES.get(content_type)

    if parser_key is None:
        # Type de fichier inconnu : on tente quand même de le lire comme du texte
        logger.warning("unsupported_content_type", content_type=content_type, filename=filename)
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            # Si ça échoue (fichier binaire comme une image), on abandonne
            raise ValueError(f"Unsupported content type: {content_type}")

    # On appelle le parseur approprié selon le type de fichier
    if parser_key == "text":
        return _parse_text(content)
    elif parser_key == "pdf":
        return _parse_pdf(content)
    elif parser_key == "docx":
        return _parse_docx(content)
    else:
        raise ValueError(f"No parser for: {parser_key}")


def _parse_text(content: bytes) -> str:
    """Parse un fichier texte : simplement décoder les octets en chaîne UTF-8.

    C'est le cas le plus simple : un fichier .txt est déjà du texte,
    il suffit de le convertir d'octets (bytes) en chaîne de caractères (str).
    """
    return content.decode("utf-8")


def _parse_pdf(content: bytes) -> str:
    """Parse un fichier PDF : extrait le texte de chaque page et les concatène.

    Un PDF est un format binaire complexe (positions de texte, polices, images...).
    La bibliothèque pypdf sait le lire et en extraire le texte lisible.

    Note : l'import est fait ici (et non en haut du fichier) pour ne pas ralentir
    le démarrage de l'application si pypdf n'est pas utilisé.
    """
    from pypdf import PdfReader

    # io.BytesIO transforme les bytes en un objet "fichier" que PdfReader peut lire
    reader = PdfReader(io.BytesIO(content))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:  # Certaines pages peuvent être vides (images sans texte)
            pages.append(text)
    # On joint toutes les pages avec un double saut de ligne entre chaque
    return "\n\n".join(pages)


def _parse_docx(content: bytes) -> str:
    """Parse un fichier Word (.docx) : extrait le texte de chaque paragraphe.

    Un fichier .docx est en réalité un fichier ZIP contenant du XML.
    La bibliothèque python-docx sait le décoder et en extraire les paragraphes.
    """
    from docx import Document

    doc = Document(io.BytesIO(content))
    # On récupère le texte de chaque paragraphe non vide
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)
