"""
Moteur RAG — le cerveau du système : chercher le contexte, construire le prompt, générer la réponse.

CE FICHIER SERT À QUOI ?
    C'est ici que tout se rassemble. Quand l'utilisateur pose une question,
    ce fichier orchestre les 3 étapes du RAG :

    1. RETRIEVE (chercher) — Trouver les chunks les plus pertinents dans ChromaDB
    2. AUGMENT (enrichir) — Injecter ces chunks dans le prompt envoyé au LLM
    3. GENERATE (générer) — Envoyer le tout à Claude pour obtenir une réponse

    Le mot "RAG" vient de ces 3 étapes : Retrieval-Augmented Generation.

POURQUOI SÉPARER RETRIEVE ET GENERATE ?
    Le LLM (Claude) est très bon pour formuler des réponses en langage naturel,
    mais il ne connaît PAS vos documents. En lui donnant les bons extraits AVANT
    de lui poser la question, on obtient une réponse basée sur des FAITS RÉELS
    plutôt qu'une invention (hallucination).
"""

from __future__ import annotations

import time

import anthropic
import structlog

from src.core.config import get_settings
from src.models.schemas import QueryRequest, QueryResponse, SourceDocument
from src.retrieval.vector_store import get_vector_store

logger = structlog.get_logger(__name__)

# =============================================================================
# PROMPTS (les instructions envoyées au LLM)
# =============================================================================

# Le SYSTEM PROMPT définit le comportement de l'IA.
# C'est un texte "caché" que l'utilisateur ne voit pas, mais qui guide Claude.
# Ici, on lui ordonne de :
# - Ne répondre QU'à partir du contexte fourni (pas d'invention !)
# - Avouer quand il ne sait pas
# - Citer ses sources
SYSTEM_PROMPT = """\
You are an enterprise AI assistant. Answer questions using ONLY the provided context documents.

Rules:
1. Base your answer strictly on the context provided below.
2. If the context does not contain enough information to answer, say so clearly.
3. Cite which source documents support your answer when possible.
4. Be concise, accurate, and professional.
"""

# Le CONTEXT TEMPLATE structure le message envoyé au LLM.
# Il contient deux sections :
# 1. Les documents retrouvés (le "contexte")
# 2. La question de l'utilisateur
# Le LLM voit ce texte et doit formuler sa réponse à partir du contexte.
CONTEXT_TEMPLATE = """\
## Retrieved Context

{context}

## User Question

{question}
"""


# =============================================================================
# CONSTRUCTION DU CONTEXTE
# =============================================================================


def _build_context_block(sources: list[SourceDocument]) -> str:
    """Formate les chunks retrouvés en un bloc de texte structuré pour le LLM.

    Chaque source est numérotée et accompagnée de son score de pertinence
    et de ses métadonnées (nom du fichier, etc.). Les sources sont séparées
    par des lignes horizontales (---) pour que le LLM les distingue clairement.

    Exemple de sortie :
        [Source 1 | score=0.87] (filename=politique-rh.pdf)
        Le télétravail est autorisé 2 jours par semaine...

        ---

        [Source 2 | score=0.72] (filename=charte-interne.pdf)
        Les employés doivent être joignables de 9h à 17h...
    """
    parts: list[str] = []
    for i, src in enumerate(sources, 1):
        # En-tête : numéro de source + score de pertinence
        header = f"[Source {i} | score={src.score}]"
        # Ajouter les métadonnées (sauf chunk_index qui n'est pas utile pour le LLM)
        meta_str = ", ".join(f"{k}={v}" for k, v in src.metadata.items() if k != "chunk_index")
        if meta_str:
            header += f" ({meta_str})"
        parts.append(f"{header}\n{src.text}")
    return "\n\n---\n\n".join(parts)


# =============================================================================
# FONCTION PRINCIPALE : QUERY
# =============================================================================


def query(request: QueryRequest) -> QueryResponse:
    """Exécute une requête RAG complète : retrieve → augment → generate.

    C'est LA fonction principale de tout le système. Elle est appelée par l'endpoint
    POST /api/v1/query quand l'utilisateur pose une question.

    Args:
        request: La requête de l'utilisateur contenant la question et les paramètres
                 (top_k, score_threshold, collection...)

    Returns:
        La réponse complète : texte généré par Claude + sources citées + métadonnées.
    """
    settings = get_settings()
    start = time.perf_counter()  # Chronomètre pour mesurer le temps total

    # ── ÉTAPE 1 : RETRIEVE ── Chercher les chunks pertinents dans ChromaDB
    store = get_vector_store(collection=request.collection)
    sources = store.search(
        query=request.question,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
    )

    logger.info(
        "retrieval_complete",
        question=request.question[:80],  # Log tronqué pour ne pas être trop long
        sources_found=len(sources),
    )

    # ── ÉTAPE 2 : AUGMENT ── Construire le prompt avec le contexte retrouvé
    if sources:
        # On a trouvé des chunks pertinents → on les injecte dans le prompt
        context_block = _build_context_block(sources)
        user_message = CONTEXT_TEMPLATE.format(
            context=context_block,
            question=request.question,
        )
    else:
        # Aucun chunk pertinent trouvé → on prévient le LLM
        user_message = (
            f"No relevant documents were found for this question.\n\n"
            f"Question: {request.question}\n\n"
            f"Please let the user know that no matching documents were found "
            f"and suggest they upload relevant documents first."
        )

    # ── ÉTAPE 3 : GENERATE ── Envoyer le prompt à Claude et obtenir la réponse

    # Vérifier si une clé API est configurée
    api_key = settings.llm.api_key
    if not api_key:
        # PAS de clé API → mode dégradé : on retourne les chunks bruts sans génération LLM
        # C'est utile pour tester la recherche même sans accès payant à Claude
        elapsed = (time.perf_counter() - start) * 1000
        return QueryResponse(
            answer=(
                "LLM API key not configured. Here are the most relevant document "
                "excerpts found:\n\n"
                + "\n\n---\n\n".join(s.text for s in sources[:3])
                if sources
                else "No API key configured and no relevant documents found."
            ),
            sources=sources,
            query=request.question,
            model="none (no API key)",
            latency_ms=round(elapsed, 1),
        )

    # Créer le client Anthropic (connexion à l'API Claude)
    client = anthropic.Anthropic(api_key=api_key)

    # Appel à l'API Claude pour générer la réponse
    response = client.messages.create(
        model=settings.llm.model,           # Ex: "claude-sonnet-4-20250514"
        max_tokens=settings.llm.max_tokens, # Longueur max de la réponse (4096 tokens)
        temperature=settings.llm.temperature,  # 0.1 = réponses factuelles et prévisibles
        system=SYSTEM_PROMPT,               # Les instructions cachées pour Claude
        messages=[{"role": "user", "content": user_message}],
        # ↑ Le message "utilisateur" contient le contexte ET la question
    )

    # Extraire le texte de la réponse de Claude
    answer = response.content[0].text
    elapsed = (time.perf_counter() - start) * 1000  # Temps total en millisecondes

    logger.info("generation_complete", latency_ms=round(elapsed, 1))

    # Retourner la réponse structurée avec toutes les informations
    return QueryResponse(
        answer=answer,                 # La réponse générée par Claude
        sources=sources,               # Les chunks qui ont servi de contexte
        query=request.question,        # La question originale
        model=settings.llm.model,      # Le modèle utilisé
        latency_ms=round(elapsed, 1),  # Le temps total de traitement
    )
