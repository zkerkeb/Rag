"""
Authentification JWT — protège les endpoints de l'API avec des tokens.

CE FICHIER SERT À QUOI ?
    Il gère la sécurité de l'API : vérifier QUI fait les requêtes.
    Sans authentification, n'importe qui pourrait uploader des documents
    ou poser des questions au système.

COMMENT ÇA MARCHE ?
    1. L'utilisateur envoie son login/mot de passe à POST /auth/token
    2. Le serveur vérifie les identifiants
    3. Si OK, il crée un TOKEN JWT (une chaîne de caractères signée)
       et le retourne au client
    4. Pour les requêtes suivantes, le client envoie ce token dans l'en-tête HTTP :
       Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
    5. Le serveur vérifie la signature du token à chaque requête

QU'EST-CE QU'UN JWT ?
    JWT = JSON Web Token. C'est un standard pour créer des "laissez-passer" numériques.
    Le token contient des informations (username, rôle, date d'expiration) et une
    signature cryptographique. Le serveur peut vérifier qu'il a bien émis ce token
    sans avoir besoin de consulter une base de données à chaque requête.

    Structure d'un JWT : header.payload.signature
    Exemple : eyJhbGciOiJIUzI1NiIs.eyJ1c2VyIjoiYWRtaW4i.SflKxwRJSMeKKF2QT4

NOTE : L'authentification est DÉSACTIVÉE par défaut (auth.enabled=false dans settings.yaml).
       C'est volontaire pour faciliter le développement et les tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.core.config import get_settings

# HTTPBearer extrait automatiquement le token depuis l'en-tête "Authorization: Bearer <token>"
# auto_error=False : si pas de token, retourne None au lieu de lever une erreur
# (utile quand l'auth est désactivée)
_bearer_scheme = HTTPBearer(auto_error=False)


# =============================================================================
# GESTION DES MOTS DE PASSE
# =============================================================================


def _hash_password(password: str) -> str:
    """Hache un mot de passe avec bcrypt.

    On ne stocke JAMAIS un mot de passe en clair. On stocke un "hash" :
    une version brouillée irréversible. bcrypt est un algorithme standard
    conçu pour être lent (exprès !), ce qui rend les attaques par force brute difficiles.
    """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


# =============================================================================
# UTILISATEURS DE DÉMONSTRATION
# =============================================================================
# En production, on utiliserait une vraie base de données.
# Ici, on crée 2 utilisateurs en dur pour la démo :
# - "admin" (mot de passe: "admin") → rôle administrateur
# - "viewer" (mot de passe: "viewer") → rôle lecteur seul
_DEMO_USERS: dict[str, dict] = {
    "admin": {
        "hashed_password": _hash_password("admin"),
        "role": "admin",
    },
    "viewer": {
        "hashed_password": _hash_password("viewer"),
        "role": "viewer",
    },
}


def verify_password(plain: str, hashed: str) -> bool:
    """Vérifie si un mot de passe en clair correspond à un hash bcrypt.

    bcrypt.checkpw fait la comparaison de manière sécurisée
    (résistante aux attaques par timing).
    """
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def authenticate_user(username: str, password: str) -> dict | None:
    """Vérifie les identifiants d'un utilisateur.

    Returns:
        Les infos de l'utilisateur si les identifiants sont valides, None sinon.
    """
    user = _DEMO_USERS.get(username)
    if user and verify_password(password, user["hashed_password"]):
        return {"username": username, **user}
    return None


# =============================================================================
# CRÉATION ET VÉRIFICATION DES TOKENS JWT
# =============================================================================


def create_access_token(data: dict) -> str:
    """Crée un token JWT signé contenant les données fournies.

    Args:
        data: Les informations à mettre dans le token (ex: {"sub": "admin", "role": "admin"})
              "sub" = subject = l'identité de l'utilisateur (convention JWT)

    Returns:
        Le token JWT sous forme de chaîne de caractères.
    """
    settings = get_settings().auth
    # Calculer la date d'expiration du token
    expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {**data, "exp": expire}  # "exp" = date d'expiration (convention JWT)
    # jwt.encode crée le token signé avec notre clé secrète
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str) -> dict:
    """Décode et vérifie un token JWT.

    Vérifie la signature (le token n'a pas été modifié) et la date d'expiration.
    Lève une exception JWTError si le token est invalide ou expiré.
    """
    settings = get_settings().auth
    return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])


# =============================================================================
# DÉPENDANCE FASTAPI — Protection des endpoints
# =============================================================================


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict | None:
    """Dépendance FastAPI qui vérifie l'authentification de chaque requête.

    COMMENT ÇA S'UTILISE ?
        On l'ajoute comme paramètre d'un endpoint avec Depends() :

        @router.get("/documents")
        async def get_docs(user: dict | None = Depends(get_current_user)):
            ...

        FastAPI appelle automatiquement get_current_user AVANT l'endpoint.
        Si l'auth est activée et le token invalide → erreur 401.
        Si l'auth est désactivée → retourne None (accès libre).

    Returns:
        Les infos de l'utilisateur (username, role) si authentifié, None si l'auth est désactivée.
    """
    settings = get_settings().auth

    # Si l'authentification est désactivée, on laisse passer tout le monde
    if not settings.enabled:
        return None

    # L'auth est activée mais pas de token fourni → refusé
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    # Vérifier et décoder le token JWT
    try:
        payload = decode_token(credentials.credentials)
        username: str | None = payload.get("sub")  # Extraire le nom d'utilisateur
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return {"username": username, "role": payload.get("role", "viewer")}
    except JWTError:
        # Token invalide, expiré, ou signature incorrecte
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
