"""UI routes serving the demo frontend."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_TEMPLATES = Jinja2Templates(directory=str(_BASE_DIR / "templates"))

router = APIRouter(include_in_schema=False)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return _TEMPLATES.TemplateResponse("index.html", {"request": request})
