/**
 * Enterprise RAG Demo — Frontend
 */

const API = "/api/v1";

// ── DOM refs ────────────────────────────────────────────────────────────────

const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const uploadStatus = document.getElementById("upload-status");
const docsList = document.getElementById("documents-list");
const refreshBtn = document.getElementById("refresh-docs");
const queryForm = document.getElementById("query-form");
const queryInput = document.getElementById("query-input");
const queryBtn = document.getElementById("query-btn");
const topKInput = document.getElementById("top-k");
const thresholdInput = document.getElementById("score-threshold");
const answerSection = document.getElementById("answer-section");
const answerContent = document.getElementById("answer-content");
const answerMeta = document.getElementById("answer-meta");
const sourceCount = document.getElementById("source-count");
const sourcesList = document.getElementById("sources-list");
const themeToggle = document.getElementById("theme-toggle");

// ── Theme ───────────────────────────────────────────────────────────────────

function initTheme() {
    const saved = localStorage.getItem("theme") || "light";
    document.documentElement.setAttribute("data-theme", saved);
}

themeToggle.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
});

// ── Upload ──────────────────────────────────────────────────────────────────

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener("change", () => {
    handleFiles(fileInput.files);
    fileInput.value = "";
});

async function handleFiles(files) {
    for (const file of files) {
        const item = createUploadItem(file.name, "loading");
        uploadStatus.prepend(item);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const resp = await fetch(`${API}/documents/upload`, {
                method: "POST",
                body: formData,
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || resp.statusText);
            }

            const doc = await resp.json();
            item.className = "upload-item success";
            item.querySelector(".status").textContent =
                `Indexed (${doc.metadata.chunk_count} chunks)`;
        } catch (err) {
            item.className = "upload-item error";
            item.querySelector(".status").textContent = `Error: ${err.message}`;
        }
    }
    refreshDocuments();
}

function createUploadItem(name, state) {
    const div = document.createElement("div");
    div.className = `upload-item ${state}`;
    div.innerHTML = `
        <span class="name">${escapeHtml(name)}</span>
        <span class="status">${state === "loading" ? '<span class="spinner"></span> Uploading...' : ""}</span>
    `;
    return div;
}

// ── Documents ───────────────────────────────────────────────────────────────

async function refreshDocuments() {
    try {
        const resp = await fetch(`${API}/documents`);
        const data = await resp.json();

        if (data.documents.length === 0) {
            docsList.innerHTML = '<p class="empty-state">No documents uploaded yet.</p>';
            return;
        }

        docsList.innerHTML = data.documents
            .map((doc) => {
                const m = doc.metadata;
                const sizeKB = (m.size_bytes / 1024).toFixed(1);
                return `
                <div class="doc-item">
                    <div>
                        <span class="doc-name">${escapeHtml(m.filename)}</span>
                        <span class="doc-meta">${sizeKB} KB &middot; ${m.chunk_count} chunks</span>
                    </div>
                    <span class="badge ${m.status}">${m.status}</span>
                </div>`;
            })
            .join("");
    } catch {
        docsList.innerHTML = '<p class="empty-state">Failed to load documents.</p>';
    }
}

refreshBtn.addEventListener("click", refreshDocuments);

// ── Query ───────────────────────────────────────────────────────────────────

queryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = queryInput.value.trim();
    if (!question) return;

    queryBtn.disabled = true;
    queryBtn.innerHTML = '<span class="spinner"></span>';
    answerSection.hidden = false;
    answerContent.textContent = "Thinking...";
    answerMeta.textContent = "";
    sourcesList.innerHTML = "";
    sourceCount.textContent = "0";

    try {
        const resp = await fetch(`${API}/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                top_k: parseInt(topKInput.value) || 5,
                score_threshold: parseFloat(thresholdInput.value) || 0.3,
            }),
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || resp.statusText);
        }

        const data = await resp.json();

        answerContent.textContent = data.answer;
        answerMeta.textContent = `Model: ${data.model} | Latency: ${data.latency_ms.toFixed(0)} ms`;
        sourceCount.textContent = data.sources.length;

        sourcesList.innerHTML = data.sources
            .map(
                (s, i) => `
                <div class="source-item">
                    <strong>Source ${i + 1}</strong> &mdash;
                    <span class="score">Score: ${s.score}</span>
                    ${s.metadata.filename ? `| ${escapeHtml(s.metadata.filename)}` : ""}
                    <p style="margin-top:0.4rem">${escapeHtml(s.text.slice(0, 300))}${s.text.length > 300 ? "..." : ""}</p>
                </div>`
            )
            .join("");
    } catch (err) {
        answerContent.textContent = `Error: ${err.message}`;
    } finally {
        queryBtn.disabled = false;
        queryBtn.textContent = "Ask";
    }
});

// ── Utils ───────────────────────────────────────────────────────────────────

function escapeHtml(str) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}

// ── Init ────────────────────────────────────────────────────────────────────

initTheme();
refreshDocuments();
