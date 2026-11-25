"""
Demo 4 - Policy-based refund assistant / grounded AI (slides 25-26).
Policy PDFs -> Python retrieves the relevant sections (simple RAG, BM25 keyword
search - no vector DB needed) -> Ollama answers ONLY from those sections with
citations -> if nothing relevant is found, the fixed refusal is returned.
"""
import math
import re
from collections import Counter

import pymupdf

from . import config, llm, prompts

STOP = set("a an the of to for and or in on is are be was were what which who how should do does with by "
           "this that it as at from can will may must if not no any into our we they he she their there".split())


def _tok(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP and len(w) > 1]


def sections_from_pdf(name, source):
    """Split one policy PDF (path or bytes) into numbered sections - the citation unit."""
    doc = pymupdf.open(stream=source, filetype="pdf") if isinstance(source, (bytes, bytearray)) else pymupdf.open(str(source))
    text = "\n".join(page.get_text() for page in doc)
    lines = [l for l in text.splitlines() if l.strip()]
    title = (lines[0] if lines else name).replace("NovaBank - ", "").strip()
    chunks = []
    parts = re.split(r"\n(?=(?:Section\s+)?\d+(?:\.\d+)*\s*[-:.)]\s+\S)", "\n" + text)
    for part in parts[1:]:
        head, _, body = part.partition("\n")
        m = re.match(r"(?:Section\s+)?(\d+(?:\.\d+)*)\s*[-:.)]\s+(.+)", head.strip())
        if m:
            chunks.append({"policy": title, "section": m.group(1), "heading": m.group(2).strip(),
                           "text": " ".join(body.split()), "file": name})
    if not chunks:   # no numbered sections - split into ~120-word passages instead
        words = text.split()
        for i in range(0, len(words), 120):
            chunks.append({"policy": title, "section": f"p{i // 120 + 1}", "heading": "passage",
                           "text": " ".join(words[i:i + 120]), "file": name})
    return chunks


def load_sections(folder=None, files=None):
    """Sections from every policy PDF in a folder, or from uploaded files [(name, bytes), ...]."""
    if files:
        return [c for n, b in files for c in sections_from_pdf(n, b)]
    folder = folder or config.POLICIES
    return [c for pdf in sorted(folder.glob("*.pdf")) for c in sections_from_pdf(pdf.name, pdf)]


class PolicyIndex:
    """BM25 ranking over section heading + text."""

    def __init__(self, chunks=None, k1=1.4, b=0.75):
        self.chunks = chunks or load_sections()
        self.docs = [_tok(c["heading"] + " " + c["heading"] + " " + c["text"]) for c in self.chunks]
        self.avgdl = sum(map(len, self.docs)) / len(self.docs)
        self.df = Counter(w for d in self.docs for w in set(d))
        self.N, self.k1, self.b = len(self.docs), k1, b

    def search(self, query, k=None):
        k = k or config.RAG_TOP_K
        q = _tok(query)
        scored = []
        for c, d in zip(self.chunks, self.docs):
            tf = Counter(d)
            s = 0.0
            for w in q:
                if w in tf:
                    idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    s += idf * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            scored.append((round(s, 2), c))
        scored.sort(key=lambda x: -x[0])
        return [dict(c, score=s) for s, c in scored[:k] if s > 0]


def answer_prompt(question, index=None):
    """(prompt or None if the policy has no relevant section, hits)."""
    index = index or PolicyIndex()
    hits = index.search(question)
    if not hits or hits[0]["score"] < config.RAG_MIN_SCORE:
        return None, hits
    context = "\n\n".join(f"[{h['policy']} - Section {h['section']}: {h['heading']}]\n{h['text']}" for h in hits)
    return prompts.POLICY_ANSWER.format(question=question, context=context, refusal=config.REFUSAL), hits


def answer(question, index=None):
    """Returns dict(answer, sources, grounded, fallback)."""
    index = index or PolicyIndex()
    hits = index.search(question)
    if not hits or hits[0]["score"] < config.RAG_MIN_SCORE:
        return {"answer": config.REFUSAL, "sources": hits, "grounded": False, "fallback": False}
    context = "\n\n".join(f"[{h['policy']} - Section {h['section']}: {h['heading']}]\n{h['text']}" for h in hits)
    txt = llm.ask(prompts.POLICY_ANSWER.format(question=question, context=context, refusal=config.REFUSAL),
                  system=prompts.SYSTEM_BANK)
    if txt:
        return {"answer": txt, "sources": hits, "grounded": config.REFUSAL not in txt, "fallback": False}
    good = [h for h in hits if h["score"] >= config.RAG_MIN_SCORE]
    lines = ["Relevant policy extracts (Ollama offline - showing retrieved text verbatim):", ""]
    for h in good:
        lines += [f"[{h['policy']} - Section {h['section']}: {h['heading']}]", h["text"], ""]
    lines += ["Source References:"] + [f"- {h['policy']} - Section {h['section']} ({h['heading']})" for h in good]
    return {"answer": "\n".join(lines), "sources": hits, "grounded": True, "fallback": True}
