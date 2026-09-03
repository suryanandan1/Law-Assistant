import json
import os
import re
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader

from loaders import load_pdfs
from splitter import split_documents
from embeddings_store import create_vectorstore
from qa_chain import build_qa_chain
from employee_data import update_employee_pdf_heading, remove_employee_pdf_heading

BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "media"
USER_PDF_DIR = MEDIA_DIR / "user_pdfs"
REGISTRY_NAME = "documents.json"


def _safe_user_id(employee_id):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(employee_id).strip())


def get_user_folder(employee_id):
    folder = USER_PDF_DIR / _safe_user_id(employee_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _registry_path(employee_id):
    return get_user_folder(employee_id) / REGISTRY_NAME


def load_registry(employee_id):
    path = _registry_path(employee_id)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_registry(employee_id, documents):
    path = _registry_path(employee_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)


def _clean_title(text):
    text = " ".join(str(text or "").replace("\x00", " ").split())
    text = re.sub(r"[^\w\s:/,&().\-]", "", text).strip(" -_|")
    if len(text) > 80:
        text = text[:80].rsplit(" ", 1)[0]
    return text


def extract_pdf_title(pdf_path):
    """Extract a readable assistant title from metadata or the first page."""
    try:
        reader = PdfReader(str(pdf_path))
        metadata_title = None
        if reader.metadata:
            metadata_title = reader.metadata.get("/Title") or reader.metadata.title
        title = _clean_title(metadata_title)
        if title and title.lower() not in {"untitled", "document"}:
            return title

        if reader.pages:
            text = reader.pages[0].extract_text() or ""
            lines = [_clean_title(line) for line in text.splitlines()]
            lines = [line for line in lines if len(line) >= 4]
            for line in lines[:12]:
                lowered = line.lower()
                if lowered not in {"page 1", "confidential"}:
                    return line
    except Exception:
        pass

    return Path(pdf_path).stem.replace("_", " ").replace("-", " ").title()




def extract_pdf_short_detail(pdf_path, max_pages=2):
    """Create a short readable detail/summary from the first pages of a PDF."""
    try:
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        collected = []
        for page in reader.pages[:max_pages]:
            text = page.extract_text() or ""
            for raw_line in text.splitlines():
                line = _clean_title(raw_line)
                if 12 <= len(line) <= 140 and not re.fullmatch(r"[\d\s./-]+", line):
                    lowered = line.lower()
                    if lowered not in {"page 1", "confidential", "table of contents"}:
                        collected.append(line)
                if len(collected) >= 3:
                    break
            if len(collected) >= 3:
                break

        detail_text = " ".join(collected[:3]).strip()
        if len(detail_text) > 220:
            detail_text = detail_text[:220].rsplit(" ", 1)[0] + "..."
        if not detail_text:
            detail_text = "PDF uploaded and ready for question answering."
        return {
            "page_count": page_count,
            "short_detail": detail_text,
        }
    except Exception:
        return {
            "page_count": 0,
            "short_detail": "PDF uploaded and ready for question answering.",
        }


def save_uploaded_pdf(employee_id, uploaded_file):
    if not uploaded_file.name.lower().endswith(".pdf"):
        return False, "Only PDF files are allowed."

    folder = get_user_folder(employee_id)
    original_name = Path(uploaded_file.name).name
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", original_name)
    destination = folder / safe_name

    counter = 1
    while destination.exists():
        destination = folder / f"{destination.stem}_{counter}{destination.suffix}"
        counter += 1

    with open(destination, "wb+") as out:
        for chunk in uploaded_file.chunks():
            out.write(chunk)

    title = extract_pdf_title(destination)
    pdf_info = extract_pdf_short_detail(destination)
    documents = load_registry(employee_id)
    documents.append({
        "filename": destination.name,
        "title": title,
        "path": str(destination),
        "page_count": pdf_info.get("page_count", 0),
        "short_detail": pdf_info.get("short_detail", ""),
    })
    save_registry(employee_id, documents)
    update_employee_pdf_heading(employee_id, title)
    clear_user_chain_cache()
    return True, f"PDF uploaded successfully: {title}"


def delete_user_pdf(employee_id, filename):
    """Delete one PDF for a user and remove its heading from employees.xlsx."""
    filename = Path(str(filename or "")).name
    if not filename:
        return False, "PDF filename is missing."

    documents = load_registry(employee_id)
    matched_doc = None
    remaining_docs = []

    for doc in documents:
        doc_filename = Path(str(doc.get("filename", ""))).name
        if doc_filename == filename and matched_doc is None:
            matched_doc = doc
        else:
            remaining_docs.append(doc)

    if matched_doc is None:
        return False, "PDF not found."

    pdf_path = Path(matched_doc.get("path", ""))
    user_folder = get_user_folder(employee_id).resolve()

    try:
        # Safety check: delete only files inside this user's PDF folder.
        resolved_pdf_path = pdf_path.resolve()
        if user_folder not in resolved_pdf_path.parents:
            return False, "Invalid PDF path."

        if resolved_pdf_path.exists():
            resolved_pdf_path.unlink()

        save_registry(employee_id, remaining_docs)
        remove_employee_pdf_heading(employee_id, matched_doc.get("title", ""))
        clear_user_chain_cache()
        return True, "PDF deleted successfully."

    except Exception as exc:
        return False, f"Unable to delete PDF: {exc}"


def get_user_documents(employee_id):
    docs = []
    for item in load_registry(employee_id):
        path = Path(item.get("path", ""))
        if path.exists() and path.suffix.lower() == ".pdf":
            info = extract_pdf_short_detail(path) if not item.get("short_detail") else {}
            docs.append({
                "filename": item.get("filename", path.name),
                "title": item.get("title") or extract_pdf_title(path),
                "path": str(path),
                "page_count": item.get("page_count") or info.get("page_count", 0),
                "short_detail": item.get("short_detail") or info.get("short_detail", "PDF uploaded and ready for question answering."),
            })
    return docs



QUESTION_KEYWORDS = [
    "leave", "policy", "rule", "eligibility", "entitlement", "allowance",
    "approval", "document", "documents", "claim", "reimbursement", "apply",
    "process", "procedure", "limit", "duration", "timeline", "deadline",
    "encashment", "sick", "casual", "privilege", "travel", "expense",
]


def _extract_pdf_lines(pdf_path, max_pages=8):
    """Extract clean, useful lines from a PDF for automatic question suggestions."""
    lines = []
    try:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages[:max_pages]:
            text = page.extract_text() or ""
            for raw_line in text.splitlines():
                line = _clean_title(raw_line)
                line = re.sub(r"^\d+(\.\d+)*\s*", "", line).strip(" :-")
                if 8 <= len(line) <= 90 and len(line.split()) >= 2:
                    lowered = line.lower()
                    if lowered in {"page", "page 1", "confidential", "table of contents"}:
                        continue
                    if not re.fullmatch(r"[\d\s./-]+", line):
                        lines.append(line)
    except Exception:
        pass
    return lines


def _line_score(line):
    lowered = line.lower()
    score = 0
    for keyword in QUESTION_KEYWORDS:
        if keyword in lowered:
            score += 4
    if any(ch.isdigit() for ch in line):
        score += 1
    if line.istitle() or sum(1 for word in line.split() if word[:1].isupper()) >= 2:
        score += 2
    if 20 <= len(line) <= 70:
        score += 2
    return score


def _topic_to_question(topic):
    topic = re.sub(r"\s+", " ", str(topic or "")).strip(" :-.")
    topic = re.sub(r"^(section|chapter|part)\s+\w+\s*", "", topic, flags=re.I).strip(" :-.")
    if not topic:
        return "What are the key points in this PDF?"

    lowered = topic.lower()
    if any(word in lowered for word in ["eligibility", "eligible"]):
        return f"Who is eligible for {topic}?"
    if any(word in lowered for word in ["approval", "approve"]):
        return f"What approval is required for {topic}?"
    if any(word in lowered for word in ["document", "claim", "reimbursement"]):
        return f"What documents or process are required for {topic}?"
    if any(word in lowered for word in ["limit", "duration", "timeline", "deadline", "days"]):
        return f"What are the limits or timelines for {topic}?"
    return f"What does the PDF say about {topic}?"


def get_pdf_question_suggestions(employee_id):
    """Return 4 automatic question suggestions generated from the user's uploaded PDFs."""
    documents = get_user_documents(employee_id)
    if not documents:
        return []

    candidates = []
    seen_topics = set()

    for doc in documents:
        # Use extracted title as a strong candidate.
        title = doc.get("title", "")
        if title:
            candidates.append((8, title))

        for line in _extract_pdf_lines(doc.get("path", "")):
            key = re.sub(r"[^a-z0-9]+", " ", line.lower()).strip()
            if not key or key in seen_topics:
                continue
            seen_topics.add(key)
            score = _line_score(line)
            if score > 0:
                candidates.append((score, line))

    candidates.sort(key=lambda item: item[0], reverse=True)

    questions = []
    seen_questions = set()
    for _score, topic in candidates:
        question = _topic_to_question(topic)
        normalized = question.lower()
        if normalized not in seen_questions:
            questions.append(question)
            seen_questions.add(normalized)
        if len(questions) == 4:
            break

    fallback_title = documents[-1].get("title", "this PDF")
    fallbacks = [
        f"What are the key points in {fallback_title}?",
        f"What rules or conditions are mentioned in {fallback_title}?",
        f"What is the process explained in {fallback_title}?",
        f"What important limits or timelines are given in {fallback_title}?",
    ]
    for question in fallbacks:
        if len(questions) == 4:
            break
        if question.lower() not in seen_questions:
            questions.append(question)
            seen_questions.add(question.lower())

    return questions[:4]



def get_user_pdf_summary(employee_id):
    """Small sidebar details for any uploaded PDF, replacing fixed PL/CL/SL boxes."""
    documents = get_user_documents(employee_id)
    if not documents:
        return {
            "total_pdfs": 0,
            "total_pages": 0,
            "latest_heading": "No PDF uploaded",
            "latest_detail": "Upload a PDF to see a short detail here.",
        }

    latest = documents[-1]
    total_pages = 0
    for doc in documents:
        try:
            total_pages += int(doc.get("page_count") or 0)
        except Exception:
            pass

    return {
        "total_pdfs": len(documents),
        "total_pages": total_pages,
        "latest_heading": latest.get("title", "Uploaded PDF"),
        "latest_detail": latest.get("short_detail", "PDF uploaded and ready for question answering."),
    }

def get_assistant_title(employee_id):
    documents = get_user_documents(employee_id)
    if not documents:
        return "Upload PDF Assistant"
    if len(documents) == 1:
        return documents[0]["title"]
    return f"{documents[-1]['title']} + {len(documents) - 1} more PDF"


def _signature(employee_id):
    documents = get_user_documents(employee_id)
    return tuple((doc["path"], os.path.getmtime(doc["path"])) for doc in documents)


@lru_cache(maxsize=32)
def _build_user_chain_cached(employee_id, signature):
    pdf_paths = [path for path, _mtime in signature]
    if not pdf_paths:
        return None
    docs = load_pdfs(pdf_paths)
    split_docs = split_documents(docs)
    vectorstore = create_vectorstore(split_docs)
    return build_qa_chain(vectorstore)


def get_user_chain(employee_id):
    sig = _signature(employee_id)
    if not sig:
        return None
    return _build_user_chain_cached(_safe_user_id(employee_id), sig)


def clear_user_chain_cache():
    _build_user_chain_cached.cache_clear()
