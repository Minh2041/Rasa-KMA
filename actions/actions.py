
import os
import re
import json
import logging
import unicodedata

import requests
from rasa_sdk import Action

logger = logging.getLogger(__name__)

RAG_URL = os.getenv(
    "KMA_RAG_URL",
    "http://127.0.0.1:8001/ask"
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434"
)

OLLAMA_MODEL = os.getenv(
    "KMA_CHAT_MODEL",
    "qwen2.5:3b"
)

MIN_INTENT = 0.82
MIN_SELECTOR = 0.65
MIN_SELECTOR_MARGIN = 0.08


def normalize(text):
    text = str(text).casefold().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def is_bot_identity(question):
    q = normalize(question)

    patterns = [
        r"^ban la ai$",
        r"^chatbot(?: kma)? la ai$",
        r"^tro ly(?: ai)? la ai$",
        r"^ban co phai (?:la )?chatbot",
        r"^ban la nguoi hay",
        r"^minh dang noi chuyen voi ai$",
        r"^ai dang tra loi minh$",
    ]

    return any(re.search(p, q) for p in patterns)


def is_person_question(question):
    q = normalize(question)

    if is_bot_identity(question):
        return False

    if re.search(
        r"(?:thong tin|tim hieu|gioi thieu)"
        r"(?: ve)? nguoi ten ",
        q
    ):
        return True

    if re.search(
        r"\b(?:dong chi|ong|ba|anh|chi|lanh dao) "
        r".+ (?:giu chuc vu gi|lam gi|cong tac o dau)$",
        q
    ):
        return True

    # Loai tru cau hoi ve chuc danh, to chuc.
    roles = [
        "giam doc", "pho giam doc", "bi thu",
        "pho bi thu", "truong phong",
        "hoc vien", "hieu truong",
        "chu nhiem khoa", "truong khoa", "uy vien"
    ]

    if q.endswith(" la ai") and not q.startswith("ai la "):
        if not any(role in q for role in roles):
            return True

    return False


def ollama_json(system_prompt, user_prompt):
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0
                }
            },
            timeout=45
        )
        response.raise_for_status()

        text = response.json()["message"]["content"]
        return json.loads(text)

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError
    ) as exc:
        logger.warning("Ollama verification failed: %s", exc)
        return None


def get_selected_faq(tracker):
    message = tracker.latest_message or {}
    intent = message.get("intent") or {}

    parent = intent.get("name", "")
    parent_conf = float(intent.get("confidence") or 0)

    selectors = message.get("response_selector") or {}

    selected = (
        selectors.get(parent)
        or selectors.get("default")
        or {}
    )

    response = selected.get("response") or {}
    key = response.get("intent_response_key", "")

    if not key.startswith(parent + "/"):
        return None

    selector_conf = float(
        response.get("confidence") or 0
    )

    ranking = selected.get("ranking") or []
    margin = 1.0

    if len(ranking) >= 2:
        margin = (
            float(ranking[0].get("confidence") or 0)
            - float(ranking[1].get("confidence") or 0)
        )

    return {
        "key": key,
        "utter": response.get("utter_action")
                 or "utter_" + key,
        "parent_conf": parent_conf,
        "selector_conf": selector_conf,
        "margin": margin
    }


def faq_is_valid(question, selected, domain):
    if not selected:
        return False

    key = selected["key"]
    utter = selected["utter"]

    if (
        selected["parent_conf"] < MIN_INTENT
        or selected["selector_conf"] < MIN_SELECTOR
        or selected["margin"] < MIN_SELECTOR_MARGIN
    ):
        return False

    responses = domain.get("responses", {})
    templates = responses.get(utter) or []

    if not templates:
        return False

    answer = templates[0].get("text", "")

    if not answer:
        return False

    # Cau hoi ve mot nguoi bat ky khong duoc
    # chap nhan nham FAQ ve chatbot hoac BCH.
    if is_person_question(question):
        return False

    if key.endswith("/bot_challenge"):
        return is_bot_identity(question)

    # Cac cau giao tiep ro rang co the xu ly nhanh.
    q = normalize(question)

    fast = {
        "faq_giaotiep/greet": {
            "chao", "xin chao", "hello", "hi", "hey"
        },
        "faq_giaotiep/thanks": {
            "cam on", "cam on ban", "thank you", "thanks"
        },
        "faq_giaotiep/goodbye": {
            "tam biet", "bye", "goodbye", "hen gap lai"
        }
    }

    if key in fast and q in fast[key]:
        return True

    # Bao ve FAQ khoi truong hop hoi sai doi tuong,
    # sai truong, sai chu de nhung confidence cao.
    system_prompt = """
Ban la bo kiem tra FAQ cho chatbot KMA.

Chi tra ve JSON:
{"match": true}
hoac:
{"match": false}

match=true CHI KHI:
- Cau tra loi giai quyet dung noi dung cau hoi.
- Dung doi tuong, to chuc va pham vi duoc hoi.
- Khong danh dong thong tin truong khac voi KMA.
- Khong danh dong nguoi bat ky voi chatbot.
- Khong suy doan tu vai tu giong nhau.

Neu chua chac chan, tra match=false.
Khong tu tra loi cau hoi.
"""

    result = ollama_json(
        system_prompt,
        json.dumps(
            {
                "question": question,
                "faq_key": key,
                "faq_answer": answer
            },
            ensure_ascii=False
        )
    )

    return (
        isinstance(result, dict)
        and result.get("match") is True
    )


def call_rag(dispatcher, question):
    try:
        response = requests.post(
            RAG_URL,
            json={"question": question},
            timeout=120
        )
        response.raise_for_status()

        data = response.json()
        answer = data.get("answer")
        answered = data.get("answered")
        sources = data.get("sources", [])

        logger.warning("RAG answer: %s", answer)
        logger.warning("RAG sources: %s", sources)

        if answered is False:
            dispatcher.utter_message(
                text="Tôi chưa tìm thấy thông tin chính xác "
                "trong tài liệu hiện có."
            )
            return

        if isinstance(answer, str) and answer.strip():
            dispatcher.utter_message(text=answer.strip())
            return

    except (
        requests.RequestException,
        ValueError,
        TypeError
    ) as exc:
        logger.exception("KMA RAG error: %s", exc)

    dispatcher.utter_message(
        response="utter_default"
    )


class ActionKmaFaqGate(Action):

    def name(self):
        return "action_kma_faq_gate"

    def run(self, dispatcher, tracker, domain):
        question = (
            tracker.latest_message.get("text") or ""
        ).strip()

        selected = get_selected_faq(tracker)

        if faq_is_valid(question, selected, domain):
            dispatcher.utter_message(
                response=selected["utter"]
            )
        else:
            call_rag(dispatcher, question)

        return []


class ActionKmaRag(Action):

    def name(self):
        return "action_kma_rag"

    def run(self, dispatcher, tracker, domain):
        question = (
            tracker.latest_message.get("text") or ""
        ).strip()

        call_rag(dispatcher, question)
        return []
