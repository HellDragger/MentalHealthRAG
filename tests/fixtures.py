"""Tiny synthetic corpus that mirrors the real raw_data layout, built in a temp dir so tests run offline."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path


def _pdf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[list[str]]) -> bytes:
    """Minimal valid PDF with Helvetica text, one line per string. pdfium can extract it."""
    objs: list[bytes] = []
    n_pages = len(pages)
    # 1 catalog, 2 pages, 3 font, then (page, content) pairs
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n_pages))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    for i, lines in enumerate(pages):
        content_ref = 5 + 2 * i
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_ref} 0 R >>".encode()
        )
        ops = ["BT", "/F1 10 Tf", "14 TL", "40 760 Td"]
        for ln in lines:
            ops.append(f"({_pdf_escape(ln)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write(f"{n} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


ANGER_URL = "https://www.mind.org.uk/information-support/types-of-mental-health-problems/anger/about-anger/"
INTRO = "Explains anger, some possible causes and how it can make you feel and act. There is practical advice."


def web_print(title: str, url: str, body_pages: list[list[str]]) -> bytes:
    n = len(body_pages)
    pages = []
    for i, body in enumerate(body_pages):
        pages.append([f"4/16/24, 9:21 AM {title} - Mind", f"{url} {i + 1}/{n}", *body])
    pages[-1] += ["© 2024 Mind We're a registered charity in England (no. 219830) and a registered company"]
    return make_pdf(pages)


def build_fixture_raw_data(root: Path) -> Path:
    """Write tests' raw_data.zip and return its path."""
    pdf1 = web_print(
        "When is anger a problem?",
        ANGER_URL,
        [
            [
                "Anger",
                INTRO,
                "Mae'r dudalen hon hefyd ar gael yn Gymraeg. This link will take you to a Welsh",
                "translation of this page.",
                "What is anger?",
                "We all feel angry at times and it is part of being human. Anger is a normal, healthy emotion that",
                "everyone experiences. There are many different reasons why we might feel angry at other people.",
                "How can anger be helpful?",
                "Feeling angry can sometimes be useful. It can help us identify problems and protect us from things",
                "that are hurting us. It can also motivate us to push for changes in the world around us.",
            ]
        ],
    )
    pdf2 = web_print(
        "How to manage anger in the moment",
        ANGER_URL.replace("about-anger", "managing-anger"),
        [
            [
                "Anger",
                INTRO,
                "What can I do to manage my anger?",
                "Try to notice the signs that you are becoming angry. Counting to ten or slowing your breathing can",
                "help you feel calmer. You could go for a walk or go to a different room to give yourself some space.",
            ]
        ],
    )
    pdf3 = web_print(
        "Sleep problems",
        "https://www.mind.org.uk/information-support/types-of-mental-health-problems/sleep-problems/",
        [
            [
                "Sleep",
                INTRO,
                "How can I improve my sleep?",
                "Try to establish a routine by going to bed and getting up at roughly the same time every day. Avoid",
                "screens for an hour before bed and keep your bedroom dark, quiet and at a comfortable temperature.",
            ]
        ],
    )
    empty = make_pdf([[" "]])  # image-only stand-in: no text layer
    faq = (
        "Question_ID,Questions,Answers\n"
        '1,What is mental health?,"Mental health includes our emotional, psychological and social well-being."\n'
        '2,Where can I find a support group?,"Many people find peer support helpful. Ask your doctor or a local charity '
        'about support groups near you."\n'
    )
    kb = {
        "intents": [
            {"tag": "greeting", "patterns": ["Hi"], "responses": ["Hello there."]},
            {"tag": "scared", "patterns": ["I'm scared"], "response": "['It is natural to feel scared.', 'Take a breath.']"},
            {"tag": "fact-1", "patterns": ["What is therapy?", "Do i need therapy?"],
             "responses": ["Therapy is a form of treatment that aims to help resolve mental or emotional issues."]},
            {"tag": "fact-10", "patterns": ["What causes mental illness?"], "responses": ["Wrong copied answer."]},
        ]
    }
    counsel = 'Context,Response\n"I cannot sleep and feel worthless.","Talk to someone you trust about how you feel."\n'
    zpath = root / "raw_data.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("raw_data/PDF_Files/anger.pdf", pdf1)
        z.writestr("raw_data/PDF_Files/anger4.pdf", pdf2)
        z.writestr("raw_data/PDF_Files/sleep.pdf", pdf3)
        z.writestr("raw_data/PDF_Files/easy-read.pdf", empty)
        z.writestr("raw_data/CSV Files/Mental_Health_FAQ.csv", faq)
        z.writestr("raw_data/CSV Files/mentalhealth.csv", faq)
        z.writestr("raw_data/CSV Files/context_response_train.csv", counsel)
        z.writestr("raw_data/JSON Files/KB.json", json.dumps(kb))
    return zpath
