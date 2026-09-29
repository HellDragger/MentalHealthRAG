"""Static checks on the frontend (CHANGELOG B15-B17, B34)."""

import re

from mhrag.config import PROJECT_ROOT

WEB = PROJECT_ROOT / "web"


def test_no_unsanitised_innerhtml():
    js = (WEB / "app.js").read_text()
    assigns = re.findall(r"\.innerHTML\s*=\s*([^;]+);", js)
    # only the sanitised markdown and a static typing indicator may use innerHTML
    assert all(a.strip() in ("clean", '"<span></span><span></span><span></span>"') for a in assigns), assigns
    assert "DOMPurify.sanitize" in js
    assert "textContent = text" in js  # user text is never parsed as HTML


def test_ui_has_disclaimer_crisis_banner_and_labels():
    html = (WEB / "index.html").read_text()
    assert "Not a substitute for professional care" in html
    assert 'id="crisis"' in html and 'role="alert"' in html
    assert 'for="msg"' in html and 'aria-label="Language model"' in html and 'role="log"' in html
    assert "<script>" not in html  # no inline scripts (CSP script-src 'self')


def test_enter_to_send_and_error_handling_present():
    js = (WEB / "app.js").read_text()
    assert 'e.key === "Enter"' in js and "AbortController" in js and "Retry" in js
    assert "sendBtn.disabled = busy" in js
