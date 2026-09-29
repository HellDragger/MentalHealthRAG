"""Download the models the CPU server needs (used at Docker build time so the Space starts quickly).

    python -m scripts.prefetch_models
"""

from __future__ import annotations

from mhrag.config import get_settings
from mhrag.llm.registry import load_catalog


def main():
    s = get_settings()
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    emb = s.embedder_cfg()
    TextEmbedding(model_name=emb.model)
    rr = s.rerankers[s.retrieval.reranker]
    TextCrossEncoder(model_name=rr.fastembed_model or rr.model)
    print("fastembed models ready:", emb.model, rr.fastembed_model)

    from huggingface_hub import hf_hub_download

    cat = load_catalog()
    # only the automatic fallback model is baked into the image; other GGUF models download on first use
    for key in s.llm.auto_preference:
        spec = cat.get(key)
        if spec and spec.backend == "llamacpp":
            path = hf_hub_download(spec.raw["gguf_repo"], spec.raw["gguf_file"])
            print("GGUF ready:", path)
    try:
        from tokenizers import Tokenizer

        Tokenizer.from_pretrained(emb.model)
    except Exception:
        pass


if __name__ == "__main__":
    main()
