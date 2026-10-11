import json

from scripts.generation_reanalysis import runtime_comparison


def test_runtime_comparison_pairs_questions_and_counts_discordant(tmp_path):
    rows = []
    for i in range(10):  # llama.cpp F16 declines all ten, transformers F16 only the first four
        rows.append({"model": "qwen2.5-1.5b-gguf-f16", "id": i, "dataset": "oos_questions", "profile": "full",
                     "generated": True, "metrics": {"abstained": True}})
        rows.append({"model": "qwen2.5-1.5b-instruct-f16", "id": i, "dataset": "oos_questions", "profile": "full",
                     "generated": True, "metrics": {"abstained": i < 4}})
    rows.append({"model": "qwen2.5-1.5b-instruct-f16", "id": 99, "dataset": "oos_questions", "profile": "full",
                 "generated": True, "metrics": {"abstained": True}})  # unpaired: left out
    path = tmp_path / "scored.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    out = runtime_comparison(path)
    c = out["qwen2.5-1.5b-gguf-f16>qwen2.5-1.5b-instruct-f16"]
    assert (c["n"], c["only_a"], c["only_b"], c["declined_b"]) == (10, 6, 0, 4)
    assert abs(c["p_mcnemar"] - 2 * 0.5 ** 6) < 1e-12
    assert len(out) == 1  # contrasts with a missing runtime are skipped
