"""scripts.import_results only brings in experiments the Kaggle run finished, and never loses a local measurement."""

import json

import scripts.import_results as ir


def test_import_only_finished_steps_and_merge_latency(tmp_path, monkeypatch):
    local, run = tmp_path / "results", tmp_path / "out" / "mhrag_results"
    local.mkdir()
    (run / "_checkpoints").mkdir(parents=True)
    (run / "generation" / "local").mkdir(parents=True)
    monkeypatch.setattr(ir, "RESULTS", local)

    (local / "retrieval_main.json").write_text('{"v": "laptop"}')
    (run / "retrieval_main.json").write_text('{"v": "laptop"}')              # seeded copy, step switched off
    cfg = __import__("yaml").safe_load((ir.PROJECT_ROOT / "configs/experiments/retrieval_chunks.yaml").read_text())
    (run / "retrieval_chunks.json").write_text(json.dumps({"v": "kaggle", "config": cfg}))  # finished on Kaggle
    (run / "_checkpoints" / "retrieval_chunks.json").write_text("{}")
    (run / "_checkpoints" / "latency.json").write_text("{}")                  # latency step finished
    (run / "generation_local.json").write_text("{}")                          # unfinished: no marker
    (run / "generation" / "local" / "m__full__faq_gen.jsonl").write_text("{}\n")
    (run / "retrieval_chunks.partial.jsonl").write_text("x")                  # never imported
    (local / "latency.json").write_text(json.dumps({"cpu": {"summary": 1}, "cuda": {"status": "TODO(run)"}}))
    (run / "latency.json").write_text(json.dumps({"cpu": {"status": "TODO(run)"}, "cuda": {"summary": 2}}))

    rep = ir.import_results(tmp_path / "out")
    assert "retrieval_chunks.json" in rep["imported"]
    assert json.loads((local / "retrieval_chunks.json").read_text())["v"] == "kaggle"
    assert rep["not_finished"] == ["generation_local"]
    assert not (local / "generation_local.json").exists() and not (local / "generation").exists()
    assert not (local / "retrieval_chunks.partial.jsonl").exists()
    lat = json.loads((local / "latency.json").read_text())
    assert lat == {"cpu": {"summary": 1}, "cuda": {"summary": 2}}  # measured entries kept and added
    assert ir.import_results(tmp_path / "out")["imported"] == []  # idempotent


def test_stale_marker_or_incomplete_judge_is_not_imported(tmp_path, monkeypatch):
    local, run = tmp_path / "results", tmp_path / "out"
    local.mkdir()
    (run / "_checkpoints").mkdir(parents=True)
    monkeypatch.setattr(ir, "RESULTS", local)
    cfg = __import__("yaml").safe_load((ir.PROJECT_ROOT / "configs/experiments/generation_local.yaml").read_text())
    (run / "_checkpoints" / "generation_local.json").write_text("{}")  # marker from an earlier session

    (run / "generation_local.json").write_text(json.dumps({"config": {**cfg, "judge_sample": None}}))  # old config
    assert ir.import_results(run)["not_finished"] == ["generation_local"]
    (run / "generation_local.json").write_text(json.dumps({"config": cfg, "judge_status": "incomplete"}))
    assert ir.import_results(run)["not_finished"] == ["generation_local"]
    assert not (local / "generation_local.json").exists()
    (run / "generation_local.json").write_text(json.dumps({"config": cfg, "judge_status": "complete"}))
    assert "generation_local.json" in ir.import_results(run)["imported"]


def test_latency_from_an_unfinished_step_is_not_imported(tmp_path, monkeypatch):
    local, run = tmp_path / "results", tmp_path / "out"
    local.mkdir()
    (run / "_checkpoints").mkdir(parents=True)
    monkeypatch.setattr(ir, "RESULTS", local)
    (run / "latency.json").write_text(json.dumps({"v2_hf_cuda": {"summary": 1}}))
    assert ir.import_results(run)["imported"] == []  # no latency marker: the step failed or is unfinished
    (run / "_checkpoints" / "latency.json").write_text("{}")
    assert ir.import_results(run)["imported"] == ["latency.json: v2_hf_cuda"]


def test_marker_older_than_a_later_session_is_stale(tmp_path, monkeypatch):
    local, run = tmp_path / "results", tmp_path / "out"
    local.mkdir()
    (run / "_checkpoints").mkdir(parents=True)
    (run / "logs").mkdir()
    monkeypatch.setattr(ir, "RESULTS", local)
    (run / "latency.json").write_text(json.dumps({"v2_hf_cuda": {"summary": 1}}))
    (run / "_checkpoints" / "latency.json").write_text(json.dumps({"finished": "2026-09-26T00:55:30"}))
    (run / "logs" / "kaggle_latency.log").write_text("===== session started 2026-09-26 00:42\n")
    assert ir.import_results(run)["finished_steps"] == ["latency"]
    with open(run / "logs" / "kaggle_latency.log", "a") as f:
        f.write("===== session started 2026-09-26 14:11\nTraceback ...\n")  # ran again later and failed
    rep = ir.import_results(run)
    assert rep["finished_steps"] == [] and rep["imported"] == ["logs/kaggle_latency.log"]


def test_latency_rows_from_the_old_benchmark_are_not_imported(tmp_path, monkeypatch):
    local, run = tmp_path / "results", tmp_path / "out"
    local.mkdir()
    (run / "_checkpoints").mkdir(parents=True)
    monkeypatch.setattr(ir, "RESULTS", local)
    (run / "_checkpoints" / "latency.json").write_text("{}")
    (run / "latency.json").write_text(json.dumps({
        "v2_hf_cuda": {"summary": 1, "rows": [{"ttft_ms": 1, "rate_limit_wait_s": 0.0}]},  # current benchmark
        "v2_api": {"summary": 2, "rows": [{"ttft_ms": 11000}]},                           # old, throttled run
    }))
    assert ir.import_results(run)["imported"] == ["latency.json: v2_hf_cuda"]
