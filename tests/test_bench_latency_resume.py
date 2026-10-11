import json

from scripts import bench_latency


def test_skip_measured_leaves_existing_measurement(tmp_path, monkeypatch):
    out = tmp_path / "latency_cpu.json"
    out.write_text(json.dumps({"v2_llamacpp_cpu": {"summary": {"n": 12}}}))
    monkeypatch.setattr(bench_latency, "run_one", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    assert bench_latency.main(["--config", "v2_llamacpp_cpu", "--out", str(out), "--skip-measured"]) == 0
    assert json.loads(out.read_text()) == {"v2_llamacpp_cpu": {"summary": {"n": 12}}}


def test_skip_measured_runs_missing_configuration(tmp_path, monkeypatch):
    out = tmp_path / "latency_cpu.json"
    out.write_text(json.dumps({"v2_llamacpp_cpu": {"summary": {"n": 12}}}))
    monkeypatch.setattr(bench_latency, "run_one", lambda name, n, model: {"config": name, "summary": {"n": 1}})
    assert bench_latency.main(["--config", "v2_hf_fp32_cpu", "--out", str(out), "--skip-measured"]) == 0
    assert set(json.loads(out.read_text())) == {"v2_llamacpp_cpu", "v2_hf_fp32_cpu"}


def test_import_keeps_local_only_keys(tmp_path):
    from scripts.import_results import _json_subset

    cur, new = tmp_path / "cur.json", tmp_path / "new.json"
    cur.write_text(json.dumps({"summary": [1], "abstention_rescored": {"answers_changed": 86}}))
    new.write_text(json.dumps({"summary": [1]}))
    assert _json_subset(new, cur)
    new.write_text(json.dumps({"summary": [2]}))  # a real change is imported
    assert not _json_subset(new, cur)
