import argparse
import json
import sys

import pytest

from scripts.generate_data import get_dataset_output_dir, main

pytestmark = pytest.mark.exp0


def _args(tmp_path, *, num_samples=4, vocab_reduction=True):
    return argparse.Namespace(
        length=6,
        dimension=2,
        mod=10,
        num_filler=4,
        num_samples=num_samples,
        seed=123,
        vocab_reduction=vocab_reduction,
        out_dir=str(tmp_path),
    )


def test_generated_data_identity_covers_sample_count_and_vocab_mode(tmp_path):
    reduced = get_dataset_output_dir(
        _args(tmp_path, num_samples=4, vocab_reduction=True)
    )
    full_vocab = get_dataset_output_dir(
        _args(tmp_path, num_samples=4, vocab_reduction=False)
    )
    larger = get_dataset_output_dir(
        _args(tmp_path, num_samples=8, vocab_reduction=True)
    )

    assert reduced != full_vocab
    assert reduced != larger
    assert full_vocab != larger


def test_generate_data_rejects_unsupported_mod_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_data.py",
            "--mod",
            "7",
            "--out_dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert list(tmp_path.iterdir()) == []


def _run_generation(tmp_path, monkeypatch, num_samples):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_data.py",
            "--length",
            "6",
            "--dimension",
            "2",
            "--num_filler",
            "4",
            "--num_samples",
            str(num_samples),
            "--seed",
            "123",
            "--out_dir",
            str(tmp_path),
        ],
    )
    main()

    output_dir = get_dataset_output_dir(_args(tmp_path, num_samples=num_samples))
    with open(output_dir / "dataset.json", encoding="utf-8") as f:
        return json.load(f)


def test_formatting_is_stable_when_dataset_size_changes(tmp_path, monkeypatch):
    samples_4 = _run_generation(tmp_path / "small", monkeypatch, 4)
    samples_6 = _run_generation(tmp_path / "large", monkeypatch, 6)

    assert samples_6[:4] == samples_4
