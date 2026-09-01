"""The environment block embedded in artifacts must survive json.dumps intact.

Two ways this has already gone wrong, one on each side:

  Merging `get_environment_info()` straight into an artifact raises
  `TypeError: Object of type BuildCapabilities is not JSON serializable`.

  Filtering it to scalars instead serializes cleanly and silently drops
  `gpu_compute_capability`, which is a tuple -- and that is the one field that
  separates sm_75 from sm_86 from sm_89. The artifact then looks complete and
  cannot answer the question it was added to answer.

So the test asserts both properties together: it serializes, and the compute
capability is still there.
"""
import json
import pathlib
from unittest import mock

from rosa_compute.diagnostics import get_artifact_environment


class _Unserializable:
    """Stands in for rosa_soft's BuildCapabilities."""


FAKE = {
    "python_version": "3.13.1",
    "torch_version": "2.6.0+cu124",
    "cuda_available": True,
    "cuda_version": "12.4",
    "gpu_name": "NVIDIA GeForce RTX 4060 Ti",
    "gpu_compute_capability": (8, 9),
    "rosa_soft_imported": True,
    "rosa_soft_version": "0.1.0",
    "rosa_soft_build_capabilities": _Unserializable(),
    "RWKV-LM_commit": "ec56ea2b",
    "rosa_soft_commit": "5cb78987",
}


def _fake_env():
    with mock.patch("rosa_compute.diagnostics.get_environment_info",
                    return_value=dict(FAKE)):
        return get_artifact_environment()


def test_the_environment_block_is_json_serializable():
    json.dumps(_fake_env())


def test_the_compute_capability_survives_as_a_list():
    env = _fake_env()
    assert env["gpu_compute_capability"] == [8, 9], (
        "the field that distinguishes sm_75/sm_86/sm_89 was dropped or left as "
        "a tuple")
    assert json.loads(json.dumps(env))["gpu_compute_capability"] == [8, 9]


def test_the_unserializable_field_is_omitted():
    assert "rosa_soft_build_capabilities" not in _fake_env()


def test_the_kernel_source_commits_are_recorded():
    env = _fake_env()
    assert env["RWKV-LM_commit"] == "ec56ea2b"
    assert env["rosa_soft_commit"] == "5cb78987"


FLOAT32_SETTINGS = (
    "float32_matmul_precision",
    "cuda_matmul_allow_tf32",
    "cudnn_allow_tf32",
)


def test_the_float32_execution_settings_are_recorded():
    """`precision: "fp32"` in a config does not establish FP32 arithmetic.

    TF32 exists on sm_80 and later and not on sm_75, so the same nominal
    precision can mean different arithmetic on different nodes. Three nodes ran
    one Qwen4 cell whose artifacts recorded only the config string, and whether
    TF32 was off could not be established afterwards from the artifacts.
    """
    env = _fake_env()
    for field in FLOAT32_SETTINGS:
        assert field in env, f"{field} is missing from the artifact environment"
    assert isinstance(env["float32_matmul_precision"], (str, type(None)))
    assert isinstance(env["cuda_matmul_allow_tf32"], (bool, type(None)))
    assert isinstance(env["cudnn_allow_tf32"], (bool, type(None)))
    assert json.loads(json.dumps(env))["float32_matmul_precision"] == \
        env["float32_matmul_precision"]


def test_an_unreadable_setting_costs_the_field_and_not_the_artifact():
    """A torch that cannot report a flag must not stop an artifact being written."""
    from rosa_compute.diagnostics import _torch_setting

    def explode():
        raise RuntimeError("this build does not expose that flag")

    assert _torch_setting(explode) is None


def test_a_cpu_only_host_records_a_null_capability_rather_than_failing():
    cpu = dict(FAKE, cuda_available=False, cuda_version=None, gpu_name=None,
               gpu_compute_capability=None)
    with mock.patch("rosa_compute.diagnostics.get_environment_info",
                    return_value=cpu):
        env = get_artifact_environment()
    assert env["gpu_compute_capability"] is None
    for field in FLOAT32_SETTINGS:
        assert field in env, (
            f"{field} must still be recorded on a host without CUDA; the "
            "settings are process state, not device state")
    json.dumps(env)


def test_the_real_helper_serializes_on_this_host():
    """No mock: whatever this machine actually reports must serialize."""
    json.dumps(get_artifact_environment())


def test_repo_provenance_records_a_full_sha_and_a_tree_hash():
    """Eight characters is readable and not auditable.

    A fleet artifact recorded `ab973d8c`, the tip of a branch that was
    squash-merged and deleted; the SHA stopped resolving from a clone. The tree
    hash survives a squash merge, so recording both lets a reviewer confirm two
    differently-named commits built identical source.
    """
    from rosa_compute.diagnostics import get_repo_provenance

    prov = get_repo_provenance(str(pathlib.Path(__file__).resolve().parents[1]))
    assert set(prov) == {"commit", "tree", "dirty"}
    for field in ("commit", "tree"):
        assert len(prov[field]) == 40, f"{field} is not a full SHA: {prov[field]}"
        assert all(c in "0123456789abcdef" for c in prov[field])
    assert isinstance(prov["dirty"], bool)
    json.dumps(prov)


def test_repo_provenance_degrades_instead_of_raising_outside_a_checkout(tmp_path):
    """An analysis tool still has to write its results outside a checkout."""
    from rosa_compute.diagnostics import get_repo_provenance

    prov = get_repo_provenance(str(tmp_path))
    assert prov["commit"] in ("unknown",) or len(prov["commit"]) == 40
    json.dumps(prov)
