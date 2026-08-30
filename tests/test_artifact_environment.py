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


def test_a_cpu_only_host_records_a_null_capability_rather_than_failing():
    cpu = dict(FAKE, cuda_available=False, cuda_version=None, gpu_name=None,
               gpu_compute_capability=None)
    with mock.patch("rosa_compute.diagnostics.get_environment_info",
                    return_value=cpu):
        env = get_artifact_environment()
    assert env["gpu_compute_capability"] is None
    json.dumps(env)


def test_the_real_helper_serializes_on_this_host():
    """No mock: whatever this machine actually reports must serialize."""
    json.dumps(get_artifact_environment())
