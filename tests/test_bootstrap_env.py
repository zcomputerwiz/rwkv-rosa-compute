import argparse

from scripts import bootstrap_env


def _args(*, cpu=False, torch_index=None):
    return argparse.Namespace(cpu=cpu, torch_index=torch_index)


def test_cuda_extra_follows_resolved_torch_source(monkeypatch):
    monkeypatch.setattr(bootstrap_env.platform, "system", lambda: "Windows")

    cpu_args = _args(cpu=True)
    cpu_index = bootstrap_env.resolve_torch_index(cpu_args)
    assert cpu_index == bootstrap_env.CPU_INDEX
    assert bootstrap_env.torch_install_uses_cuda(cpu_args, cpu_index) is False

    # --torch-index explicitly overrides --cpu, so a CUDA index must still
    # install the [cuda] extra (ninja) required by torch.cpp_extension.
    explicit_cuda = _args(
        cpu=True,
        torch_index="https://download.pytorch.org/whl/cu126",
    )
    cuda_index = bootstrap_env.resolve_torch_index(explicit_cuda)
    assert cuda_index.endswith("/cu126")
    assert bootstrap_env.torch_install_uses_cuda(explicit_cuda, cuda_index) is True

    explicit_cpu = _args(
        cpu=False,
        torch_index="https://download.pytorch.org/whl/cpu",
    )
    resolved_cpu = bootstrap_env.resolve_torch_index(explicit_cpu)
    assert bootstrap_env.torch_install_uses_cuda(explicit_cpu, resolved_cpu) is False


def test_default_platform_cuda_extra_selection(monkeypatch):
    args = _args()

    monkeypatch.setattr(bootstrap_env.platform, "system", lambda: "Linux")
    assert bootstrap_env.resolve_torch_index(args) is None
    assert bootstrap_env.torch_install_uses_cuda(args, None) is True

    monkeypatch.setattr(bootstrap_env.platform, "system", lambda: "Darwin")
    assert bootstrap_env.resolve_torch_index(args) is None
    assert bootstrap_env.torch_install_uses_cuda(args, None) is False
