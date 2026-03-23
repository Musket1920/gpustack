from gpustack.schemas.inference_backend import (
    get_built_in_backend,
    is_built_in_backend,
)
from gpustack.schemas.models import BackendEnum


def test_get_built_in_backend_includes_llama_cpp():
    backend_names = {backend.backend_name for backend in get_built_in_backend()}

    assert BackendEnum.LLAMA_CPP.value in backend_names


def test_is_built_in_backend_recognizes_llama_cpp():
    assert is_built_in_backend(BackendEnum.LLAMA_CPP.value) is True
