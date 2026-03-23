import yaml

from gpustack.routes.inference_backend import _process_version_configs
from gpustack.utils.compat_importlib import pkg_resources


def test_part2_community_backends_have_direct_process_contracts():
    yaml_file = pkg_resources.files("gpustack.assets").joinpath(
        "community-inference-backends.yaml"
    )
    yaml_data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    expected = {
        "Kokoro-FastAPI": {
            "latest-gpu": "python -m uvicorn api.src.main:app --host 0.0.0.0 --port {{port}} --log-level debug",
            "latest-cpu": "python -m uvicorn api.src.main:app --host 0.0.0.0 --port {{port}} --log-level debug",
        },
        "MinerU": {
            "v2.7.0": "mineru-vllm-server --port {{port}} --served-model-name {{model_name}}",
        },
        "PaddleX-GenAI-Server": {
            "latest-nvidia-gpu": "paddleocr genai_server --model_name {{MODEL_NAME}} --model_dir {{model_path}} --host 0.0.0.0 --port {{port}} --backend vllm",
            "latest-nvidia-gpu-sm120": "paddleocr genai_server --model_name {{MODEL_NAME}} --model_dir {{model_path}} --host 0.0.0.0 --port {{port}} --backend vllm",
            "latest-huawei-npu": "paddleocr genai_server --model_name {{MODEL_NAME}} --model_dir {{model_path}} --host 0.0.0.0 --port {{port}} --backend vllm",
            "latest-hygon-dcu": "paddleocr genai_server --model_name {{MODEL_NAME}} --model_dir {{model_path}} --host 0.0.0.0 --port {{port}} --backend vllm",
            "latest-iluvatar-gpu": "paddleocr genai_server --model_name {{model_name}} --model_dir {{model_path}} --host 0.0.0.0 --port {{port}} --backend fastdeploy",
            "latest-metax-gpu": "paddleocr genai_server --model_name {{model_name}} --model_dir {{model_path}} --host 0.0.0.0 --port {{port}} --backend fastdeploy",
        },
        "Text-Embeddings-Inference": {
            "cpu-1.8": "text-embeddings-router --model-id {{model_path}} --port {{port}}",
            "cuda-sm89-1.8": "text-embeddings-router --model-id {{model_path}} --port {{port}}",
            "cuda-sm86-1.8": "text-embeddings-router --model-id {{model_path}} --port {{port}}",
            "cuda-sm80-1.8": "text-embeddings-router --model-id {{model_path}} --port {{port}}",
        },
    }

    backends = {item["backend_name"]: item for item in yaml_data}

    for backend_name, expected_versions in expected.items():
        version_configs = backends[backend_name]["version_configs"]
        parsed = _process_version_configs(version_configs)

        assert set(parsed.root.keys()) >= set(expected_versions.keys())

        for version_name, expected_command in expected_versions.items():
            version_config = parsed.root[version_name]
            contract = version_config.direct_process_contract
            assert contract is not None
            assert contract.command_template == expected_command
            assert contract.startup_timeout_seconds == 120
            assert contract.stop_signal == "SIGTERM"
            assert contract.stop_timeout_seconds == 30
