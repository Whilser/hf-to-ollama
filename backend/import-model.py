import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import signal
import base64
from urllib.error import HTTPError
from urllib.request import Request, urlopen


SCRIPTS = Path(__file__).resolve().parent
CACHE = Path.home() / "Library" / "Caches" / "HF to Ollama"
API = "http://127.0.0.1:11434"


def command(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def download_weights(download, shards, weights, total, environment):
    process = subprocess.Popen([*download, *shards], env={**environment, "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    samples = []
    try:
        while True:
            downloaded = 0
            for shard in shards:
                path = weights / shard
                if path.is_file():
                    downloaded += path.stat().st_size
                    continue
                key = base64.urlsafe_b64encode(hashlib.sha1((Path(shard).name + ".metadata").encode()).digest()).decode()
                candidates = list((weights / ".cache/huggingface/download").glob(key + ".*.incomplete"))
                if candidates:
                    latest = max(candidates, key=lambda item: item.stat().st_mtime)
                    downloaded += latest.stat().st_size
            now = time.monotonic()
            samples.append((now, downloaded))
            samples = [item for item in samples if now - item[0] <= 20]
            elapsed = now - samples[0][0]
            speed = max(0, (downloaded - samples[0][1]) / elapsed) if elapsed > 0 else 0
            print("@@PROGRESS " + json.dumps({"done": downloaded, "total": total, "speed": speed}), flush=True)
            result = process.poll()
            if result is not None:
                if result:
                    raise subprocess.CalledProcessError(result, process.args)
                return
            time.sleep(1)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()


def api(path, payload=None, timeout=10):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(API + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        raise RuntimeError(error.read().decode(errors="replace")) from error


def repository(value):
    value = value.removeprefix("https://huggingface.co/").strip("/")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", value, flags=re.ASCII):
        raise ValueError("Укажи ссылку на репозиторий Hugging Face или имя вида автор/модель.")
    if any(part in (".", "..") for part in value.split("/")):
        raise ValueError("Недопустимое имя репозитория.")
    return value


def validate_architecture(config, version):
    architectures = config.get("architectures", [])
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    known_version = match and tuple(map(int, match.groups())) <= (0, 40, 16)
    if known_version and ("GptOssForCausalLM" in architectures or config.get("model_type") == "gpt_oss"):
        raise RuntimeError(
            f"Ollama {version} не поддерживает GPT-OSS через MLX-импорт. "
            "Для запуска в Ollama доступна команда: ollama run gpt-oss:20b. "
            "Загрузка весов остановлена после проверки конфигурации."
        )


def main():
    parser = argparse.ArgumentParser(description="Импорт Safetensors из Hugging Face в Ollama MLX")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo")
    source.add_argument("--local", type=Path)
    parser.add_argument("--name", required=True, help="Имя для новой модели в Ollama")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Требуется Mac с Apple Silicon.")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*(?::[a-z0-9][a-z0-9._-]*)?", args.name):
        raise ValueError("Для имени Ollama используй строчные латинские буквы, цифры, точки, дефисы и подчёркивания.")
    os.environ["OLLAMA_HOST"] = API
    version = api("/api/version")["version"]
    print(f"Ollama {version}", flush=True)
    name = args.name if ":" in args.name else args.name + ":latest"
    if any(model["name"] == name for model in api("/api/tags")["models"]):
        raise RuntimeError(f"Имя {name} уже занято. Выбери новое имя.")
    if args.repo:
        repo = repository(args.repo)
        key = hashlib.sha256(f"{repo}@{args.revision}".encode()).hexdigest()[:20]
        weights = CACHE / key
        print(f"Скачиваю {repo} ({args.revision}) в {weights}", flush=True)
        download = ["hf", "download", repo, "--revision", args.revision, "--local-dir", str(weights)]
        environment = {**os.environ, "HF_HUB_DISABLE_XET": os.environ.get("HF_HUB_DISABLE_XET", "1")}
        print("Проверяю архитектуру модели…", flush=True)
        command(*download, "config.json", env=environment)
        validate_architecture(json.loads((weights / "config.json").read_text()), version)
        command(*download, "--include", "*.json", "--include", "*.jinja",
                "--include", "*.model", "--include", "*.txt", "--exclude", "*/*", env=environment)
        index_file = weights / "model.safetensors.index.json"
        if index_file.is_file():
            index = json.loads(index_file.read_text())
            shards = sorted(set(index["weight_map"].values()))
            if any(Path(shard).is_absolute() or ".." in Path(shard).parts for shard in shards):
                raise ValueError("Недопустимые пути в индексе весов.")
            print(f"Загружаю файлы весов: {len(shards)}", flush=True)
            download_weights(download, shards, weights, index.get("metadata", {}).get("total_size", 0), environment)
        else:
            command(*download, "--include", "*.safetensors", "--exclude", "*/*", env=environment)
    else:
        weights = args.local.expanduser().resolve()
    config = json.loads((weights / "config.json").read_text())
    validate_architecture(config, version)
    if not list(weights.glob("*.safetensors")):
        raise RuntimeError("Папка должна содержать файлы Safetensors.")
    split_experts = False
    for shard in weights.glob("*.safetensors"):
        with shard.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if header_size > 100_000_000:
                raise ValueError(f"Слишком большой заголовок Safetensors: {shard.name}")
            names = json.loads(stream.read(header_size))
        split_experts |= any(".experts.switch_glu." in key for key in names)
    quant = config.get("quantization") or config.get("quantization_config")
    if quant is not None and not isinstance(quant, dict):
        raise RuntimeError("Не удалось определить формат квантования.")
    mixed = {key: value for key, value in (quant or {}).items() if isinstance(value, dict)}
    with tempfile.TemporaryDirectory(prefix="hf-ollama-") as temporary:
        work = Path(temporary)
        input_dir = work / "weights"
        input_dir.symlink_to(weights, target_is_directory=True)
        if split_experts:
            supported = (
                config.get("architectures") == ["Gemma4ForConditionalGeneration"]
                and quant and (quant.get("mode"), quant.get("bits"), quant.get("group_size")) == ("mxfp4", 4, 32)
                and (weights / "model.safetensors.index.json").is_file()
                and all(isinstance(v, dict) and v.get("bits") == 8 and v.get("group_size") == 64 and v.get("mode", "affine") == "affine"
                        and re.search(r"\.(?:mlp\.(?:gate|up|down)_proj|router\.proj)$", k) for k, v in mixed.items())
            )
            if not supported:
                raise RuntimeError("Эта раскладка экспертов пока не поддерживается. Проверенный профиль: Gemma 4 26B MXFP4.")
            if mixed and not shutil.which("uv"):
                raise RuntimeError("Для преобразования смешанной разрядности установи uv.")
            print("Объединяю веса экспертов для Ollama.", flush=True)
            command(sys.executable, str(SCRIPTS / "repack.py"), str(work))
            input_dir = work / "ollama-weights"
            if mixed:
                print("Преобразую 8-битные слои MLP в MXFP4. Веса маршрутизатора распаковываю без дополнительного квантования.", flush=True)
                command("uv", "run", "--no-project", "--python", "3.13", "--with", "mlx",
                        "python", str(SCRIPTS / "normalize.py"), str(work))
                input_dir = work / "normalized-weights"
        if quant and quant.get("bits") == 5 and quant.get("mode", "affine") == "affine":
            if quant.get("group_size") != 32 or mixed:
                raise RuntimeError("Преобразование 5-битных весов поддерживает affine group_size=32 без смешанной разрядности.")
            print("Преобразую 5-битные веса в совместимую 8-битную копию. Размер увеличится.", flush=True)
            converted = work / "affine8"
            command("uv", "run", "--no-project", "--python", "3.13", "--with", "mlx",
                    "python", str(SCRIPTS / "convert-affine5.py"), str(input_dir), str(converted))
            input_dir = converted
        modelfile = work / "Modelfile"
        modelfile.write_text(f"FROM ./{input_dir.name}\n")
        create = ["ollama", "create", name, "-f", str(modelfile)]
        if not quant:
            create.extend(["--quantize", "mxfp4"])
        print(f"Импортирую {name}.", flush=True)
        command(*create)
        try:
            result = api("/api/chat", {
                "model": name, "messages": [{"role": "user", "content": "What is 2 + 2? Reply with just the number."}],
                "think": False, "stream": False, "keep_alive": 0,
                "options": {"num_ctx": 4096, "num_predict": 32},
            }, timeout=180)
            answer = result.get("message", {}).get("content", "").strip()
            if not result.get("done") or answer.rstrip(".") != "4":
                raise RuntimeError(f"Неожиданный ответ: {answer!r}")
        except (RuntimeError, OSError) as error:
            raise RuntimeError(f"Импорт завершился, но проверка запуска не прошла. Запись {name} осталась в Ollama. {error}") from error
    print(f"Готово. Проверочный ответ: {answer}\nЗапуск: ollama run {name}")


if __name__ == "__main__":
    try:
        if os.getpgrp() != os.getpid():
            os.setsid()
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(130))
        main()
    except (OSError, ValueError, RuntimeError, KeyError, struct.error, subprocess.CalledProcessError) as error:
        sys.exit(f"Ошибка: {error}")
