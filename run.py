import modal
import subprocess

app = modal.App("bangla-training-app")
volume = modal.Volume.from_name("datasets-volume", create_if_missing=True)

image = (
    modal.Image.from_registry("tensorflow/tensorflow:2.16.1-gpu")
    # Pre-install blinker with ignore-installed to avoid distutils conflict
    .run_commands([
        "python -m pip install --ignore-installed blinker==1.9.0"
    ])
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("train.py", remote_path="/root/train.py")
    .add_local_file("config.py", remote_path="/root/config.py")
)

@app.function(
    image=image,
    volumes={"/root/datasets": volume},
    gpu="L40S",
    secrets=[modal.Secret.from_name("project-config")],
    timeout=86400 # 24 hour timeout for training
)
def run_training():
    print("Starting model training on Modal...")
    subprocess.run(["python", "/root/train.py"], check=True)

@app.local_entrypoint()
def main():
    run_training.remote()
