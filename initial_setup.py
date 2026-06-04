import modal

app = modal.App("initial-setup")

volume = modal.Volume.from_name("datasets-volume", create_if_missing=True)

image = (
    modal.Image.from_registry("tensorflow/tensorflow:2.16.1-gpu")
    .pip_install_from_requirements("requirements.txt")
    .add_local_file(
        "BanglaLekha-Isolated.zip",
        remote_path="/tmp/BanglaLekha-Isolated.zip",
    )
    .add_local_file(
        "labels.json",
        remote_path="/tmp/labels.json",
    )
)


@app.function(
    image=image,
    volumes={"/root/datasets": volume},
)
def upload_and_unzip():
    import os
    import zipfile
    import shutil

    zip_path = "/tmp/BanglaLekha-Isolated.zip"
    labels_src = "/tmp/labels.json"

    extract_path = "/root/datasets"

    print("📦 Extracting dataset zip...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    print("✅ Dataset extraction complete!")

    # Move labels.json into the volume
    labels_dst = os.path.join(extract_path, "labels.json")
    shutil.copy2(labels_src, labels_dst)

    print("🏷️ labels.json uploaded!")

    # Verify contents
    total_files = 0
    for root, dirs, files in os.walk(extract_path):
        total_files += len(files)
        print(root, "->", len(files), "files")

    print(f"📊 Total files in volume: {total_files}")

    # Commit changes
    volume.commit()
    print("💾 Volume commit successful!")


@app.local_entrypoint()
def main():
    upload_and_unzip.remote()