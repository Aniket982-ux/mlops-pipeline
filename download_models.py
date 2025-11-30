import mlflow
import os
from mlflow.tracking import MlflowClient
import shutil

# Ensure stable HF cache in CI
os.environ["HF_HOME"] = "./.hf_cache"

# Set MLflow URI once globally
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
client = MlflowClient()

def download_latest_model(model_name, destination_path):
    try:
        # Include all stages so CI doesn't break
        latest_versions = client.get_latest_versions(
            model_name,
            stages=["None", "Staging", "Production"]
        )

        if not latest_versions:
            print(f"❌ No versions found for model '{model_name}'")
            return

        latest = sorted(latest_versions, key=lambda x: int(x.version), reverse=True)[0]
        print(f"⬇️ Downloading {model_name} (version {latest.version})")

        local_path = mlflow.artifacts.download_artifacts(
            run_id=latest.run_id,
            artifact_path="model"
        )

        os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)

        found = False
        for root, _, files in os.walk(local_path):
            for file in files:
                src = os.path.join(root, file)

                if model_name == "RefinerModel" and file.endswith(".pth"):
                    shutil.copyfile(src, destination_path)
                    found = True

                elif model_name == "LGBMModel" and (file.endswith(".txt") or file.endswith(".lgb")):
                    shutil.copyfile(src, destination_path)
                    found = True

        if found:
            print(f"✅ Saved {model_name} to {destination_path}")
        else:
            print(f"⚠️ Could not find correct file for {model_name}")

    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")


if __name__ == "__main__":
    print(f"Using MLflow URI: {mlflow.get_tracking_uri()}")

    download_latest_model("RefinerModel", "embedding_refiner_checkpoint.pth")
    download_latest_model("LGBMModel", "trained_lgbm_model.txt")

    # Download HuggingFace models
    from transformers import AutoModel, AutoTokenizer

    models = [
        "sentence-transformers/all-mpnet-base-v2",
        "google/vit-base-patch16-224-in21k"
    ]

    for model_name in models:
        try:
            print(f"⬇️ Downloading HuggingFace model: {model_name}")
            AutoModel.from_pretrained(model_name)
            AutoTokenizer.from_pretrained(model_name)
            print(f"🧠 {model_name} cached successfully")
        except Exception as e:
            print(f"⚠️ Failed to download {model_name}: {e}")
