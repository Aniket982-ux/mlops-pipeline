import mlflow
import os
from mlflow.tracking import MlflowClient
import shutil

# Stable HF cache for CI
os.environ["HF_HOME"] = "./.hf_cache"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
client = MlflowClient()

def download_latest_model(model_name, destination_path):

    try:
        # Get most recent version from registry (any stage)
        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            print(f"❌ No versions found for '{model_name}'")
            return

        latest = max(versions, key=lambda v: int(v.version))
        print(f"⬇️ Downloading {model_name} (version {latest.version})")

        # ✅ CRITICAL FIX:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=latest.source
        )

        os.makedirs(os.path.dirname(destination_path) or ".", exist_ok=True)

        found = False
        for root, _, files in os.walk(local_path):
            for f in files:
                src = os.path.join(root, f)

                if model_name == "RefinerModel" and f.endswith(".pth"):
                    shutil.copyfile(src, destination_path)
                    found = True

                elif model_name == "LGBMModel" and (f.endswith(".txt") or f.endswith(".lgb")):
                    shutil.copyfile(src, destination_path)
                    found = True

        if found:
            print(f"✅ Saved {model_name} to {destination_path}")
        else:
            print(f"⚠️ Model files not found inside downloaded artifact structure!")

    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")


if __name__ == "__main__":
    print("Using MLflow URI:", mlflow.get_tracking_uri())

    download_latest_model("RefinerModel", "embedding_refiner_checkpoint.pth")
    download_latest_model("LGBMModel", "trained_lgbm_model.txt")

    from transformers import AutoModel, AutoTokenizer

    hf_models = [
        "sentence-transformers/all-mpnet-base-v2",
        "google/vit-base-patch16-224-in21k"
    ]

    for m in hf_models:
        try:
            print(f"⬇️ Downloading HF model: {m}")
            AutoModel.from_pretrained(m)
            AutoTokenizer.from_pretrained(m)
            print(f"🧠 {m} cached successfully")
        except Exception as e:
            print(f"⚠️ Failed to download {m}: {e}")
