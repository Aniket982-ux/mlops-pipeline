import mlflow
import os
from mlflow.tracking import MlflowClient
from pathlib import Path
import shutil

def download_latest_model(model_name, destination_path):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    client = MlflowClient()

    try:
        latest_versions = client.get_latest_versions(model_name, stages=["Production", "Staging"])
        if not latest_versions:
            print(f"❌ No versions found for model '{model_name}' in Production/Staging")
            return

        latest = sorted(latest_versions, key=lambda x: int(x.version), reverse=True)[0]
        print(f"⬇️ Downloading {model_name} (version {latest.version})")

        local_path = mlflow.artifacts.download_artifacts(
            run_id=latest.run_id,
            artifact_path="model",  # Always correct for mlflow.log_model
        )

        found = False
        for root, _, files in os.walk(local_path):
            for file in files:
                if model_name == "RefinerModel" and file.endswith(".pth"):
                    shutil.copy(os.path.join(root, file), destination_path)
                    found = True
                elif model_name == "LGBMModel" and (file.endswith(".txt") or file.endswith(".lgb")):
                    shutil.copy(os.path.join(root, file), destination_path)
                    found = True

        if found:
            print(f"✅ Saved {model_name} to {destination_path}")
        else:
            print(f"⚠️ Could not find correct file for {model_name} in {local_path}")

    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")

if __name__ == "__main__":
    print(f"Using MLflow URI: {os.getenv('MLFLOW_TRACKING_URI')}")
    
    download_latest_model("RefinerModel", "embedding_refiner_checkpoint.pth")
    download_latest_model("LGBMModel", "trained_lgbm_model.txt")

    # Download HuggingFace models (needed for tests)
    from transformers import AutoModel, AutoTokenizer

    try:
        AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        AutoModel.from_pretrained('google/vit-base-patch16-224-in21k')
        AutoTokenizer.from_pretrained('google/vit-base-patch16-224-in21k')
        print("🧠 HuggingFace models cached successfully")
    except Exception as e:
        print(f"⚠️ Failed to download HuggingFace models: {e}")
