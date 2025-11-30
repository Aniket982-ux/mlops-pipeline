import mlflow
import os
from mlflow.tracking import MlflowClient

def download_latest_model(model_name, destination_path):
    client = MlflowClient()
    try:
        # Get the latest version of the model
        latest_version_info = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
        if not latest_version_info:
            print(f"❌ No versions found for model '{model_name}'")
            return
        
        # Sort by version number descending to get the absolute latest
        latest_version_info.sort(key=lambda x: int(x.version), reverse=True)
        latest_version = latest_version_info[0]
        
        print(f"⬇️  Downloading '{model_name}' version {latest_version.version}...")
        
        # Download the artifact
        # The artifact path inside the run is usually "model" based on our training scripts
        local_path = mlflow.artifacts.download_artifacts(
            run_id=latest_version.run_id,
            artifact_path="model",
            dst_path="."
        )
        
        # Move/Rename the downloaded file to the expected destination
        # mlflow downloads a directory. We need to find the actual file inside.
        # For PyTorch, it's usually 'data/model.pth' or similar inside the artifact dir, 
        # BUT our training script logged the model object directly.
        # Let's check how mlflow.pytorch.log_model saves it. 
        # Usually it saves a 'data' dir and 'MLmodel' file.
        # However, for simplicity in this specific project context where we want a single file:
        
        # Actually, mlflow.pytorch.log_model saves a directory structure. 
        # But our inference.py expects a raw .pth file for the refiner 
        # and a text file for LGBM.
        
        # Wait, our training script used:
        # mlflow.pytorch.log_model(model, "model", ...)
        # This creates a folder named "model" with "MLmodel", "conda.yaml", "data/model.pth" etc.
        
        # We need to extract the actual model file.
        
        if model_name == "RefinerModel":
            # PyTorch model
            # The actual state dict is likely in data/model.pth or similar.
            # Let's look at the downloaded structure.
            # download_artifacts returns the local path to the downloaded directory.
            
            # Common structure for mlflow.pytorch:
            # model/
            #   MLmodel
            #   conda.yaml
            #   data/
            #     model.pth  <-- This is what we usually want, OR pickle.
            
            # BUT, our inference.py uses `torch.load(path)`.
            # If we load the MLflow wrapper, we should use `mlflow.pytorch.load_model`.
            # However, to keep inference.py independent of MLflow (as per current design),
            # we should extract the state_dict file.
            
            # Let's try to find the .pth file in the downloaded directory.
            found = False
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    if file.endswith(".pth"):
                        source = os.path.join(root, file)
                        # Rename/Move
                        if os.path.exists(destination_path):
                            os.remove(destination_path)
                        os.rename(source, destination_path)
                        print(f"✅ Saved {model_name} to {destination_path}")
                        found = True
                        break
                if found: break
            
            if not found:
                # Fallback: maybe it's a pickle?
                print(f"⚠️ Could not find .pth file for {model_name} in {local_path}")

        elif model_name == "LGBMModel":
            # LightGBM model
            # mlflow.lightgbm.log_model saves it.
            # Structure:
            # model/
            #   MLmodel
            #   model.lgb (or similar)
            
            found = False
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    # LightGBM usually saves as model.lgb or just 'model'
                    if file == "model.lgb" or file.endswith(".txt"):
                        source = os.path.join(root, file)
                        if os.path.exists(destination_path):
                            os.remove(destination_path)
                        os.rename(source, destination_path)
                        print(f"✅ Saved {model_name} to {destination_path}")
                        found = True
                        break
                if found: break
            
            if not found:
                 print(f"⚠️ Could not find model file for {model_name} in {local_path}")

    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")

if __name__ == "__main__":
    # Ensure URI is set (should be passed via env var in CI/CD, or set here for local test)
    # mlflow.set_tracking_uri("http://34.61.12.226:5000") 
    # We assume MLFLOW_TRACKING_URI is set in environment
    
    print(f"Using MLflow URI: {mlflow.get_tracking_uri()}")
    
    download_latest_model("RefinerModel", "embedding_refiner_checkpoint.pth")
    download_latest_model("LGBMModel", "trained_lgbm_model.txt")
