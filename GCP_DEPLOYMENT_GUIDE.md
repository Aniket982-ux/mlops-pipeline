# GCP CI/CD Deployment Guide

This guide explains how to set up a Continuous Deployment (CD) pipeline to a Google Cloud Platform (GCP) VM using GitHub Actions and `gcloud compute ssh`.

## Prerequisites

Before running the pipeline, you need to configure a GCP Service Account and GitHub Secrets.

### 1. Create a GCP Service Account

1.  Go to the [GCP Console IAM & Admin > Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts).
2.  Select your project (`buri-buri-zaimon`).
3.  Click **+ CREATE SERVICE ACCOUNT**.
4.  **Service account details**:
    *   Name: `github-actions-deployer` (or similar)
    *   Description: "Service account for GitHub Actions deployment"
    *   Click **CREATE AND CONTINUE**.
5.  **Grant this service account access to project**:
    *   Add the following roles:
        *   **Compute Instance Admin (v1)** (Allows SSH and VM management)
        *   **Service Account User** (Required to run operations as the service account)
        *   **IAP-secured Tunnel User** (If using IAP for SSH, otherwise optional but good practice)
    *   Click **CONTINUE** and then **DONE**.

### 2. Generate a JSON Key

1.  Click on the newly created service account email in the list.
2.  Go to the **KEYS** tab.
3.  Click **ADD KEY** > **Create new key**.
4.  Select **JSON** and click **CREATE**.
5.  A `.json` file will be downloaded to your computer. **Keep this file secure!**

### 3. Configure GitHub Secrets

1.  Go to your GitHub repository.
2.  Navigate to **Settings** > **Secrets and variables** > **Actions**.
3.  Click **New repository secret**.
4.  Add the following secrets:

| Secret Name | Value Description |
| :--- | :--- |
| `GCP_SA_KEY` | **Paste the entire content of the JSON key file you downloaded.** |
| `GCP_PROJECT_ID` | Your GCP Project ID (e.g., `buri-buri-zaimon`). |
| `GCP_ZONE` | The zone where your VM is located (e.g., `us-central1-b`). |
| `VM_NAME` | The name of your VM instance (e.g., `instance-20251129-202835`). |
| `VM_USER` | The username to log in as (usually your Google username or a specific user on the VM). |
| `MLFLOW_TRACKING_URI` | (Existing) Your MLflow tracking URI. |

> [!NOTE]
> The `VM_IP` secret is no longer strictly required if we use `gcloud compute ssh` with the instance name and zone, but you can keep it if you use it elsewhere.

## Workflow Explanation

The updated `cicd.yml` workflow performs the following steps for deployment:

1.  **Authenticate to Google Cloud**: Uses `google-github-actions/auth` with the `GCP_SA_KEY` to log in as the service account.
2.  **Set up Cloud SDK**: Uses `google-github-actions/setup-gcloud` to install and configure the `gcloud` CLI tool.
3.  **Deploy to VM**: Uses `gcloud compute ssh` to securely connect to your VM instance. It executes a script on the VM to:
    *   Navigate to the project directory.
    *   Pull the latest code from git.
    *   Set up/activate the virtual environment.
    *   Install dependencies.
    *   Run the model download script.
    *   Restart the application service.

## Troubleshooting

*   **Permission Denied**: Ensure the Service Account has the "Compute Instance Admin (v1)" role.
*   **SSH Connection Failed**: Check if the firewall rules allow SSH (port 22) or if IAP is required.
*   **Host Key Verification**: `gcloud compute ssh` handles host keys automatically, but if you see issues, you might need to clear old entries in `~/.ssh/known_hosts` on the runner (though the runner is ephemeral, so this is rarely an issue).
