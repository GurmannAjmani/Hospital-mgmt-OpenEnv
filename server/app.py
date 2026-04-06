import os
import argparse
import uvicorn
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from openenv.core.env_server.http_server import create_app

# 1. HANDLE IMPORTS
# This block ensures the server finds your environment files 
# whether you run it locally or inside the Docker container.
try:
    from ..models import HospitalMgmtAction, HospitalMgmtObservation
    from .hospital_mgmt_env_environment import HospitalMgmtEnvironment
except (ModuleNotFoundError, ImportError):
    from models import HospitalMgmtAction, HospitalMgmtObservation
    from server.hospital_mgmt_env_environment import HospitalMgmtEnvironment

# 2. FACTORY FUNCTION
# OpenEnv requires a function that returns a fresh environment instance
# for every new session (like the working OpenSpiel example).
def create_hospital_environment():
    """Factory function that creates the environment instance."""
    return HospitalMgmtEnvironment()

# 3. CREATE THE FASTAPI APP
# This creates the /reset, /step, and /state endpoints automatically.
app = create_app(
    create_hospital_environment,
    HospitalMgmtAction,
    HospitalMgmtObservation,
    env_name="hospital_mgmt_env"
)

# 4. HEALTH CHECK ENDPOINT
# Required for Hugging Face to know your container is "Live".
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# 5. VALIDATOR-FRIENDLY MAIN FUNCTION
# We use a parameterless main() so the openenv scanner can call it directly.
def main():
    """
    Entry point for the server. 
    Reads the port from environment variables (standard for HF/Docker).
    """
    import uvicorn
    # Hugging Face usually provides the port via the PORT env var
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"--- Starting Hospital Mgmt Server on {host}:{port} ---")
    uvicorn.run(app, host=host, port=port)

# 6. STANDARD EXECUTION GUARD
# This is the exact pattern 'openenv validate' looks for.
if __name__ == "__main__":
    main()