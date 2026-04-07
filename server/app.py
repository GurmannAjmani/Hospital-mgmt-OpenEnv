import os
import argparse
import uvicorn
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from openenv.core.env_server.http_server import create_app
try:
    from ..models import HospitalMgmtAction, HospitalMgmtObservation
    from .hospital_mgmt_env_environment import HospitalMgmtEnvironment
except (ModuleNotFoundError, ImportError):
    from models import HospitalMgmtAction, HospitalMgmtObservation
    from server.hospital_mgmt_env_environment import HospitalMgmtEnvironment
def create_hospital_environment():
    return HospitalMgmtEnvironment()
app = create_app(
    create_hospital_environment,
    HospitalMgmtAction,
    HospitalMgmtObservation,
    env_name="hospital_mgmt_env"
)
@app.get("/health")
def health_check():
    return {"status": "healthy"}
def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"--- Starting Hospital Mgmt Server on {host}:{port} ---")
    uvicorn.run(app, host=host, port=port)
if __name__ == "__main__":
    main()