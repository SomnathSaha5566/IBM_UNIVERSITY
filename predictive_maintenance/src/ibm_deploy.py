"""Deploy the trained model to IBM Watson Machine Learning (IBM Cloud Lite).

Setup (one time):
 1. Create an IBM Cloud Lite account and provision 'Watson Machine Learning' (Lite plan).
 2. Create a Watson Studio project + deployment space; copy the Space ID.
 3. Create an API key (IBM Cloud -> Manage -> Access (IAM) -> API keys).
 4. export IBM_API_KEY=...  IBM_SPACE_ID=...  IBM_URL=https://us-south.ml.cloud.ibm.com
 5. python src/ibm_deploy.py

WML runtimes support specific scikit-learn versions (e.g. 1.3). If the upload
complains about a version mismatch, retrain with that version (see README).
"""
import os
import sys

import joblib
import pandas as pd
from ibm_watsonx_ai import APIClient, Credentials

sys.path.insert(0, os.path.dirname(__file__))
from features import RAW_FEATURES, add_features  # noqa: E402

creds = Credentials(url=os.environ.get("IBM_URL", "https://us-south.ml.cloud.ibm.com"),
                    api_key=os.environ["IBM_API_KEY"])
client = APIClient(creds)
client.set.default_space(os.environ["IBM_SPACE_ID"])

bundle = joblib.load("models/model.joblib")
model = bundle["model"]

sw_spec_id = client.software_specifications.get_id_by_name("runtime-24.1-py3.11")
meta = {
    client.repository.ModelMetaNames.NAME: "predictive-maintenance-failure-type",
    client.repository.ModelMetaNames.TYPE: "scikit-learn_1.3",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
}
details = client.repository.store_model(model=model, meta_props=meta)
model_id = client.repository.get_model_id(details)

dep = client.deployments.create(model_id, meta_props={
    client.deployments.ConfigurationMetaNames.NAME: "pm-failure-type-online",
    client.deployments.ConfigurationMetaNames.ONLINE: {},
})
dep_id = client.deployments.get_id(dep)
print("Deployment ID:", dep_id)

# Smoke test with one engineered row
row = add_features(pd.DataFrame([{
    "Type": "L", "Air temperature [K]": 298.5, "Process temperature [K]": 308.6,
    "Rotational speed [rpm]": 1400, "Torque [Nm]": 45, "Tool wear [min]": 210,
}])[RAW_FEATURES])[bundle["features"]]
payload = {"input_data": [{"fields": list(row.columns), "values": row.values.tolist()}]}
print(client.deployments.score(dep_id, payload))
