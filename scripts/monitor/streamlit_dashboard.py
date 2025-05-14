import streamlit as st
import json
import boto3
from datetime import datetime

st.title("MLOps Model Performance Dashboard")

BUCKET = "modelmlopss"
KEY = "registry/registry.json"

# Load registry from S3
s3 = boto3.client("s3")
content = s3.get_object(Bucket=BUCKET, Key=KEY)
data = json.loads(content["Body"].read().decode())

# Compare current vs history
current = data["current"]
history = data["history"][-5:][::-1]  # show latest 5

st.subheader("Current Deployed Model")
st.write(current)

st.subheader("History")
for h in history:
    st.markdown(f"**{h['version']}** — acc: `{h['accuracy']}` — endpoint: `{h['endpoint']}`")

# Optional: show chart
import pandas as pd
chart_data = pd.DataFrame([h for h in data["history"]] + [current])
chart_data['timestamp'] = pd.to_datetime(chart_data['timestamp'])
chart_data = chart_data.sort_values("timestamp")
st.line_chart(chart_data.set_index("timestamp")["accuracy"])