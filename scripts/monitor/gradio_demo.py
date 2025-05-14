# import gradio as gr
# import boto3
# import json

# # ⚙️ สร้าง SageMaker Runtime Client
# runtime = boto3.client("sagemaker-runtime", region_name="ap-southeast-2")

# # ✅ ตั้งชื่อ endpoint ที่คุณสร้างไว้ (ปรับให้ตรงกับของคุณ)
# ENDPOINT_NAME = "Financial-endpoint"

# # 🧠 ฟังก์ชันเรียก API ไปยัง SageMaker Endpoint
# def answer_from_endpoint(question, context):
#     payload = {
#         "question": question,
#         "context": context
#     }

#     response = runtime.invoke_endpoint(
#         EndpointName=ENDPOINT_NAME,
#         ContentType="application/json",
#         Body=json.dumps(payload)
#     )

#     result = json.loads(response["Body"].read())
#     return result.get("answer", "[No answer found]")

# # 🎨 Gradio UI
# gr.Interface(
#     fn=answer_from_endpoint,
#     inputs=[
#         gr.Textbox(lines=2, label="Question"),
#         gr.Textbox(lines=5, label="Context")
#     ],
#     outputs="text",
#     title="Ask My SageMaker Model",
#     description="Enter a question and a context. The model will try to answer it!"
# ).launch()



import gradio as gr
import boto3
import json
import time
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError

# --- CONFIG ---
ENDPOINT_NAME = "Financial-endpoint"
REGION = "ap-southeast-2"
MAX_RETRY = 3  # จะลองเช็กสถานะ Endpoint ซ้ำสูงสุดกี่ครั้ง (ถ้ายังไม่ InService)

# --- INIT AWS CLIENTS ---
sm_client = boto3.client("sagemaker", region_name=REGION)
runtime_client = boto3.client("sagemaker-runtime", region_name=REGION)

# --- CHECK ENDPOINT STATUS ---
def is_endpoint_ready():
    try:
        for i in range(MAX_RETRY):
            resp = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
            status = resp["EndpointStatus"]

            if status == "InService":
                return True
            elif status == "Failed":
                return False
            else:
                time.sleep(5)
        return False
    except ClientError as e:
        print(f"❌ ClientError: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

# --- MAIN INFERENCE FUNCTION ---
def answer_from_endpoint(question, context):
    if not is_endpoint_ready():
        return "❌ Endpoint ยังไม่พร้อมใช้งาน กรุณาลองใหม่ภายหลัง"

    payload = {
        "question": question,
        "context": context
    }

    try:
        response = runtime_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        result = json.loads(response["Body"].read())
        return result.get("answer", "[No answer found]")
    except ClientError as e:
        return f"❌ ClientError: {e}"
    except Exception as e:
        return f"❌ Error: {e}"

# --- GRADIO UI ---
gr.Interface(
    fn=answer_from_endpoint,
    inputs=[
        gr.Textbox(lines=2, label="Question"),
        gr.Textbox(lines=5, label="Context")
    ],
    outputs="text",
    title="🧠 Ask My SageMaker Model",
    description="ใส่คำถามและเนื้อหา แล้วให้โมเดลตอบ"
).launch()
