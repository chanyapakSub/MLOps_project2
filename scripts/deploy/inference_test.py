import boto3
import json
import time
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError

# ----- CONFIG -----
ENDPOINT_NAME = "Financial-endpoint"  # 🔧 เปลี่ยนชื่อ endpoint ของคุณ
REGION = "ap-southeast-2"
INPUT_TEXT = "ช่วยสรุปข่าววันนี้ให้หน่อย"

MAX_RETRY = 3  # หากยังไม่ InService จะรอซ้ำสูงสุดกี่ครั้ง


def check_endpoint_exists(sm_client):
    try:
        response = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        return response["EndpointStatus"]
    except sm_client.exceptions.ClientError as e:
        if "Could not find" in str(e):
            print(f"Endpoint '{ENDPOINT_NAME}' ไม่พบใน SageMaker")
        else:
            print(f"ClientError: {e}")
        return None


def wait_for_endpoint_inservice(sm_client):
    print(f"ตรวจสอบสถานะ Endpoint: '{ENDPOINT_NAME}'")
    for i in range(MAX_RETRY):
        status = check_endpoint_exists(sm_client)
        if status == "InService":
            print("Endpoint พร้อมใช้งานแล้ว")
            return True
        elif status == "Failed":
            print("Endpoint สร้างไม่สำเร็จ")
            return False
        else:
            print(f"⌛ Status = {status}, รอ 10 วิ...")
            time.sleep(10)
    print("Endpoint ยังไม่พร้อมใช้งานหลังรอครบ")
    return False


def run_inference(runtime_client):
    payload = {
        "inputs": INPUT_TEXT,
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.7
        }
    }

    try:
        print("📤 ส่งคำขอเข้า Endpoint...")
        response = runtime_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        result = json.loads(response["Body"].read())
        print("\nผลลัพธ์จากโมเดล:\n", json.dumps(result, indent=2, ensure_ascii=False))

    except ClientError as e:
        print(f"ClientError ระหว่างเรียก Endpoint: {e}")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดไม่คาดคิด: {e}")


def main():
    try:
        sm_client = boto3.client("sagemaker", region_name=REGION)
        runtime_client = boto3.client("sagemaker-runtime", region_name=REGION)

        if wait_for_endpoint_inservice(sm_client):
            run_inference(runtime_client)

    except NoCredentialsError:
        print("ไม่พบ AWS credentials — โปรดตรวจสอบ `aws configure` หรือ GitHub Secrets")
    except EndpointConnectionError:
        print(f"เชื่อมต่อกับ region '{REGION}' ไม่ได้ — โปรดตรวจสอบอินเทอร์เน็ตหรือ AWS region")
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")


if __name__ == "__main__":
    main()
