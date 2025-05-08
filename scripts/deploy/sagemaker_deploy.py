# สำหรับไฟล์ scripts/deploy/sagemaker_deploy.py

import boto3
import json
from sagemaker import get_execution_role

def deploy_model_to_sagemaker(model_path, config_path):
    """
    Deploy model to SageMaker using the IAM role with attached policy
    """
    # อ่านการตั้งค่า
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # ใช้ IAM role ที่มีนโยบายที่กำหนดไว้
    role = get_execution_role()  # โดยที่ role นี้ควรมีนโยบายที่คุณมีแนบไว้แล้ว
    
    # เริ่มกระบวนการ deploy
    sagemaker_session = boto3.Session()
    sm_client = sagemaker_session.client('sagemaker')
    
    # สร้าง SageMaker model
    model_name = config.get('model_name', 'finqa-model')
    
    # ส่วนต่อไปสำหรับการตั้งค่าโมเดลและ endpoint
    # ...
    
    return endpoint_name