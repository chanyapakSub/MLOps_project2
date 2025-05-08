# สำหรับไฟล์ mlops/huggingface_estimator.py

from sagemaker.huggingface import HuggingFace

def create_huggingface_estimator(role_arn, config):
    """
    สร้าง SageMaker HuggingFace estimator ด้วย IAM role ที่มีนโยบายที่เหมาะสม
    """
    # สร้าง estimator ด้วย IAM role ที่มีนโยบายที่คุณมี
    estimator = HuggingFace(
        entry_point='train.py',
        source_dir='scripts/train',
        role=role_arn,  # IAM role ที่มีนโยบายที่คุณมี
        instance_count=config['instance_count'],
        instance_type=config['instance_type'],
        transformers_version=config['transformers_version'],
        pytorch_version=config['pytorch_version'],
        py_version=config['py_version'],
    )
    
    return estimator