import os
import tarfile
import boto3

# ----- CONFIG -----
MODEL_DIR = "models/hybrid_model"
TAR_PATH = "model.tar.gz"
S3_BUCKET = "modelmlops"
S3_KEY = "Model/model.tar.gz"

# ----- 1. Compress -----
def compress_model():
    print(f" Compressing model directory: {MODEL_DIR}")
    with tarfile.open(TAR_PATH, "w:gz") as tar:
        tar.add(MODEL_DIR, arcname=".")  # ใส่ทุกไฟล์ใน dir แต่ไม่มีชื่อโฟลเดอร์ซ้อน
    print(f" Model compressed to: {TAR_PATH}")

# ----- 2. Upload to S3 -----
def upload_to_s3():
    print(f" Uploading to s3://{S3_BUCKET}/{S3_KEY}")
    s3 = boto3.client("s3")
    s3.upload_file(TAR_PATH, S3_BUCKET, S3_KEY)
    print(" Upload complete.")

# ----- 3. Verify file exists on S3 -----
def verify_upload():
    s3 = boto3.client("s3")
    response = s3.head_object(Bucket=S3_BUCKET, Key=S3_KEY)
    size_mb = response["ContentLength"] / (1024 * 1024)
    print(f" File verified on S3: {size_mb:.2f} MB")

# ----- MAIN -----
if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"{MODEL_DIR} not found! Train or compress first.")

    compress_model()
    upload_to_s3()
    verify_upload()
