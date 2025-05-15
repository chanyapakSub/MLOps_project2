import boto3, json

s3 = boto3.client('s3')
sm = boto3.client('sagemaker')

def upload_to_s3(local_path, bucket, s3_path):
    s3.upload_file(local_path, bucket, s3_path)

def read_registry(bucket, key="registry/registry.json"):
    content = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(content["Body"].read().decode())

def write_registry(bucket, data, key="registry/registry.json"):
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(data))

def delete_endpoint(endpoint_name):
    sm.delete_endpoint(EndpointName=endpoint_name)
