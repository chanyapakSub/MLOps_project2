from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.model_step import ModelStep

# สร้างขั้นตอนการฝึก
training_step = TrainingStep(
    name="TrainingStep",
    estimator=huggingface_estimator,
    inputs={
        'training': 's3://your-bucket/training-data'
    }
)

# สร้างขั้นตอนการ deploy โมเดล
model_step = ModelStep(
    name="ModelStep",
    model=huggingface_estimator.create_model(),
    inputs=training_step.properties.ModelArtifacts.S3ModelArtifacts
)

# สร้าง Pipeline
pipeline = Pipeline(
    name="HuggingFacePipeline",
    steps=[training_step, model_step]
)

# รัน pipeline
pipeline.upsert()
pipeline.start()
