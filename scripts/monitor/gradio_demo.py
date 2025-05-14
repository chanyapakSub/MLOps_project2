import gradio as gr
from transformers import pipeline

# Load model (assume it's a QA model or similar)
qa = pipeline("question-answering",
             model="models/quantize_model",
             tokenizer="models/quantize_model")

def answer(q, c):
    return qa(question=q, context=c)["answer"]

gr.Interface(
    fn=answer,
    inputs=["text", "text"],
    outputs="text",
    title="💬 Ask Me Anything!",
    description="Enter a question and context."
).launch()
