import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load from the uploaded model folder in this Space
model_dir = "./my_imdb_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Use GPU if available in Space
device = 0 if torch.cuda.is_available() else -1
model.to(device) if device == 0 else None

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=device
)

def predict_sentiment(text):
    if not text or not text.strip():
        return "Please enter a movie review!"
    
    try:
        result = sentiment_pipeline(text)[0]
        label = "Positive 😊" if result['label'] == "LABEL_1" else "Negative 😞"
        score = result['score']
        
        if score < 0.60:
            conf_text = "low confidence"
        elif score < 0.85:
            conf_text = "moderate confidence"
        else:
            conf_text = "high confidence"
        
        return f"**{label}**\nConfidence: {score:.1%} ({conf_text})"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=6,
        placeholder="Write your movie review here...",
        label="Your Movie Review"
    ),
    outputs=gr.Markdown(label="Result"),
    title="🎬 Movie Review Sentiment Analyzer",
    description="Fine-tuned DistilBERT on IMDB reviews. Enter a review to see if it's positive or negative.",
    examples=[
        ["This movie was absolutely fantastic and heartwarming!"],
        ["Worst film I've seen in years. Terrible acting."],
        ["It was okay, nothing special but not bad either."],
        ["The plot twists kept me on the edge of my seat!"]
    ],
    theme="soft",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
