import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import matplotlib.pyplot as plt

# --- Page setup ---
st.set_page_config(page_title="Geopolitical Classifier", layout="centered")
st.title("Geopolitical Tension Classifier")
st.write("Paste or type any text below to classify it into a geopolitical risk category.")

# --- Load model and tokenizer (cached) ---
@st.cache_resource
def load_model():
    model_id = "durrani95/eurobert-geopolitical-multiclass"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
    return tokenizer, model

tokenizer, model = load_model()

# --- Text input ---
text_input = st.text_area(
    "Enter your text here:",
    placeholder="e.g. Russia cut off gas supplies to Europe amid rising tensions.",
    height=150
)

# --- Inference ---
if st.button("Classify"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer([text_input], padding=True, truncation=True, max_length=512, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0].cpu()

        # Prepare data
        labels = [model.config.id2label[i] for i in range(len(probs))]
        df = pd.DataFrame({"Category": labels, "Probability": probs.numpy()})
        df = df.sort_values("Probability", ascending=True)  # sorted for horizontal bars

        # Predicted label
        label_id = int(probs.argmax())
        label = model.config.id2label[label_id]
        confidence = float(probs[label_id])

        # --- Text output ---
        st.markdown("### 🧭 Prediction")
        st.write(f"**Predicted Category:** {label}")
        st.write(f"**Confidence:** {confidence:.2%}")
        st.write(f"**Input Text:** {text_input}")
        st.divider()

        # --- Matplotlib bar chart ---
        fig, ax = plt.subplots(figsize=(7, 0.4 * len(df) + 1))
        bars = ax.barh(df["Category"], df["Probability"], color="skyblue", edgecolor="black")

        # Highlight predicted category
        for bar, cat in zip(bars, df["Category"]):
            if cat == label:
                bar.set_color("steelblue")
                bar.set_edgecolor("black")
                bar.set_linewidth(1.2)

        # Annotate bars with percentages
        for i, v in enumerate(df["Probability"]):
            ax.text(v + 0.01, i, f"{v:.1%}", va="center", fontsize=10)

        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability", fontsize=11)
        ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        st.pyplot(fig)

# --- Footer ---
st.markdown(
    "<small>Model: [durrani95/eurobert-geopolitical-multiclass](https://huggingface.co/durrani95/eurobert-geopolitical-multiclass)</small>",
    unsafe_allow_html=True
)
