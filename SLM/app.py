from flask import Flask, request, jsonify
import torch
import pickle

from model import GPTLanguageModel

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

stoi = tokenizer["stoi"]
itos = tokenizer["itos"]

vocab_size = len(stoi)

# Load model
model = GPTLanguageModel(vocab_size)
model.load_state_dict(
    torch.load(
        "victorian_gpt_model.pt",
        map_location=device
    )
)

model.to(device)
model.eval()


def encode(text):
    return [stoi[c] for c in text if c in stoi]


def decode(tokens):
    return "".join([itos[i] for i in tokens])


@app.route("/generate", methods=["POST"])
def generate():

    data = request.json

    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 300)
    temperature = data.get("temperature", 1.0)

    context = torch.tensor(
        [encode(prompt)],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            context,
            max_tokens,
            temperature
        )

    generated_text = decode(
        output[0].tolist()
    )

    return jsonify({
        "generated_text": generated_text
    })


@app.route("/")
def home():
    return "Victorian GPT API Running"


if __name__ == "__main__":
    app.run(debug=True)