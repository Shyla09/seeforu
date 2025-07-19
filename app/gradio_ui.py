import gradio as gr
import torch
from torchvision import transforms
from inference.generate_caption import generate_caption
from inference.tts_speech import speak_caption
from model.encoder import EncoderCNN
from model.decoder import DecoderRNN
from utils.vocab import Vocabulary
import pickle

# Load vocabulary
with open("utils/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = EncoderCNN().to(device)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(vocab)).to(device)

encoder.load_state_dict(torch.load("checkpoints/encoder.pth", map_location=device))
decoder.load_state_dict(torch.load("checkpoints/decoder.pth", map_location=device))
encoder.eval()
decoder.eval()

# Interface function
def caption_and_speak(image):
    caption = generate_caption(image, encoder, decoder, vocab, device, transform)
    speak_caption(caption)
    return caption

# Gradio interface
iface = gr.Interface(
    fn=caption_and_speak,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="SeeForU: Image Captioning with Voice",
    description="Upload an image to get an AI-generated caption and audio narration."
)

if __name__ == "__main__":
    iface.launch()
