import torch
import torchvision.transforms as transforms
from PIL import Image

def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

def generate_caption(image, encoder, decoder, vocab, device, transform, max_len=20):
    image_tensor = load_image(image, transform).to(device)
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.itos.get(word_id, "<UNK>")
        if word == "<EOS>":
            break
        sampled_caption.append(word)

    return " ".join(sampled_caption)
