# seeforu
SeeForU(Work In Progress)
SeeForU is an AI-powered image captioning and audio narration system designed to enhance accessibility for visually impaired users. It generates natural language descriptions of images using a deep learning model and vocalizes them in real time using text-to-speech synthesis.

Tech Stack
Encoder: ResNet-50 CNN (feature extractor)
Decoder: LSTM-based RNN trained from scratch
Dataset: Flickr8k
Libraries: PyTorch, torchvision, NLTK, gTTS, Gradio
Interface: Gradio Web UI for interactive captioning
Narration: Google Text-to-Speech (gTTS)

Key Features
Extracts visual features using ResNet-50
Generates natural captions using a custom-trained LSTM decoder
Converts generated captions to audio using gTTS
User-friendly Gradio web interface
Custom vocabulary and tokenizer pipeline using NLTK
Focused on accessibility and social impact

How It Works
Upload an image via Gradio
ResNet-50 extracts image features (no gradients during inference)
LSTM decoder generates captions using greedy decoding
gTTS converts captions to audio and plays them
Output: Real-time caption with narration
