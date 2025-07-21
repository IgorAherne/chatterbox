import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)

text = "Tonight, our brave troubadour is poised for his grandest romantic gesture! <Gist> He's about to be utterly, spectacularly, and hilariously ghosted live, for the entire world to witness! A painful comedy!"
wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)

# # If you want to synthesize with a different voice, specify the audio prompt
# AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
# wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
# ta.save("test-2.wav", wav, model.sr)
