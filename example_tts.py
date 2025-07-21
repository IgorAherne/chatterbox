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

# text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus."
# wav = model.generate(text)
# ta.save("test-default-voice.wav", wav, model.sr)


# If you want to synthesize with a different voice, specify the audio prompt:
AUDIO_PROMPT_PATH = "C:\\_myDrive\\repos\\auto-vlog\\assets\\audio_sample1.wav"
texts_batch = [ "Tonight, our brave troubadour is poised for his grandest romantic gesture! <Gist> He's about to be utterly, spectacularly, and hilariously ghosted live, for the entire world to witness! A painful comedy!"]
num_variations = 3


# # Batching - list of strings to synthesize multiple different texts in a single batch.
# # This is the most efficient way to process multiple, different prompts at once.
# # Careful: 1 text = 1 additional KV Cache (Vram)
# wavs_batch = model.generate(texts_batch, audio_prompt_path=AUDIO_PROMPT_PATH)
# for i, wav in enumerate(wavs_batch):
#     ta.save(f"test-batch-{i+1}.wav", wav, model.sr)


# Batching - Use num_return_sequences to generate multiple variations for each text.
# This is highly efficient for creating diverse samples, as the prompt is only processed once.
# Without making extra KV Caches. 
wavs_batch_multi = model.generate(texts_batch, audio_prompt_path=AUDIO_PROMPT_PATH, num_return_sequences=num_variations)
for i, group in enumerate(wavs_batch_multi):
    for j, wav in enumerate(group):
        ta.save(f"test-batch-{i+1}-variant-{j+1}.wav", wav, model.sr)
