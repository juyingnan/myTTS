# xtts

Voice-cloning text-to-speech using [F5-TTS](https://github.com/SWivid/F5-TTS). Provide a short audio sample of your voice and generate speech from text that sounds like you.

## Usage

```bash
python xtts.py --text-file "script.txt" --ref "my_voice.m4a" --out "output.wav"
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--text` / `-t` | | Text to speak |
| `--text-file` / `-f` | | Read text from file |
| `--ref` / `-r` | `Recording.m4a` | Reference audio of your voice |
| `--out` / `-o` | `output.wav` | Output wav path |
| `--cfg-strength` | `2.0` | Voice similarity vs fluency (higher = more like you) |
| `--speed` | `1.0` | Speech rate |
| `--nfe-step` | `32` | Diffusion steps (higher = better quality, slower) |
| `--no-gpu` | | Run on CPU |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install f5-tts
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Install [ffmpeg](https://ffmpeg.org/) for audio conversion:

```bash
winget install ffmpeg
```
