# Echo / Nyx

Echo is a local desktop AI assistant. Its assistant persona is called **Nyx**, and it also responds to **Echo**. The application runs its language and image models locally and provides a small PySide6 chat interface with optional voice and Spotify controls.

## Features

- Local text chat with the bundled Qwen 2.5 3B model
- Local text-to-image generation with Stable Diffusion 3.5 Medium
- Offline speech recognition through Vosk
- Optional text-to-speech through `pyttsx3`
- Wake words: `Echo` and `Nyx`
- Explicit approval before the assistant reads a file
- File and directory commands for the project `data` folder
- Optional always-on-top response overlay
- Optional Spotify playback, search, volume, shuffle, and repeat controls
- Separate user profiles and optional saved conversations

## Requirements

- Windows 10 or later
- Python 3.10 or newer
- A working microphone for voice input
- Enough disk space for the bundled models
- A CUDA-capable GPU is recommended. Text mode uses 4-bit loading; image mode can run on CPU but will be considerably slower.

On Windows, `pyaudio` and `bitsandbytes` may require compatible prebuilt wheels or additional system setup. If voice or 4-bit model loading fails, start by checking those two packages and the selected Python interpreter.

## Installation

Open PowerShell in the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The model files are expected to remain in these bundled locations:

```text
assets/models/qwen2.5 3b/
assets/models/SD3.5 medium/
assets/vosk/
```

Do not rename these directories unless the paths in `main.py` are updated as well.

## Run

```powershell
python main.py
```

On startup:

1. Choose a username, or leave it blank for the public profile.
2. Choose **Text assistant** or **Image generator**.
3. Wait for the selected model and optional voice components to load.
4. Enter a message, press Enter, or use the **Voice** button.

Image prompts entered in image mode are saved as PNG files in the active user's `images` folder.

## Example Commands

Text mode accepts normal questions. It also supports these direct commands:

```text
read notes.txt
list data
spotify play
spotify pause
spotify next
spotify back
spotify volume 50
spotify shuffle on
spotify repeat track
spotify search artist INTERWORLD
spotify play playlist My Playlist
```

Relative file paths are resolved under the project `data` directory. The assistant cannot execute code, edit files, or browse the internet.

## Spotify Setup

Spotify controls require a Spotify Developer application and an active Spotify playback device.

Add credentials to the active profile's `config.yaml`:

```yaml
spotify:
	client_id: YOUR_CLIENT_ID
	client_secret: YOUR_CLIENT_SECRET
	redirect_uri: http://127.0.0.1:8888/callback
	device_id: ''
```

Register the redirect URI exactly in the Spotify Developer Dashboard. The first Spotify request opens the authorization flow and stores a token in `.spotify_cache`. Never commit `client_id`, `client_secret`, or token files to source control. Rotate credentials if they have already been committed.

## Configuration

The app creates profiles below `user/`:

```text
user/<username>/cfg/public/config.yaml
user/<username>/debug/logs.log
user/<username>/images/
user/<username>/past-conversations/
```

The blank username uses `user/.public/`. Most settings can be changed from the **Settings** dialog:

- Overlays and overlay position
- Text-to-speech and voice selection
- Vosk voice recognition and microphone
- Conversation saving

## Build A Windows Executable

Install PyInstaller, then build with the included specification file:

```powershell
python -m pip install pyinstaller
pyinstaller main.spec
```

The bundled application is written to `dist/main`. Keep the `assets`, `data`, and `user` directories available beside the executable so the local models and runtime files can be found.

## Troubleshooting

- **Model not loaded:** verify the model directories and available memory, then inspect `user/<username>/debug/logs.log`.
- **No microphone input:** select the correct device in **Settings** and confirm Windows microphone permissions.
- **No speech output:** verify that a Windows speech voice is installed and that TTS is enabled.
- **Spotify errors:** start Spotify on an active device, verify the credentials and redirect URI, and retry authorization.
- **Slow image generation:** use a CUDA-enabled PyTorch installation and a compatible GPU.

## Model and Dependency Licenses

Echo's dependencies and bundled models have their own licenses. Review the files in `LICENSE/`, `assets/models/qwen2.5 3b/`, and `assets/models/SD3.5 medium/` before redistribution or commercial use.
