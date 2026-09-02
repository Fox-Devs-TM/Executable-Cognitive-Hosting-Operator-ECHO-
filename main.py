import os
import sys
import json
import re
import subprocess
import threading
import time
import logging
import copy
import yaml
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLineEdit,
    QVBoxLayout, QWidget, QPushButton, QHBoxLayout,
    QLabel, QCheckBox, QDialog, QDialogButtonBox,
    QSpinBox, QFormLayout, QMessageBox
    , QComboBox, QRadioButton
)
from PySide6.QtCore import Qt, QPoint, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QColor, QPalette, QGuiApplication, QPixmap
from vosk import Model, KaldiRecognizer, SetLogLevel
import pyaudio
import pyttsx3
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
import torch
import signal
try:
    from diffusers import StableDiffusion3Pipeline
except ImportError:
    StableDiffusion3Pipeline = None
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException

SYSTEM_PROMPT = (
    "You are Nyx, a highly efficient local AI assistant running on open-source software called 'echo'. "
    "You also respond to the name 'echo'.\n\n"
    
    "### PERSONA & TONE\n"
    "- Adopt a professional, concise, and slightly witty tone.\n"
    "- Be direct and capable. Avoid fluff, excessive politeness, or robotic boilerplate.\n"
    "- You are transparent about your nature: You are a local LLM on open-source infrastructure. Never hide this fact.\n\n"
    
    "### TOOL PROTOCOLS\n"
    "Use tools ONLY when explicitly requested to read a file or list a directory.\n"
    "When a tool is required, output ONLY valid JSON. Do not include any other text, markdown, or explanations around the JSON object:\n"
    "For reading files use:\n"
    '{"tool": "read_file", "arguments": {"path": "notes.txt"}}\n'
    "For listing every child of the parent directory:\n"
    '{"tool": "list_files", "arguments": {"path": "data"}}\n'
    "For Spotify requests, use:\n"
    '{"tool": "spotify_skip"}\n'
    '{"tool": "spotify_play"}\n'
    '{"tool": "spotify_pause"}\n' 
    '{"tool": "spotify_back"}\n' 
    '{"tool": "spotify_volume", "arguments": {"volume": 50}}\n' 
    '{"tool": "spotify_shuffle", "arguments": {"state": true}}\n' 
    '{"tool": "spotify_repeat", "arguments": {"state": "track"}}\n' 
    '{"tool": "spotify_start"}\n' 
    '{"tool": "spotify_search", "arguments": {"query": "INTERWORLD", "type": "artist"}}\n' 
    "Spotify search types are track, artist, album, playlist, episode, or show. Use artist for people/bands and show for podcasts.\n\n"
    
    "### RESPONSE GUIDELINES\n"
    "- Keep answers short, clear, and actionable.\n"
    "- Do not explain the tool call unless explicitly asked.\n"
    "- If no tool is needed, respond naturally in plain text with precision.\n"
    "- If a tool fails or returns an error, acknowledge it briefly and suggest an alternative approach.\n"
    "- If asked about something outside your knowledge cutoff (cutoff ), state this clearly rather than guessing.\n"
    "- For ambiguous requests, ask one clarifying question rather than making assumptions.\n"
    "- Be mindful of conversation length. Summarize key points if the discussion becomes lengthy.\n"
    "- You cannot access the internet, execute code, or modify files. You can only read files and control Spotify as specified."

)

LLM_TOOLS = {
    "read_file": {
        "description": "Read a file from the app workspace or data folder.",
        "parameters": {
            "path": {
                "type": "string",
                "description": "Relative path under the project data folder or an absolute file path."
            }
        }
    },
    "list_files": {
        "description": "List files in a folder inside the app workspace or data folder.",
        "parameters": {
            "path": {
                "type": "string",
                "description": "Folder path relative to the data folder or an absolute folder path. Default is 'data'.",
                "default": "data"
            }
        }
    },
    "spotify_skip": {
        "description": "Skip to the next Spotify track.",
        "parameters": {}
    },
    "spotify_play": {
        "description": "Resume Spotify playback.",
        "parameters": {}
    },
    "spotify_pause": {
        "description": "Pause Spotify playback.",
        "parameters": {}
    },
    "spotify_back": {
        "description": "Go back to the previous Spotify track.",
        "parameters": {}
    },
    "spotify_volume": {
        "description": "Set Spotify volume from 0 to 100 percent.",
        "parameters": {
            "volume": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "The target volume percentage, from 0 to 100."
            }
        },
        "required": ["volume"]
    },
    "spotify_shuffle": {
        "description": "Turn Spotify shuffle on or off.",
        "parameters": {
            "state": {
                "type": "boolean",
                "description": "True to enable shuffle, false to disable it."
            }
        },
        "required": ["state"]
    },
    "spotify_repeat": {
        "description": "Set Spotify repeat mode to off, track, or context.",
        "parameters": {
            "state": {
                "type": "string",
                "enum": ["off", "track", "context"],
                "description": "Use off for no repeat, track for the current song, or context for the current album/playlist."
            }
        },
        "required": ["state"]
    },
    "spotify_start": {
        "description": "Start a Spotify track, album, or playlist by Spotify URI, URL, or search name.",
        "parameters": {
            "target": {"type": "string"},
            "type": {"type": "string", "enum": ["track", "album", "playlist"], "default": "track"}
        }
    },
    "spotify_search": {
        "description": "Search Spotify for tracks, artists, albums, playlists, podcast episodes, or podcast shows.",
        "parameters": {
            "query": {
                "type": "string",
                "description": "The song, artist, album, playlist, podcast episode, or podcast show to search for."
            },
            "type": {
                "type": "string",
                "enum": ["track", "artist", "album", "playlist", "episode", "show"],
                "default": "track",
                "description": "The result type: track, artist, album, playlist, episode, or show."
            }
        },
        "required": ["query"]
    }
}

# --- Suppress Vosk logs ---
SetLogLevel(-1)

APP_DIR = os.path.dirname(
    sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__)
)
ASSETS_DIR = os.path.join(APP_DIR, "assets")
DATA_DIR = os.path.join(APP_DIR, "data")
USER_ROOT = os.path.join(APP_DIR, "user")
CURRENT_USERNAME = None
USER_DIR = os.path.join(USER_ROOT, ".public")
IMAGE_DIR = os.path.join(USER_DIR, "images")
CONVERSATIONS_DIR = os.path.join(USER_DIR, "past-conversations")
LOG_PATH = os.path.join(USER_ROOT, ".public", "debug", "logs.log")

CONFIG_PATH = os.path.join(USER_ROOT, ".public", "cfg", "public", "config.yaml")

logger = logging.getLogger("echo")


def configure_user(username=None):
    global CURRENT_USERNAME, CONFIG_PATH, LOG_PATH
    global USER_DIR, IMAGE_DIR, CONVERSATIONS_DIR

    username = (username or "").strip()
    if username and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", username):
        raise ValueError("Username may contain only letters, numbers, underscores, and hyphens.")

    CURRENT_USERNAME = username or None
    user_directory = os.path.join(USER_ROOT, username or ".public")
    config_directory = os.path.join(user_directory, "cfg", "public")
    log_directory = os.path.join(user_directory, "debug")
    os.makedirs(config_directory, exist_ok=True)
    os.makedirs(log_directory, exist_ok=True)
    CONFIG_PATH = os.path.join(config_directory, "config.yaml")
    LOG_PATH = os.path.join(log_directory, "logs.log")
    USER_DIR = user_directory
    IMAGE_DIR = os.path.join(USER_DIR, "images")
    CONVERSATIONS_DIR = os.path.join(USER_DIR, "past-conversations")

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.info("User context selected: %s", CURRENT_USERNAME or "public")

DEFAULT_SETTINGS = {
    "overlays": True,
    "tts": True,
    "vosk": True,
    "overlay_x": 100,
    "overlay_y": 100,
    "input_device": -1,
    "tts_voice": "",
    "save_conversations": False,
    "spotify": {
        "client_id": "",
        "client_secret": "",
        "redirect_uri": "http://127.0.0.1:8888/callback",
        "device_id": ""
    }
}


def load_runtime_components(model_mode="text", status_callback=None):
    if status_callback:
        status_callback(
            "Loading image model..." if model_mode == "image" else "Loading language model...",
            15
        )
    loaded_model, loaded_tokenizer = None, None
    try:
        if model_mode == "image":
            if StableDiffusion3Pipeline is None:
                raise RuntimeError("Install diffusers to use image mode.")
            loaded_model = StableDiffusion3Pipeline.from_pretrained(
                os.path.join(ASSETS_DIR, "models", "SD3.5 medium"),
                torch_dtype=torch.float16
            )
            loaded_model.to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            loaded_model = AutoModelForCausalLM.from_pretrained(
                os.path.join(ASSETS_DIR, "models", "qwen2.5 3b"),
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                device_map="auto",
                torch_dtype=torch.float16
            )
            loaded_tokenizer = AutoTokenizer.from_pretrained(
                os.path.join(ASSETS_DIR, "models", "qwen2.5 3b")
            )
    except Exception as e:
        logger.exception("Failed to load %s model", model_mode)

    if status_callback:
        status_callback("Loading voice recognition...", 60)
    try:
        loaded_vosk_model = Model(os.path.join(ASSETS_DIR, "vosk"))
    except Exception as e:
        logger.exception("Failed to load Vosk model")
        loaded_vosk_model = None

    if status_callback:
        status_callback("Starting text-to-speech...", 85)
    loaded_engine = None

    if status_callback:
        status_callback("Finalizing startup...", 100)

    return {
        "model": loaded_model,
        "tokenizer": loaded_tokenizer,
        "vosk_model": loaded_vosk_model,
        "engine": loaded_engine,
    }


class StartupLoadingWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nyx/Echo starting...")
        self.resize(460, 180)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet("background: #111827; color: white; font-family: Segoe UI;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)

        self.title = QLabel("Starting Nyx/Echo")
        self.title.setStyleSheet("font-size: 20px; font-weight: bold;")

        self.percent = QLabel("0%")
        self.percent.setStyleSheet("font-size: 26px; color: #60a5fa; font-weight: bold;")
        self.percent.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status = QLabel("Preparing your assistant...")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-size: 13px; color: #d1d5db;")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress = QLabel(".")
        self.progress.setStyleSheet("font-size: 30px; color: #60a5fa; letter-spacing: 6px;")
        self.progress.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.dot_cycle = [".", "..", "...", "...."]
        self.dot_index = 0
        self.dot_timer = QTimer(self)
        self.dot_timer.timeout.connect(self.animate_dots)
        self.dot_timer.start(300)

        layout.addWidget(self.title)
        layout.addWidget(self.percent)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)

    def animate_dots(self):
        self.dot_index = (self.dot_index + 1) % len(self.dot_cycle)
        self.progress.setText(self.dot_cycle[self.dot_index])
        QApplication.processEvents()

    def set_status(self, message, percent=0):
        self.status.setText(message)
        self.percent.setText(f"{percent}%")
        QApplication.processEvents()

    def closeEvent(self, event):
        self.dot_timer.stop()
        super().closeEvent(event)


class UserChoiceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose user")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username, or leave blank for public")

        layout = QFormLayout(self)
        layout.addRow("Username:", self.username_input)
        layout.addRow(QLabel("Leave the username blank to use the public profile."))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    @property
    def username(self):
        return self.username_input.text().strip()

    def accept(self):
        if self.username and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.username):
            QMessageBox.warning(
                self,
                "Invalid username",
                "Use only letters, numbers, underscores, or hyphens."
            )
            return
        super().accept()


class ModelChoiceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Nyx model")
        self.text_radio = QRadioButton("Text assistant")
        self.image_radio = QRadioButton("Image generator")
        self.text_radio.setChecked(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose what Nyx should load for this session:"))
        layout.addWidget(self.text_radio)
        layout.addWidget(self.image_radio)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def model_mode(self):
        return "image" if self.image_radio.isChecked() else "text"


class StartupLoaderThread(QThread):
    status_signal = Signal(str, int)
    finished_signal = Signal(dict)

    def __init__(self, model_mode):
        super().__init__()
        self.model_mode = model_mode

    def run(self):
        result = load_runtime_components(self.model_mode, self.status_signal.emit)
        self.finished_signal.emit(result)


class ResponseThread(QThread):
    chunk_signal = Signal(str)
    image_signal = Signal(str)
    response_signal = Signal(str)

    def __init__(self, assistant, user_text, tool_name=None, tool_args=None):
        super().__init__()
        self.assistant = assistant
        self.user_text = user_text
        self.tool_name = tool_name
        self.tool_args = tool_args or {}
        self.streaming = not tool_name and assistant.model_mode == "text"

    def run(self):
        try:
            if self.tool_name:
                response = self.assistant.execute_tool_call(self.tool_name, self.tool_args)
            elif self.assistant.model_mode == "image" and model:
                image_path = self.assistant.generate_image(self.user_text)
                self.image_signal.emit(image_path)
                response = "Image generated."
            elif model and tokenizer:
                response = self.assistant.generate_response_live(
                    self.user_text,
                    self.chunk_signal.emit
                )
            else:
                response = "Model not loaded. Please check your model files."
        except Exception as error:
            logger.exception("Response worker failed")
            response = f"Error responding: {error}"
        self.response_signal.emit(response)


# --- Overlay Window ---
class OverlayWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("PA Overlay")

        # Layout and text label
        layout = QVBoxLayout()
        self.text_label = QLabel()
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setStyleSheet(
            "color: #00FF00; "
            "font-size: 16px; "
            "font-weight: bold; "
            "background-color: rgba(0, 0, 0, 0);"
        )
        layout.addWidget(self.text_label)
        self.setLayout(layout)

        # Default position (bottom-right)
        self.move_to_position(100, 100)

    def move_to_position(self, x, y):
        self.move(x, y)

    def show_response(self, text):
        self.text_label.setText(text)
        self.adjustSize()
        self.show()

    def closeEvent(self, event):
        event.ignore()

# --- Vosk Listener Thread ---
class VoskListener(QThread):
    text_signal = Signal(str)

    def __init__(self, model, input_device_index=None, sample_rate=16000, parent=None):
        super().__init__(parent)
        self.model = model
        self.input_device_index = input_device_index
        self.sample_rate = sample_rate
        self.running = True
        self.recognizer = KaldiRecognizer(model, sample_rate)
        self.recognizer.SetWords(True)
        self.p = pyaudio.PyAudio()
        self.stream = None

    def run(self):
        if not self.model or not self.p:
            return

        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=8000,
                input_device_index=self.input_device_index
            )
            self.stream.start_stream()

            wake_words = ["nyx", "echo", "nick"]
            while self.running and not self.isInterruptionRequested():
                try:
                    data = self.stream.read(4000, exception_on_overflow=False)
                except (OSError, IOError):
                    if self.running:
                        raise
                    break

                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip().lower()
                    if text:
                        self.text_signal.emit(text)
        finally:
            if self.stream:
                try:
                    self.stream.stop_stream()
                except (OSError, IOError):
                    pass
                try:
                    self.stream.close()
                except (OSError, IOError):
                    pass
                self.stream = None
            if self.p:
                self.p.terminate()
                self.p = None

    def stop(self):
        self.running = False
        self.requestInterruption()
        if self.stream:
            self.stream.close()

# --- Settings Dialog ---
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout()

        # Toggle settings
        self.overlay_cb = QCheckBox("Enable Overlays")
        self.tts_cb = QCheckBox("Enable TTS")
        self.vosk_cb = QCheckBox("Enable Vosk Voice Recognition")
        self.save_conversations_cb = QCheckBox("Save Conversations")
        self.input_device_combo = QComboBox()
        self.tts_voice_combo = QComboBox()
        self.populate_audio_devices()
        self.populate_tts_voices()

        # Overlay position settings
        self.overlay_x = QSpinBox()
        self.overlay_x.setRange(0, QGuiApplication.primaryScreen().geometry().width())
        self.overlay_y = QSpinBox()
        self.overlay_y.setRange(0, QGuiApplication.primaryScreen().geometry().height())

        layout.addRow("Enable Overlays:", self.overlay_cb)
        layout.addRow("Overlay X Position:", self.overlay_x)
        layout.addRow("Overlay Y Position:", self.overlay_y)
        layout.addRow("Enable TTS:", self.tts_cb)
        layout.addRow("Save Conversations:", self.save_conversations_cb)
        layout.addRow("TTS Voice:", self.tts_voice_combo)
        layout.addRow("Enable Vosk:", self.vosk_cb)
        layout.addRow("Voice Recognition Device:", self.input_device_combo)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

        self.setLayout(layout)

    def populate_audio_devices(self):
        audio = pyaudio.PyAudio()
        try:
            self.input_device_combo.addItem("System default microphone", None)
            for index in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(index)
                if info.get("maxInputChannels", 0) > 0:
                    self.input_device_combo.addItem(
                        f"{info['name']} (device {index})", index
                    )
        finally:
            audio.terminate()

    def populate_tts_voices(self):
        if not engine:
            self.tts_voice_combo.addItem("System default voice", "")
            return

        voices = engine.getProperty("voices") or []
        self.tts_voice_combo.addItem("System default voice", "")
        for voice in voices:
            self.tts_voice_combo.addItem(voice.name or voice.id, voice.id)

# --- File Access Dialog ---
class FileAccessDialog(QDialog):
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("File Access Request")
        self.file_path = file_path
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"The model requested access to:\n{self.file_path}"))
        layout.addWidget(QLabel("Do you want to allow access?"))

        # Buttons
        self.allow_once_button = QPushButton("Allow Once")
        self.allow_message_button = QPushButton("Allow for This Message")
        self.deny_button = QPushButton("Don't Allow")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.allow_once_button)
        button_layout.addWidget(self.allow_message_button)
        button_layout.addWidget(self.deny_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Connect buttons
        self.allow_once_button.clicked.connect(self.accept_allow_once)
        self.allow_message_button.clicked.connect(self.accept_allow_message)
        self.deny_button.clicked.connect(self.reject)

    def accept_allow_once(self):
        self.done(1)  # Allow this single file request

    def accept_allow_message(self):
        self.done(2)  # Allow all file requests in this message

# --- Main App ---
class PAApp(QMainWindow):
    def __init__(self, model_mode="text"):
        super().__init__()
        self.setWindowTitle("Nyx/Echo PA")
        self.setup_ui()
        self.load_settings()
        self.overlay = OverlayWindow()
        self.model_mode = model_mode
        self.update_overlay_position()
        self.apply_tts_voice()
        self.setup_connections()
        self.vosk_thread = None
        self.response_thread = None
        self.voice_listening = False
        self.start_vosk_listener()
        self.allow_all_files = False  # Flag for "Allow for This Message"

    def setup_ui(self):
        # Main chat UI
        self.text_display = QTextEdit(readOnly=True)
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setMaximumHeight(500)
        self.image_display.hide()
        self.last_image_path = None
        self.text_input = QLineEdit()
        self.text_input.returnPressed.connect(self.send_message)

        # Buttons
        self.voice_button = QPushButton("Voice")
        self.file_button = QPushButton("Files")
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.open_settings)

        # Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.voice_button)
        button_layout.addWidget(self.file_button)
        button_layout.addWidget(self.settings_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.text_display)
        main_layout.addWidget(self.image_display)
        main_layout.addWidget(self.text_input)
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_settings(self):
        self.settings_dict = copy.deepcopy(DEFAULT_SETTINGS)
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
                saved_settings = yaml.safe_load(config_file) or {}
            if isinstance(saved_settings, dict):
                self.settings_dict.update({
                    key: saved_settings[key]
                    for key in DEFAULT_SETTINGS
                    if key in saved_settings
                })
        except FileNotFoundError:
            self.save_settings()
        except (OSError, yaml.YAMLError) as error:
            logger.exception("Failed to load config")

    def save_settings(self):
        try:
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as config_file:
                yaml.safe_dump(self.settings_dict, config_file, sort_keys=False)
        except OSError as error:
            logger.exception("Failed to save config")

    @staticmethod
    def select_combo_value(combo, value):
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                combo.setCurrentIndex(index)
                return
        combo.setCurrentIndex(0)

    def apply_tts_voice(self):
        if not engine:
            return
        voice_id = self.settings_dict["tts_voice"]
        if not voice_id:
            return
        try:
            engine.setProperty("voice", voice_id)
        except Exception as error:
            logger.exception("Failed to set TTS voice")
            self.settings_dict["tts_voice"] = ""
            self.save_settings()

    def update_overlay_position(self):
        self.overlay.move_to_position(
            self.settings_dict["overlay_x"],
            self.settings_dict["overlay_y"]
        )

    def setup_connections(self):
        self.voice_button.clicked.connect(self.start_listening)
        self.file_button.clicked.connect(self.list_files)

    def start_vosk_listener(self):
        if not vosk_model or not self.settings_dict["vosk"]:
            return
        input_device = self.settings_dict["input_device"]
        self.vosk_thread = VoskListener(
            vosk_model,
            input_device_index=None if input_device < 0 else input_device,
            parent=self
        )
        self.vosk_thread.text_signal.connect(self.on_wake_word_detected)
        self.vosk_thread.start()

    def on_wake_word_detected(self, text):
        wake_words = ("nyx", "echo", "nick's", "nyx")
        words = text.split()
        wake_index = next(
            (index for index, word in enumerate(words) if word in wake_words),
            None
        )

        if self.voice_listening:
            self.voice_listening = False
            self.text_input.setText(text)
            self.send_message()
        elif wake_index is not None:
            command = " ".join(words[wake_index + 1:]).strip()
            if command:
                self.text_input.setText(command)
                self.send_message()
            else:
                self.voice_listening = True
                self.text_input.setFocus()
                self.text_input.setText("Listening... (Say your query)")

    def start_listening(self):
        if not vosk_model or not self.settings_dict["vosk"]:
            QMessageBox.warning(self, "Error", "Vosk is not enabled or loaded.")
            return
        self.voice_listening = True
        self.text_input.setFocus()
        self.text_input.setText("Listening... (Say your query)")

    def open_settings(self):
        dialog = SettingsDialog(self)
        dialog.overlay_cb.setChecked(self.settings_dict["overlays"])
        dialog.tts_cb.setChecked(self.settings_dict["tts"])
        dialog.vosk_cb.setChecked(self.settings_dict["vosk"])
        dialog.save_conversations_cb.setChecked(self.settings_dict.get("save_conversations", False))
        dialog.overlay_x.setValue(self.settings_dict["overlay_x"])
        dialog.overlay_y.setValue(self.settings_dict["overlay_y"])
        self.select_combo_value(dialog.input_device_combo, self.settings_dict["input_device"])
        self.select_combo_value(dialog.tts_voice_combo, self.settings_dict["tts_voice"])

        if dialog.exec():
            old_vosk_setting = self.settings_dict["vosk"]
            old_input_device = self.settings_dict["input_device"]
            self.settings_dict.update({
                "overlays": dialog.overlay_cb.isChecked(),
                "tts": dialog.tts_cb.isChecked(),
                "vosk": dialog.vosk_cb.isChecked(),
                "save_conversations": dialog.save_conversations_cb.isChecked(),
                "overlay_x": dialog.overlay_x.value(),
                "overlay_y": dialog.overlay_y.value(),
                "input_device": dialog.input_device_combo.currentData(),
                "tts_voice": dialog.tts_voice_combo.currentData() or ""
            })
            if self.settings_dict["input_device"] is None:
                self.settings_dict["input_device"] = -1
            self.save_settings()
            self.update_overlay_position()
            self.apply_tts_voice()

            # Restart Vosk listener if setting changed
            if (
                old_vosk_setting != self.settings_dict["vosk"]
                or old_input_device != self.settings_dict["input_device"]
            ):
                if self.vosk_thread:
                    self.vosk_thread.stop()
                    self.vosk_thread.wait()
                if self.settings_dict["vosk"]:
                    self.start_vosk_listener()

            if self.settings_dict["overlays"]:
                self.overlay.show_response("Overlay enabled")
            else:
                self.overlay.hide()

    def extract_tool_calls(self, text):
        calls = []
        text = text.strip()

        # First: direct JSON object/array format
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                if "tool" in parsed or "name" in parsed:
                    calls.append(parsed)
            elif isinstance(parsed, list):
                calls.extend(item for item in parsed if isinstance(item, dict) and ("tool" in item or "name" in item))
        except Exception:
            pass

        # Second: find JSON blocks that look like tool calls anywhere in the text
        json_pattern = re.compile(r'\{[^\{\}]*?"tool"\s*:\s*"[^"]+"[^\{\}]*\}', re.IGNORECASE | re.DOTALL)
        for match in json_pattern.finditer(text):
            candidate = match.group(0)
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and ("tool" in parsed or "name" in parsed):
                    calls.append(parsed)
            except Exception:
                continue

        return calls

    def execute_tool_call(self, tool_name, arguments=None):
        arguments = arguments or {}
        tool_name = (tool_name or "").strip()

        if tool_name == "read_file":
            file_path = arguments.get("path") or arguments.get("file_path") or ""
            return self.read_file(file_path)

        if tool_name == "list_files":
            dir_path = arguments.get("path") or arguments.get("directory") or "data"
            return self.list_files(dir_path)

        spotify_actions = {
            "spotify_skip": "next",
            "spotify_play": "play",
            "spotify_pause": "pause",
            "spotify_back": "previous"
        }
        if tool_name in spotify_actions:
            return self.spotify_control(spotify_actions[tool_name])

        if tool_name == "spotify_volume":
            return self.spotify_control("volume", arguments)
        if tool_name == "spotify_shuffle":
            return self.spotify_control("shuffle", arguments)
        if tool_name == "spotify_repeat":
            return self.spotify_control("repeat", arguments)
        if tool_name == "spotify_start":
            return self.spotify_control("start", arguments)
        if tool_name == "spotify_search":
            return self.spotify_control("search", arguments)

        return f"Unknown tool: {tool_name}"

    def parse_direct_tool_request(self, text):
        text = text.strip()
        read_match = re.match(r'^(?:read|open)\s+["\']?(.+?)["\']?\s*$', text, re.IGNORECASE)
        if read_match:
            return "read_file", {"path": read_match.group(1)}

        list_match = re.match(r'^(?:list|ls)\s+["\']?(.+?)["\']?\s*$', text, re.IGNORECASE)
        if list_match:
            return "list_files", {"path": list_match.group(1) or "data"}

        spotify_match = re.match(
            r'^(?:spotify\s+)?(skip|next|play|pause|back|previous)(?:\s+track)?\s*$',
            text,
            re.IGNORECASE
        )
        if spotify_match:
            spotify_tools = {
                "skip": "spotify_skip",
                "next": "spotify_skip",
                "play": "spotify_play",
                "pause": "spotify_pause",
                "back": "spotify_back",
                "previous": "spotify_back"
            }
            return spotify_tools[spotify_match.group(1).lower()], {}

        volume_match = re.match(r'^(?:spotify\s+)?(?:volume|vol)\s+(\d{1,3})%?\s*$', text, re.IGNORECASE)
        if volume_match:
            return "spotify_volume", {"volume": int(volume_match.group(1))}

        shuffle_match = re.match(r'^(?:spotify\s+)?shuffle\s+(on|off)\s*$', text, re.IGNORECASE)
        if shuffle_match:
            return "spotify_shuffle", {"state": shuffle_match.group(1).lower() == "on"}

        repeat_match = re.match(r'^(?:spotify\s+)?repeat\s+(off|track|context|song|album|playlist)\s*$', text, re.IGNORECASE)
        if repeat_match:
            state = repeat_match.group(1).lower()
            state = {"song": "track", "album": "context", "playlist": "context"}.get(state, state)
            return "spotify_repeat", {"state": state}

        search_match = re.match(
            r'^(?:spotify\s+)?search(?:\s+spotify)?\s+(?:(track|artist|album|playlist|episode|show)\s+)?(.+?)\s*$',
            text,
            re.IGNORECASE
        )
        if search_match:
            return "spotify_search", {
                "query": search_match.group(2),
                "type": (search_match.group(1) or "track").lower()
            }

        start_match = re.match(
            r'^(?:spotify\s+)?play\s+(?:(track|album|playlist)\s+)?(.+?)\s*$',
            text,
            re.IGNORECASE
        )
        if start_match and start_match.group(2).lower() not in {"track", "music"}:
            return "spotify_start", {
                "type": (start_match.group(1) or "track").lower(),
                "target": start_match.group(2)
            }

        return None, None

    def spotify_control(self, action, arguments=None):
        arguments = arguments or {}
        spotify_settings = self.settings_dict.get("spotify", {})
        client_id = spotify_settings.get("client_id", "").strip()
        client_secret = spotify_settings.get("client_secret", "").strip()
        redirect_uri = spotify_settings.get(
            "redirect_uri", "http://127.0.0.1:8888/callback"
        ).strip()
        device_id = spotify_settings.get("device_id", "").strip()
        if not client_id or not client_secret:
            return "Spotify is not configured. Add spotify.client_id and spotify.client_secret to config.yaml."

        try:
            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope="user-modify-playback-state user-read-playback-state",
                cache_path=os.path.join(APP_DIR, ".spotify_cache")
            )
            spotify_client = spotipy.Spotify(auth_manager=auth_manager)
            device_kwargs = {"device_id": device_id} if device_id else {}
            if action == "next":
                spotify_client.next_track(**device_kwargs)
            elif action == "previous":
                spotify_client.previous_track(**device_kwargs)
            elif action == "play":
                spotify_client.start_playback(**device_kwargs)
            elif action == "pause":
                spotify_client.pause_playback(**device_kwargs)
            elif action == "volume":
                volume = int(arguments.get("volume", -1))
                if not 0 <= volume <= 100:
                    return "Spotify volume must be between 0 and 100."
                spotify_client.volume(volume, **device_kwargs)
                return f"Spotify volume set to {volume}%."
            elif action == "shuffle":
                state = arguments.get("state")
                if isinstance(state, str):
                    state = state.lower() in {"on", "true", "1"}
                spotify_client.shuffle(bool(state), **device_kwargs)
                return f"Spotify shuffle {'on' if state else 'off'}."
            elif action == "repeat":
                state = str(arguments.get("state", "off")).lower()
                if state not in {"off", "track", "context"}:
                    return "Spotify repeat must be off, track, or context."
                spotify_client.repeat(state, **device_kwargs)
                return f"Spotify repeat set to {state}."
            elif action == "search":
                query = str(arguments.get("query", "")).strip()
                search_type = str(arguments.get("type", "track")).lower()
                valid_search_types = {"track", "artist", "album", "playlist", "episode", "show"}
                if not query or search_type not in valid_search_types:
                    return "Provide a search query and type track, artist, album, playlist, episode, or show."
                results = spotify_client.search(q=query, type=search_type, limit=5)
                items = results.get(f"{search_type}s", {}).get("items", [])
                if not items:
                    return f"No Spotify {search_type} results found for '{query}'."
                return "\n".join(
                    f"{index}. {item.get('name', 'Unknown')} - {item.get('external_urls', {}).get('spotify', item.get('uri', ''))}"
                    for index, item in enumerate(items, 1)
                )
            elif action == "start":
                target = str(arguments.get("target", "")).strip()
                target_type = str(arguments.get("type", "track")).lower()
                if not target or target_type not in {"track", "album", "playlist"}:
                    return "Provide a track, album, or playlist target."
                if target.startswith("spotify:") or "open.spotify.com/" in target:
                    uri = target
                    if "open.spotify.com/" in uri:
                        uri = "spotify:" + uri.split("open.spotify.com/", 1)[1].replace("/", ":").split("?", 1)[0]
                else:
                    results = spotify_client.search(q=target, type=target_type, limit=1)
                    item = results.get(f"{target_type}s", {}).get("items", [None])[0]
                    if not item:
                        return f"No Spotify {target_type} found for '{target}'."
                    uri = item["uri"]
                spotify_client.start_playback(context_uri=uri, **device_kwargs) if target_type != "track" else spotify_client.start_playback(uris=[uri], **device_kwargs)
                return f"Started Spotify {target_type}: {target}."
            else:
                return f"Unknown Spotify action: {action}"
            return f"Spotify {action} command sent."
        except SpotifyException as error:
            if error.http_status == 404:
                return "Spotify could not find an active player. Start Spotify on a device first."
            if error.http_status == 401:
                return "Spotify authorization expired. Try the command again to reauthorize."
            return f"Spotify API error ({error.http_status})."
        except Exception as error:
            return f"Could not connect to Spotify: {error}"

    def request_file_access(self, file_path):
        dialog = FileAccessDialog(file_path, self)
        result = dialog.exec()
        if result == 1:  # Allow Once
            return True, False
        elif result == 2:  # Allow for This Message
            return True, True
        else:  # Don't Allow
            return False, False

    def read_file(self, file_path):
        try:
            if not file_path:
                return "No file path was provided."
            if not os.path.isabs(file_path):
                file_path = os.path.join(DATA_DIR, file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"

    def list_files(self, dir_path="data"):
        try:
            if not dir_path or dir_path == "data":
                dir_path = DATA_DIR
            elif not os.path.isabs(dir_path):
                dir_path = os.path.join(DATA_DIR, dir_path)
            files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            return f"Files in {dir_path}: {', '.join(files) if files else 'none'}"
        except Exception as e:
            return f"Error listing files: {e}"

    def send_message(self):
        user_text = self.text_input.text()
        if not user_text or self.response_thread and self.response_thread.isRunning():
            return

        logger.info("Request received: %s", user_text)
        self.text_display.append(f"You: {user_text}")
        self.last_image_path = None
        tool_name, tool_args = self.parse_direct_tool_request(user_text)
        if tool_name:
            logger.info("Direct tool selected: %s args=%s", tool_name, tool_args)
        self.text_input.clear()
        self.text_input.setEnabled(False)
        self.response_thread = ResponseThread(self, user_text, tool_name, tool_args)
        if self.response_thread.streaming:
            self.text_display.insertPlainText("PA: ")
        self.response_thread.chunk_signal.connect(self.on_response_chunk)
        self.response_thread.image_signal.connect(self.on_image_ready)
        self.response_thread.response_signal.connect(self.on_response_ready)
        self.response_thread.finished.connect(self.on_response_finished)
        self.response_thread.start()

    @Slot(str)
    def on_response_chunk(self, chunk):
        self.text_display.insertPlainText(chunk)
        self.text_display.ensureCursorVisible()

    @Slot(str)
    def on_image_ready(self, image_path):
        self.last_image_path = image_path
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            self.image_display.setPixmap(
                pixmap.scaled(
                    self.image_display.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )
            self.image_display.show()

    @Slot(str)
    def on_response_ready(self, response):
        logger.info("Response completed: %s", response)
        if self.response_thread:
            self.save_conversation(
                self.response_thread.user_text,
                response,
                self.last_image_path
            )
        if not self.response_thread or not self.response_thread.streaming:
            self.text_display.append(f"PA: {response}")

        if self.settings_dict["overlays"]:
            self.overlay.show_response(response)

        if self.settings_dict["tts"] and engine:
            self.speak(response)

    @Slot()
    def on_response_finished(self):
        self.text_input.setEnabled(True)
        self.text_input.setFocus()
        self.response_thread.deleteLater()
        self.response_thread = None

    def generate_image(self, prompt):
        logger.info("Image generation started")
        os.makedirs(IMAGE_DIR, exist_ok=True)
        result = model(prompt=prompt, num_inference_steps=28)
        image_path = os.path.join(IMAGE_DIR, f"generated_{int(time.time() * 1000)}.png")
        result.images[0].save(image_path)
        logger.info("Image generated: %s", image_path)
        return image_path

    def save_conversation(self, user_text, response, image_path=None):
        if not self.settings_dict.get("save_conversations", False):
            return
        try:
            os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
            conversation = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "username": CURRENT_USERNAME or "public",
                "user": user_text,
                "nyx": response
            }
            if image_path:
                conversation["image"] = image_path
            conversation_path = os.path.join(
                CONVERSATIONS_DIR,
                f"conversation_{int(time.time() * 1000)}.yaml"
            )
            with open(conversation_path, "w", encoding="utf-8") as conversation_file:
                yaml.safe_dump(conversation, conversation_file, sort_keys=False)
            logger.info("Conversation saved: %s", conversation_path)
        except (OSError, yaml.YAMLError):
            logger.exception("Failed to save conversation")

    def stream_model_response(self, messages):
        if hasattr(tokenizer, "apply_chat_template"):
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
        else:
            combined = "\n\n".join(
                f"{message['role']}: {message['content']}" for message in messages
            )
            inputs = tokenizer(combined, return_tensors="pt")

        inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )
        generation_kwargs = {"max_new_tokens": 512, "streamer": streamer}
        if isinstance(inputs, torch.Tensor):
            generation_kwargs["input_ids"] = inputs
        else:
            generation_kwargs.update(inputs)

        def generate():
            try:
                model.generate(**generation_kwargs)
            except Exception:
                streamer.on_finalized_text("", stream_end=True)

        generation_thread = threading.Thread(target=generate, daemon=True)
        generation_thread.start()
        yield from streamer

    def generate_response_live(self, text, chunk_callback):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
        reply_parts = []
        for chunk in self.stream_model_response(messages):
            reply_parts.append(chunk)
            chunk_callback(chunk)

        reply = "".join(reply_parts)
        tool_calls = self.extract_tool_calls(reply)
        if not tool_calls:
            return reply

        tool_results = []
        for tool in tool_calls:
            tool_name = tool.get("tool") or tool.get("name")
            tool_args = tool.get("arguments") or tool.get("args") or {}
            result = self.execute_tool_call(tool_name, tool_args)
            tool_results.append(f"Tool {tool_name}: {result}")

        followup_messages = messages + [
            {"role": "assistant", "content": reply},
            {
                "role": "user",
                "content": "Tool results:\n" + "\n".join(tool_results) +
                "\n\nNow answer the user's request based only on those tool results and keep it brief."
            }
        ]
        final_parts = []
        for chunk in self.stream_model_response(followup_messages):
            final_parts.append(chunk)
            chunk_callback(chunk)
        return "".join(final_parts)

    def generate_response(self, text):
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ]

            if hasattr(tokenizer, "apply_chat_template"):
                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
            else:
                combined = "\n\n".join(
                    f"{message['role']}: {message['content']}" for message in messages
                )
                inputs = tokenizer(combined, return_tensors="pt")

            inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
            if isinstance(inputs, torch.Tensor):
                outputs = model.generate(input_ids=inputs, max_new_tokens=512)
            else:
                outputs = model.generate(**inputs, max_new_tokens=512)

            if isinstance(outputs, torch.Tensor) and inputs is not None and hasattr(inputs, "shape"):
                generated = outputs[0][inputs.shape[-1]:]
            else:
                generated = outputs[0]

            reply = tokenizer.decode(generated, skip_special_tokens=True)

            tool_calls = self.extract_tool_calls(reply)
            if tool_calls:
                tool_results = []
                for tool in tool_calls:
                    tool_name = tool.get("tool") or tool.get("name")
                    tool_args = tool.get("arguments") or tool.get("args") or {}
                    result = self.execute_tool_call(tool_name, tool_args)
                    tool_results.append(f"Tool {tool_name}: {result}")

                tool_context = "\n".join(tool_results)
                followup_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": "Tool results:\n" + tool_context + "\n\nNow answer the user's request based only on those tool results and keep it brief."}
                ]

                followup_inputs = tokenizer.apply_chat_template(
                    followup_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to("cuda" if torch.cuda.is_available() else "cpu")
                if isinstance(followup_inputs, torch.Tensor):
                    followup_outputs = model.generate(
                        input_ids=followup_inputs,
                        max_new_tokens=512
                    )
                else:
                    followup_outputs = model.generate(**followup_inputs, max_new_tokens=512)
                generated_reply = followup_outputs[0][followup_inputs.shape[-1]:]
                return tokenizer.decode(generated_reply, skip_special_tokens=True)

            return reply
        except Exception as e:
            return f"Error generating response: {e}"

    def speak(self, text):
        voice_id = self.settings_dict["tts_voice"]
        try:
            if getattr(sys, "frozen", False):
                command = [sys.executable, "--tts-worker", text, voice_id]
            else:
                speech_script = (
                    "import pyttsx3, sys; "
                    "engine = pyttsx3.init(); "
                    "voice = sys.argv[2]; "
                    "engine.setProperty('voice', voice) if voice else None; "
                    "engine.say(sys.argv[1]); "
                    "engine.runAndWait(); "
                    "engine.stop()"
                )
                command = [sys.executable, "-c", speech_script, text, voice_id]
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as error:
            logger.exception("TTS error")

    def closeEvent(self, event):
        if self.response_thread and self.response_thread.isRunning():
            self.response_thread.requestInterruption()
            self.response_thread.wait()
        if self.vosk_thread:
            self.vosk_thread.stop()
            self.vosk_thread.wait()
        self.overlay.close()
        super().closeEvent(event)


def run_tts_worker():
    import pyttsx3

    text = sys.argv[2]
    voice_id = sys.argv[3]
    speech_engine = pyttsx3.init()
    if voice_id:
        speech_engine.setProperty("voice", voice_id)
    speech_engine.say(text)
    speech_engine.runAndWait()
    speech_engine.stop()

def signal_handler(sig, frame):
    logger.info("KeyboardInterrupt detected. Closing Nyx/Echo...")
    app = QApplication.instance()
    if app:
        app.quit()

# --- Run the App ---
if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--tts-worker":
        run_tts_worker()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    app = QApplication(sys.argv)

    user_choice = UserChoiceDialog()
    if user_choice.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)
    try:
        configure_user(user_choice.username)
    except (OSError, ValueError) as error:
        QMessageBox.critical(app.activeWindow(), "User setup failed", str(error))
        sys.exit(1)

    splash = StartupLoadingWindow()
    screen = QGuiApplication.primaryScreen().availableGeometry()
    splash.move(
        (screen.width() - splash.width()) // 2,
        (screen.height() - splash.height()) // 2,
    )
    splash.show()
    splash.raise_()
    app.processEvents()

    model_choice = ModelChoiceDialog()
    if model_choice.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)
    selected_model_mode = model_choice.model_mode
    logger.info("Model mode selected: %s", selected_model_mode)

    loader = StartupLoaderThread(selected_model_mode)
    loader.status_signal.connect(lambda message, percent: splash.set_status(message, percent))
    window = None

    def on_loaded(result):
        global model, tokenizer, vosk_model, engine, window
        model = result["model"]
        tokenizer = result["tokenizer"]
        vosk_model = result["vosk_model"]
        try:
            engine = pyttsx3.init()
        except Exception as error:
            logger.exception("Failed to initialize TTS")
            engine = None
        splash.hide()
        window = PAApp(selected_model_mode)
        window.show()

    loader.finished_signal.connect(on_loaded)
    app.aboutToQuit.connect(lambda: loader.wait() if loader.isRunning() else None)
    loader.start()

    sys.exit(app.exec())