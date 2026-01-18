"""
TwinWave Audio Engine
Audio capture, file loading, and stream management
"""

import numpy as np
import wave
import io
import threading
import queue
import time
from typing import Optional, Callable, Tuple, Generator
from dataclasses import dataclass
from enum import Enum

# Try to import sounddevice, but make it optional for systems without audio hardware
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError):
    SOUNDDEVICE_AVAILABLE = False
    sd = None


class AudioSource(Enum):
    """Audio source types"""
    FILE = "file"
    MICROPHONE = "microphone"
    STREAM = "stream"
    TEST_TONE = "test_tone"


@dataclass
class AudioConfig:
    """Audio configuration"""
    sample_rate: int = 44100
    channels: int = 2
    block_size: int = 1024
    dtype: str = 'float32'


class AudioBuffer:
    """Thread-safe audio buffer"""
    
    def __init__(self, max_size: int = 100):
        self.queue = queue.Queue(maxsize=max_size)
        self.is_running = False
        
    def put(self, data: np.ndarray) -> bool:
        """Add audio data to buffer"""
        try:
            self.queue.put_nowait(data)
            return True
        except queue.Full:
            return False
    
    def get(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get audio data from buffer"""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def clear(self):
        """Clear buffer"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
    
    @property
    def size(self) -> int:
        """Current buffer size"""
        return self.queue.qsize()


class TestToneGenerator:
    """Generate test tones for testing and calibration"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.phase_left = 0.0
        self.phase_right = 0.0
        
    def generate_stereo_test(self, num_samples: int,
                             left_freq: float = 440.0,
                             right_freq: float = 554.37) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate stereo test tones
        Default: A4 (440Hz) on left, C#5 (554.37Hz) on right
        """
        t = np.arange(num_samples) / self.sample_rate
        
        # Left channel - lower frequency
        left = 0.5 * np.sin(2 * np.pi * left_freq * t + self.phase_left)
        self.phase_left = (self.phase_left + 2 * np.pi * left_freq * num_samples / self.sample_rate) % (2 * np.pi)
        
        # Right channel - higher frequency
        right = 0.5 * np.sin(2 * np.pi * right_freq * t + self.phase_right)
        self.phase_right = (self.phase_right + 2 * np.pi * right_freq * num_samples / self.sample_rate) % (2 * np.pi)
        
        return left.astype(np.float32), right.astype(np.float32)
    
    def generate_sweep(self, num_samples: int,
                       start_freq: float = 100.0,
                       end_freq: float = 8000.0) -> np.ndarray:
        """Generate frequency sweep"""
        t = np.linspace(0, 1, num_samples)
        # Logarithmic frequency sweep
        freq = start_freq * np.exp(t * np.log(end_freq / start_freq))
        phase = 2 * np.pi * np.cumsum(freq) / self.sample_rate
        return (0.5 * np.sin(phase)).astype(np.float32)
    
    def generate_pink_noise(self, num_samples: int) -> np.ndarray:
        """Generate pink noise (1/f spectrum)"""
        # Generate white noise
        white = np.random.randn(num_samples)
        
        # Apply 1/f filter using Voss-McCartney algorithm approximation
        # Simple IIR approximation
        b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
        a = np.array([1, -2.494956002, 2.017265875, -0.522189400])
        
        from scipy.signal import lfilter
        pink = lfilter(b, a, white)
        
        # Normalize
        pink = pink / np.max(np.abs(pink)) * 0.5
        
        return pink.astype(np.float32)


class AudioFileLoader:
    """Load and process audio files"""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        
    def load_wav(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load WAV file and return audio data with sample rate"""
        with wave.open(file_path, 'rb') as wav:
            sample_rate = wav.getframerate()
            n_channels = wav.getnchannels()
            n_frames = wav.getnframes()
            sample_width = wav.getsampwidth()
            
            # Read raw data
            raw_data = wav.readframes(n_frames)
            
            # Convert to numpy array based on sample width
            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            audio = np.frombuffer(raw_data, dtype=dtype)
            
            # Reshape for multi-channel
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels)
            
            # Convert to float32 (-1.0 to 1.0)
            if dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128) / 128
            else:
                audio = audio.astype(np.float32) / np.iinfo(dtype).max
            
            return audio, sample_rate
    
    def load_from_bytes(self, data: bytes) -> Tuple[np.ndarray, int]:
        """Load WAV from bytes"""
        with io.BytesIO(data) as f:
            return self.load_wav_from_file_object(f)
    
    def load_wav_from_file_object(self, file_obj) -> Tuple[np.ndarray, int]:
        """Load WAV from file-like object"""
        with wave.open(file_obj, 'rb') as wav:
            sample_rate = wav.getframerate()
            n_channels = wav.getnchannels()
            n_frames = wav.getnframes()
            sample_width = wav.getsampwidth()
            
            raw_data = wav.readframes(n_frames)
            
            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            else:
                dtype = np.int32
            
            audio = np.frombuffer(raw_data, dtype=dtype)
            
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels)
            
            if dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128) / 128
            else:
                audio = audio.astype(np.float32) / np.iinfo(dtype).max
            
            return audio, sample_rate
    
    def resample(self, audio: np.ndarray, 
                 orig_rate: int, 
                 target_rate: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if orig_rate == target_rate:
            return audio
        
        from scipy.signal import resample
        
        ratio = target_rate / orig_rate
        new_length = int(len(audio) * ratio)
        
        if audio.ndim == 1:
            return resample(audio, new_length).astype(np.float32)
        else:
            # Resample each channel
            resampled = np.zeros((new_length, audio.shape[1]), dtype=np.float32)
            for ch in range(audio.shape[1]):
                resampled[:, ch] = resample(audio[:, ch], new_length)
            return resampled


class MicrophoneCapture:
    """Capture audio from microphone"""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.buffer = AudioBuffer()
        self.stream = None
        self.is_running = False
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input"""
        if status:
            print(f"Audio input status: {status}")
        
        # Copy data to buffer
        self.buffer.put(indata.copy())
    
    def start(self) -> bool:
        """Start microphone capture"""
        if not SOUNDDEVICE_AVAILABLE:
            print("SoundDevice not available - microphone capture disabled")
            return False
            
        try:
            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.block_size,
                dtype=self.config.dtype,
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_running = True
            return True
        except Exception as e:
            print(f"Failed to start microphone: {e}")
            return False
    
    def stop(self):
        """Stop microphone capture"""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def get_audio(self) -> Optional[np.ndarray]:
        """Get audio from buffer"""
        return self.buffer.get()


class AudioStreamProcessor:
    """
    Main audio stream processor
    Manages audio source and provides processed audio blocks
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.source = AudioSource.TEST_TONE
        
        # Components
        self.file_loader = AudioFileLoader(self.config)
        self.mic_capture = MicrophoneCapture(self.config)
        self.test_tone = TestToneGenerator(self.config.sample_rate)
        
        # File playback state
        self.file_audio = None
        self.file_position = 0
        self.file_sample_rate = self.config.sample_rate
        
        # State
        self.is_running = False
        self.is_looping = True
        
    def set_source_file(self, file_path: str) -> bool:
        """Set audio source to file"""
        try:
            audio, sample_rate = self.file_loader.load_wav(file_path)
            
            # Resample if needed
            if sample_rate != self.config.sample_rate:
                audio = self.file_loader.resample(
                    audio, sample_rate, self.config.sample_rate
                )
            
            self.file_audio = audio
            self.file_position = 0
            self.file_sample_rate = self.config.sample_rate
            self.source = AudioSource.FILE
            return True
        except Exception as e:
            print(f"Failed to load file: {e}")
            return False
    
    def set_source_file_bytes(self, data: bytes) -> bool:
        """Set audio source from bytes"""
        try:
            audio, sample_rate = self.file_loader.load_from_bytes(data)
            
            if sample_rate != self.config.sample_rate:
                audio = self.file_loader.resample(
                    audio, sample_rate, self.config.sample_rate
                )
            
            self.file_audio = audio
            self.file_position = 0
            self.file_sample_rate = self.config.sample_rate
            self.source = AudioSource.FILE
            return True
        except Exception as e:
            print(f"Failed to load file from bytes: {e}")
            return False
    
    def set_source_microphone(self) -> bool:
        """Set audio source to microphone"""
        if self.mic_capture.start():
            self.source = AudioSource.MICROPHONE
            return True
        return False
    
    def set_source_test_tone(self):
        """Set audio source to test tone"""
        self.source = AudioSource.TEST_TONE
    
    def get_next_block(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get next audio block as stereo (left, right)
        Returns tuple of (left_channel, right_channel)
        """
        block_size = self.config.block_size
        
        if self.source == AudioSource.TEST_TONE:
            return self.test_tone.generate_stereo_test(block_size)
        
        elif self.source == AudioSource.FILE:
            if self.file_audio is None:
                return self._get_silence()
            
            end_pos = self.file_position + block_size
            
            if end_pos > len(self.file_audio):
                if self.is_looping:
                    self.file_position = 0
                    end_pos = block_size
                else:
                    return self._get_silence()
            
            audio = self.file_audio[self.file_position:end_pos]
            self.file_position = end_pos
            
            # Split to stereo
            if audio.ndim == 1:
                return audio.astype(np.float32), audio.astype(np.float32)
            else:
                left = audio[:, 0].astype(np.float32)
                right = audio[:, 1] if audio.shape[1] > 1 else left.copy()
                return left, right.astype(np.float32)
        
        elif self.source == AudioSource.MICROPHONE:
            audio = self.mic_capture.get_audio()
            if audio is None:
                return self._get_silence()
            
            if audio.ndim == 1:
                return audio.astype(np.float32), audio.astype(np.float32)
            else:
                left = audio[:, 0].astype(np.float32)
                right = audio[:, 1] if audio.shape[1] > 1 else left.copy()
                return left, right.astype(np.float32)
        
        return self._get_silence()
    
    def _get_silence(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return silence"""
        silence = np.zeros(self.config.block_size, dtype=np.float32)
        return silence, silence.copy()
    
    def get_stream_generator(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generator that yields audio blocks"""
        self.is_running = True
        
        while self.is_running:
            yield self.get_next_block()
    
    def stop(self):
        """Stop streaming"""
        self.is_running = False
        self.mic_capture.stop()
    
    def seek(self, position_samples: int):
        """Seek to position in file"""
        if self.file_audio is not None:
            self.file_position = max(0, min(position_samples, len(self.file_audio)))
    
    def get_duration_ms(self) -> float:
        """Get file duration in milliseconds"""
        if self.file_audio is None:
            return 0.0
        return (len(self.file_audio) / self.config.sample_rate) * 1000
    
    def get_position_ms(self) -> float:
        """Get current position in milliseconds"""
        return (self.file_position / self.config.sample_rate) * 1000


def audio_to_bytes(audio: np.ndarray, sample_rate: int = 44100) -> bytes:
    """Convert numpy audio array to bytes for transmission"""
    # Ensure float32
    audio = audio.astype(np.float32)
    return audio.tobytes()


def bytes_to_audio(data: bytes) -> np.ndarray:
    """Convert bytes back to numpy audio array"""
    return np.frombuffer(data, dtype=np.float32)
