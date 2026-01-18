"""
TwinWave DSP Engine
Professional-grade audio processing with cinema-quality filters
"""

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class EQPreset(Enum):
    """Cinema EQ Presets"""
    THEATRE = "theatre"
    NIGHT_MODE = "night_mode"
    DIALOGUE_BOOST = "dialogue_boost"
    BASS_MONSTER = "bass_monster"
    FLAT = "flat"


@dataclass
class DSPConfig:
    """DSP Configuration Parameters"""
    sample_rate: int = 44100
    block_size: int = 1024
    rms_target: float = 0.3
    limiter_threshold: float = 0.9
    limiter_ratio: float = 10.0
    dialogue_band_low: float = 1000.0
    dialogue_band_high: float = 4000.0
    width_phase_offset_ms: float = 0.8
    room_reverb_decay: float = 0.3
    room_reflection_delay_ms: float = 15.0


class DSPEngine:
    """
    Main DSP Processing Engine
    Handles all audio processing from normalization to final output
    """
    
    def __init__(self, config: Optional[DSPConfig] = None):
        self.config = config or DSPConfig()
        self.sample_rate = self.config.sample_rate
        self.block_size = self.config.block_size
        
        # Initialize filter states
        self._init_filters()
        
        # RMS smoothing state
        self.rms_history = []
        self.rms_window = 10
        
        # Limiter state for smooth operation
        self.limiter_gain = 1.0
        self.limiter_attack = 0.001
        self.limiter_release = 0.05
        
    def _init_filters(self):
        """Initialize all filter coefficients"""
        # Dialogue enhancement bandpass filter (1kHz - 4kHz)
        self.dialogue_bp_b, self.dialogue_bp_a = signal.butter(
            4, 
            [self.config.dialogue_band_low, self.config.dialogue_band_high],
            btype='band',
            fs=self.sample_rate
        )
        self.dialogue_zi = signal.lfilter_zi(self.dialogue_bp_b, self.dialogue_bp_a)
        
        # Bass emphasis low-pass (< 200Hz)
        self.bass_lp_b, self.bass_lp_a = signal.butter(
            4, 200, btype='low', fs=self.sample_rate
        )
        self.bass_zi = signal.lfilter_zi(self.bass_lp_b, self.bass_lp_a)
        
        # Treble emphasis high-pass (> 4kHz)
        self.treble_hp_b, self.treble_hp_a = signal.butter(
            4, 4000, btype='high', fs=self.sample_rate
        )
        self.treble_zi = signal.lfilter_zi(self.treble_hp_b, self.treble_hp_a)
        
        # Low-pass dominance for left phone (< 800Hz emphasis)
        self.left_lp_b, self.left_lp_a = signal.butter(
            2, 800, btype='low', fs=self.sample_rate
        )
        
        # High-pass dominance for right phone (> 2kHz emphasis)
        self.right_hp_b, self.right_hp_a = signal.butter(
            2, 2000, btype='high', fs=self.sample_rate
        )
        
        # EQ preset filters
        self._init_eq_filters()
        
    def _init_eq_filters(self):
        """Initialize EQ preset filter coefficients"""
        self.eq_filters = {}
        
        # Theatre preset - enhanced bass and presence
        self.eq_filters[EQPreset.THEATRE] = {
            'bass_boost': self._create_shelf_filter(100, 3.0, 'low'),
            'presence': self._create_peak_filter(3000, 2.0, 1.5),
            'air': self._create_shelf_filter(10000, 1.5, 'high')
        }
        
        # Night mode - reduced bass, enhanced dialogue
        self.eq_filters[EQPreset.NIGHT_MODE] = {
            'bass_cut': self._create_shelf_filter(100, -6.0, 'low'),
            'dialogue': self._create_peak_filter(2500, 4.0, 2.0),
            'treble_cut': self._create_shelf_filter(8000, -3.0, 'high')
        }
        
        # Dialogue boost - focus on speech frequencies
        self.eq_filters[EQPreset.DIALOGUE_BOOST] = {
            'low_cut': self._create_shelf_filter(200, -4.0, 'low'),
            'presence': self._create_peak_filter(2000, 5.0, 3.0),
            'clarity': self._create_peak_filter(4000, 3.0, 2.0)
        }
        
        # Bass monster - maximum low-end
        self.eq_filters[EQPreset.BASS_MONSTER] = {
            'sub_bass': self._create_shelf_filter(60, 6.0, 'low'),
            'bass': self._create_peak_filter(100, 8.0, 4.0),
            'punch': self._create_peak_filter(200, 3.0, 2.0)
        }
        
    def _create_shelf_filter(self, freq: float, gain_db: float, 
                             shelf_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create a shelf filter"""
        # Using butterworth approximation for shelf
        if shelf_type == 'low':
            b, a = signal.butter(2, freq, btype='low', fs=self.sample_rate)
        else:
            b, a = signal.butter(2, freq, btype='high', fs=self.sample_rate)
        
        # Apply gain
        gain = 10 ** (gain_db / 20)
        b = b * gain
        return b, a
    
    def _create_peak_filter(self, freq: float, gain_db: float, 
                           q: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create a parametric peak/notch filter"""
        # Peaking EQ filter design
        A = 10 ** (gain_db / 40)
        w0 = 2 * np.pi * freq / self.sample_rate
        alpha = np.sin(w0) / (2 * q)
        
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
        
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1, a1/a0, a2/a0])
        return b, a
    
    def rms_normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        RMS normalization with smooth gain adjustment
        Prevents sudden volume jumps
        """
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio ** 2)) + 1e-10
        
        # Smooth RMS over time
        self.rms_history.append(rms)
        if len(self.rms_history) > self.rms_window:
            self.rms_history.pop(0)
        
        smoothed_rms = np.mean(self.rms_history)
        
        # Calculate and apply gain
        gain = self.config.rms_target / smoothed_rms
        gain = np.clip(gain, 0.1, 10.0)  # Limit gain range
        
        return audio * gain
    
    def soft_limiter(self, audio: np.ndarray) -> np.ndarray:
        """
        Soft limiter for cinema punch
        Uses smooth knee compression to prevent harsh clipping
        """
        threshold = self.config.limiter_threshold
        ratio = self.config.limiter_ratio
        
        # Calculate the soft knee curve
        abs_audio = np.abs(audio)
        
        # Create soft knee around threshold
        knee_width = 0.1
        knee_start = threshold - knee_width / 2
        knee_end = threshold + knee_width / 2
        
        # Linear region (below threshold)
        output = np.copy(audio)
        
        # Knee region
        knee_mask = (abs_audio >= knee_start) & (abs_audio <= knee_end)
        if np.any(knee_mask):
            x = abs_audio[knee_mask]
            # Smooth transition using quadratic curve
            knee_gain = 1 - ((x - knee_start) ** 2) / (4 * knee_width * threshold)
            output[knee_mask] = audio[knee_mask] * knee_gain
        
        # Compression region (above threshold)
        compress_mask = abs_audio > knee_end
        if np.any(compress_mask):
            x = abs_audio[compress_mask]
            compressed = threshold + (x - threshold) / ratio
            output[compress_mask] = np.sign(audio[compress_mask]) * compressed
        
        return output
    
    def enhance_dialogue(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhance dialogue frequencies (1-4kHz band)
        Pushes speech forward without affecting music too much
        """
        # Extract dialogue band
        dialogue, self.dialogue_zi = signal.lfilter(
            self.dialogue_bp_b, self.dialogue_bp_a, 
            audio, zi=self.dialogue_zi * audio[0]
        )
        
        # Mix enhanced dialogue back (subtle boost)
        enhanced = audio + dialogue * 0.3
        return enhanced
    
    def apply_eq_preset(self, audio: np.ndarray, 
                        preset: EQPreset) -> np.ndarray:
        """Apply cinema EQ preset"""
        if preset == EQPreset.FLAT or preset not in self.eq_filters:
            return audio
        
        filters = self.eq_filters[preset]
        result = audio.copy()
        
        for filter_name, (b, a) in filters.items():
            result = signal.lfilter(b, a, result)
        
        return result
    
    def process_master(self, audio: np.ndarray, 
                       preset: EQPreset = EQPreset.THEATRE) -> np.ndarray:
        """
        Master DSP chain: normalization -> EQ -> dialogue -> limiter
        """
        # Step 1: RMS normalization
        audio = self.rms_normalize(audio)
        
        # Step 2: Apply EQ preset
        audio = self.apply_eq_preset(audio, preset)
        
        # Step 3: Dialogue enhancement
        audio = self.enhance_dialogue(audio)
        
        # Step 4: Soft limiting for cinema punch
        audio = self.soft_limiter(audio)
        
        return audio


class StereoEngine:
    """
    Stereo processing with true L/R split and Mid-Side processing
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
    def split_stereo(self, stereo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split stereo to left and right channels"""
        if stereo.ndim == 1:
            # Mono input - duplicate to both channels
            return stereo.copy(), stereo.copy()
        
        left = stereo[:, 0] if stereo.shape[1] > 0 else stereo.flatten()
        right = stereo[:, 1] if stereo.shape[1] > 1 else left.copy()
        return left, right
    
    def to_mid_side(self, left: np.ndarray, 
                    right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert L/R to Mid-Side representation"""
        mid = (left + right) / 2
        side = (left - right) / 2
        return mid, side
    
    def from_mid_side(self, mid: np.ndarray, 
                      side: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert Mid-Side back to L/R"""
        left = mid + side
        right = mid - side
        return left, right
    
    def widen_stereo(self, left: np.ndarray, right: np.ndarray, 
                     width: float = 1.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Widen stereo image using Mid-Side processing
        width > 1.0 = wider, < 1.0 = narrower
        """
        mid, side = self.to_mid_side(left, right)
        
        # Boost side signal for wider stereo
        side = side * width
        
        return self.from_mid_side(mid, side)


class PsychoacousticProcessor:
    """
    Psychoacoustic tricks for theatre-like sound
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
    def apply_haas_effect(self, left: np.ndarray, right: np.ndarray,
                          delay_ms: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Haas effect - slight delay on one channel
        Creates perception of wider sound stage
        Safe values: 0.5-1.0ms (Haas fusion zone)
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        
        if delay_samples <= 0:
            return left, right
        
        # Delay right channel slightly
        delayed_right = np.zeros_like(right)
        delayed_right[delay_samples:] = right[:-delay_samples]
        
        return left, delayed_right
    
    def apply_phase_offset(self, left: np.ndarray, right: np.ndarray,
                           offset_ms: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply slight phase offset between channels
        Creates subtle widening effect
        """
        offset_samples = int(offset_ms * self.sample_rate / 1000)
        
        if offset_samples <= 0:
            return left, right
        
        # Create phase-offset version
        # Use allpass filter for phase shift without amplitude change
        offset_samples = min(offset_samples, len(right) - 1)
        
        # Simple delay-based phase approximation
        phase_right = np.roll(right, offset_samples)
        phase_right[:offset_samples] = right[:offset_samples]  # Avoid wraparound artifacts
        
        # Blend for subtle effect
        right_out = right * 0.7 + phase_right * 0.3
        
        return left, right_out


class RoomSimulator:
    """
    Room simulation with early reflections and reverb
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
        # Early reflection delays (in ms) - different for each phone
        self.left_reflections = [5, 12, 18, 25]  # ms
        self.right_reflections = [7, 15, 22, 30]  # ms
        
        # Reflection gains (decaying)
        self.reflection_gains = [0.3, 0.2, 0.15, 0.1]
        
    def add_early_reflections(self, audio: np.ndarray, 
                              is_left: bool = True) -> np.ndarray:
        """Add early reflections for room effect"""
        reflections = self.left_reflections if is_left else self.right_reflections
        
        output = audio.copy()
        
        for delay_ms, gain in zip(reflections, self.reflection_gains):
            delay_samples = int(delay_ms * self.sample_rate / 1000)
            
            if delay_samples < len(audio):
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples] * gain
                output += delayed
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 0.95:
            output = output * (0.95 / max_val)
        
        return output
    
    def add_subtle_reverb(self, audio: np.ndarray, 
                          decay: float = 0.3) -> np.ndarray:
        """
        Add subtle reverb tail
        Uses simple feedback delay network approximation
        """
        output = audio.copy()
        
        # Multiple delay taps for diffuse reverb
        delays_ms = [23, 37, 47, 61]  # Prime numbers for less metallic sound
        
        for delay_ms in delays_ms:
            delay_samples = int(delay_ms * self.sample_rate / 1000)
            
            if delay_samples < len(audio):
                delayed = np.zeros_like(audio)
                delayed[delay_samples:] = audio[:-delay_samples] * (decay / len(delays_ms))
                output += delayed
        
        return output


class PhoneDSP:
    """
    Phone-specific DSP processing
    Left phone: Bass emphasis, low-pass dominance
    Right phone: Treble emphasis, high-pass dominance
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
        # Bass emphasis filter for left phone (shelf at 200Hz, +4dB)
        self.bass_b, self.bass_a = signal.butter(2, 200, btype='low', fs=sample_rate)
        
        # Treble emphasis filter for right phone (shelf at 4kHz, +3dB)
        self.treble_b, self.treble_a = signal.butter(2, 4000, btype='high', fs=sample_rate)
        
        # Low-pass dominance for left (gentle rolloff above 8kHz)
        self.lowpass_b, self.lowpass_a = signal.butter(2, 8000, btype='low', fs=sample_rate)
        
        # High-pass dominance for right (gentle boost above 2kHz)
        self.highpass_b, self.highpass_a = signal.butter(1, 2000, btype='high', fs=sample_rate)
        
        # Room simulator
        self.room = RoomSimulator(sample_rate)
        
        # Psychoacoustic processor
        self.psycho = PsychoacousticProcessor(sample_rate)
        
    def process_left(self, audio: np.ndarray, 
                     room_sim: bool = True) -> np.ndarray:
        """
        Process audio for LEFT phone
        - Bass emphasis
        - Low-pass dominance  
        - Room simulation with left-side reflections
        """
        # Extract and boost bass
        bass = signal.lfilter(self.bass_b, self.bass_a, audio)
        audio = audio + bass * 0.4  # Add bass boost
        
        # Apply gentle low-pass for warmth
        audio = signal.lfilter(self.lowpass_b, self.lowpass_a, audio)
        
        # Add room simulation
        if room_sim:
            audio = self.room.add_early_reflections(audio, is_left=True)
            audio = self.room.add_subtle_reverb(audio, decay=0.25)
        
        return audio
    
    def process_right(self, audio: np.ndarray,
                      room_sim: bool = True) -> np.ndarray:
        """
        Process audio for RIGHT phone
        - Treble emphasis
        - High-pass dominance
        - Room simulation with right-side reflections
        """
        # Extract and boost treble
        treble = signal.lfilter(self.treble_b, self.treble_a, audio)
        audio = audio + treble * 0.3  # Add treble boost
        
        # Boost high frequencies slightly
        highs = signal.lfilter(self.highpass_b, self.highpass_a, audio)
        audio = audio + highs * 0.2
        
        # Add room simulation
        if room_sim:
            audio = self.room.add_early_reflections(audio, is_left=False)
            audio = self.room.add_subtle_reverb(audio, decay=0.3)
        
        return audio


class SceneDetector:
    """
    Dynamic scene detection for adaptive DSP
    Detects loud vs dialogue scenes and adjusts processing
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.rms_history = []
        self.history_length = 50  # Number of blocks to analyze
        
        # Dialogue band filter
        self.dialogue_b, self.dialogue_a = signal.butter(
            4, [1000, 4000], btype='band', fs=sample_rate
        )
        
    def analyze(self, audio: np.ndarray) -> dict:
        """
        Analyze audio block and return scene characteristics
        """
        # Overall RMS
        rms = np.sqrt(np.mean(audio ** 2))
        self.rms_history.append(rms)
        if len(self.rms_history) > self.history_length:
            self.rms_history.pop(0)
        
        avg_rms = np.mean(self.rms_history)
        
        # Dialogue presence (1-4kHz energy ratio)
        dialogue_band = signal.lfilter(self.dialogue_b, self.dialogue_a, audio)
        dialogue_rms = np.sqrt(np.mean(dialogue_band ** 2))
        dialogue_ratio = dialogue_rms / (rms + 1e-10)
        
        # Determine scene type
        is_loud = rms > avg_rms * 1.5
        is_dialogue = dialogue_ratio > 0.4
        
        return {
            'rms': float(rms),
            'avg_rms': float(avg_rms),
            'dialogue_ratio': float(dialogue_ratio),
            'is_loud': is_loud,
            'is_dialogue': is_dialogue,
            'scene_type': 'dialogue' if is_dialogue else ('action' if is_loud else 'normal')
        }
    
    def get_adaptive_params(self, scene_info: dict) -> dict:
        """
        Get adaptive DSP parameters based on scene
        """
        if scene_info['scene_type'] == 'dialogue':
            return {
                'stereo_width': 0.8,  # Narrower for center focus
                'haas_delay': 0.3,    # Less widening
                'room_sim': False,     # Less reverb
                'dialogue_boost': 0.5
            }
        elif scene_info['scene_type'] == 'action':
            return {
                'stereo_width': 1.4,  # Wider for immersion
                'haas_delay': 1.0,    # More widening
                'room_sim': True,      # Full room simulation
                'dialogue_boost': 0.2
            }
        else:
            return {
                'stereo_width': 1.2,
                'haas_delay': 0.8,
                'room_sim': True,
                'dialogue_boost': 0.3
            }
