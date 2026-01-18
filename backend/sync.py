"""
TwinWave Sync Engine
Time synchronization and drift correction for dual-phone playback
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class SyncFrame:
    """Audio frame with synchronization metadata"""
    frame_id: int
    timestamp_ms: float
    sample_rate: int
    samples_count: int
    audio_data: bytes
    channel: str  # 'left' or 'right'
    
    def to_dict(self) -> dict:
        """Convert to dictionary for transmission"""
        return {
            'frame_id': self.frame_id,
            'timestamp_ms': self.timestamp_ms,
            'sample_rate': self.sample_rate,
            'samples_count': self.samples_count,
            'channel': self.channel
        }


@dataclass
class ClientState:
    """State tracking for each connected client"""
    client_id: str
    channel: str  # 'left' or 'right'
    connected_at: float = field(default_factory=time.time)
    last_ping: float = field(default_factory=time.time)
    rtt_history: list = field(default_factory=list)
    current_frame_id: int = 0
    playback_offset_ms: float = 0.0
    buffer_size_ms: float = 100.0
    is_ready: bool = False
    drift_correction: float = 1.0  # Playback rate multiplier
    
    def update_rtt(self, rtt_ms: float):
        """Update RTT history and calculate average"""
        self.rtt_history.append(rtt_ms)
        if len(self.rtt_history) > 20:
            self.rtt_history.pop(0)
    
    @property
    def avg_rtt(self) -> float:
        """Average round-trip time"""
        if not self.rtt_history:
            return 0.0
        return np.mean(self.rtt_history)
    
    @property
    def rtt_jitter(self) -> float:
        """RTT jitter (standard deviation)"""
        if len(self.rtt_history) < 2:
            return 0.0
        return np.std(self.rtt_history)


class SyncEngine:
    """
    Main synchronization engine
    Handles timestamp generation, drift detection, and correction
    """
    
    def __init__(self, sample_rate: int = 44100, block_size: int = 1024):
        self.sample_rate = sample_rate
        self.block_size = block_size
        
        # Timing
        self.start_time = time.time()
        self.frame_counter = 0
        self.samples_sent = 0
        
        # Client tracking
        self.clients: Dict[str, ClientState] = {}
        self.lock = threading.Lock()
        
        # Sync parameters
        self.max_drift_ms = 30.0  # Maximum allowed drift before correction
        self.resync_interval_ms = 5000.0  # Resync every 5 seconds
        self.last_resync = time.time() * 1000
        
        # Buffer sizing
        self.min_buffer_ms = 50.0
        self.max_buffer_ms = 200.0
        self.target_buffer_ms = 100.0
        
    def get_current_timestamp(self) -> float:
        """Get current playback timestamp in milliseconds"""
        elapsed = time.time() - self.start_time
        return elapsed * 1000
    
    def get_sample_timestamp(self) -> float:
        """Get timestamp based on samples sent"""
        return (self.samples_sent / self.sample_rate) * 1000
    
    def create_frame(self, audio_left: bytes, audio_right: bytes) -> Tuple[SyncFrame, SyncFrame]:
        """Create synchronized frames for both channels"""
        timestamp = self.get_sample_timestamp()
        frame_id = self.frame_counter
        
        left_frame = SyncFrame(
            frame_id=frame_id,
            timestamp_ms=timestamp,
            sample_rate=self.sample_rate,
            samples_count=self.block_size,
            audio_data=audio_left,
            channel='left'
        )
        
        right_frame = SyncFrame(
            frame_id=frame_id,
            timestamp_ms=timestamp,
            sample_rate=self.sample_rate,
            samples_count=self.block_size,
            audio_data=audio_right,
            channel='right'
        )
        
        self.frame_counter += 1
        self.samples_sent += self.block_size
        
        return left_frame, right_frame
    
    def register_client(self, client_id: str, channel: str) -> ClientState:
        """Register a new client (phone)"""
        with self.lock:
            client = ClientState(client_id=client_id, channel=channel)
            self.clients[client_id] = client
            return client
    
    def unregister_client(self, client_id: str):
        """Remove a client"""
        with self.lock:
            if client_id in self.clients:
                del self.clients[client_id]
    
    def handle_ping(self, client_id: str, client_timestamp: float) -> dict:
        """
        Handle ping from client for RTT measurement
        Returns response with server timestamp and calculated offset
        """
        server_time = time.time() * 1000
        
        with self.lock:
            if client_id not in self.clients:
                return {'error': 'Client not registered'}
            
            client = self.clients[client_id]
            
            # Calculate RTT
            rtt = server_time - client_timestamp
            client.update_rtt(rtt)
            client.last_ping = time.time()
            
            # Calculate optimal playback offset
            # Client should buffer enough to account for network jitter
            buffer_needed = client.avg_rtt / 2 + client.rtt_jitter * 2
            buffer_needed = np.clip(buffer_needed, self.min_buffer_ms, self.max_buffer_ms)
            
            client.buffer_size_ms = buffer_needed
            
            return {
                'server_timestamp': server_time,
                'client_timestamp': client_timestamp,
                'rtt': rtt,
                'avg_rtt': client.avg_rtt,
                'recommended_buffer': buffer_needed,
                'playback_offset': client.playback_offset_ms
            }
    
    def report_playback_position(self, client_id: str, 
                                  frame_id: int,
                                  playback_time_ms: float) -> dict:
        """
        Client reports current playback position
        Used for drift detection and correction
        """
        with self.lock:
            if client_id not in self.clients:
                return {'error': 'Client not registered'}
            
            client = self.clients[client_id]
            client.current_frame_id = frame_id
            
            # Calculate expected playback time
            expected_time = (frame_id * self.block_size / self.sample_rate) * 1000
            
            # Calculate drift
            drift_ms = playback_time_ms - expected_time
            
            # Determine correction
            correction = self._calculate_drift_correction(drift_ms)
            client.drift_correction = correction
            
            return {
                'expected_time': expected_time,
                'reported_time': playback_time_ms,
                'drift_ms': drift_ms,
                'playback_rate_correction': correction,
                'needs_resync': abs(drift_ms) > self.max_drift_ms * 2
            }
    
    def _calculate_drift_correction(self, drift_ms: float) -> float:
        """
        Calculate playback rate correction to fix drift
        Returns multiplier for playbackRate (0.95 - 1.05)
        """
        if abs(drift_ms) < 5.0:
            # Drift too small, no correction needed
            return 1.0
        
        if abs(drift_ms) > self.max_drift_ms:
            # Significant drift - apply stronger correction
            correction = 0.02  # 2% speed adjustment
        else:
            # Minor drift - gentle correction
            correction = 0.005  # 0.5% speed adjustment
        
        if drift_ms > 0:
            # Playing ahead - slow down
            return 1.0 - correction
        else:
            # Playing behind - speed up
            return 1.0 + correction
    
    def check_sync_status(self) -> dict:
        """Check synchronization status of all clients"""
        with self.lock:
            if len(self.clients) < 2:
                return {
                    'synced': False,
                    'reason': 'Need both phones connected',
                    'clients': len(self.clients)
                }
            
            # Get left and right clients
            left_client = None
            right_client = None
            
            for client in self.clients.values():
                if client.channel == 'left':
                    left_client = client
                elif client.channel == 'right':
                    right_client = client
            
            if not left_client or not right_client:
                return {
                    'synced': False,
                    'reason': 'Need one LEFT and one RIGHT phone',
                    'clients': len(self.clients)
                }
            
            # Check frame difference
            frame_diff = abs(left_client.current_frame_id - right_client.current_frame_id)
            time_diff_ms = (frame_diff * self.block_size / self.sample_rate) * 1000
            
            is_synced = time_diff_ms < self.max_drift_ms
            
            return {
                'synced': is_synced,
                'left_frame': left_client.current_frame_id,
                'right_frame': right_client.current_frame_id,
                'frame_diff': frame_diff,
                'time_diff_ms': time_diff_ms,
                'left_rtt': left_client.avg_rtt,
                'right_rtt': right_client.avg_rtt,
                'left_correction': left_client.drift_correction,
                'right_correction': right_client.drift_correction
            }
    
    def needs_resync(self) -> bool:
        """Check if periodic resync is needed"""
        current_time = time.time() * 1000
        return (current_time - self.last_resync) > self.resync_interval_ms
    
    def perform_resync(self) -> dict:
        """
        Perform periodic resynchronization
        Returns sync parameters for all clients
        """
        self.last_resync = time.time() * 1000
        
        with self.lock:
            sync_data = {
                'resync_timestamp': self.last_resync,
                'master_frame': self.frame_counter,
                'master_sample_time': self.get_sample_timestamp(),
                'clients': {}
            }
            
            for client_id, client in self.clients.items():
                sync_data['clients'][client_id] = {
                    'channel': client.channel,
                    'target_frame': self.frame_counter,
                    'buffer_size': client.buffer_size_ms,
                    'playback_rate': client.drift_correction
                }
            
            return sync_data
    
    def get_optimal_buffer_config(self) -> dict:
        """Get optimal buffer configuration based on network conditions"""
        with self.lock:
            if not self.clients:
                return {
                    'buffer_size_ms': self.target_buffer_ms,
                    'chunk_duration_ms': (self.block_size / self.sample_rate) * 1000
                }
            
            # Find worst-case RTT
            max_rtt = max(c.avg_rtt for c in self.clients.values()) if self.clients else 0
            max_jitter = max(c.rtt_jitter for c in self.clients.values()) if self.clients else 0
            
            # Calculate buffer needed
            buffer_needed = max_rtt + max_jitter * 3
            buffer_needed = np.clip(buffer_needed, self.min_buffer_ms, self.max_buffer_ms)
            
            return {
                'buffer_size_ms': buffer_needed,
                'chunk_duration_ms': (self.block_size / self.sample_rate) * 1000,
                'max_rtt': max_rtt,
                'max_jitter': max_jitter
            }


class DynamicBuffer:
    """
    Dynamic buffer that resizes based on network conditions
    Used on client side for smooth playback
    """
    
    def __init__(self, target_size_ms: float = 100.0,
                 sample_rate: int = 44100):
        self.target_size_ms = target_size_ms
        self.sample_rate = sample_rate
        self.min_size_ms = 30.0
        self.max_size_ms = 300.0
        
        # Buffer storage
        self.buffer = deque()
        self.total_samples = 0
        
        # Adaptive sizing
        self.underrun_count = 0
        self.overflow_count = 0
        self.last_adjust_time = time.time()
        self.adjust_interval = 1.0  # Adjust every second
        
    def add_frame(self, frame: SyncFrame) -> bool:
        """Add frame to buffer"""
        current_size_ms = self.get_size_ms()
        
        if current_size_ms > self.max_size_ms:
            self.overflow_count += 1
            return False  # Buffer full
        
        self.buffer.append(frame)
        self.total_samples += frame.samples_count
        return True
    
    def get_frame(self) -> Optional[SyncFrame]:
        """Get next frame from buffer"""
        if not self.buffer:
            self.underrun_count += 1
            return None
        
        frame = self.buffer.popleft()
        self.total_samples -= frame.samples_count
        return frame
    
    def get_size_ms(self) -> float:
        """Get current buffer size in milliseconds"""
        return (self.total_samples / self.sample_rate) * 1000
    
    def adjust_size(self) -> float:
        """Adjust target buffer size based on performance"""
        current_time = time.time()
        
        if current_time - self.last_adjust_time < self.adjust_interval:
            return self.target_size_ms
        
        self.last_adjust_time = current_time
        
        # Increase buffer on underruns
        if self.underrun_count > 0:
            self.target_size_ms = min(
                self.target_size_ms * 1.2,
                self.max_size_ms
            )
        
        # Decrease buffer on overflows (but slower)
        elif self.overflow_count > 0:
            self.target_size_ms = max(
                self.target_size_ms * 0.95,
                self.min_size_ms
            )
        
        # Reset counters
        self.underrun_count = 0
        self.overflow_count = 0
        
        return self.target_size_ms
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        self.total_samples = 0
