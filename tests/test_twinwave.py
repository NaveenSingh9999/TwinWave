"""
TwinWave Unit Tests
Tests for audio processing, DSP, and synchronization components
"""

import pytest
import numpy as np
import time

from backend.audio import (
    AudioStreamProcessor, AudioConfig, AudioBuffer, 
    TestToneGenerator, audio_to_bytes, bytes_to_audio
)
from backend.dsp import (
    DSPEngine, DSPConfig, StereoEngine, PhoneDSP,
    PsychoacousticProcessor, RoomSimulator, SceneDetector, EQPreset
)
from backend.sync import SyncEngine, SyncFrame, ClientState, DynamicBuffer


class TestAudioComponents:
    """Tests for audio.py components"""
    
    def test_audio_config_defaults(self):
        config = AudioConfig()
        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.block_size == 1024
        assert config.dtype == 'float32'
    
    def test_audio_buffer_basic(self):
        buffer = AudioBuffer(max_size=10)
        data = np.zeros(100, dtype=np.float32)
        
        # Test put
        assert buffer.put(data) is True
        assert buffer.size == 1
        
        # Test get
        result = buffer.get(timeout=0.1)
        assert result is not None
        assert len(result) == 100
    
    def test_audio_buffer_full(self):
        buffer = AudioBuffer(max_size=2)
        data = np.zeros(100, dtype=np.float32)
        
        assert buffer.put(data) is True
        assert buffer.put(data) is True
        assert buffer.put(data) is False  # Buffer full
    
    def test_test_tone_generator(self):
        gen = TestToneGenerator(sample_rate=44100)
        left, right = gen.generate_stereo_test(1024)
        
        assert len(left) == 1024
        assert len(right) == 1024
        assert left.dtype == np.float32
        assert right.dtype == np.float32
        
        # Verify they are different (different frequencies)
        assert not np.allclose(left, right)
    
    def test_audio_bytes_conversion(self):
        original = np.array([0.5, -0.5, 0.0, 1.0], dtype=np.float32)
        
        # Convert to bytes
        data = audio_to_bytes(original)
        assert isinstance(data, bytes)
        
        # Convert back
        recovered = bytes_to_audio(data)
        assert np.allclose(original, recovered)
    
    def test_audio_stream_processor_test_tone(self):
        config = AudioConfig()
        processor = AudioStreamProcessor(config)
        processor.set_source_test_tone()
        
        left, right = processor.get_next_block()
        
        assert left.shape == (config.block_size,)
        assert right.shape == (config.block_size,)
        assert np.max(np.abs(left)) <= 1.0
        assert np.max(np.abs(right)) <= 1.0


class TestDSPComponents:
    """Tests for dsp.py components"""
    
    def test_dsp_config_defaults(self):
        config = DSPConfig()
        assert config.sample_rate == 44100
        assert config.rms_target == 0.3
        assert config.limiter_threshold == 0.9
    
    def test_dsp_engine_rms_normalize(self):
        dsp = DSPEngine()
        
        # Create quiet signal
        quiet = np.ones(1024, dtype=np.float32) * 0.1
        normalized = dsp.rms_normalize(quiet)
        
        # Should be amplified
        assert np.max(np.abs(normalized)) > np.max(np.abs(quiet))
    
    def test_dsp_engine_soft_limiter(self):
        dsp = DSPEngine()
        
        # Create signal exceeding threshold
        loud = np.ones(1024, dtype=np.float32) * 1.5
        limited = dsp.soft_limiter(loud)
        
        # Should be limited
        assert np.max(np.abs(limited)) <= 1.0
    
    def test_dsp_engine_eq_presets(self):
        dsp = DSPEngine()
        signal = np.sin(np.linspace(0, 10 * np.pi, 1024)).astype(np.float32)
        
        for preset in EQPreset:
            result = dsp.apply_eq_preset(signal, preset)
            assert result.shape == signal.shape
            assert not np.any(np.isnan(result))
    
    def test_dsp_engine_process_master(self):
        dsp = DSPEngine()
        signal = np.sin(np.linspace(0, 10 * np.pi, 1024)).astype(np.float32) * 0.5
        
        result = dsp.process_master(signal, EQPreset.THEATRE)
        
        assert result.shape == signal.shape
        assert not np.any(np.isnan(result))
        assert np.max(np.abs(result)) <= 1.0
    
    def test_stereo_engine_split(self):
        engine = StereoEngine()
        
        # Test stereo input
        stereo = np.column_stack([
            np.ones(100) * 0.5,
            np.ones(100) * -0.5
        ])
        left, right = engine.split_stereo(stereo)
        
        assert np.allclose(left, 0.5)
        assert np.allclose(right, -0.5)
    
    def test_stereo_engine_mid_side(self):
        engine = StereoEngine()
        
        left = np.ones(100) * 0.5
        right = np.ones(100) * 0.3
        
        # Convert to mid-side
        mid, side = engine.to_mid_side(left, right)
        
        # Convert back
        left2, right2 = engine.from_mid_side(mid, side)
        
        assert np.allclose(left, left2)
        assert np.allclose(right, right2)
    
    def test_stereo_engine_widen(self):
        engine = StereoEngine()
        
        left = np.ones(100) * 0.5
        right = np.ones(100) * 0.3
        
        # Widen should increase difference
        l_wide, r_wide = engine.widen_stereo(left, right, width=1.5)
        
        orig_diff = np.mean(np.abs(left - right))
        new_diff = np.mean(np.abs(l_wide - r_wide))
        
        assert new_diff >= orig_diff
    
    def test_phone_dsp_left(self):
        phone_dsp = PhoneDSP()
        signal = np.sin(np.linspace(0, 10 * np.pi, 1024)).astype(np.float32) * 0.5
        
        result = phone_dsp.process_left(signal, room_sim=True)
        
        assert result.shape == signal.shape
        assert not np.any(np.isnan(result))
    
    def test_phone_dsp_right(self):
        phone_dsp = PhoneDSP()
        signal = np.sin(np.linspace(0, 10 * np.pi, 1024)).astype(np.float32) * 0.5
        
        result = phone_dsp.process_right(signal, room_sim=True)
        
        assert result.shape == signal.shape
        assert not np.any(np.isnan(result))
    
    def test_psychoacoustic_haas_effect(self):
        psycho = PsychoacousticProcessor()
        
        left = np.sin(np.linspace(0, 10 * np.pi, 1024)).astype(np.float32)
        right = left.copy()
        
        left_out, right_out = psycho.apply_haas_effect(left, right, delay_ms=1.0)
        
        # Right should be delayed
        assert not np.allclose(right, right_out)
        assert np.allclose(left, left_out)
    
    def test_room_simulator(self):
        room = RoomSimulator()
        signal = np.sin(np.linspace(0, 10 * np.pi, 1024)).astype(np.float32) * 0.5
        
        with_reflections = room.add_early_reflections(signal, is_left=True)
        with_reverb = room.add_subtle_reverb(signal, decay=0.3)
        
        assert with_reflections.shape == signal.shape
        assert with_reverb.shape == signal.shape
    
    def test_scene_detector(self):
        detector = SceneDetector()
        
        # Analyze some audio
        signal = np.sin(np.linspace(0, 10 * np.pi, 1024)).astype(np.float32) * 0.5
        scene_info = detector.analyze(signal)
        
        assert 'rms' in scene_info
        assert 'dialogue_ratio' in scene_info
        assert 'scene_type' in scene_info
        
        params = detector.get_adaptive_params(scene_info)
        assert 'stereo_width' in params
        assert 'haas_delay' in params


class TestSyncComponents:
    """Tests for sync.py components"""
    
    def test_sync_frame_creation(self):
        frame = SyncFrame(
            frame_id=0,
            timestamp_ms=0.0,
            sample_rate=44100,
            samples_count=1024,
            audio_data=b'test',
            channel='left'
        )
        
        assert frame.frame_id == 0
        assert frame.channel == 'left'
        
        d = frame.to_dict()
        assert 'frame_id' in d
        assert 'timestamp_ms' in d
    
    def test_client_state(self):
        client = ClientState(client_id='test', channel='left')
        
        # Update RTT
        client.update_rtt(10.0)
        client.update_rtt(15.0)
        client.update_rtt(12.0)
        
        assert client.avg_rtt > 0
        assert client.rtt_jitter >= 0
    
    def test_sync_engine_frame_creation(self):
        sync = SyncEngine()
        
        left_bytes = b'left_audio'
        right_bytes = b'right_audio'
        
        left_frame, right_frame = sync.create_frame(left_bytes, right_bytes)
        
        assert left_frame.frame_id == 0
        assert right_frame.frame_id == 0
        assert left_frame.channel == 'left'
        assert right_frame.channel == 'right'
        
        # Create another
        left2, right2 = sync.create_frame(left_bytes, right_bytes)
        assert left2.frame_id == 1
    
    def test_sync_engine_client_registration(self):
        sync = SyncEngine()
        
        client1 = sync.register_client('client1', 'left')
        client2 = sync.register_client('client2', 'right')
        
        assert len(sync.clients) == 2
        assert client1.channel == 'left'
        assert client2.channel == 'right'
        
        sync.unregister_client('client1')
        assert len(sync.clients) == 1
    
    def test_sync_engine_ping_handling(self):
        sync = SyncEngine()
        sync.register_client('test', 'left')
        
        response = sync.handle_ping('test', time.time() * 1000)
        
        assert 'server_timestamp' in response
        assert 'rtt' in response
    
    def test_sync_engine_drift_correction(self):
        sync = SyncEngine()
        sync.register_client('test', 'left')
        
        # Report position with some drift
        response = sync.report_playback_position('test', 100, 2500.0)
        
        assert 'drift_ms' in response
        assert 'playback_rate_correction' in response
    
    def test_dynamic_buffer(self):
        buffer = DynamicBuffer(target_size_ms=100.0)
        
        frame = SyncFrame(
            frame_id=0,
            timestamp_ms=0.0,
            sample_rate=44100,
            samples_count=1024,
            audio_data=b'test',
            channel='left'
        )
        
        assert buffer.add_frame(frame) is True
        assert buffer.get_size_ms() > 0
        
        retrieved = buffer.get_frame()
        assert retrieved is not None
        assert retrieved.frame_id == 0


class TestIntegration:
    """Integration tests for the full pipeline"""
    
    def test_full_audio_pipeline(self):
        """Test complete audio processing pipeline"""
        # Audio source
        config = AudioConfig()
        processor = AudioStreamProcessor(config)
        processor.set_source_test_tone()
        
        # Get raw audio
        left_raw, right_raw = processor.get_next_block()
        
        # Master DSP
        dsp = DSPEngine()
        mono = (left_raw + right_raw) / 2
        processed = dsp.process_master(mono, EQPreset.THEATRE)
        
        # Stereo engine
        stereo = StereoEngine()
        left, right = stereo.widen_stereo(
            left_raw * 0.5 + processed * 0.5,
            right_raw * 0.5 + processed * 0.5,
            width=1.2
        )
        
        # Psychoacoustic
        psycho = PsychoacousticProcessor()
        left, right = psycho.apply_haas_effect(left, right, delay_ms=0.8)
        
        # Phone DSP
        phone_dsp = PhoneDSP()
        left_final = phone_dsp.process_left(left, room_sim=True)
        right_final = phone_dsp.process_right(right, room_sim=True)
        
        # Sync frames
        sync = SyncEngine()
        left_bytes = audio_to_bytes(left_final)
        right_bytes = audio_to_bytes(right_final)
        left_frame, right_frame = sync.create_frame(left_bytes, right_bytes)
        
        # Verify output
        assert left_frame.frame_id == right_frame.frame_id
        assert left_frame.timestamp_ms == right_frame.timestamp_ms
        assert left_frame.channel == 'left'
        assert right_frame.channel == 'right'
        
        # Verify audio integrity
        left_recovered = bytes_to_audio(left_bytes)
        right_recovered = bytes_to_audio(right_bytes)
        
        assert np.allclose(left_final, left_recovered)
        assert np.allclose(right_final, right_recovered)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
