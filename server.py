"""
TwinWave Flask Server
WebSocket-based audio streaming server for dual-phone theatre audio
"""

import os
import time
import base64
import threading
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import numpy as np

from backend.audio import AudioStreamProcessor, AudioConfig, audio_to_bytes
from backend.dsp import (
    DSPEngine, DSPConfig, StereoEngine, PhoneDSP, 
    PsychoacousticProcessor, SceneDetector, EQPreset
)
from backend.sync import SyncEngine, SyncFrame


# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'twinwave-secret-key-2024')

# Initialize SocketIO with eventlet
socketio = SocketIO(app, 
                    cors_allowed_origins="*",
                    async_mode='threading',
                    ping_timeout=60,
                    ping_interval=25)

# Audio configuration
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
CHANNELS = 2

# Initialize audio components
audio_config = AudioConfig(
    sample_rate=SAMPLE_RATE,
    channels=CHANNELS,
    block_size=BLOCK_SIZE
)

dsp_config = DSPConfig(
    sample_rate=SAMPLE_RATE,
    block_size=BLOCK_SIZE
)

# Global components
audio_processor = AudioStreamProcessor(audio_config)
dsp_engine = DSPEngine(dsp_config)
stereo_engine = StereoEngine(SAMPLE_RATE)
phone_dsp = PhoneDSP(SAMPLE_RATE)
psycho_processor = PsychoacousticProcessor(SAMPLE_RATE)
scene_detector = SceneDetector(SAMPLE_RATE)
sync_engine = SyncEngine(SAMPLE_RATE, BLOCK_SIZE)

# Streaming state
streaming_active = False
streaming_thread = None
current_preset = EQPreset.THEATRE
psycho_enabled = True
room_sim_enabled = True
scene_adaptive = False


# ─────────────────────────────────────────────────────────────
# HTTP Routes
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Main page - phone selection"""
    return render_template('index.html')


@app.route('/left')
def left_phone():
    """Left phone client page"""
    return render_template('player.html', channel='left')


@app.route('/right')
def right_phone():
    """Right phone client page"""
    return render_template('player.html', channel='right')


@app.route('/control')
def control_panel():
    """Control panel for managing the system"""
    return render_template('control.html')


@app.route('/api/status')
def api_status():
    """Get system status"""
    sync_status = sync_engine.check_sync_status()
    buffer_config = sync_engine.get_optimal_buffer_config()
    
    return jsonify({
        'streaming': streaming_active,
        'sample_rate': SAMPLE_RATE,
        'block_size': BLOCK_SIZE,
        'preset': current_preset.value,
        'psycho_enabled': psycho_enabled,
        'room_sim_enabled': room_sim_enabled,
        'scene_adaptive': scene_adaptive,
        'sync': sync_status,
        'buffer': buffer_config,
        'source': audio_processor.source.value
    })


@app.route('/api/presets')
def api_presets():
    """Get available EQ presets"""
    return jsonify({
        'presets': [p.value for p in EQPreset]
    })


@app.route('/api/set_preset', methods=['POST'])
def api_set_preset():
    """Set EQ preset"""
    global current_preset
    data = request.get_json()
    preset_name = data.get('preset', 'theatre')
    
    try:
        current_preset = EQPreset(preset_name)
        return jsonify({'success': True, 'preset': current_preset.value})
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid preset'}), 400


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Upload audio file"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    try:
        data = file.read()
        if audio_processor.set_source_file_bytes(data):
            return jsonify({
                'success': True,
                'duration_ms': audio_processor.get_duration_ms()
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to load file'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/source', methods=['POST'])
def api_set_source():
    """Set audio source"""
    data = request.get_json()
    source = data.get('source', 'test_tone')
    
    if source == 'test_tone':
        audio_processor.set_source_test_tone()
        return jsonify({'success': True, 'source': 'test_tone'})
    elif source == 'microphone':
        if audio_processor.set_source_microphone():
            return jsonify({'success': True, 'source': 'microphone'})
        else:
            return jsonify({'success': False, 'error': 'Microphone not available'}), 400
    else:
        return jsonify({'success': False, 'error': 'Invalid source'}), 400


# ─────────────────────────────────────────────────────────────
# WebSocket Events
# ─────────────────────────────────────────────────────────────

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    emit('connected', {'sid': request.sid})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")
    sync_engine.unregister_client(request.sid)


@socketio.on('register')
def handle_register(data):
    """Register client as left or right phone"""
    channel = data.get('channel', 'left')
    
    if channel not in ['left', 'right']:
        emit('error', {'message': 'Invalid channel. Use "left" or "right"'})
        return
    
    # Register with sync engine
    client = sync_engine.register_client(request.sid, channel)
    
    # Join channel room
    join_room(channel)
    
    emit('registered', {
        'channel': channel,
        'client_id': request.sid,
        'sample_rate': SAMPLE_RATE,
        'block_size': BLOCK_SIZE,
        'buffer_config': sync_engine.get_optimal_buffer_config()
    })
    
    # Notify other clients
    socketio.emit('client_joined', {
        'channel': channel,
        'total_clients': len(sync_engine.clients)
    }, broadcast=True)


@socketio.on('ping_sync')
def handle_ping_sync(data):
    """Handle sync ping from client"""
    client_timestamp = data.get('timestamp', time.time() * 1000)
    
    response = sync_engine.handle_ping(request.sid, client_timestamp)
    emit('pong_sync', response)


@socketio.on('report_position')
def handle_report_position(data):
    """Handle playback position report from client"""
    frame_id = data.get('frame_id', 0)
    playback_time = data.get('playback_time', 0)
    
    correction = sync_engine.report_playback_position(
        request.sid, frame_id, playback_time
    )
    emit('sync_correction', correction)


@socketio.on('ready')
def handle_ready(data):
    """Client signals it's ready to receive audio"""
    if request.sid in sync_engine.clients:
        sync_engine.clients[request.sid].is_ready = True
    
    # Check if both clients are ready
    sync_status = sync_engine.check_sync_status()
    if sync_status.get('synced') or len(sync_engine.clients) >= 2:
        socketio.emit('all_ready', sync_status, broadcast=True)


@socketio.on('start_stream')
def handle_start_stream(data=None):
    """Start audio streaming"""
    global streaming_active, streaming_thread
    
    if streaming_active:
        emit('error', {'message': 'Streaming already active'})
        return
    
    streaming_active = True
    streaming_thread = threading.Thread(target=stream_audio_loop)
    streaming_thread.daemon = True
    streaming_thread.start()
    
    socketio.emit('stream_started', {
        'timestamp': time.time() * 1000
    }, broadcast=True)


@socketio.on('stop_stream')
def handle_stop_stream(data=None):
    """Stop audio streaming"""
    global streaming_active
    streaming_active = False
    
    socketio.emit('stream_stopped', {}, broadcast=True)


@socketio.on('set_options')
def handle_set_options(data):
    """Set DSP options"""
    global psycho_enabled, room_sim_enabled, scene_adaptive, current_preset
    
    if 'psycho' in data:
        psycho_enabled = data['psycho']
    if 'room_sim' in data:
        room_sim_enabled = data['room_sim']
    if 'scene_adaptive' in data:
        scene_adaptive = data['scene_adaptive']
    if 'preset' in data:
        try:
            current_preset = EQPreset(data['preset'])
        except ValueError:
            pass
    
    socketio.emit('options_updated', {
        'psycho': psycho_enabled,
        'room_sim': room_sim_enabled,
        'scene_adaptive': scene_adaptive,
        'preset': current_preset.value
    }, broadcast=True)


# ─────────────────────────────────────────────────────────────
# Audio Streaming Loop
# ─────────────────────────────────────────────────────────────

def stream_audio_loop():
    """Main audio streaming loop"""
    global streaming_active
    
    # Calculate timing
    block_duration = BLOCK_SIZE / SAMPLE_RATE
    
    print(f"Starting audio stream (block duration: {block_duration*1000:.2f}ms)")
    
    while streaming_active:
        loop_start = time.time()
        
        try:
            # Get raw audio block
            left_raw, right_raw = audio_processor.get_next_block()
            
            # Mix to mono for master processing
            mono = (left_raw + right_raw) / 2
            
            # Apply master DSP
            preset = current_preset
            if scene_adaptive:
                scene_info = scene_detector.analyze(mono)
                params = scene_detector.get_adaptive_params(scene_info)
                # Adjust preset based on scene (simplified)
                if scene_info['scene_type'] == 'dialogue':
                    preset = EQPreset.DIALOGUE_BOOST
            
            processed = dsp_engine.process_master(mono, preset)
            
            # Split to stereo with widening
            left, right = stereo_engine.widen_stereo(
                left_raw * 0.5 + processed * 0.5,
                right_raw * 0.5 + processed * 0.5,
                width=1.2
            )
            
            # Apply psychoacoustic processing
            if psycho_enabled:
                left, right = psycho_processor.apply_haas_effect(
                    left, right, delay_ms=0.8
                )
            
            # Apply phone-specific DSP
            left_final = phone_dsp.process_left(left, room_sim=room_sim_enabled)
            right_final = phone_dsp.process_right(right, room_sim=room_sim_enabled)
            
            # Create sync frames
            left_bytes = audio_to_bytes(left_final)
            right_bytes = audio_to_bytes(right_final)
            
            left_frame, right_frame = sync_engine.create_frame(left_bytes, right_bytes)
            
            # Send to respective channels
            # Encode audio as base64 for JSON transmission
            left_data = {
                **left_frame.to_dict(),
                'audio': base64.b64encode(left_bytes).decode('ascii')
            }
            right_data = {
                **right_frame.to_dict(),
                'audio': base64.b64encode(right_bytes).decode('ascii')
            }
            
            socketio.emit('audio_frame', left_data, room='left')
            socketio.emit('audio_frame', right_data, room='right')
            
            # Check for periodic resync
            if sync_engine.needs_resync():
                sync_data = sync_engine.perform_resync()
                socketio.emit('resync', sync_data, broadcast=True)
            
            # Maintain timing
            elapsed = time.time() - loop_start
            sleep_time = max(0, block_duration - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
        except Exception as e:
            print(f"Streaming error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.01)
    
    print("Audio stream stopped")


# ─────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # SSL/TLS configuration via environment variables
    ssl_cert = os.environ.get('SSL_CERT')
    ssl_key = os.environ.get('SSL_KEY')
    port = int(os.environ.get('PORT', 5000))
    
    # Determine if SSL is enabled
    ssl_context = None
    protocol = 'http'
    if ssl_cert and ssl_key:
        if os.path.isfile(ssl_cert) and os.path.isfile(ssl_key):
            ssl_context = (ssl_cert, ssl_key)
            protocol = 'https'
        else:
            print(f"Warning: SSL certificate or key file not found.")
            print(f"  SSL_CERT: {ssl_cert} (exists: {os.path.isfile(ssl_cert)})")
            print(f"  SSL_KEY: {ssl_key} (exists: {os.path.isfile(ssl_key)})")
            print("Falling back to HTTP mode.")
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    TwinWave Audio System                     ║
║           Dual Phone Theatre Audio Experience                ║
╠══════════════════════════════════════════════════════════════╣
║  Server running on {protocol}://0.0.0.0:{port:<27}║
║                                                              ║
║  1. Open /left on LEFT phone                                 ║
║  2. Open /right on RIGHT phone                               ║
║  3. Open /control on any device to manage                    ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    if ssl_context:
        socketio.run(app, host='0.0.0.0', port=port, debug=False, ssl_context=ssl_context)
    else:
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
