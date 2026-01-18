# TwinWave - Dual Phone Theatre Audio System

A professional-grade audio streaming system that converts TWO ANDROID PHONES into a synchronized, stereo cinema sound system.

## 🎬 Core Concept

- **One phone = LEFT speaker** (Bass emphasis, warm sound)
- **One phone = RIGHT speaker** (Treble emphasis, crisp sound)
- Both phones receive the SAME MASTER STREAM
- Stereo separation + psychoacoustic tricks create a theatre illusion

## 🏗️ System Architecture

```
Audio Source (File/Mic/Test Tone)
         │
         ▼
┌─────────────────────────────────┐
│     MASTER DSP ENGINE           │
│  • RMS Normalization            │
│  • Soft Limiter (cinema punch)  │
│  • Dialogue Enhancement         │
│  • EQ Presets                   │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│     STEREO ENGINE               │
│  • True L/R Split               │
│  • Mid-Side Processing          │
│  • Stereo Width Control         │
└─────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ LEFT  │ │ RIGHT │
│ PHONE │ │ PHONE │
│  DSP  │ │  DSP  │
└───────┘ └───────┘
    │         │
    ▼         ▼
┌─────────────────────────────────┐
│     TIME-SYNC ENGINE            │
│  • Timestamped Frames           │
│  • RTT Measurement              │
│  • Drift Correction             │
│  • Dynamic Buffer Sizing        │
└─────────────────────────────────┘
         │
         ▼
   WebSocket Streaming
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ LEFT  │ │ RIGHT │
│ WEB   │ │ WEB   │
│ AUDIO │ │ AUDIO │
└───────┘ └───────┘
```

## 🎛️ Features

### DSP Processing
- **RMS Normalization**: Smooth gain control to prevent volume jumps
- **Soft Limiter**: Cinema-quality punch without harsh clipping
- **Dialogue Enhancement**: 1-4kHz band boost for clear speech

### Cinema EQ Presets
- **Theatre**: Enhanced bass and presence for immersive sound
- **Night Mode**: Reduced bass, clear dialogue for quiet listening
- **Dialogue Boost**: Focus on speech frequencies
- **Bass Monster**: Maximum low-end impact

### Psychoacoustic Width Expander
- Haas effect (0.5-1ms delay)
- Phase offset between channels
- Creates perception of wider sound stage

### Room Simulation
- Early reflections (different profile per phone)
- Subtle reverb tail
- Creates sense of space

### Phone-Specific DSP
- **LEFT Phone**: Bass emphasis, low-pass dominance, warm character
- **RIGHT Phone**: Treble emphasis, high-pass dominance, crisp character

### Time Synchronization
- Each audio chunk includes Frame ID and timestamp
- RTT measurement for optimal buffer sizing
- Drift correction via playbackRate adjustments
- Automatic resync every 5 seconds

## 📁 Project Structure

```
TwinWave/
├── backend/
│   ├── __init__.py       # Package initialization
│   ├── audio.py          # Audio capture and file loading
│   ├── dsp.py            # DSP filters and processing
│   └── sync.py           # Time synchronization engine
├── static/
│   └── js/
│       └── audio-player.js  # Web Audio API client
├── templates/
│   ├── index.html        # Phone selection page
│   ├── player.html       # Phone player page
│   └── control.html      # Control panel
├── server.py             # Flask/SocketIO server
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python server.py
```

The server will start on `http://0.0.0.0:5000`

### 3. Connect Phones

1. **LEFT Phone**: Open `http://<server-ip>:5000/left`
2. **RIGHT Phone**: Open `http://<server-ip>:5000/right`
3. **Control Panel**: Open `http://<server-ip>:5000/control` on any device

### 4. Start Streaming

- Use the Control Panel to start audio streaming
- Upload an audio file or use the test tone
- Adjust EQ presets and DSP options

## 🛠️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Server port number | `5000` |
| `SSL_CERT` | Path to SSL certificate file (.pem) | None |
| `SSL_KEY` | Path to SSL private key file (.pem) | None |
| `SECRET_KEY` | Flask secret key | Auto-generated |

### HTTPS/SSL Setup

To enable HTTPS, set the `SSL_CERT` and `SSL_KEY` environment variables:

```bash
# Generate a self-signed certificate for testing
openssl req -x509 -nodes -newkey rsa:2048 \
  -keyout key.pem -out cert.pem -days 365 \
  -subj "/CN=localhost"

# Start server with HTTPS
SSL_CERT=cert.pem SSL_KEY=key.pem python server.py
```

For production, use certificates from a trusted CA (e.g., Let's Encrypt).

### Audio Settings (in `server.py`)

```python
SAMPLE_RATE = 44100  # Audio sample rate
BLOCK_SIZE = 1024    # Samples per frame (~23ms at 44.1kHz)
CHANNELS = 2         # Stereo input
```

### DSP Settings (in `backend/dsp.py`)

```python
@dataclass
class DSPConfig:
    sample_rate: int = 44100
    block_size: int = 1024
    rms_target: float = 0.3           # Target RMS level
    limiter_threshold: float = 0.9     # Limiter threshold
    limiter_ratio: float = 10.0        # Compression ratio
    dialogue_band_low: float = 1000.0  # Dialogue band start
    dialogue_band_high: float = 4000.0 # Dialogue band end
    width_phase_offset_ms: float = 0.8 # Haas effect delay
```

## 📡 API Endpoints

### HTTP Routes
- `GET /` - Phone selection page
- `GET /left` - Left phone player
- `GET /right` - Right phone player
- `GET /control` - Control panel
- `GET /api/status` - System status
- `GET /api/presets` - Available EQ presets
- `POST /api/set_preset` - Set EQ preset
- `POST /api/upload` - Upload audio file
- `POST /api/source` - Set audio source

### WebSocket Events
- `register` - Register as left/right phone
- `ping_sync` - Measure RTT
- `report_position` - Report playback position
- `start_stream` - Start audio streaming
- `stop_stream` - Stop streaming
- `set_options` - Update DSP options

## 🔧 Latency Optimization Tips

1. **Use WiFi 5GHz** - Less interference, lower latency
2. **Same Router** - Both phones on same access point
3. **Reduce Buffer** - Lower buffer = lower latency (at cost of stability)
4. **Close Other Apps** - Reduce CPU competition
5. **Keep Screen On** - Use wake lock to prevent throttling

## 🐛 Troubleshooting

### No Audio Playing
- Check if AudioContext is suspended (tap screen to resume on mobile)
- Verify both phones are registered in Control Panel
- Check browser console for errors

### Audio Out of Sync
- Increase buffer size for more stability
- Check WiFi signal strength
- Resync will automatically occur every 5 seconds

### Audio Crackling
- Reduce DSP processing (disable room simulation)
- Increase buffer size
- Check CPU usage on server

### Connection Drops
- Check WiFi stability
- Ensure phones don't go to sleep (use wake lock)

## 📜 License

MIT License - Feel free to use and modify!

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.