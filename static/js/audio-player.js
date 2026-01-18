/**
 * TwinWave Audio Player
 * Web Audio API-based audio player with precision timing and sync
 */

class TwinWavePlayer {
    constructor(channel) {
        this.channel = channel; // 'left' or 'right'
        this.socket = null;
        this.audioContext = null;
        this.gainNode = null;
        this.analyser = null;
        
        // Playback state
        this.isPlaying = false;
        this.isConnected = false;
        this.isReady = false;
        
        // Buffer management
        this.audioBuffer = [];
        this.bufferSizeMs = 100;
        this.minBufferMs = 50;
        this.targetBufferMs = 100;
        
        // Sync state
        this.currentFrameId = 0;
        this.playbackStartTime = 0;
        this.sampleRate = 44100;
        this.blockSize = 1024;
        this.driftCorrection = 1.0;
        
        // Stats
        this.stats = {
            latency: 0,
            bufferSize: 0,
            frameId: 0,
            drift: 0,
            synced: false
        };
        
        // Callbacks
        this.onStatusChange = null;
        this.onStatsUpdate = null;
        this.onVisualizerData = null;
        
        // Visualization
        this.visualizerData = new Float32Array(128);
        
        // Ping interval
        this.pingInterval = null;
        this.reportInterval = null;
    }

    async connect() {
        this._setStatus('Connecting...', 'Establishing WebSocket connection');
        
        try {
            // Check if Socket.IO is loaded
            if (typeof io === 'undefined') {
                throw new Error('Socket.IO library not loaded. Please check your network connection.');
            }
            
            // Initialize Audio Context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 44100,
                latencyHint: 'interactive'
            });
            
            // Create audio graph
            this.gainNode = this.audioContext.createGain();
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = 256;
            
            this.gainNode.connect(this.analyser);
            this.analyser.connect(this.audioContext.destination);
            
            // Resume audio context (needed for mobile)
            await this._resumeAudioContext();
            
            // Connect to server
            this.socket = io({
                transports: ['websocket'],
                upgrade: false
            });
            
            this._setupSocketEvents();
            
        } catch (error) {
            this._setStatus('Error', error.message);
            console.error('Connection error:', error);
        }
    }

    _setupSocketEvents() {
        this.socket.on('connect', () => {
            console.log('Socket connected');
            this._setStatus('Connected', 'Registering as ' + this.channel + ' speaker');
            
            // Register with channel
            this.socket.emit('register', { channel: this.channel });
        });

        this.socket.on('registered', (data) => {
            console.log('Registered:', data);
            this.sampleRate = data.sample_rate;
            this.blockSize = data.block_size;
            this.isConnected = true;
            
            this._setStatus('Connected', `${this.channel.toUpperCase()} speaker ready`);
            
            // Start ping interval for RTT measurement
            this._startPingInterval();
            
            // Signal ready
            this.socket.emit('ready', {});
        });

        this.socket.on('audio_frame', (data) => {
            this._handleAudioFrame(data);
        });

        this.socket.on('pong_sync', (data) => {
            this.stats.latency = data.rtt / 2;
            this.targetBufferMs = data.recommended_buffer;
        });

        this.socket.on('sync_correction', (data) => {
            this.stats.drift = data.drift_ms;
            this.driftCorrection = data.playback_rate_correction;
            this.stats.synced = !data.needs_resync;
        });

        this.socket.on('resync', (data) => {
            console.log('Resync received:', data);
            this._performResync(data);
        });

        this.socket.on('stream_started', (data) => {
            console.log('Stream started');
            this._setStatus('Buffering', 'Filling audio buffer...');
            this.playbackStartTime = data.timestamp;
            this.isPlaying = true;
        });

        this.socket.on('stream_stopped', () => {
            console.log('Stream stopped');
            this._setStatus('Connected', 'Stream stopped');
            this.isPlaying = false;
            this.audioBuffer = [];
        });

        this.socket.on('disconnect', () => {
            console.log('Socket disconnected');
            this._setStatus('Disconnected', 'Connection lost');
            this.isConnected = false;
            this._stopPingInterval();
        });

        this.socket.on('error', (error) => {
            console.error('Socket error:', error);
            this._setStatus('Error', error.message);
        });
    }

    _handleAudioFrame(data) {
        // Decode base64 audio data
        const audioData = this._base64ToFloat32(data.audio);
        
        // Add to buffer
        this.audioBuffer.push({
            frameId: data.frame_id,
            timestamp: data.timestamp_ms,
            audio: audioData
        });

        this.currentFrameId = data.frame_id;
        this.stats.frameId = data.frame_id;
        
        // Update buffer stats
        const bufferDurationMs = (this.audioBuffer.length * this.blockSize / this.sampleRate) * 1000;
        this.stats.bufferSize = bufferDurationMs;
        
        // Start playback when buffer is full enough
        if (!this.isReady && bufferDurationMs >= this.targetBufferMs) {
            this.isReady = true;
            this._setStatus('Playing', 'Audio streaming');
            this._startPlayback();
            this._startReportInterval();
        }
        
        // Update visualizer
        this._updateVisualizer(audioData);
        
        // Update stats
        if (this.onStatsUpdate) {
            this.onStatsUpdate(this.stats);
        }
    }

    _startPlayback() {
        const scheduleNextBuffer = () => {
            if (!this.isPlaying || this.audioBuffer.length === 0) {
                if (this.isPlaying) {
                    // Buffer underrun
                    this._setStatus('Buffering', 'Buffer underrun, refilling...');
                    this.isReady = false;
                }
                return;
            }
            
            const frame = this.audioBuffer.shift();
            
            // Create audio buffer
            const audioBuffer = this.audioContext.createBuffer(
                1, // Mono - we're either left or right
                frame.audio.length,
                this.sampleRate
            );
            
            // Copy audio data
            audioBuffer.copyToChannel(frame.audio, 0);
            
            // Create source
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            
            // Apply drift correction via playback rate
            source.playbackRate.value = this.driftCorrection;
            
            // Connect to gain node
            source.connect(this.gainNode);
            
            // Schedule playback
            source.start();
            
            // Calculate when to schedule next
            const bufferDuration = frame.audio.length / this.sampleRate;
            const adjustedDuration = bufferDuration / this.driftCorrection;
            
            // Schedule next buffer slightly before current ends
            setTimeout(scheduleNextBuffer, adjustedDuration * 1000 * 0.95);
        };
        
        scheduleNextBuffer();
    }

    _performResync(data) {
        // Clear current buffer for clean resync
        this.audioBuffer = [];
        this.isReady = false;
        this._setStatus('Buffering', 'Resyncing...');
        
        // Apply sync parameters
        if (data.clients && data.clients[this.socket.id]) {
            const clientData = data.clients[this.socket.id];
            this.driftCorrection = clientData.playback_rate;
            this.targetBufferMs = clientData.buffer_size;
        }
    }

    _base64ToFloat32(base64) {
        // Efficient base64 to Float32Array conversion using Uint8Array.from
        const binaryString = atob(base64);
        const bytes = Uint8Array.from(binaryString, c => c.charCodeAt(0));
        return new Float32Array(bytes.buffer);
    }

    _updateVisualizer(audioData) {
        // Simple RMS-based visualization
        const blockSize = Math.floor(audioData.length / this.visualizerData.length);
        
        for (let i = 0; i < this.visualizerData.length; i++) {
            let sum = 0;
            for (let j = 0; j < blockSize; j++) {
                const sample = audioData[i * blockSize + j] || 0;
                sum += sample * sample;
            }
            this.visualizerData[i] = Math.sqrt(sum / blockSize);
        }
        
        if (this.onVisualizerData) {
            this.onVisualizerData(this.visualizerData);
        }
    }

    _startPingInterval() {
        // Ping every 2 seconds for RTT measurement
        this.pingInterval = setInterval(() => {
            if (this.isConnected) {
                this.socket.emit('ping_sync', {
                    timestamp: Date.now()
                });
            }
        }, 2000);
    }

    _stopPingInterval() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    _startReportInterval() {
        // Report playback position every 500ms
        this.reportInterval = setInterval(() => {
            if (this.isConnected && this.isPlaying) {
                this.socket.emit('report_position', {
                    frame_id: this.currentFrameId,
                    playback_time: (this.currentFrameId * this.blockSize / this.sampleRate) * 1000
                });
            }
        }, 500);
    }

    async _resumeAudioContext() {
        if (this.audioContext.state === 'suspended') {
            // Try to resume immediately
            await this.audioContext.resume();
            
            // Also add touch/click handler for mobile
            const resumeOnInteraction = async () => {
                await this.audioContext.resume();
                document.removeEventListener('touchstart', resumeOnInteraction);
                document.removeEventListener('click', resumeOnInteraction);
            };
            
            document.addEventListener('touchstart', resumeOnInteraction, { once: true });
            document.addEventListener('click', resumeOnInteraction, { once: true });
        }
    }

    _setStatus(status, detail) {
        if (this.onStatusChange) {
            this.onStatusChange(status, detail);
        }
    }

    setVolume(value) {
        if (this.gainNode) {
            this.gainNode.gain.value = Math.max(0, Math.min(1, value));
        }
    }

    disconnect() {
        this._stopPingInterval();
        if (this.reportInterval) {
            clearInterval(this.reportInterval);
        }
        if (this.socket) {
            this.socket.disconnect();
        }
        if (this.audioContext) {
            this.audioContext.close();
        }
    }
}

// Export for use
if (typeof window !== 'undefined') {
    window.TwinWavePlayer = TwinWavePlayer;
}
