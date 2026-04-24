// ============================================
// Ollama Agent - Modern Frontend Application
// ============================================

class OllamaAgentApp {
    constructor() {
        this.socket = null;
        this.currentModel = '';
        this.sessionId = this.generateSessionId();
        this.isConnected = false;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isGenerating = false;
        this.thinkingEnabled = true;
        this.cameraStream = null;
        this.audioTimer = null;
        this.audioSeconds = 0;
        this.currentAssistantMessage = null;
        this.theme = localStorage.getItem('theme') || 'dark';

        this.init();
    }

    init() {
        this.initTheme();
        this.initParticles();
        this.initSocket();
        this.initEventListeners();
        this.loadModels();
        this.loadSettings();
        this.setupDragAndDrop();
        this.initCodeInterpreter();
    }

    // Theme Management
    initTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);
        this.updateThemeIcon();
    }

    toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', this.theme);
        localStorage.setItem('theme', this.theme);
        this.updateThemeIcon();
    }

    updateThemeIcon() {
        const btn = document.getElementById('theme-toggle');
        if (btn) {
            btn.innerHTML = `<i class="fas fa-${this.theme === 'dark' ? 'sun' : 'moon'}"></i>`;
        }
    }

    // Particle Background
    initParticles() {
        const canvas = document.getElementById('particle-bg');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        let particles = [];
        let animationId = null;
        let isActive = true;

        const resize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };
        resize();
        window.addEventListener('resize', resize);

        const createParticles = () => {
            particles = [];
            const count = Math.min(Math.floor(window.innerWidth * window.innerHeight / 15000), 80);
            for (let i = 0; i < count; i++) {
                particles.push({
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    vx: (Math.random() - 0.5) * 0.3,
                    vy: (Math.random() - 0.5) * 0.3,
                    radius: Math.random() * 2 + 0.5,
                    opacity: Math.random() * 0.5 + 0.1
                });
            }
        };
        createParticles();

        const draw = () => {
            if (!isActive) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            const isDark = this.theme === 'dark';
            const color = isDark ? '147, 197, 253' : '99, 102, 241';

            particles.forEach((p, i) => {
                p.x += p.vx;
                p.y += p.vy;

                if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
                if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${color}, ${p.opacity})`;
                ctx.fill();

                // Draw connections
                for (let j = i + 1; j < particles.length; j++) {
                    const p2 = particles[j];
                    const dx = p.x - p2.x;
                    const dy = p.y - p2.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist < 120) {
                        ctx.beginPath();
                        ctx.moveTo(p.x, p.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.strokeStyle = `rgba(${color}, ${0.1 * (1 - dist / 120)})`;
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                }
            });

            animationId = requestAnimationFrame(draw);
        };
        draw();

        // Pause when tab hidden
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                isActive = false;
                if (animationId) cancelAnimationFrame(animationId);
            } else {
                isActive = true;
                draw();
            }
        });
    }

    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    // Socket.IO
    initSocket() {
        this.socket = io();

        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.isConnected = true;
            this.updateConnectionStatus(true);
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.isConnected = false;
            this.updateConnectionStatus(false);
        });

        this.socket.on('connected', (data) => {
            console.log('Server response:', data);
        });

        this.socket.on('message_start', () => {
            this.showTypingIndicator();
        });

        this.socket.on('message_chunk', (data) => {
            this.updateAssistantMessage(data.delta, data.full_response);
        });

        this.socket.on('message_end', () => {
            this.hideTypingIndicator();
        });

        this.socket.on('message_response', (data) => {
            this.hideTypingIndicator();
            if (data.error) {
                this.showError(data.error);
            } else {
                this.addMessage('assistant', data.response);
            }
        });

        this.socket.on('error', (data) => {
            this.hideTypingIndicator();
            this.showError(data.error);
        });

        this.socket.on('generation_stopped', () => {
            this.isGenerating = false;
            this.hideStopButton();
            this.hideTypingIndicator();
        });
    }

    updateConnectionStatus(connected) {
        const dot = document.getElementById('pulse-dot');
        const text = document.getElementById('connection-text');
        const pill = document.getElementById('connection-pill');

        if (connected) {
            dot.className = 'pulse-dot connected';
            text.textContent = '已连接';
            pill.style.borderColor = 'rgba(16, 185, 129, 0.3)';
        } else {
            dot.className = 'pulse-dot disconnected';
            text.textContent = '连接断开';
            pill.style.borderColor = 'rgba(239, 68, 68, 0.3)';
        }
    }

    // Event Listeners
    initEventListeners() {
        // Theme toggle
        document.getElementById('theme-toggle').addEventListener('click', () => this.toggleTheme());

        // Send message
        document.getElementById('send-btn').addEventListener('click', () => this.sendMessage());
        document.getElementById('message-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        const textarea = document.getElementById('message-input');
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });

        // Stop generation
        document.getElementById('stop-btn').addEventListener('click', () => this.stopGeneration());

        // Clear chat
        document.getElementById('clear-chat-btn').addEventListener('click', () => this.clearChat());

        // Settings modal
        document.getElementById('settings-btn').addEventListener('click', () => this.showSettings());
        document.getElementById('close-settings').addEventListener('click', () => this.hideSettings());
        document.getElementById('save-settings-btn').addEventListener('click', () => this.saveSettings());
        document.getElementById('test-connection-btn').addEventListener('click', () => this.testConnection());

        // Close modal on backdrop click
        document.getElementById('settings-modal').addEventListener('click', (e) => {
            if (e.target === e.currentTarget) this.hideSettings();
        });

        // Model select
        document.getElementById('model-select').addEventListener('change', (e) => {
            this.currentModel = e.target.value;
            this.updateModelInfo();
        });

        // Thinking toggle
        document.getElementById('thinking-toggle').addEventListener('change', (e) => {
            this.thinkingEnabled = e.target.checked;
        });

        // Camera
        document.getElementById('camera-btn').addEventListener('click', () => this.toggleCamera());
        document.getElementById('capture-btn').addEventListener('click', () => this.capturePhoto());
        document.getElementById('close-camera-btn').addEventListener('click', () => this.closeCamera());

        // File upload
        document.getElementById('file-upload').addEventListener('change', (e) => this.handleFileUpload(e));

        // Audio
        document.getElementById('audio-record-btn').addEventListener('click', () => this.toggleAudioRecording());
        document.getElementById('stop-audio-btn').addEventListener('click', () => this.stopAudioRecording());

        // Error toast
        document.getElementById('close-error-btn').addEventListener('click', () => this.hideError());
    }

    // Load Models
    async loadModels() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();

            if (data.success) {
                this.populateModelSelect(data.models);
                this.updateConnectionStatus(data.ollama_connected);
                this.updateSystemInfo(data);
            } else {
                this.showError('加载模型失败: ' + data.error);
            }
        } catch (error) {
            console.error('加载模型失败:', error);
            this.showError('加载模型失败，请检查网络连接');
        }
    }

    populateModelSelect(models) {
        const select = document.getElementById('model-select');
        select.innerHTML = '<option value="">请选择模型</option>';

        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.name;
            option.textContent = this.formatModelName(model.name);
            option.dataset.isVision = model.is_vision;
            option.dataset.isAudio = model.is_audio;
            option.dataset.size = model.size || 0;
            select.appendChild(option);
        });

        if (models.length > 0) {
            select.value = models[0].name;
            this.currentModel = models[0].name;
            this.updateModelInfo();
        }
    }

    formatModelName(name) {
        const parts = name.split(':');
        if (parts.length > 1) {
            return parts[0] + ' (' + parts[1] + ')';
        }
        return name;
    }

    updateModelInfo() {
        const select = document.getElementById('model-select');
        const selectedOption = select.options[select.selectedIndex];
        const badge = document.getElementById('model-badge');
        const sizeEl = document.getElementById('model-size');

        if (!selectedOption || !selectedOption.value) {
            badge.textContent = '文本';
            badge.className = 'model-badge';
            sizeEl.textContent = '--';
            return;
        }

        const isVision = selectedOption.dataset.isVision === 'true';
        const isAudio = selectedOption.dataset.isAudio === 'true';

        if (isVision) {
            badge.textContent = '视觉';
            badge.className = 'model-badge vision';
        } else if (isAudio) {
            badge.textContent = '音频';
            badge.className = 'model-badge audio';
        } else {
            badge.textContent = '文本';
            badge.className = 'model-badge';
        }

        sizeEl.textContent = this.formatFileSize(selectedOption.dataset.size || 0);
    }

    formatFileSize(bytes) {
        if (bytes === 0 || !bytes) return '未知大小';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Send Message
    sendMessage() {
        const input = document.getElementById('message-input');
        const message = input.value.trim();

        if (!message || !this.isConnected || this.isGenerating) {
            return;
        }

        this.addMessage('user', message);
        input.value = '';
        input.style.height = 'auto';

        this.isGenerating = true;
        this.showStopButton();

        this.socket.emit('send_message', {
            session_id: this.sessionId,
            message: message,
            model: this.currentModel,
            thinking: this.thinkingEnabled
        });
    }

    // Add Message
    addMessage(role, content, metadata = {}) {
        const chatMessages = document.getElementById('chat-messages');

        // Remove welcome screen
        const welcomeScreen = chatMessages.querySelector('.welcome-screen');
        if (welcomeScreen) {
            welcomeScreen.remove();
        }

        const messageEl = document.createElement('div');
        messageEl.className = `message ${role}`;

        const icon = role === 'user' ? 'fa-user' : 'fa-robot';
        const sender = role === 'user' ? '你' : 'AI助手';
        const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });

        messageEl.innerHTML = `
            <div class="message-avatar">
                <i class="fas ${icon}"></i>
            </div>
            <div class="message-body">
                <div class="message-header">
                    <span class="message-sender">${sender}</span>
                    <span class="message-time">${time}</span>
                </div>
                <div class="message-bubble">${this.formatMessage(content)}</div>
            </div>
        `;

        chatMessages.appendChild(messageEl);
        this.scrollToBottom();

        return messageEl;
    }

    formatMessage(content) {
        if (!content) return '';

        // Escape HTML first
        let formatted = content
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Code blocks
        formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            return `<pre data-language="${lang || 'code'}"><code>${code.trim()}</code></pre>`;
        });

        // Inline code
        formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold
        formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Italic
        formatted = formatted.replace(/\*(.+?)\*/g, '<em>$1</em>');

        // Links
        formatted = formatted.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

        // Line breaks
        formatted = formatted.replace(/\n/g, '<br>');

        return formatted;
    }

    scrollToBottom() {
        const chatMessages = document.getElementById('chat-messages');
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Typing Indicator
    showTypingIndicator() {
        const chatMessages = document.getElementById('chat-messages');

        const welcomeScreen = chatMessages.querySelector('.welcome-screen');
        if (welcomeScreen) welcomeScreen.remove();

        const existing = document.getElementById('typing-indicator');
        if (existing) existing.remove();

        const typingEl = document.createElement('div');
        typingEl.className = 'message assistant typing-indicator';
        typingEl.id = 'typing-indicator';
        typingEl.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-body">
                <div class="message-header">
                    <span class="message-sender">AI助手</span>
                </div>
                <div class="message-bubble">
                    <div class="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
        `;

        chatMessages.appendChild(typingEl);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
        this.isGenerating = false;
        this.hideStopButton();
        this.currentAssistantMessage = null;
    }

    // Stop Button
    showStopButton() {
        document.getElementById('send-btn').classList.add('hidden');
        document.getElementById('stop-btn').classList.remove('hidden');
    }

    hideStopButton() {
        document.getElementById('send-btn').classList.remove('hidden');
        document.getElementById('stop-btn').classList.add('hidden');
    }

    stopGeneration() {
        if (this.isGenerating) {
            this.socket.emit('stop_generation', { session_id: this.sessionId });
            this.isGenerating = false;
            this.hideStopButton();
            this.hideTypingIndicator();
        }
    }

    // Update Assistant Message (streaming)
    updateAssistantMessage(delta, fullResponse) {
        if (!this.currentAssistantMessage) {
            const chatMessages = document.getElementById('chat-messages');
            const indicator = document.getElementById('typing-indicator');
            if (indicator) indicator.remove();

            this.currentAssistantMessage = this.addMessage('assistant', fullResponse || delta);
        } else {
            const bubble = this.currentAssistantMessage.querySelector('.message-bubble');
            if (bubble) {
                bubble.innerHTML = this.formatMessage(fullResponse || delta);
            }
            this.scrollToBottom();
        }
    }

    // Clear Chat
    clearChat() {
        if (!confirm('确定要清空所有对话记录吗？')) return;

        const chatMessages = document.getElementById('chat-messages');
        chatMessages.innerHTML = `
            <div class="welcome-screen">
                <div class="welcome-hero">
                    <div class="hero-icon">
                        <i class="fas fa-atom"></i>
                    </div>
                    <h2>欢迎使用 Ollama Agent</h2>
                    <p>基于本地大语言模型的智能对话助手</p>
                </div>
                <div class="feature-grid">
                    <div class="feature-card">
                        <div class="feature-icon"><i class="fas fa-eye"></i></div>
                        <h4>视觉理解</h4>
                        <p>支持图像分析与描述</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon"><i class="fas fa-file-alt"></i></div>
                        <h4>文件分析</h4>
                        <p>智能解析各类文档</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon"><i class="fas fa-microphone"></i></div>
                        <h4>语音输入</h4>
                        <p>便捷的语音交互</p>
                    </div>
                    <div class="feature-card">
                        <div class="feature-icon"><i class="fas fa-code"></i></div>
                        <h4>代码执行</h4>
                        <p>Python 沙盒环境</p>
                    </div>
                </div>
            </div>
        `;

        fetch(`/api/chat/clear/${this.sessionId}`, { method: 'POST' }).catch(() => {});
    }

    // Camera
    async toggleCamera() {
        const preview = document.getElementById('camera-preview');
        const btn = document.getElementById('camera-btn');

        if (preview.classList.contains('hidden')) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480 }
                });

                const video = document.getElementById('video');
                video.srcObject = stream;
                this.cameraStream = stream;

                preview.classList.remove('hidden');
                btn.classList.add('active');
                btn.innerHTML = '<i class="fas fa-video"></i><span>关闭</span>';
            } catch (error) {
                console.error('Camera error:', error);
                this.showError('无法访问摄像头，请检查权限设置');
            }
        } else {
            this.closeCamera();
        }
    }

    closeCamera() {
        const preview = document.getElementById('camera-preview');
        const btn = document.getElementById('camera-btn');

        if (this.cameraStream) {
            this.cameraStream.getTracks().forEach(track => track.stop());
            this.cameraStream = null;
        }

        preview.classList.add('hidden');
        btn.classList.remove('active');
        btn.innerHTML = '<i class="fas fa-camera"></i><span>摄像头</span>';
    }

    capturePhoto() {
        const video = document.getElementById('video');
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        canvas.toBlob((blob) => {
            const file = new File([blob], `photo_${Date.now()}.png`, { type: 'image/png' });
            this.processImageFile(file);
        }, 'image/png');
    }

    processImageFile(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const imageData = e.target.result;
            const message = `[图片: ${file.name}] 请分析这张图片`;

            this.addMessage('user', message);

            this.socket.emit('send_message', {
                session_id: this.sessionId,
                message: message,
                model: this.currentModel,
                thinking: this.thinkingEnabled,
                image: imageData
            });

            this.showUploadedFile(file.name);
        };
        reader.readAsDataURL(file);
    }

    // File Upload
    async handleFileUpload(event) {
        const files = Array.from(event.target.files);
        for (const file of files) {
            await this.uploadFile(file);
        }
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.showUploadedFile(file.name);

                if (file.type.startsWith('image/')) {
                    this.processImageFile(file);
                }
            } else {
                this.showError('文件上传失败: ' + data.error);
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showError('文件上传失败');
        }
    }

    showUploadedFile(filename) {
        const container = document.getElementById('uploaded-files');
        const item = document.createElement('div');
        item.className = 'uploaded-file-item';
        item.innerHTML = `<i class="fas fa-file"></i><span>${filename}</span>`;
        container.appendChild(item);
    }

    // Drag & Drop
    setupDragAndDrop() {
        const chatMessages = document.getElementById('chat-messages');

        chatMessages.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
            chatMessages.style.background = 'var(--bg-active)';
        });

        chatMessages.addEventListener('dragleave', () => {
            chatMessages.style.background = '';
        });

        chatMessages.addEventListener('drop', (e) => {
            e.preventDefault();
            chatMessages.style.background = '';
            const files = Array.from(e.dataTransfer.files);
            files.forEach(file => this.uploadFile(file));
        });
    }

    // Audio Recording
    async toggleAudioRecording() {
        if (this.isRecording) {
            this.stopAudioRecording();
            return;
        }

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = () => {
                const blob = new Blob(this.audioChunks, { type: 'audio/wav' });
                const file = new File([blob], `audio_${Date.now()}.wav`, { type: 'audio/wav' });
                this.uploadFile(file);
                stream.getTracks().forEach(track => track.stop());
            };

            this.mediaRecorder.start();
            this.isRecording = true;
            this.audioSeconds = 0;

            const btn = document.getElementById('audio-record-btn');
            const recording = document.getElementById('audio-recording');

            btn.classList.add('active');
            btn.innerHTML = '<i class="fas fa-stop"></i><span>停止</span>';
            recording.classList.remove('hidden');

            this.audioTimer = setInterval(() => {
                this.audioSeconds++;
                const mins = Math.floor(this.audioSeconds / 60).toString().padStart(2, '0');
                const secs = (this.audioSeconds % 60).toString().padStart(2, '0');
                document.querySelector('.audio-timer').textContent = `${mins}:${secs}`;
            }, 1000);

        } catch (error) {
            console.error('Audio error:', error);
            this.showError('无法访问麦克风，请检查权限设置');
        }
    }

    stopAudioRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;

            clearInterval(this.audioTimer);

            const btn = document.getElementById('audio-record-btn');
            const recording = document.getElementById('audio-recording');

            btn.classList.remove('active');
            btn.innerHTML = '<i class="fas fa-microphone"></i><span>语音</span>';
            recording.classList.add('hidden');
            document.querySelector('.audio-timer').textContent = '00:00';
        }
    }

    // Settings
    showSettings() {
        const modal = document.getElementById('settings-modal');
        modal.classList.add('show');
    }

    hideSettings() {
        const modal = document.getElementById('settings-modal');
        modal.classList.remove('show');
    }

    async loadSettings() {
        try {
            const response = await fetch('/api/config');
            const data = await response.json();

            if (data.external_api_key !== undefined) {
                document.getElementById('api-key').value = data.external_api_key || '';
                document.getElementById('api-base').value = data.external_api_base || '';
                document.getElementById('thinking-toggle').checked = data.thinking_enabled !== false;
            }
        } catch (error) {
            console.error('加载设置失败:', error);
        }
    }

    async saveSettings() {
        const settings = {
            external_api_key: document.getElementById('api-key').value,
            external_api_base: document.getElementById('api-base').value,
            thinking_enabled: document.getElementById('thinking-toggle').checked
        };

        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });

            const data = await response.json();

            if (data.success) {
                this.showSuccess('设置已保存');
                this.hideSettings();
            } else {
                this.showError('保存设置失败: ' + data.error);
            }
        } catch (error) {
            console.error('保存设置失败:', error);
            this.showError('保存设置失败');
        }
    }

    async testConnection() {
        const btn = document.getElementById('test-connection-btn');
        const originalHTML = btn.innerHTML;

        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 测试中...';
        btn.disabled = true;

        try {
            const response = await fetch('/api/models');
            const data = await response.json();

            if (data.success) {
                this.showSuccess('连接测试成功');
                this.updateSystemInfo(data);
            } else {
                this.showError('连接测试失败: ' + data.error);
            }
        } catch (error) {
            this.showError('连接测试失败');
        } finally {
            btn.innerHTML = originalHTML;
            btn.disabled = false;
        }
    }

    updateSystemInfo(data) {
        const statusEl = document.getElementById('ollama-status');
        const countEl = document.getElementById('available-models-count');

        if (data.ollama_connected) {
            statusEl.textContent = '已连接';
            statusEl.className = 'status-badge connected';
        } else {
            statusEl.textContent = '未连接';
            statusEl.className = 'status-badge disconnected';
        }

        countEl.textContent = data.models ? data.models.length : 0;
    }

    // Notifications
    showError(message) {
        const toast = document.getElementById('error-toast');
        const msgEl = document.getElementById('error-message');
        msgEl.textContent = message;
        toast.classList.remove('hidden');

        setTimeout(() => this.hideError(), 5000);
    }

    hideError() {
        document.getElementById('error-toast').classList.add('hidden');
    }

    showSuccess(message) {
        const toast = document.getElementById('success-toast');
        const msgEl = document.getElementById('success-message');
        msgEl.textContent = message;
        toast.classList.remove('hidden');

        setTimeout(() => {
            toast.classList.add('hidden');
        }, 3000);
    }

    // Code Interpreter
    initCodeInterpreter() {
        this.codeInterpreter = new CodeInterpreter(this);
    }
}

// ============================================
// Code Interpreter Module
// ============================================
class CodeInterpreter {
    constructor(app) {
        this.app = app;
        this.isRunning = false;
        this.execResults = [];
        this.init();
    }

    init() {
        document.getElementById('code-interpreter-btn').addEventListener('click', () => this.togglePanel());
        document.getElementById('code-panel-close').addEventListener('click', () => this.hidePanel());
        document.getElementById('code-run-btn').addEventListener('click', () => this.runCode());
        document.getElementById('code-stop-btn').addEventListener('click', () => this.stopCode());
        document.getElementById('code-insert-btn').addEventListener('click', () => this.insertToChat());

        const editor = document.getElementById('code-editor');
        editor.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                e.preventDefault();
                const start = editor.selectionStart;
                const end = editor.selectionEnd;
                editor.value = editor.value.substring(0, start) + '    ' + editor.value.substring(end);
                editor.selectionStart = editor.selectionEnd = start + 4;
            }
        });

        this.app.socket.on('code_execution_start', () => this.onExecutionStart());
        this.app.socket.on('code_execution_result', (data) => this.onExecutionResult(data));
        this.app.socket.on('code_execution_error', (data) => this.onExecutionError(data));
        this.app.socket.on('code_execution_stopped', () => this.onExecutionStopped());
    }

    togglePanel() {
        const panel = document.getElementById('code-interpreter-panel');
        const btn = document.getElementById('code-interpreter-btn');

        if (panel.classList.contains('hidden')) {
            panel.classList.remove('hidden');
            btn.classList.add('active');
        } else {
            this.hidePanel();
        }
    }

    hidePanel() {
        document.getElementById('code-interpreter-panel').classList.add('hidden');
        document.getElementById('code-interpreter-btn').classList.remove('active');
    }

    getCode() {
        return document.getElementById('code-editor').value;
    }

    runCode() {
        const code = this.getCode();
        if (!code.trim()) {
            this.showOutput('error', '请输入要执行的代码');
            return;
        }

        this.isRunning = true;
        document.getElementById('code-run-btn').classList.add('hidden');
        document.getElementById('code-stop-btn').classList.remove('hidden');
        this.setStatus('运行中...');
        this.clearOutput();

        this.app.socket.emit('code_execute', {
            session_id: this.app.sessionId,
            code: code,
            timeout: 30
        });
    }

    stopCode() {
        this.app.socket.emit('code_stop', { session_id: this.app.sessionId });
    }

    onExecutionStart() {
        this.setStatus('正在执行...');
    }

    onExecutionResult(data) {
        this.isRunning = false;
        document.getElementById('code-run-btn').classList.remove('hidden');
        document.getElementById('code-stop-btn').classList.add('hidden');

        const result = data.result;
        if (!result) return;

        const outputDiv = document.getElementById('code-output');
        outputDiv.innerHTML = '';

        if (result.error) {
            this.showOutput('error', '执行出错:');
            this.showOutput('error-text', result.error);
            this.setStatus('执行失败');
            return;
        }

        if (result.stdout) this.showOutput('stdout', result.stdout);
        if (result.stderr) this.showOutput('stderr', result.stderr);
        if (result.plots) {
            result.plots.forEach(plot => this.showOutput('plot', plot));
        }
        if (result.files) {
            result.files.forEach(file => this.showOutput('file', file));
        }

        if (!result.stdout && !result.stderr && !result.plots && !result.files) {
            this.showOutput('info', '代码已执行完成（无输出）');
        }

        this.setStatus(`完成 (${result.execution_time || 0}秒)`);
        this.execResults.push(result);
    }

    onExecutionError(data) {
        this.isRunning = false;
        document.getElementById('code-run-btn').classList.remove('hidden');
        document.getElementById('code-stop-btn').classList.add('hidden');
        this.showOutput('error', '执行出错: ' + data.error);
        this.setStatus('出错');
    }

    onExecutionStopped() {
        this.isRunning = false;
        document.getElementById('code-run-btn').classList.remove('hidden');
        document.getElementById('code-stop-btn').classList.add('hidden');
        this.showOutput('info', '执行已停止');
        this.setStatus('已停止');
    }

    showOutput(type, content) {
        const outputDiv = document.getElementById('code-output');
        const div = document.createElement('div');
        div.className = 'code-output-item';

        switch (type) {
            case 'stdout':
                div.innerHTML = `<pre class="code-output-stdout">${this.escapeHtml(content)}</pre>`;
                break;
            case 'stderr':
                div.innerHTML = `<pre class="code-output-stderr">${this.escapeHtml(content)}</pre>`;
                break;
            case 'error':
                div.innerHTML = `<div class="code-output-error-header"><i class="fas fa-times-circle"></i> ${this.escapeHtml(content)}</div>`;
                break;
            case 'error-text':
                div.innerHTML = `<pre class="code-output-stderr">${this.escapeHtml(content)}</pre>`;
                break;
            case 'info':
                div.innerHTML = `<div class="code-output-info"><i class="fas fa-info-circle"></i> ${this.escapeHtml(content)}</div>`;
                break;
            case 'plot':
                div.innerHTML = `
                    <div class="code-output-plot">
                        <img src="${content.url}" alt="${content.filename}" onclick="window.open('${content.url}', '_blank')">
                        <div class="code-output-file-name"><i class="fas fa-image"></i> ${content.filename}</div>
                    </div>`;
                break;
            case 'file':
                const isImage = content.filename.match(/\.(png|jpg|jpeg|gif|webp|svg)$/i);
                if (isImage) {
                    div.innerHTML = `
                        <div class="code-output-plot">
                            <img src="${content.url}" alt="${content.filename}" onclick="window.open('${content.url}', '_blank')">
                            <div class="code-output-file-name"><i class="fas fa-file-image"></i> ${content.filename}</div>
                        </div>`;
                } else {
                    div.innerHTML = `
                        <div class="code-output-file">
                            <i class="fas fa-file"></i>
                            <a href="${content.url}" target="_blank">${content.filename}</a>
                            <span class="code-output-file-size">(${this.formatBytes(content.size)})</span>
                        </div>`;
                }
                break;
        }

        outputDiv.appendChild(div);
        outputDiv.scrollTop = outputDiv.scrollHeight;
    }

    clearOutput() {
        document.getElementById('code-output').innerHTML = '';
    }

    setStatus(text) {
        const statusEl = document.getElementById('code-status');
        statusEl.textContent = text;
        statusEl.className = 'code-status';

        if (text.includes('运行') || text.includes('执行')) {
            statusEl.classList.add('running');
        } else if (text.includes('失败') || text.includes('出错')) {
            statusEl.classList.add('error');
        } else if (text.includes('完成')) {
            statusEl.classList.add('success');
        }
    }

    insertToChat() {
        const code = this.getCode();
        if (!code.trim()) return;

        const lastResult = this.execResults.length > 0 ? this.execResults[this.execResults.length - 1] : null;
        let message = '```python\n' + code + '\n```';

        if (lastResult) {
            if (lastResult.plots && lastResult.plots.length > 0) {
                message += '\n\n生成的图表：';
                lastResult.plots.forEach(p => {
                    message += `\n![${p.filename}](${p.url})`;
                });
            }
            if (lastResult.stdout) {
                message += '\n\n输出：\n```\n' + lastResult.stdout.slice(0, 500) + '\n```';
            }
        }

        const input = document.getElementById('message-input');
        input.value = message;
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';

        if (this.app.isConnected && !this.app.isGenerating) {
            this.app.sendMessage();
        }

        this.hidePanel();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    formatBytes(bytes) {
        if (!bytes || bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// ============================================
// Initialize App
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    window.ollamaApp = new OllamaAgentApp();
});
