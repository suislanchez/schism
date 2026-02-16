/**
 * WebSocket client for streaming generation.
 */

class SchismSocket {
    constructor() {
        this.ws = null;
        this.onToken = null;
        this.onStart = null;
        this.onDone = null;
        this.onError = null;
        this.onStatusChange = null;
        this._reconnectTimer = null;
    }

    connect() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${protocol}//${location.host}/ws/generate`;

        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
            this._setStatus('connected');
        };

        this.ws.onclose = () => {
            this._setStatus('disconnected');
            // Auto-reconnect after 2s
            this._reconnectTimer = setTimeout(() => this.connect(), 2000);
        };

        this.ws.onerror = () => {
            this._setStatus('error');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            switch (data.type) {
                case 'start':
                    if (this.onStart) this.onStart(data);
                    break;
                case 'token':
                    if (this.onToken) this.onToken(data.token, data.side);
                    break;
                case 'done':
                    if (this.onDone) this.onDone();
                    break;
                case 'error':
                    if (this.onError) this.onError(data.error);
                    break;
            }
        };
    }

    generate(request) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            if (this.onError) this.onError('Not connected to server');
            return;
        }
        this.ws.send(JSON.stringify(request));
    }

    disconnect() {
        if (this._reconnectTimer) {
            clearTimeout(this._reconnectTimer);
        }
        if (this.ws) {
            this.ws.close();
        }
    }

    _setStatus(status) {
        if (this.onStatusChange) this.onStatusChange(status);
    }
}
