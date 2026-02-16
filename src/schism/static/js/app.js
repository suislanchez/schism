/**
 * Schism - Main application logic.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    let generating = false;
    let compareMode = false;

    // --- Elements ---
    const statusDot = document.getElementById('status-dot');
    const modelSelect = document.getElementById('model-select');
    const promptInput = document.getElementById('prompt-input');
    const generateBtn = document.getElementById('generate-btn');
    const compareToggle = document.getElementById('compare-toggle');
    const outputArea = document.getElementById('output-area');
    const steeredBody = document.getElementById('steered-body');
    const vanillaPanel = document.getElementById('vanilla-panel');
    const vanillaBody = document.getElementById('vanilla-body');
    const presetSelect = document.getElementById('preset-select');
    const savePresetBtn = document.getElementById('save-preset-btn');
    const resetBtn = document.getElementById('reset-btn');
    const tempInput = document.getElementById('temp-input');
    const maxTokensInput = document.getElementById('max-tokens-input');

    // Modal
    const modalOverlay = document.getElementById('modal-overlay');
    const modalNameInput = document.getElementById('modal-name');
    const modalDescInput = document.getElementById('modal-desc');
    const modalSaveBtn = document.getElementById('modal-save');
    const modalCancelBtn = document.getElementById('modal-cancel');

    // Drop zone
    const dropZone = document.getElementById('drop-zone');

    // Toast
    const toastContainer = document.getElementById('toast-container');

    // --- Init Slider Manager ---
    const sidebarEl = document.querySelector('.sidebar');
    const sliderManager = new SliderManager(sidebarEl);

    // --- Init Preset Manager ---
    const presetManager = new PresetManager();

    // --- Init WebSocket ---
    const socket = new SchismSocket();

    socket.onStatusChange = (status) => {
        statusDot.className = 'status-dot ' + status;
    };

    socket.onStart = (data) => {
        generating = true;
        generateBtn.textContent = 'Generating...';
        generateBtn.disabled = true;

        steeredBody.textContent = '';
        steeredBody.classList.remove('empty');
        steeredBody.innerHTML = '<span class="cursor"></span>';

        if (data.compare) {
            vanillaBody.textContent = '';
            vanillaBody.classList.remove('empty');
            vanillaBody.innerHTML = '<span class="cursor"></span>';
        }
    };

    socket.onToken = (token, side) => {
        const target = side === 'vanilla' ? vanillaBody : steeredBody;
        // Remove cursor, append token, re-add cursor
        const cursor = target.querySelector('.cursor');
        if (cursor) cursor.remove();
        target.appendChild(document.createTextNode(token));
        const newCursor = document.createElement('span');
        newCursor.className = 'cursor';
        target.appendChild(newCursor);

        // Auto-scroll
        target.scrollTop = target.scrollHeight;
    };

    socket.onDone = () => {
        generating = false;
        generateBtn.textContent = 'Generate';
        generateBtn.disabled = false;

        // Remove cursors
        document.querySelectorAll('.cursor').forEach(c => c.remove());
    };

    socket.onError = (err) => {
        generating = false;
        generateBtn.textContent = 'Generate';
        generateBtn.disabled = false;
        document.querySelectorAll('.cursor').forEach(c => c.remove());
        showToast(err, 'error');
    };

    socket.connect();

    // --- Compare Mode Toggle ---
    compareToggle.addEventListener('click', () => {
        compareMode = !compareMode;
        compareToggle.classList.toggle('active', compareMode);

        if (compareMode) {
            outputArea.classList.remove('single');
            vanillaPanel.style.display = 'flex';
        } else {
            outputArea.classList.add('single');
            vanillaPanel.style.display = 'none';
        }
    });

    // Start in single mode
    outputArea.classList.add('single');
    vanillaPanel.style.display = 'none';

    // --- Generate ---
    function doGenerate() {
        if (generating) return;
        const prompt = promptInput.value.trim();
        if (!prompt) {
            showToast('Enter a prompt first', 'error');
            return;
        }

        const sliders = sliderManager.getValues();
        const model = modelSelect.value;
        const temperature = parseFloat(tempInput.value) || 0.7;
        const max_tokens = parseInt(maxTokensInput.value) || 256;

        socket.generate({
            prompt,
            sliders,
            model,
            compare: compareMode,
            temperature,
            max_tokens,
        });
    }

    generateBtn.addEventListener('click', doGenerate);

    // Ctrl+Enter / Cmd+Enter to generate
    promptInput.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            doGenerate();
        }
    });

    // --- Presets ---
    async function loadPresets() {
        await presetManager.fetchPresets();
        presetManager.renderSelect(presetSelect);
    }

    presetSelect.addEventListener('change', () => {
        const name = presetSelect.value;
        if (!name) return;

        const preset = presetManager.getPreset(name);
        if (preset) {
            sliderManager.setValues(preset.sliders);
            showToast(`Loaded preset: ${preset.name}`, 'success');
        }
    });

    savePresetBtn.addEventListener('click', () => {
        modalOverlay.classList.add('visible');
        modalNameInput.value = '';
        modalDescInput.value = '';
        modalNameInput.focus();
    });

    modalCancelBtn.addEventListener('click', () => {
        modalOverlay.classList.remove('visible');
    });

    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) {
            modalOverlay.classList.remove('visible');
        }
    });

    modalSaveBtn.addEventListener('click', async () => {
        const name = modalNameInput.value.trim();
        if (!name) {
            showToast('Enter a preset name', 'error');
            return;
        }

        const sliders = sliderManager.getValues();
        if (Object.keys(sliders).length === 0) {
            showToast('Adjust at least one slider first', 'error');
            return;
        }

        await presetManager.savePreset(name, modalDescInput.value.trim(), sliders);
        presetManager.renderSelect(presetSelect);
        modalOverlay.classList.remove('visible');
        showToast(`Saved preset: ${name}`, 'success');
    });

    resetBtn.addEventListener('click', () => {
        sliderManager.reset();
        presetSelect.value = '';
        showToast('Sliders reset', 'info');
    });

    // --- Drag & Drop Presets ---
    document.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('visible');
    });

    document.addEventListener('dragleave', (e) => {
        if (!e.relatedTarget || e.relatedTarget === document.documentElement) {
            dropZone.classList.remove('visible');
        }
    });

    document.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('visible');

        const file = e.dataTransfer.files[0];
        if (!file || !file.name.endsWith('.json')) {
            showToast('Drop a .json preset file', 'error');
            return;
        }

        const reader = new FileReader();
        reader.onload = (ev) => {
            try {
                const data = JSON.parse(ev.target.result);
                if (data.sliders) {
                    sliderManager.setValues(data.sliders);
                    showToast(`Loaded: ${data.name || file.name}`, 'success');
                } else {
                    showToast('Invalid preset file', 'error');
                }
            } catch {
                showToast('Failed to parse JSON', 'error');
            }
        };
        reader.readAsText(file);
    });

    // --- Load Models ---
    async function loadModels() {
        const resp = await fetch('/api/models');
        const models = await resp.json();
        modelSelect.innerHTML = '';
        for (const m of models) {
            const opt = document.createElement('option');
            opt.value = m.name;
            opt.textContent = m.name + (m.loaded ? ' (loaded)' : '');
            modelSelect.appendChild(opt);
        }
    }

    // --- Toast ---
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transition = 'opacity 0.3s';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // --- Init ---
    async function init() {
        await sliderManager.loadFeatures();
        await loadPresets();
        await loadModels();
    }

    init().catch(err => {
        console.error('Init failed:', err);
        showToast('Failed to connect to server', 'error');
    });
});
