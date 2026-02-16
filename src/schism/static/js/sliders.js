/**
 * Slider components and preset management.
 */

class SliderManager {
    constructor(container) {
        this.container = container;
        this.sliders = {};
        this.features = {};
        this.onChange = null;
    }

    async loadFeatures() {
        const resp = await fetch('/api/features');
        this.features = await resp.json();
        this.render();
    }

    render() {
        // Keep preset bar at the bottom
        const presetBar = this.container.querySelector('.preset-bar');

        // Clear sliders but not the title or preset bar
        const existingGroups = this.container.querySelectorAll('.slider-group');
        existingGroups.forEach(g => g.remove());

        const names = Object.keys(this.features).sort();

        for (const name of names) {
            const info = this.features[name];
            const value = this.sliders[name] || 0;

            const group = document.createElement('div');
            group.className = 'slider-group' + (Math.abs(value) > 0.01 ? ' active' : '');
            group.dataset.feature = name;

            group.innerHTML = `
                <div class="slider-label">
                    <span class="slider-name">${name}</span>
                    <span class="slider-value ${value > 0 ? 'positive' : value < 0 ? 'negative' : ''}">${value > 0 ? '+' : ''}${value.toFixed(1)}</span>
                </div>
                <div class="slider-description">${info.description}</div>
                <input type="range" min="-1" max="1" step="0.1" value="${value}" data-feature="${name}">
            `;

            const input = group.querySelector('input[type="range"]');
            const valueDisplay = group.querySelector('.slider-value');

            input.addEventListener('input', (e) => {
                const val = parseFloat(e.target.value);
                this.sliders[name] = val;

                valueDisplay.textContent = (val > 0 ? '+' : '') + val.toFixed(1);
                valueDisplay.className = 'slider-value' + (val > 0 ? ' positive' : val < 0 ? ' negative' : '');
                group.className = 'slider-group' + (Math.abs(val) > 0.01 ? ' active' : '');

                if (this.onChange) this.onChange(this.getValues());
            });

            // Double-click to reset
            input.addEventListener('dblclick', () => {
                input.value = 0;
                this.sliders[name] = 0;
                valueDisplay.textContent = '0.0';
                valueDisplay.className = 'slider-value';
                group.className = 'slider-group';
                if (this.onChange) this.onChange(this.getValues());
            });

            // Insert before preset bar
            if (presetBar) {
                this.container.insertBefore(group, presetBar);
            } else {
                this.container.appendChild(group);
            }
        }
    }

    getValues() {
        const values = {};
        for (const [name, val] of Object.entries(this.sliders)) {
            if (Math.abs(val) > 0.01) {
                values[name] = val;
            }
        }
        return values;
    }

    setValues(sliders) {
        this.sliders = { ...sliders };
        this.render();
    }

    reset() {
        this.sliders = {};
        this.render();
    }
}


class PresetManager {
    constructor() {
        this.presets = [];
        this.onLoad = null;
    }

    async fetchPresets() {
        const resp = await fetch('/api/presets');
        this.presets = await resp.json();
        return this.presets;
    }

    async savePreset(name, description, sliders) {
        const resp = await fetch('/api/presets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, description, sliders }),
        });
        const data = await resp.json();
        await this.fetchPresets();
        return data;
    }

    getPreset(name) {
        return this.presets.find(p => p.name.toLowerCase() === name.toLowerCase());
    }

    renderSelect(selectEl) {
        selectEl.innerHTML = '<option value="">-- Select Preset --</option>';
        for (const p of this.presets) {
            const opt = document.createElement('option');
            opt.value = p.name;
            opt.textContent = `${p.name} ${p.source === 'default' ? '(built-in)' : ''}`;
            selectEl.appendChild(opt);
        }
    }
}
