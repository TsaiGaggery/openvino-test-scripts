/* OpenVINO Benchmark Studio — Frontend Application */

// ─── State ──────────────────────────────────────────────────────────────────

let config = null;           // Current benchmark.json config
let devices = [];            // Detected devices
let benchmarkRunning = false;
let speedChart = null;
let loadTimeChart = null;
let powerChart = null;
const isElectron = typeof window.electronAPI !== 'undefined';

// ─── API Helpers ────────────────────────────────────────────────────────────

async function api(path, options = {}) {
    const res = await fetch(path, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
    });
    return res.json();
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// ─── Tab Navigation ─────────────────────────────────────────────────────────

document.querySelectorAll('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`tab-${tab}`).classList.add('active');

        if (tab === 'settings') loadSettings();
        if (tab === 'run') loadRunSummary();
        if (tab === 'results') loadResults();
    });
});

// ─── Initialization ─────────────────────────────────────────────────────────

async function init() {
    // Detect Electron vs Web mode for local browse
    if (isElectron) {
        document.getElementById('localBrowseElectron').style.display = 'flex';
        document.getElementById('localBrowseWeb').style.display = 'none';
    }

    // Load config
    try {
        config = await api('/api/config');
        if (!config.models) config.models = [];
        if (!config.benchmark_config) config.benchmark_config = {};
        renderModelList();
    } catch (e) {
        console.error('Failed to load config:', e);
        config = { models: [], benchmark_config: {} };
        showToast('Failed to load config — using defaults', 'error');
    }

    // Load devices
    try {
        const data = await api('/api/devices');
        devices = data.devices || [];
    } catch (e) {
        console.error('Failed to detect devices:', e);
    }

    // Setup event listeners
    setupModelListeners();
    setupSettingsListeners();
    setupRunListeners();
    setupResultsListeners();
}

// ─── Models Tab ─────────────────────────────────────────────────────────────

function renderModelList() {
    const list = document.getElementById('modelList');
    const models = config?.models || [];
    document.getElementById('modelCount').textContent = models.length;

    if (models.length === 0) {
        list.innerHTML = '<div class="empty-state">No models configured. Add models below.</div>';
        return;
    }

    list.innerHTML = models.map((m, i) => `
        <div class="model-card" data-index="${i}">
            <div class="model-card-toggle">
                <label class="toggle">
                    <input type="checkbox" ${m.enabled ? 'checked' : ''} data-action="toggle" data-index="${i}">
                    <span class="toggle-slider"></span>
                </label>
            </div>
            <div class="model-card-info">
                <div class="model-card-name">${escapeHtml(m.name)}</div>
                <div class="model-card-meta">
                    ${m.size ? `<span class="badge badge-sm">${m.size}</span>` : ''}
                    ${m.quantization ? `<span class="badge badge-sm">${m.quantization}</span>` : ''}
                    ${(m.recommended_devices || []).map(d => `<span class="badge-device badge-${d.toLowerCase()}">${d}</span>`).join('')}
                </div>
                <div class="model-card-id">${escapeHtml(m.model_id)}</div>
            </div>
            <div class="model-card-actions">
                <button class="btn btn-sm btn-ghost" data-action="remove" data-index="${i}" title="Remove">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6"/><path d="M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
                </button>
            </div>
        </div>
    `).join('');
}

function setupModelListeners() {
    // Toggle / Remove model
    document.getElementById('modelList').addEventListener('click', async (e) => {
        const target = e.target.closest('[data-action]');
        if (!target) return;
        const action = target.dataset.action;
        const index = parseInt(target.dataset.index);

        if (action === 'toggle') {
            config.models[index].enabled = target.checked;
            await saveConfig();
        } else if (action === 'remove') {
            config.models.splice(index, 1);
            await saveConfig();
            renderModelList();
            if (hfSearchResults.length > 0) renderHFResults(hfSearchResults);
            showToast('Model removed', 'info');
        }
    });

    // Browse local (Electron)
    document.getElementById('browseLocalBtn')?.addEventListener('click', async () => {
        const result = await window.electronAPI.selectModelFolder();
        if (result.canceled) return;
        await registerLocalModel(result.folderPath);
    });

    // Add local path (Web)
    document.getElementById('addLocalPathBtn')?.addEventListener('click', async () => {
        const input = document.getElementById('localPathInput');
        const path = input.value.trim();
        if (!path) return;
        await registerLocalModel(path);
        input.value = '';
    });

    // HuggingFace search
    document.getElementById('hfSearchBtn').addEventListener('click', searchHuggingFace);
    document.getElementById('hfSearchInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') searchHuggingFace();
    });

    // Filter tabs
    document.querySelectorAll('.filter-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            filterHFResults(tab.dataset.filter);
        });
    });
}

async function registerLocalModel(folderPath) {
    try {
        const result = await api('/api/model/register-local', {
            method: 'POST',
            body: JSON.stringify({ folder_path: folderPath }),
        });
        if (result.status === 'success') {
            // Add to benchmark config
            const modelInfo = result.model;
            config.models.push({
                name: modelInfo.name,
                model_id: modelInfo.local_path,
                size: '',
                quantization: '',
                recommended_devices: devices.map(d => d.name),
                description: `Local model: ${modelInfo.local_path}`,
                enabled: true,
            });
            await saveConfig();
            renderModelList();
            showToast(result.message, 'success');
        } else {
            showToast(result.message, 'error');
        }
    } catch (e) {
        showToast('Failed to register model', 'error');
    }
}

let hfSearchResults = [];

async function searchHuggingFace() {
    const query = document.getElementById('hfSearchInput').value.trim();
    if (!query) return;

    const container = document.getElementById('hfResults');
    container.innerHTML = '<div class="empty-state">Searching...</div>';

    try {
        const data = await api(`/api/models/search?q=${encodeURIComponent(query)}&limit=20`);
        hfSearchResults = data.results || [];
        renderHFResults(hfSearchResults);
    } catch (e) {
        container.innerHTML = '<div class="empty-state text-error">Search failed</div>';
    }
}

function renderHFResults(results) {
    const container = document.getElementById('hfResults');
    if (results.length === 0) {
        container.innerHTML = '<div class="empty-state">No models found</div>';
        return;
    }

    container.innerHTML = results.map(m => {
        const existing = (config?.models || []).some(cm => cm.model_id === m.id);
        const deviceType = detectModelDevice(m.id);
        return `
            <div class="hf-result-card" data-device="${deviceType}" data-model-id="${escapeHtml(m.id)}">
                <div class="hf-result-info">
                    <div class="hf-result-name">${escapeHtml(m.name)}</div>
                    <div class="hf-result-author">${escapeHtml(m.author)}</div>
                </div>
                <div class="hf-result-stats">
                    <span class="badge-device badge-${deviceType}">${deviceType.toUpperCase()}</span>
                    <span>${formatDownloads(m.downloads)}</span>
                    ${existing
                        ? '<span class="text-muted">Added</span>'
                        : `<button class="btn btn-sm btn-primary" onclick="addHFModel('${escapeHtml(m.id)}', '${escapeHtml(m.name)}')">Add</button>`
                    }
                </div>
            </div>
        `;
    }).join('');

    // Re-apply the active filter tab
    const activeFilter = document.querySelector('.filter-tab.active')?.dataset.filter || 'all';
    if (activeFilter !== 'all') filterHFResults(activeFilter);
}

function filterHFResults(filter) {
    const cards = document.querySelectorAll('.hf-result-card');
    cards.forEach(card => {
        const device = card.dataset.device;
        if (filter === 'all') card.style.display = '';
        else if (filter === 'npu') card.style.display = device === 'npu' ? '' : 'none';
        else card.style.display = device !== 'npu' ? '' : 'none';
    });
}

function detectModelDevice(modelId) {
    const id = modelId.toLowerCase();
    if (id.includes('-cw-ov') || id.includes('-npu') || id.includes('_npu')) return 'npu';
    return 'gpu';
}

async function addHFModel(modelId, modelName) {
    // Extract quantization from model name
    let quant = 'INT8';
    if (modelId.toLowerCase().includes('int4')) quant = 'INT4';
    else if (modelId.toLowerCase().includes('fp16')) quant = 'FP16';

    config.models.push({
        name: modelName.replace(/-int[48]-ov$/, '').replace(/-fp16-ov$/, '').replace(/-cw-ov$/, ''),
        model_id: modelId,
        size: extractSize(modelName),
        quantization: quant,
        recommended_devices: devices.map(d => d.name),
        description: `From HuggingFace: ${modelId}`,
        enabled: true,
    });
    await saveConfig();
    renderModelList();
    renderHFResults(hfSearchResults); // Refresh to show "Added"
    showToast(`Added ${modelName}`, 'success');
}

function extractSize(name) {
    const match = name.match(/(\d+\.?\d*)[Bb]/);
    return match ? match[1] + 'B' : '';
}

// ─── Settings Tab ───────────────────────────────────────────────────────────

function loadSettings() {
    if (!config) return;
    const bc = config.benchmark_config || {};
    const gc = bc.generation_config || {};

    // Devices
    renderDeviceCheckboxes(bc.devices_to_test || []);

    // Generation config
    setSlider('maxTokensSlider', 'maxTokensValue', gc.max_new_tokens || 100);
    setSlider('temperatureSlider', 'temperatureValue', gc.temperature || 0.7);
    setSlider('topPSlider', 'topPValue', gc.top_p || 0.9);
    document.getElementById('doSampleToggle').checked = gc.do_sample !== false;
    document.getElementById('warmupToggle').checked = bc.run_warmup !== false;

    // Prompts
    renderPrompts(bc.test_prompts || []);

    // HF token status
    loadHfTokenStatus();
}

// ─── HF Token ───────────────────────────────────────────────────────────────

async function loadHfTokenStatus() {
    try {
        const data = await api('/api/hf-token/status');
        const statusEl = document.getElementById('hfTokenStatus');
        const input = document.getElementById('hfTokenInput');

        if (data.configured) {
            const sourceLabel = { env: 'environment variable', saved: 'saved token', ui: 'session override' }[data.source] || data.source;
            statusEl.innerHTML = `<span style="color: var(--success-color);">Token configured</span> (source: ${sourceLabel}, ${data.masked})`;
            input.placeholder = data.masked;
            input.value = '';
        } else {
            statusEl.textContent = 'No token configured';
            input.placeholder = 'hf_...';
        }
    } catch (e) {
        console.error('Failed to load HF token status:', e);
    }
}

async function saveHfToken() {
    const input = document.getElementById('hfTokenInput');
    const token = input.value.trim();
    if (!token) {
        showToast('Enter a token first', 'error');
        return;
    }
    try {
        const result = await api('/api/hf-token', {
            method: 'POST',
            body: JSON.stringify({ token }),
        });
        if (result.status === 'success') {
            showToast(result.message || 'Token saved', 'success');
            input.value = '';
            await loadHfTokenStatus();
        } else {
            showToast(result.message || 'Failed to save token', 'error');
        }
    } catch (e) {
        showToast('Failed to save token', 'error');
    }
}

async function clearHfToken() {
    try {
        const result = await api('/api/hf-token', { method: 'DELETE' });
        if (result.status === 'success') {
            showToast('Token cleared', 'info');
            await loadHfTokenStatus();
        }
    } catch (e) {
        showToast('Failed to clear token', 'error');
    }
}

function renderDeviceCheckboxes(selectedDevices) {
    const container = document.getElementById('deviceCheckboxes');
    if (devices.length === 0) {
        container.innerHTML = '<div class="text-muted">No devices detected</div>';
        return;
    }

    container.innerHTML = devices.map(d => {
        const checked = selectedDevices.includes(d.name);
        return `
            <label class="device-checkbox ${checked ? 'checked' : ''}" data-device="${d.name}">
                <input type="checkbox" ${checked ? 'checked' : ''}>
                <div class="check-icon">${checked ? '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3"><path d="M5 12l5 5L20 7"/></svg>' : ''}</div>
                <div>
                    <div class="device-name">${d.name}</div>
                    <div class="device-fullname">${escapeHtml(d.full_name)}</div>
                </div>
            </label>
        `;
    }).join('');

    // Toggle handler
    container.querySelectorAll('.device-checkbox').forEach(el => {
        el.addEventListener('click', (e) => {
            if (e.target.tagName === 'INPUT') return;
            const input = el.querySelector('input');
            input.checked = !input.checked;
            el.classList.toggle('checked', input.checked);
            const icon = el.querySelector('.check-icon');
            icon.innerHTML = input.checked ? '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3"><path d="M5 12l5 5L20 7"/></svg>' : '';
        });
    });
}

function renderPrompts(prompts) {
    const list = document.getElementById('promptList');
    list.innerHTML = prompts.map((p, i) => `
        <div class="prompt-item">
            <input type="text" class="input prompt-input" value="${escapeHtml(p)}" data-index="${i}">
            <button class="btn btn-sm btn-ghost" onclick="removePrompt(${i})">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
        </div>
    `).join('');
}

function removePrompt(index) {
    const bc = config.benchmark_config || {};
    const prompts = bc.test_prompts || [];
    prompts.splice(index, 1);
    renderPrompts(prompts);
}

function setSlider(sliderId, valueId, value) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(valueId);
    slider.value = value;
    display.textContent = value;
}

function setupSettingsListeners() {
    // Sliders
    ['maxTokensSlider', 'temperatureSlider', 'topPSlider'].forEach(id => {
        const slider = document.getElementById(id);
        const valueId = id.replace('Slider', 'Value');
        slider.addEventListener('input', () => {
            document.getElementById(valueId).textContent = slider.value;
        });
    });

    // Add prompt
    document.getElementById('addPromptBtn').addEventListener('click', () => {
        if (!config.benchmark_config) config.benchmark_config = {};
        if (!config.benchmark_config.test_prompts) config.benchmark_config.test_prompts = [];
        config.benchmark_config.test_prompts.push('');
        renderPrompts(config.benchmark_config.test_prompts);
        // Focus the new input
        const inputs = document.querySelectorAll('.prompt-input');
        inputs[inputs.length - 1]?.focus();
    });

    // Save settings
    document.getElementById('saveSettingsBtn').addEventListener('click', async () => {
        collectSettings();
        await saveConfig();
        showToast('Settings saved', 'success');
    });

    // HF Token
    document.getElementById('hfTokenSaveBtn').addEventListener('click', saveHfToken);
    document.getElementById('hfTokenClearBtn').addEventListener('click', clearHfToken);
    document.getElementById('hfTokenInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') saveHfToken();
    });
    document.getElementById('hfTokenToggleVisibility').addEventListener('click', () => {
        const input = document.getElementById('hfTokenInput');
        const isPassword = input.type === 'password';
        input.type = isPassword ? 'text' : 'password';
        const icon = document.getElementById('hfTokenEyeIcon');
        if (isPassword) {
            icon.innerHTML = '<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/>';
        } else {
            icon.innerHTML = '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>';
        }
    });
}

function collectSettings() {
    if (!config.benchmark_config) config.benchmark_config = {};
    const bc = config.benchmark_config;

    // Only collect from DOM if the Settings tab has been rendered,
    // otherwise we'd overwrite config values with empty arrays
    // because the checkboxes/sliders don't exist in the DOM yet.
    const settingsRendered = document.querySelectorAll('#deviceCheckboxes input[type="checkbox"]').length > 0;
    if (!settingsRendered) return;

    // Devices
    const selectedDevices = [];
    document.querySelectorAll('#deviceCheckboxes input[type="checkbox"]:checked').forEach(input => {
        const label = input.closest('.device-checkbox');
        selectedDevices.push(label.dataset.device);
    });
    bc.devices_to_test = selectedDevices;

    // Generation config
    bc.generation_config = {
        max_new_tokens: parseInt(document.getElementById('maxTokensSlider').value),
        temperature: parseFloat(document.getElementById('temperatureSlider').value),
        top_p: parseFloat(document.getElementById('topPSlider').value),
        do_sample: document.getElementById('doSampleToggle').checked,
    };

    bc.run_warmup = document.getElementById('warmupToggle').checked;

    // Prompts
    const prompts = [];
    document.querySelectorAll('.prompt-input').forEach(input => {
        const val = input.value.trim();
        if (val) prompts.push(val);
    });
    bc.test_prompts = prompts;

    if (!bc.cache_dir) bc.cache_dir = './ov_cache';
}

// ─── Run Tab ────────────────────────────────────────────────────────────────

function loadRunSummary() {
    if (!config) return;
    const models = (config.models || []).filter(m => m.enabled);
    const bc = config.benchmark_config || {};
    document.getElementById('summaryModels').textContent = models.length;
    document.getElementById('summaryDevices').textContent = (bc.devices_to_test || []).length;
    document.getElementById('summaryPrompts').textContent = (bc.test_prompts || []).length;

    document.getElementById('runBenchmarkBtn').disabled = models.length === 0;
}

function setupRunListeners() {
    document.getElementById('runBenchmarkBtn').addEventListener('click', startBenchmark);
    document.getElementById('cancelBenchmarkBtn').addEventListener('click', cancelBenchmark);
    document.getElementById('clearLogBtn').addEventListener('click', () => {
        document.getElementById('logContent').textContent = '';
    });
    document.getElementById('viewResultsBtn').addEventListener('click', () => {
        document.querySelector('[data-tab="results"]').click();
    });
}

async function startBenchmark() {
    if (benchmarkRunning) return;
    benchmarkRunning = true;

    // First save current settings
    collectSettings();
    await saveConfig();

    const logContent = document.getElementById('logContent');
    logContent.textContent = '';

    document.getElementById('runBenchmarkBtn').style.display = 'none';
    document.getElementById('cancelBenchmarkBtn').style.display = '';
    document.getElementById('viewResultsBtn').style.display = 'none';
    document.getElementById('progressSection').style.display = '';
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('progressText').textContent = 'Starting benchmark...';

    try {
        const response = await fetch('/api/benchmark/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let totalModels = 0;
        let currentModel = 0;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const event = JSON.parse(line.slice(6));
                    handleBenchmarkEvent(event);
                } catch (e) {
                    // Skip malformed events
                }
            }
        }
    } catch (e) {
        appendLog(`\nError: ${e.message}`);
    }

    benchmarkRunning = false;
    document.getElementById('cancelBenchmarkBtn').style.display = 'none';
    document.getElementById('runBenchmarkBtn').style.display = '';
}

function handleBenchmarkEvent(event) {
    if (event.type === 'output') {
        appendLog(event.data);
        parseProgress(event.data);
    } else if (event.type === 'done') {
        const success = event.exit_code === 0;
        appendLog(`\n--- ${event.data} ---`);
        document.getElementById('progressBar').style.width = '100%';
        document.getElementById('progressText').textContent = event.data;
        if (success) {
            document.getElementById('viewResultsBtn').style.display = '';
            showToast('Benchmark completed!', 'success');
        } else {
            showToast('Benchmark failed', 'error');
        }
    } else if (event.type === 'error') {
        appendLog(`\nError: ${event.data}`);
        showToast(event.data, 'error');
    }
}

function appendLog(text) {
    const logContent = document.getElementById('logContent');
    logContent.textContent += text + '\n';
    logContent.scrollTop = logContent.scrollHeight;
}

function parseProgress(line) {
    // Detect "MODEL X/Y" pattern
    const modelMatch = line.match(/MODEL\s+(\d+)\/(\d+)/i);
    if (modelMatch) {
        const current = parseInt(modelMatch[1]);
        const total = parseInt(modelMatch[2]);
        const pct = Math.round((current - 1) / total * 100);
        document.getElementById('progressBar').style.width = pct + '%';
        document.getElementById('progressText').textContent = `Model ${current} of ${total}`;
    }
}

async function cancelBenchmark() {
    try {
        await api('/api/benchmark/cancel', { method: 'POST' });
        showToast('Benchmark cancelled', 'info');
    } catch (e) {
        showToast('Failed to cancel', 'error');
    }
}

// ─── Results Tab ────────────────────────────────────────────────────────────

const DEVICE_COLORS = {
    CPU: { bg: 'rgba(76, 175, 80, 0.7)', border: '#4CAF50' },
    GPU: { bg: 'rgba(33, 150, 243, 0.7)', border: '#2196F3' },
    NPU: { bg: 'rgba(255, 152, 0, 0.7)', border: '#FF9800' },
};

function setupResultsListeners() {
    document.getElementById('historySelect').addEventListener('change', async (e) => {
        const val = e.target.value;
        if (val === 'latest') {
            await loadResults();
        } else {
            try {
                const data = await api(`/api/results/${val}`);
                renderResults(data);
            } catch (e) {
                showToast('Failed to load results', 'error');
            }
        }
    });

    // Table sorting
    document.querySelectorAll('#resultsTable th[data-sort]').forEach(th => {
        th.addEventListener('click', () => {
            const key = th.dataset.sort;
            sortResultsTable(key, th);
        });
    });
}

let currentSortKey = null;
let currentSortAsc = false;

function sortResultsTable(key, thElement) {
    const tbody = document.getElementById('resultsTableBody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    if (rows.length === 0) return;

    // Toggle direction
    if (currentSortKey === key) {
        currentSortAsc = !currentSortAsc;
    } else {
        currentSortKey = key;
        currentSortAsc = key === 'model' || key === 'device'; // asc for text, desc for numbers
    }

    // Update header indicators
    document.querySelectorAll('#resultsTable th[data-sort]').forEach(th => {
        th.textContent = th.textContent.replace(/ [▲▼]$/, '');
    });
    thElement.textContent += currentSortAsc ? ' ▲' : ' ▼';

    // Find column index by matching data-sort attribute
    const headers = Array.from(document.querySelectorAll('#resultsTable th[data-sort]'));
    const colIndex = headers.findIndex(th => th.dataset.sort === key);
    if (colIndex < 0) return;
    const isNumeric = colIndex >= 2;

    rows.sort((a, b) => {
        const aVal = a.cells[colIndex]?.textContent?.trim() || '';
        const bVal = b.cells[colIndex]?.textContent?.trim() || '';
        let cmp;
        if (isNumeric) {
            cmp = (parseFloat(aVal) || 0) - (parseFloat(bVal) || 0);
        } else {
            cmp = aVal.localeCompare(bVal);
        }
        return currentSortAsc ? cmp : -cmp;
    });

    rows.forEach(row => tbody.appendChild(row));
}

async function loadResults() {
    try {
        const data = await api('/api/results/latest');
        if (data.error) {
            document.getElementById('noResults').style.display = '';
            document.getElementById('resultsContent').style.display = 'none';
            return;
        }
        renderResults(data);

        // Load history
        const history = await api('/api/results/history');
        const select = document.getElementById('historySelect');
        select.innerHTML = '<option value="latest">Latest Run</option>';
        (history.runs || []).forEach(run => {
            const opt = document.createElement('option');
            opt.value = run.id;
            opt.textContent = `${run.timestamp} (${run.models_tested.length} models)`;
            select.appendChild(opt);
        });
    } catch (e) {
        document.getElementById('noResults').style.display = '';
        document.getElementById('resultsContent').style.display = 'none';
    }
}

function renderResults(data) {
    const results = data.results || {};
    const hasResults = Object.keys(results).some(m => Object.keys(results[m]).length > 0);

    if (!hasResults) {
        document.getElementById('noResults').style.display = '';
        document.getElementById('noResults').innerHTML = `
            <div>No benchmark data available yet.</div>
            <div style="margin-top:8px; font-size:12px; color:var(--text-muted)">
                Last run: ${data.timestamp || 'N/A'} — ${(data.models_tested || []).length} model(s) tested but no successful results recorded.
            </div>
        `;
        document.getElementById('resultsContent').style.display = 'none';
        return;
    }

    document.getElementById('noResults').style.display = 'none';
    document.getElementById('resultsContent').style.display = '';

    // Summary
    document.getElementById('resultsTimestamp').textContent = data.timestamp || '-';

    // Find best performer
    let bestSpeed = 0, bestModel = '', bestDevice = '';
    for (const [model, deviceResults] of Object.entries(results)) {
        for (const [device, metrics] of Object.entries(deviceResults)) {
            if (metrics.avg_speed > bestSpeed) {
                bestSpeed = metrics.avg_speed;
                bestModel = model;
                bestDevice = device;
            }
        }
    }
    document.getElementById('resultsBest').textContent = bestSpeed > 0
        ? `${bestModel} on ${bestDevice}: ${bestSpeed.toFixed(1)} tok/s`
        : '-';

    // Charts
    renderSpeedChart(data);
    renderLoadTimeChart(data);
    renderPowerChart(data);

    // Table
    renderResultsTable(data);
}

function renderSpeedChart(data) {
    const ctx = document.getElementById('speedChart').getContext('2d');
    if (speedChart) speedChart.destroy();

    const models = Object.keys(data.results || {});
    const allDevices = data.devices_tested || [];

    const datasets = allDevices.map(device => ({
        label: device,
        data: models.map(m => data.results[m]?.[device]?.avg_speed || 0),
        backgroundColor: DEVICE_COLORS[device]?.bg || 'rgba(150,150,150,0.7)',
        borderColor: DEVICE_COLORS[device]?.border || '#999',
        borderWidth: 1,
    }));

    speedChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: models, datasets },
        options: chartOptions('tokens/sec'),
    });
}

function renderLoadTimeChart(data) {
    const ctx = document.getElementById('loadTimeChart').getContext('2d');
    if (loadTimeChart) loadTimeChart.destroy();

    const models = Object.keys(data.results || {});
    const allDevices = data.devices_tested || [];

    const datasets = allDevices.map(device => ({
        label: device,
        data: models.map(m => data.results[m]?.[device]?.load_time || 0),
        backgroundColor: DEVICE_COLORS[device]?.bg || 'rgba(150,150,150,0.7)',
        borderColor: DEVICE_COLORS[device]?.border || '#999',
        borderWidth: 1,
    }));

    loadTimeChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: models, datasets },
        options: chartOptions('seconds'),
    });
}

function renderPowerChart(data) {
    const card = document.getElementById('powerChartCard');
    const results = data.results || {};

    // Check if any result has power data
    let hasPower = false;
    for (const deviceResults of Object.values(results)) {
        for (const m of Object.values(deviceResults)) {
            if (m.power && Object.keys(m.power).length > 0) {
                hasPower = true;
                break;
            }
        }
        if (hasPower) break;
    }

    if (!hasPower) {
        card.style.display = 'none';
        return;
    }
    card.style.display = '';

    const ctx = document.getElementById('powerChart').getContext('2d');
    if (powerChart) powerChart.destroy();

    const models = Object.keys(results);
    const allDevices = data.devices_tested || [];

    const datasets = allDevices.map(device => ({
        label: device,
        data: models.map(m => {
            const power = results[m]?.[device]?.power || {};
            const pkg = power['package-0'] || power['psys'] || {};
            return pkg.avg_watts || 0;
        }),
        backgroundColor: DEVICE_COLORS[device]?.bg || 'rgba(150,150,150,0.7)',
        borderColor: DEVICE_COLORS[device]?.border || '#999',
        borderWidth: 1,
    }));

    powerChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: models, datasets },
        options: chartOptions('Watts'),
    });
}

function chartOptions(yLabel) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { labels: { color: '#94A3B8', font: { family: "'IBM Plex Sans'" } } },
        },
        scales: {
            x: { ticks: { color: '#64748B', font: { size: 11 } }, grid: { color: 'rgba(148,163,184,0.06)' } },
            y: {
                ticks: { color: '#64748B' },
                grid: { color: 'rgba(148,163,184,0.06)' },
                title: { display: true, text: yLabel, color: '#94A3B8' },
            },
        },
    };
}

function renderResultsTable(data) {
    const tbody = document.getElementById('resultsTableBody');
    const thead = document.querySelector('#resultsTable thead tr');
    const rows = [];
    let maxSpeed = 0;

    // Detect power domains across all results
    const powerDomains = [];
    let hasPower = false;
    for (const deviceResults of Object.values(data.results || {})) {
        for (const m of Object.values(deviceResults)) {
            if (m.power) {
                for (const d of Object.keys(m.power)) {
                    hasPower = true;
                    if (!powerDomains.includes(d)) powerDomains.push(d);
                }
            }
        }
    }

    // Rebuild header with power columns
    let headerHtml = `
        <th data-sort="model">Model</th>
        <th data-sort="device">Device</th>
        <th data-sort="speed">Speed (tok/s)</th>
        <th data-sort="load">Load Time (s)</th>
        <th data-sort="avg">Avg Time (s)</th>
        <th data-sort="tokens">Total Tokens</th>`;
    for (const d of powerDomains) {
        headerHtml += `<th data-sort="power_${d}">${d} (W)</th>`;
    }
    if (hasPower) {
        headerHtml += `<th data-sort="tokj">tok/J</th>`;
    }
    thead.innerHTML = headerHtml;

    // Re-attach sort listeners
    thead.querySelectorAll('th[data-sort]').forEach(th => {
        th.addEventListener('click', () => sortResultsTable(th.dataset.sort, th));
    });

    for (const [model, deviceResults] of Object.entries(data.results || {})) {
        for (const [device, m] of Object.entries(deviceResults)) {
            rows.push({ model, device, ...m });
            if (m.avg_speed > maxSpeed) maxSpeed = m.avg_speed;
        }
    }

    rows.sort((a, b) => b.avg_speed - a.avg_speed);

    tbody.innerHTML = rows.map(r => {
        const power = r.power || {};
        let powerCells = '';
        let pkgWatts = null;
        for (const d of powerDomains) {
            const w = power[d]?.avg_watts;
            powerCells += `<td>${w != null ? w.toFixed(1) : '-'}</td>`;
            if ((d === 'package-0' || d === 'psys') && w != null) pkgWatts = w;
        }
        if (hasPower) {
            const tokJ = pkgWatts && pkgWatts > 0 ? (r.avg_speed / pkgWatts).toFixed(2) : '-';
            powerCells += `<td>${tokJ}</td>`;
        }
        return `
        <tr class="${r.avg_speed === maxSpeed ? 'best-row' : ''}">
            <td>${escapeHtml(r.model)}</td>
            <td><span class="badge-device badge-${r.device.toLowerCase()}">${r.device}</span></td>
            <td>${r.avg_speed?.toFixed(1) || '-'}</td>
            <td>${r.load_time?.toFixed(2) || '-'}</td>
            <td>${r.avg_time?.toFixed(2) || '-'}</td>
            <td>${r.total_tokens || '-'}</td>
            ${powerCells}
        </tr>`;
    }).join('');
}

// ─── Config Persistence ─────────────────────────────────────────────────────

async function saveConfig() {
    try {
        await api('/api/config', {
            method: 'POST',
            body: JSON.stringify(config),
        });
    } catch (e) {
        showToast('Failed to save config', 'error');
    }
}

// ─── Utilities ──────────────────────────────────────────────────────────────

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function formatDownloads(n) {
    if (!n) return '0';
    if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
    return n.toString();
}

// ─── Connection Monitor ─────────────────────────────────────────────────────

async function checkConnection() {
    const dot = document.querySelector('.status-dot');
    const text = document.querySelector('.status-text');
    try {
        const res = await fetch('/api/health');
        if (res.ok) {
            dot.className = 'status-dot';
            text.textContent = 'Connected';
        } else {
            dot.className = 'status-dot error';
            text.textContent = 'Server error';
        }
    } catch {
        dot.className = 'status-dot error';
        text.textContent = 'Disconnected';
    }
}

setInterval(checkConnection, 10000);

// ─── Start ──────────────────────────────────────────────────────────────────

init();
