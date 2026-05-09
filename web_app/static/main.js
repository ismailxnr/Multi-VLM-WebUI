document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const imagePreview = document.getElementById('image-preview');
    const removeImageBtn = document.getElementById('remove-image');
    const compareBtn = document.getElementById('compare-btn');
    const promptInput = document.getElementById('prompt-input');
    const modelGroups = document.getElementById('model-groups');
    const resultsEmpty = document.getElementById('results-empty');
    const progressContainer = document.getElementById('progress-container');
    const progressText = document.getElementById('progress-text');
    const progressCount = document.getElementById('progress-count');
    const progressFill = document.getElementById('progress-fill');
    const resultCards = document.getElementById('result-cards');

    let currentFile = null;
    let selectedModels = new Set();
    let isComparing = false;

    // ─── Load Models ───
    async function loadModels() {
        try {
            const res = await fetch('/api/models');
            const grouped = await res.json();
            modelGroups.innerHTML = '';

            for (const [groupName, models] of Object.entries(grouped)) {
                const groupDiv = document.createElement('div');

                const title = document.createElement('div');
                title.className = 'model-group-title';
                title.textContent = groupName;
                groupDiv.appendChild(title);

                const chipsDiv = document.createElement('div');
                chipsDiv.className = 'model-checkboxes';

                models.forEach(m => {
                    const itemContainer = document.createElement('div');
                    itemContainer.style.display = 'flex';
                    itemContainer.style.alignItems = 'center';
                    itemContainer.style.gap = '0.3rem';
                    itemContainer.style.marginBottom = '0.3rem';

                    const chip = document.createElement('label');
                    chip.className = 'model-chip';
                    chip.innerHTML = `
                        <input type="checkbox" value="${m.key}">
                        <span class="chip-check"><i class="fa-solid fa-check"></i></span>
                        <span>${m.name}</span>
                    `;

                    const cb = chip.querySelector('input');
                    // restore selected state if any
                    if (selectedModels.has(m.key)) {
                        cb.checked = true;
                        chip.classList.add('selected');
                    }

                    cb.addEventListener('change', () => {
                        if (cb.checked) {
                            selectedModels.add(m.key);
                            chip.classList.add('selected');
                        } else {
                            selectedModels.delete(m.key);
                            chip.classList.remove('selected');
                        }
                        updateCompareBtn();
                    });

                    itemContainer.appendChild(chip);

                    // Add remove button for custom models
                    if (m.key.startsWith('custom/')) {
                        const removeBtn = document.createElement('button');
                        removeBtn.className = 'remove-model-btn';
                        removeBtn.innerHTML = '<i class="fa-solid fa-trash"></i>';
                        removeBtn.title = "Modeli Kaldır";
                        removeBtn.addEventListener('click', async (e) => {
                            e.preventDefault();
                            if (confirm(`"${m.name}" modelini listeden kaldırmak istiyor musunuz?`)) {
                                const fd = new FormData();
                                fd.append('key', m.key);
                                try {
                                    await fetch('/api/models/remove', { method: 'DELETE', body: fd });
                                    selectedModels.delete(m.key);
                                    updateCompareBtn();
                                    await loadModels();
                                } catch(err) {
                                    console.error('Silme hatası:', err);
                                }
                            }
                        });
                        itemContainer.appendChild(removeBtn);
                    }

                    chipsDiv.appendChild(itemContainer);
                });

                groupDiv.appendChild(chipsDiv);
                modelGroups.appendChild(groupDiv);
            }
        } catch (err) {
            modelGroups.innerHTML = '<p style="color: var(--danger); font-size: 0.8rem;">Model listesi yüklenemedi.</p>';
        }
    }

    loadModels();

    // ─── Update Button State ───
    function updateCompareBtn() {
        const canCompare = currentFile && selectedModels.size > 0 && !isComparing;
        compareBtn.disabled = !canCompare;
    }

    // ─── Drag & Drop ───
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev =>
        dropZone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); }, false)
    );
    ['dragenter', 'dragover'].forEach(ev =>
        dropZone.addEventListener(ev, () => dropZone.classList.add('dragover'), false)
    );
    ['dragleave', 'drop'].forEach(ev =>
        dropZone.addEventListener(ev, () => dropZone.classList.remove('dragover'), false)
    );

    dropZone.addEventListener('drop', e => handleFiles(e.dataTransfer.files), false);
    dropZone.addEventListener('click', () => { if (!currentFile) fileInput.click(); });
    fileInput.addEventListener('change', function() { handleFiles(this.files); });

    function handleFiles(files) {
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            currentFile = files[0];
            const reader = new FileReader();
            reader.readAsDataURL(files[0]);
            reader.onloadend = () => {
                imagePreview.src = reader.result;
                uploadPlaceholder.classList.add('hidden');
                imagePreview.classList.remove('hidden');
                removeImageBtn.classList.remove('hidden');
                updateCompareBtn();
            };
        }
    }

    removeImageBtn.addEventListener('click', e => {
        e.stopPropagation();
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        uploadPlaceholder.classList.remove('hidden');
        imagePreview.classList.add('hidden');
        removeImageBtn.classList.add('hidden');
        updateCompareBtn();
    });

    // ─── Compare ───
    compareBtn.addEventListener('click', runComparison);

    async function runComparison() {
        if (!currentFile || selectedModels.size === 0 || isComparing) return;

        isComparing = true;
        compareBtn.disabled = true;
        compareBtn.querySelector('span').textContent = 'İşleniyor...';
        compareBtn.querySelector('i').className = 'fa-solid fa-spinner fa-spin';

        // Reset results
        resultsEmpty.classList.add('hidden');
        progressContainer.classList.remove('hidden');
        resultCards.innerHTML = '';
        progressFill.style.width = '0%';

        const modelKeys = Array.from(selectedModels);
        const formData = new FormData();
        formData.append('image', currentFile);
        formData.append('models', JSON.stringify(modelKeys));
        if (promptInput.value.trim()) {
            formData.append('prompt', promptInput.value.trim());
        }

        try {
            const response = await fetch('/api/compare', {
                method: 'POST',
                body: formData
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // keep incomplete line

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            handleSSEvent(data);
                        } catch (e) { /* skip parse errors */ }
                    }
                }
            }
        } catch (err) {
            console.error('Comparison error:', err);
            resultCards.innerHTML += `
                <div class="result-card error" style="animation-delay: 0s">
                    <div class="result-caption">Bağlantı hatası: ${err.message}</div>
                </div>`;
        } finally {
            isComparing = false;
            compareBtn.disabled = false;
            compareBtn.querySelector('span').textContent = 'Karşılaştır';
            compareBtn.querySelector('i').className = 'fa-solid fa-code-compare';
            updateCompareBtn();
        }
    }

    function handleSSEvent(data) {
        if (data.type === 'loading') {
            progressText.textContent = `Yükleniyor: ${data.model_name}`;
            progressCount.textContent = `${data.index + 1}/${data.total}`;

            // Add loading card
            const card = document.createElement('div');
            card.className = 'result-card loading';
            card.id = `card-${data.index}`;
            card.innerHTML = `
                <div class="result-card-header">
                    <div class="result-model-name">
                        <i class="fa-solid fa-brain" style="color: var(--primary)"></i>
                        ${data.model_name}
                    </div>
                </div>
                <div class="result-caption">
                    <div class="pulse-dot"></div>
                    Model yükleniyor ve çıktı üretiliyor...
                </div>`;
            resultCards.appendChild(card);
            card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        } else if (data.type === 'result') {
            const pct = ((data.index + 1) / data.total * 100).toFixed(0);
            progressFill.style.width = pct + '%';
            progressText.textContent = `Tamamlandı: ${data.model_name}`;
            progressCount.textContent = `${data.index + 1}/${data.total}`;

            const card = document.getElementById(`card-${data.index}`);
            if (card) {
                card.className = 'result-card';
                card.innerHTML = `
                    <div class="result-card-header">
                        <div class="result-model-name">
                            <i class="fa-solid fa-brain" style="color: var(--success)"></i>
                            ${data.model_name}
                            <span class="model-badge">${data.model_key.split('/').pop()}</span>
                        </div>
                        <div class="result-times">
                            <span><i class="fa-solid fa-download"></i> ${data.load_time}s yükleme</span>
                            <span><i class="fa-solid fa-bolt"></i> ${data.infer_time}s çıktı</span>
                        </div>
                    </div>
                    <div class="result-caption">${data.caption}</div>`;
            }

        } else if (data.type === 'error') {
            const card = document.getElementById(`card-${data.index}`);
            if (card) {
                card.className = 'result-card error';
                card.innerHTML = `
                    <div class="result-card-header">
                        <div class="result-model-name">
                            <i class="fa-solid fa-triangle-exclamation" style="color: var(--danger)"></i>
                            ${data.model_name}
                        </div>
                    </div>
                    <div class="result-caption">Hata: ${data.error}</div>`;
            }

        } else if (data.type === 'done') {
            progressText.textContent = 'Tüm modeller tamamlandı!';
            progressFill.style.width = '100%';
        }
    }

    // ─── Custom Model Add Logic ───
    const toggleAddModel = document.getElementById('toggle-add-model');
    const addModelBody = document.getElementById('add-model-body');
    const addModelIcon = document.getElementById('add-model-icon');
    const addModelBtn = document.getElementById('add-model-btn');
    const customName = document.getElementById('custom-name');
    const customPath = document.getElementById('custom-path');
    const customFamily = document.getElementById('custom-family');
    const addModelStatus = document.getElementById('add-model-status');

    toggleAddModel.addEventListener('click', () => {
        addModelBody.classList.toggle('hidden');
        toggleAddModel.classList.toggle('open');
    });

    addModelBtn.addEventListener('click', async () => {
        const nameVal = customName.value.trim();
        const pathVal = customPath.value.trim();
        const familyVal = customFamily.value;

        if (!nameVal || !pathVal) {
            addModelStatus.textContent = 'Lütfen isim ve yol/ID girin.';
            addModelStatus.style.color = 'var(--warning)';
            return;
        }

        addModelBtn.disabled = true;
        addModelBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Ekleniyor...';
        addModelStatus.textContent = '';

        const formData = new FormData();
        formData.append('name', nameVal);
        formData.append('path', pathVal);
        formData.append('family', familyVal);

        try {
            const res = await fetch('/api/models/add', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (data.status === 'success') {
                addModelStatus.textContent = 'Model eklendi! Listede görebilirsiniz.';
                addModelStatus.style.color = 'var(--success)';
                customName.value = '';
                customPath.value = '';
                // Reload model list
                await loadModels();
            } else {
                addModelStatus.textContent = 'Hata: ' + data.message;
                addModelStatus.style.color = 'var(--danger)';
            }
        } catch (err) {
            addModelStatus.textContent = 'Bağlantı hatası.';
            addModelStatus.style.color = 'var(--danger)';
        } finally {
            addModelBtn.disabled = false;
            addModelBtn.innerHTML = '<i class="fa-solid fa-download"></i> Kütüphaneye Ekle';
        }
    });

});
