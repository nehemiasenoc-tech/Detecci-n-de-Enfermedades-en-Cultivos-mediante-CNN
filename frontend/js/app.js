const API_URL = window.location.origin + '/predict';
const HEALTH_URL = window.location.origin + '/health';
const imageInput = document.getElementById('imageInput');
const uploadArea = document.getElementById('uploadArea');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const resultCard = document.getElementById('resultCard');
const errorMessage = document.getElementById('errorMessage');
const apiDot = document.getElementById('apiDot');
const apiStatus = document.getElementById('apiStatus');

let selectedFile = null;

async function checkAPI() {
    try {
        const res = await fetch(HEALTH_URL);
        if (res.ok) {
            apiDot.className = 'dot online';
            apiStatus.textContent = 'API conectada';
        } else {
            throw new Error('API error');
        }
    } catch (e) {
        apiDot.className = 'dot offline';
        apiStatus.textContent = 'API desconectada';
    }
}

checkAPI();
setInterval(checkAPI, 30000);

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

function handleFile(file) {
    errorMessage.style.display = 'none';

    if (file.size > 10 * 1024 * 1024) {
        showError('El archivo supera 10MB. Seleccione uno más pequeño.');
        return;
    }

    const validTypes = ['image/jpeg', 'image/png', 'image/bmp', 'image/gif'];
    if (!validTypes.includes(file.type)) {
        showError('Formato no válido. Use JPG, PNG o BMP.');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    loading.style.display = 'block';
    resultCard.style.display = 'none';
    errorMessage.style.display = 'none';
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.mensaje || data.error || 'Error del servidor');
        }

        document.getElementById('resultCultivo').textContent = data.cultivo;
        document.getElementById('resultEnfermedad').textContent = data.enfermedad;
        document.getElementById('resultProbabilidad').textContent = data.probabilidad + '%';

        const fill = document.getElementById('probabilityFill');
        fill.style.width = '0%';
        setTimeout(() => {
            fill.style.width = data.probabilidad + '%';
        }, 100);

        const badge = document.getElementById('statusBadge');
        if (data.enfermedad.toLowerCase().includes('healthy') ||
            data.enfermedad.toLowerCase().includes('sano')) {
            badge.textContent = 'SANO';
            badge.className = 'status-badge status-healthy';
        } else {
            badge.textContent = 'ENFERMEDAD DETECTADA';
            badge.className = 'status-badge status-diseased';
        }

        resultCard.style.display = 'block';

    } catch (error) {
        showError('Error: ' + error.message);
    } finally {
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
    }
});

function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.style.display = 'block';
}
