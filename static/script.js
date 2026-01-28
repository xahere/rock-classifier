document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultCard = document.getElementById('result-card');
    const resultType = document.getElementById('result-type');
    const resultConfidence = document.getElementById('result-confidence');
    const loader = document.querySelector('.loader');

    let currentFile = null;

    // Handle Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('dragover');
    }

    function unhighlight(e) {
        dropZone.classList.remove('dragover');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // Handle Click
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

    function handleFiles(files) {
        if (files.length > 0) {
            currentFile = files[0];
            showPreview(currentFile);
            analyzeBtn.disabled = false;
            hideResult();
        }
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            dropZone.style.display = 'none';
            previewContainer.classList.remove('hidden');
        }
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // prevent triggering dropZone click if nested (it isn't now but safe practice)
        resetUI();
    });

    function resetUI() {
        currentFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        dropZone.style.display = 'block';
        analyzeBtn.disabled = true;
        hideResult();
    }

    function hideResult() {
        resultCard.classList.remove('visible');
        setTimeout(() => {
            if (!resultCard.classList.contains('visible')) {
                resultCard.classList.add('hidden');
            }
        }, 500); // match transition
    }

    // Analysis
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        setLoading(true);
        hideResult();

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const result = await response.json();
            showResult(result);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis. Please try again.');
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        analyzeBtn.disabled = isLoading;
        const btnText = analyzeBtn.querySelector('span');
        if (isLoading) {
            loader.classList.remove('hidden');
            btnText.style.opacity = '0';
        } else {
            loader.classList.add('hidden');
            btnText.style.opacity = '1';
        }
    }

    function showResult(data) {
        resultType.textContent = data.class;
        resultConfidence.textContent = data.confidence;
        
        // Color coding based on type? (Optional)
        
        resultCard.classList.remove('hidden');
        // Trigger reflow
        void resultCard.offsetWidth;
        resultCard.classList.add('visible');
    }
});
