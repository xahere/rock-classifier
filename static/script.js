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

    // Feedback V2 Elements
    const feedbackStep1 = document.getElementById('feedback-step-1');
    const feedbackStep2 = document.getElementById('feedback-step-2');
    const yesBtn = document.getElementById('feedback-yes-btn');
    const noBtn = document.getElementById('feedback-no-btn');
    const submitBtn = document.getElementById('feedback-submit-btn');
    const correctLabelSelect = document.getElementById('correct-label-select');
    const feedbackTitle = document.getElementById('feedback-q');

    let currentFile = null;
    let currentServerFilename = null;

    // --- Drag & Drop ---
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

    function highlight(e) { dropZone.classList.add('dragover'); }
    function unhighlight(e) { dropZone.classList.remove('dragover'); }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

    function handleFiles(files) {
        if (files.length > 0) {
            currentFile = files[0];
            showPreview(currentFile);
            analyzeBtn.disabled = false;
            hideResult();
            resetFeedbackState(); // Ensure feedback UI is ready for new analysis
        }
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function () {
            imagePreview.src = reader.result;
            dropZone.style.display = 'none';
            previewContainer.classList.remove('hidden');
        }
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUI();
    });

    function resetUI() {
        currentFile = null;
        currentServerFilename = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        dropZone.style.display = 'block';
        analyzeBtn.disabled = true;
        hideResult();
        resetFeedbackState();
    }

    function resetFeedbackState() {
        feedbackStep1.classList.remove('hidden');
        feedbackStep2.classList.add('hidden');
        correctLabelSelect.selectedIndex = 0;
        if (feedbackTitle) feedbackTitle.textContent = "Is this result correct?";
    }

    function hideResult() {
        resultCard.classList.remove('visible');
        setTimeout(() => {
            if (!resultCard.classList.contains('visible')) {
                resultCard.classList.add('hidden');
            }
        }, 500);
    }

    // --- Analyze ---
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

            if (!response.ok) throw new Error('Analysis failed');

            const result = await response.json();
            showResult(result);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred. Please try again.');
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
        currentServerFilename = data.filename;

        resetFeedbackState(); // Reset questions
        resultCard.classList.remove('hidden');
        void resultCard.offsetWidth;
        resultCard.classList.add('visible');
    }

    // --- Feedback Flow V2 ---

    yesBtn.addEventListener('click', () => {
        // 1. Appreciate
        alert("Great! I'm glad I got it right.");
        // 2. Reset UI immediately
        resetUI();
    });

    noBtn.addEventListener('click', () => {
        // Switch to Step 2 (Dropdown)
        feedbackStep1.classList.add('hidden');
        feedbackStep2.classList.remove('hidden');
    });

    submitBtn.addEventListener('click', async () => {
        const selectedLabel = correctLabelSelect.value;
        if (!selectedLabel) {
            alert("Please select the correct rock type.");
            return;
        }

        const originalText = submitBtn.textContent;
        submitBtn.textContent = "Processing...";
        submitBtn.disabled = true;

        try {
            // Trigger backend training (terminal will show progress)
            const response = await fetch('/submit_feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename: currentServerFilename,
                    correct_label: selectedLabel
                })
            });

            if (response.ok) {
                // Simple acknowledgement
                alert("Thank you. I have corrected this entry and will learn from it.");
                resetUI();
            } else {
                throw new Error("Server error");
            }
        } catch (e) {
            alert("Error: " + e.message);
        } finally {
            submitBtn.textContent = originalText;
            submitBtn.disabled = false;
        }
    });

});
