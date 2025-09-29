'use strict';

const API_BASE_URL = 'http://localhost:8000';

class ReviewCSVApp {
    constructor(root = document) {
        this.root = root;
        this.state = {
            file: null,
            lastResult: null,
        };

        this.elements = this.mapElements();
        if (!this.elements) {
            console.warn('CSV upload UI not detected on this page.');
            return;
        }

        this.bindEvents();
        this.resetUI();
    }

    mapElements() {
        const get = (id) => this.root.getElementById ? this.root.getElementById(id) : document.getElementById(id);
        const uploadArea = get('uploadArea');
        const fileInput = get('csvFile');
        const processBtn = get('processBtn');

        const elements = {
            uploadArea,
            fileInput,
            removeFile: get('removeFile'),
            fileInfo: get('fileInfo'),
            fileName: get('fileName'),
            fileSize: get('fileSize'),
            processBtn,
            btnText: processBtn ? processBtn.querySelector('.btn-text') : null,
            btnSpinner: processBtn ? processBtn.querySelector('.btn-spinner') : null,
            progressContainer: get('progressContainer'),
            progressBarFill: get('progressBarFill'),
            progressStatus: get('progressStatus'),
            resultCard: get('resultCard'),
            processedRows: get('processedRows'),
            positiveReviews: get('positiveReviews'),
            negativeReviews: get('negativeReviews'),
            processingTime: get('processingTime'),
            downloadBtn: get('downloadBtn'),
            errorMessage: get('errorMessage'),
            errorText: get('errorText'),
        };

        const missing = ['uploadArea', 'fileInput', 'processBtn'].filter((key) => !elements[key]);
        if (missing.length) {
            console.error(`Missing required CSV elements: ${missing.join(', ')}`);
            return null;
        }

        return elements;
    }

    bindEvents() {
        const { uploadArea, fileInput, removeFile, processBtn, downloadBtn, errorMessage } = this.elements;

        uploadArea.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', (event) => {
            const [file] = event.target.files || [];
            this.handleFileSelection(file);
        });

        if (removeFile) {
            removeFile.addEventListener('click', () => this.resetFileSelection());
        }

        if (processBtn) {
            processBtn.addEventListener('click', () => this.submitForAnalysis());
        }

        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.downloadLatestResult());
        }

        ['dragenter', 'dragover'].forEach((eventName) => {
            uploadArea.addEventListener(eventName, (event) => {
                this.prevent(event);
                uploadArea.classList.add('dragover');
            });
        });

        ['dragleave', 'dragend', 'drop'].forEach((eventName) => {
            uploadArea.addEventListener(eventName, (event) => {
                this.prevent(event);
                uploadArea.classList.remove('dragover');
            });
        });

        uploadArea.addEventListener('drop', (event) => {
            const files = event.dataTransfer?.files;
            const file = files && files[0];
            this.handleFileSelection(file);
        });

        if (errorMessage) {
            errorMessage.addEventListener('click', () => this.hideError());
        }
    }

    prevent(event) {
        event.preventDefault();
        event.stopPropagation();
    }

    handleFileSelection(file) {
        if (!file) return;

        if (!this.isCSV(file)) {
            this.showError('Выберите файл с расширением .csv');
            return;
        }

        if (file.size === 0) {
            this.showError('Файл пустой. Загрузите файл с данными.');
            return;
        }

        this.state.file = file;
        this.elements.fileInput.value = '';
        this.updateFileInfo(file);
        this.toggleProcessButton(true);
        this.hideError();
        this.hideResult();
    }

    isCSV(file) {
        return file.name.toLowerCase().endsWith('.csv');
    }

    updateFileInfo(file) {
        const { fileInfo, fileName, fileSize } = this.elements;
        if (!fileInfo) return;

        if (fileName) fileName.textContent = file.name;
        if (fileSize) fileSize.textContent = this.formatFileSize(file.size);
        fileInfo.classList.remove('hidden');
    }

    resetFileSelection() {
        this.state.file = null;
        if (this.elements.fileInput) {
            this.elements.fileInput.value = '';
        }
        if (this.elements.fileInfo) {
            this.elements.fileInfo.classList.add('hidden');
        }
        this.hideResult();
        this.toggleProcessButton(false);
    }

    toggleProcessButton(enabled) {
        const { processBtn } = this.elements;
        if (!processBtn) return;
        processBtn.disabled = !enabled;
    }

    async submitForAnalysis() {
        if (!this.state.file) {
            this.showError('Сначала выберите CSV файл');
            return;
        }

        this.hideError();
        this.showResultPlaceholder(false);
        this.setProcessingState(true, 'Uploading...');
        this.updateProgress(10, 'Отправляем файл на сервер...');

        try {
            const formData = new FormData();
            formData.append('file', this.state.file);

            const response = await fetch(`${API_BASE_URL}/analyze-csv`, {
                method: 'POST',
                body: formData,
            });
            
            if (!response.ok) {
                const errorPayload = await this.safeParseJSON(response);
                const message = errorPayload?.detail || 'Ошибка сервера при анализе файла';
                throw new Error(message);
            }
            
            this.updateProgress(70, 'Анализируем данные нейросетью...');
                const result = await response.json();

            this.state.lastResult = result;
            this.renderResult(result);
            this.updateProgress(100, 'Готово!');
            setTimeout(() => this.resetProgress(), 1200);
        } catch (error) {
            console.error('Analyze CSV error:', error);
            this.updateProgress(0, 'Ошибка');
            this.showError(error.message || 'Не удалось обработать CSV файл');
        } finally {
            this.setProcessingState(false);
        }
    }

    renderResult(result) {
        const { resultCard, processedRows, positiveReviews, negativeReviews, processingTime } = this.elements;
        if (!resultCard) return;

        if (processedRows) processedRows.textContent = this.formatNumber(result.total_reviews);
        if (positiveReviews) positiveReviews.textContent = this.formatNumber(result.clean_reviews);
        if (negativeReviews) negativeReviews.textContent = this.formatNumber(result.flagged_reviews);
        if (processingTime) processingTime.textContent = result.processing_time.toFixed(2);

        resultCard.classList.remove('hidden');
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
        this.showResultPlaceholder(true);
    }

    hideResult() {
        const { resultCard } = this.elements;
        if (resultCard) {
            resultCard.classList.add('hidden');
        }
        this.state.lastResult = null;
        this.showResultPlaceholder(false);
    }

    showResultPlaceholder(hasResult) {
        if (this.elements.downloadBtn) {
            this.elements.downloadBtn.disabled = !hasResult;
        }
    }

    setProcessingState(isProcessing, label) {
        const { processBtn, btnText, btnSpinner } = this.elements;
        if (!processBtn) return;

        processBtn.disabled = isProcessing || !this.state.file;

        if (btnText) {
            btnText.textContent = isProcessing ? label || 'Processing...' : 'Analyze the answers';
        }

        if (btnSpinner) {
            btnSpinner.classList.toggle('hidden', !isProcessing);
        }
    }

    updateProgress(percent, status) {
        const { progressContainer, progressBarFill, progressStatus } = this.elements;
        if (!progressContainer) return;

        progressContainer.classList.remove('hidden');
        if (progressBarFill) {
            const clamped = Math.min(100, Math.max(0, percent));
            progressBarFill.style.width = `${clamped}%`;
        }

        if (progressStatus && status) {
            progressStatus.textContent = status;
        }
    }

    resetProgress() {
        const { progressContainer, progressBarFill, progressStatus } = this.elements;
        if (progressContainer) progressContainer.classList.add('hidden');
        if (progressBarFill) progressBarFill.style.width = '0%';
        if (progressStatus) progressStatus.textContent = '';
    }

    async downloadLatestResult() {
        if (!this.state.lastResult) {
            this.showError('Нет данных для скачивания');
            return;
        }

        this.setDownloadState(true);

        try {
            const response = await fetch(`${API_BASE_URL}/download-result`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.state.lastResult),
            });

            if (!response.ok) {
                throw new Error('Не удалось сформировать CSV файл');
            }

            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `analyzed_reviews_${Date.now()}.csv`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            setTimeout(() => URL.revokeObjectURL(url), 100);
        } catch (error) {
            console.error('Download error:', error);
            this.showError(error.message || 'Ошибка скачивания файла');
        } finally {
            this.setDownloadState(false);
        }
    }

    setDownloadState(isDownloading) {
        const { downloadBtn } = this.elements;
        if (!downloadBtn) return;
        downloadBtn.disabled = isDownloading || !this.state.lastResult;
        downloadBtn.classList.toggle('loading', isDownloading);
    }

    showError(message) {
        const { errorMessage, errorText } = this.elements;
        if (!errorMessage) {
            alert(message);
            return;
        }

        if (errorText) {
        errorText.textContent = message;
        }

        errorMessage.classList.remove('hidden');
        errorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });

        clearTimeout(this.errorTimeout);
        this.errorTimeout = setTimeout(() => this.hideError(), 5000);
    }

    hideError() {
        const { errorMessage } = this.elements;
        if (errorMessage) {
            errorMessage.classList.add('hidden');
        }
        if (this.errorTimeout) {
            clearTimeout(this.errorTimeout);
        }
    }

    resetUI() {
        this.resetFileSelection();
        this.resetProgress();
        this.hideError();
        this.showResultPlaceholder(false);
        if (this.elements.downloadBtn) {
            this.elements.downloadBtn.classList.remove('loading');
        }
    }

    formatFileSize(bytes) {
        if (!Number.isFinite(bytes)) return '0 B';
        if (bytes === 0) return '0 B';

        const units = ['B', 'KB', 'MB', 'GB'];
        const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
        const value = bytes / Math.pow(1024, i);
        const formatted = value >= 10 || i === 0 ? value.toFixed(0) : value.toFixed(1);
        return `${formatted} ${units[i]}`;
    }

    formatNumber(value) {
        return typeof value === 'number' ? value.toLocaleString('ru-RU') : '0';
    }

    async safeParseJSON(response) {
        try {
            return await response.json();
        } catch (error) {
            return null;
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new ReviewCSVApp();
});