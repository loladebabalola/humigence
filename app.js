// Constants for default values
const DEFAULTS_FOR_BASIC = {
    'epochs': '3',
    'learning-rate': '', // Always start empty as per spec
    'warmup-ratio': '0.03',
    'weight-decay': '0',
    'scheduler': 'linear',
    'max-seq-len': '2048',
    'packing': 'on',
    'template': 'qwen_chat_basic_v1',
    'train-split': '80',
    'val-split': '10',
    'test-split': '10',
    'lora-rank': '8',
    'lora-alpha': '16',
    'lora-dropout': '0.05',
    'dataset-schema': 'chat_messages',
    'precision-mode': 'qlora_nf4'
};

// Function to prefill advanced defaults when in Basic mode
function prefillAdvancedDefaults() {
    for (const [fieldId, value] of Object.entries(DEFAULTS_FOR_BASIC)) {
        const element = document.getElementById(fieldId);
        if (element && fieldId !== 'learning-rate') {
            element.value = value;
        }
    }
    // Always keep learning rate empty
    const learningRate = document.getElementById('learning-rate');
    if (learningRate) {
        learningRate.value = '';
    }
}

// Application state
const state = {
    currentStep: 1,
    maxUnlockedStep: 1,
    mode: 'basic',
    config: {
        project: 'myproject',
        compute: {
            gpu: 'GPU0: RTX 4080 16GB'
        },
        model: {
            repo: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            precision_mode: 'qlora_nf4'
        },
        data: {
            raw_path: 'data/raw/oa.jsonl',
            schema: 'chat_messages',
            max_seq_len: 1024,
            packing: true,
            template: 'qwen_chat_basic_v1',
            splits: {
                train: 80,
                val: 10,
                test: 10
            }
        },
        train: {
            lr: 0.0001,
            warmup_ratio: 0.05,
            weight_decay: 0.01,
            scheduler: 'cosine',
            epochs: 2,
            enabled: false,
            lora: {
                r: 16,
                alpha: 32,
                dropout: 0.05
            }
        },
        eval: {
            curated_prompts_path: 'configs/curated_eval_prompts.jsonl',
            sampling: 'off'
        }
    },
    planGenerated: false
};

// DOM elements
const elements = {
    getStartedBtn: document.getElementById('get-started-btn'),
    wizardSection: document.getElementById('wizard-section'),
    toastContainer: document.getElementById('toast-container'),
    jsonPreview: document.getElementById('json-preview'),
    commandPreview: document.getElementById('command-preview')
};

// Utility functions
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    elements.toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease-out forwards';
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }, 3000);
}

function smoothScrollTo(element) {
    element.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'start' 
    });
}

function updateProgressBar() {
    const progressSteps = document.querySelectorAll('.progress-step');
    
    progressSteps.forEach((step, index) => {
        const stepNumber = index + 1;
        
        if (stepNumber < state.currentStep) {
            step.classList.add('completed');
            step.classList.remove('active');
        } else if (stepNumber === state.currentStep) {
            step.classList.add('active');
            step.classList.remove('completed');
        } else {
            step.classList.remove('active', 'completed');
        }
    });
}

function updateJsonPreview() {
    const jsonString = JSON.stringify(state.config, null, 2);
    if (elements.jsonPreview) {
        elements.jsonPreview.innerHTML = `<code>${jsonString}</code>`;
    }
}

function updateCommandPreview() {
    let commandHtml = '';
    
    if (state.mode === 'basic') {
        commandHtml = `
            <div class="command-section">
                <code>humigence init --mode basic --run plan</code>
                <div class="command-comment"># then (optional, to train):</div>
                <code>TRAIN=1 humigence init --mode basic --run pipeline</code>
                <div class="command-separator"></div>
                <div class="command-comment"># Make targets:</div>
                <code>make setup-basic</code>
                <code>make setup-advanced</code>
            </div>
        `;
    } else {
        commandHtml = `
            <div class="command-section">
                <code>humigence pipeline --config configs/humigence.basic.json --train</code>
                <div class="command-separator"></div>
                <div class="command-comment"># Make targets:</div>
                <code>make setup-basic</code>
                <code>make setup-advanced</code>
            </div>
        `;
    }
    
    if (elements.commandPreview) {
        elements.commandPreview.innerHTML = commandHtml;
    }
}

function showStep(stepNumber) {
    // Hide all steps
    document.querySelectorAll('.wizard-step').forEach(step => {
        step.classList.remove('active');
        step.classList.add('hidden');
    });
    
    // Show current step
    const currentStepElement = document.getElementById(`step-${stepNumber}`);
    if (currentStepElement) {
        currentStepElement.classList.add('active');
        currentStepElement.classList.remove('hidden');
    }
    
    state.currentStep = stepNumber;
    updateProgressBar();
}

function unlockNextStep() {
    if (state.currentStep < 5) {
        state.maxUnlockedStep = Math.max(state.maxUnlockedStep, state.currentStep + 1);
        showStep(state.currentStep + 1);
        
        // Smooth scroll to the new step
        const nextStepElement = document.getElementById(`step-${state.currentStep}`);
        if (nextStepElement) {
            setTimeout(() => smoothScrollTo(nextStepElement), 100);
        }
    }
}

function validateStep1() {
    const gpu = document.getElementById('gpu-device').value;
    const model = document.getElementById('base-model').value;
    const datasetSource = document.getElementById('dataset-source').value;
    
    let isValid = true;
    
    // Check if local file path is provided when needed
    if (datasetSource === 'Local JSONL File') {
        const filePath = document.getElementById('file-path').value;
        if (!filePath.trim()) {
            isValid = false;
        }
    }
    
    const continueBtn = document.getElementById('step-1-continue');
    continueBtn.disabled = !isValid;
    
    return isValid;
}

function validateStep2() {
    const continueBtn = document.getElementById('step-2-continue');
    
    // In basic mode, skip validation since advanced fields are hidden
    if (state.mode === 'basic') {
        continueBtn.disabled = false;
        return true;
    }
    
    // In advanced mode, validate split ratios
    const trainSplit = parseInt(document.getElementById('train-split').value);
    const valSplit = parseInt(document.getElementById('val-split').value);
    const testSplit = parseInt(document.getElementById('test-split').value);
    
    const total = trainSplit + valSplit + testSplit;
    const validationText = document.getElementById('split-validation');
    
    if (total === 100) {
        validationText.textContent = `✓ Total: ${total}% (valid)`;
        validationText.className = 'validation-text success';
        continueBtn.disabled = false;
        return true;
    } else {
        validationText.textContent = `⚠ Total: ${total}% (must equal 100%)`;
        validationText.className = 'validation-text error';
        continueBtn.disabled = true;
        return false;
    }
}

function validateStep4() {
    const continueBtn = document.getElementById('step-4-continue');
    continueBtn.disabled = !state.planGenerated;
    return state.planGenerated;
}

function updateConfigFromForm() {
    // Step 1 - Mode & Essentials
    if (state.mode === 'basic') {
        state.config.compute.gpu = document.getElementById('gpu-device').value;
        state.config.model.repo = document.getElementById('base-model').value;
        state.config.model.precision_mode = document.getElementById('precision-mode-basic').value;
    } else {
        state.config.compute.gpu = document.getElementById('gpu-device-adv').value;
        state.config.model.repo = document.getElementById('base-model-adv').value;
        state.config.model.precision_mode = document.getElementById('precision-mode').value;
    }
    
    const datasetSource = document.getElementById('dataset-source').value;
    if (datasetSource === 'Local JSONL File') {
        state.config.data.raw_path = document.getElementById('file-path').value;
    } else {
        state.config.data.raw_path = 'data/raw/oa.jsonl';
    }
    
    state.config.data.schema = document.getElementById('dataset-schema').value;
    
    // Step 2 - Data Details
    if (document.getElementById('max-seq-len')) {
        state.config.data.max_seq_len = parseInt(document.getElementById('max-seq-len').value);
        state.config.data.packing = document.getElementById('packing').value === 'on';
        state.config.data.template = document.getElementById('template').value;
        state.config.data.splits.train = parseInt(document.getElementById('train-split').value);
        state.config.data.splits.val = parseInt(document.getElementById('val-split').value);
        state.config.data.splits.test = parseInt(document.getElementById('test-split').value);
    }
    
    // Step 3 - Training Plan
    if (document.getElementById('training-toggle')) {
        state.config.train.enabled = document.getElementById('training-toggle').value === 'enabled';
        state.config.train.lr = parseFloat(document.getElementById('learning-rate').value);
        state.config.train.warmup_ratio = parseFloat(document.getElementById('warmup-ratio').value);
        state.config.train.weight_decay = parseFloat(document.getElementById('weight-decay').value);
        state.config.train.scheduler = document.getElementById('scheduler').value;
        state.config.train.epochs = parseInt(document.getElementById('epochs').value);
        
        // LoRA settings
        state.config.train.lora.r = parseInt(document.getElementById('lora-rank').value);
        state.config.train.lora.alpha = parseInt(document.getElementById('lora-alpha').value);
        state.config.train.lora.dropout = parseFloat(document.getElementById('lora-dropout').value);
    }
}

function handleDatasetSourceChange() {
    const datasetSource = document.getElementById('dataset-source').value;
    const fileInputGroup = document.getElementById('file-input-group');
    
    if (datasetSource === 'Local JSONL File') {
        fileInputGroup.style.display = 'block';
    } else {
        fileInputGroup.style.display = 'none';
    }
    
    validateStep1();
}

function handleTabSwitch(tabName) {
    console.log('handleTabSwitch called with:', tabName);
    
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    
    // Activate selected tab
    const tabButton = document.getElementById(`tab-${tabName}`);
    const tabContent = document.getElementById(`${tabName}-content`);
    
    console.log('Tab elements found:', { tabButton, tabContent });
    
    if (tabButton && tabContent) {
        tabButton.classList.add('active');
        tabContent.classList.add('active');
    } else {
        console.error('Tab elements not found for:', tabName);
    }
    
    // Update mode indicator
    const modeIndicator = document.getElementById('mode-indicator');
    if (modeIndicator) {
        modeIndicator.textContent = tabName === 'basic' ? 'Basic' : 'Advanced';
        modeIndicator.className = tabName === 'basic' ? 'mode-indicator basic' : 'mode-indicator';
    }
    
    // Update state
    state.mode = tabName;
    
    if (tabName === 'basic') {
        // Hide other steps and progress bar in Basic mode
        document.getElementById('step-2').classList.add('hidden');
        document.getElementById('step-3').classList.add('hidden');
        document.getElementById('step-4').classList.add('hidden');
        document.getElementById('step-5').classList.add('hidden');
        document.querySelector('.progress-bar').classList.add('hidden');
        
        // Apply defaults for basic mode
        prefillAdvancedDefaults();
    } else {
        // Show all steps and progress bar in Advanced mode
        document.getElementById('step-2').classList.remove('hidden');
        document.getElementById('step-3').classList.remove('hidden');
        document.getElementById('step-4').classList.remove('hidden');
        document.getElementById('step-5').classList.remove('hidden');
        document.querySelector('.progress-bar').classList.remove('hidden');
    }
    
    // Update previews
    updateConfigFromForm();
    updateJsonPreview();
    updateCommandPreview();
    
    showToast(`Switched to ${tabName === 'basic' ? 'Basic' : 'Advanced'} mode`);
}

function setupFileBrowser() {
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('file-browse-btn');
    const filePathInput = document.getElementById('file-path');
    
    if (!fileInput || !browseBtn || !filePathInput) return;
    
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            // Show only filename, not full path for security
            filePathInput.value = file.name;
            filePathInput.removeAttribute('readonly');
            filePathInput.setAttribute('placeholder', 'File selected: ' + file.name);
            validateStep1();
        }
    });
    
    // Allow manual editing of file path
    filePathInput.addEventListener('click', () => {
        if (filePathInput.hasAttribute('readonly')) {
            filePathInput.removeAttribute('readonly');
            filePathInput.focus();
        }
    });
    
    filePathInput.addEventListener('blur', () => {
        if (!filePathInput.value.trim()) {
            filePathInput.setAttribute('readonly', '');
            filePathInput.value = '';
            filePathInput.setAttribute('placeholder', 'Select a file or paste a path');
        }
        validateStep1();
    });
}

function handleTrainingToggleChange() {
    const trainingEnabled = document.getElementById('training-toggle').value === 'enabled';
    const trainingNotice = document.getElementById('training-notice');
    
    if (trainingEnabled) {
        trainingNotice.classList.remove('hidden');
        showToast('Training enabled - use with caution', 'warning');
    } else {
        trainingNotice.classList.add('hidden');
    }
}

function generatePlan() {
    updateConfigFromForm();
    updateJsonPreview();
    updateCommandPreview();
    state.planGenerated = true;
    validateStep4();
    showToast('Plan generated successfully');
}

function runValidation() {
    // Toggle all validation checks to success (demo behavior)
    const checkIcons = document.querySelectorAll('.check-icon');
    checkIcons.forEach(icon => {
        icon.classList.remove('check-warning');
        icon.classList.add('check-success');
        icon.innerHTML = `<path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />`;
    });
    
    showToast('Validation complete - all checks passed');
}

function handleTrainAction() {
    if (!state.config.train.enabled) {
        showToast('Training is disabled by default. Enable training in Step 3 or use --train / TRAIN=1.', 'warning');
        return;
    }
    
    showToast('Training would start with current plan');
}

// Event listeners
function setupEventListeners() {
    // Get Started button
    elements.getStartedBtn.addEventListener('click', () => {
        smoothScrollTo(elements.wizardSection);
    });
    
    // Tab buttons
    const tabBasic = document.getElementById('tab-basic');
    const tabAdvanced = document.getElementById('tab-advanced');
    
    if (tabBasic && tabAdvanced) {
        tabBasic.addEventListener('click', () => {
            console.log('Basic tab clicked');
            handleTabSwitch('basic');
        });
        tabAdvanced.addEventListener('click', () => {
            console.log('Advanced tab clicked');
            handleTabSwitch('advanced');
        });
    } else {
        console.error('Tab buttons not found:', { tabBasic, tabAdvanced });
    }
    
    // Step 1 form elements
    const step1Elements = [
        'gpu-device', 'base-model', 'download-model', 'dataset-source', 
        'dataset-schema', 'precision-mode', 'precision-mode-basic', 'file-path',
        'gpu-device-adv', 'base-model-adv', 'dataset-source-adv'
    ];
    
    step1Elements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('change', () => {
                updateConfigFromForm();
                updateJsonPreview();
                updateCommandPreview();
                validateStep1();
                
                if (id === 'dataset-source') {
                    handleDatasetSourceChange();
                }
            });
            
            if (element.tagName === 'INPUT') {
                element.addEventListener('input', () => {
                    updateConfigFromForm();
                    updateJsonPreview();
                    updateCommandPreview();
                    validateStep1();
                });
            }
        }
    });
    
    // Step 1 continue button
    document.getElementById('step-1-continue').addEventListener('click', () => {
        if (validateStep1()) {
            unlockNextStep();
        }
    });
    
    // Step 2 form elements
    const step2Elements = [
        'max-seq-len', 'packing', 'template', 
        'train-split', 'val-split', 'test-split'
    ];
    
    step2Elements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('change', () => {
                updateConfigFromForm();
                updateJsonPreview();
                updateCommandPreview();
                validateStep2();
            });
        }
    });
    
    // Step 2 continue button
    document.getElementById('step-2-continue').addEventListener('click', () => {
        if (validateStep2()) {
            unlockNextStep();
        }
    });
    
    // Step 3 form elements
    const step3Elements = [
        'training-toggle', 'learning-rate', 'warmup-ratio', 'weight-decay',
        'scheduler', 'epochs', 'lora-rank', 'lora-alpha', 'lora-dropout'
    ];
    
    step3Elements.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('change', () => {
                updateConfigFromForm();
                updateJsonPreview();
                updateCommandPreview();
                
                if (id === 'training-toggle') {
                    handleTrainingToggleChange();
                }
            });
        }
    });
    
    // Step 3 continue button
    document.getElementById('step-3-continue').addEventListener('click', () => {
        unlockNextStep();
    });
    
    // Step 4 buttons
    document.getElementById('generate-plan-btn').addEventListener('click', function() {
        this.classList.add('loading');
        this.disabled = true;
        
        setTimeout(() => {
            generatePlan();
            this.classList.remove('loading');
            this.disabled = false;
        }, 1000);
    });
    
    document.getElementById('run-validation-btn').addEventListener('click', function() {
        this.classList.add('loading');
        this.disabled = true;
        
        setTimeout(() => {
            runValidation();
            this.classList.remove('loading');
            this.disabled = false;
        }, 1500);
    });
    
    // Step 4 continue button
    document.getElementById('step-4-continue').addEventListener('click', () => {
        if (validateStep4()) {
            unlockNextStep();
        }
    });
    
    // Step 5 final action buttons
    document.getElementById('final-plan-btn').addEventListener('click', function() {
        this.classList.add('loading');
        this.disabled = true;
        
        setTimeout(() => {
            generatePlan();
            this.classList.remove('loading');
            this.disabled = false;
        }, 500);
    });
    
    document.getElementById('final-validate-btn').addEventListener('click', function() {
        this.classList.add('loading');
        this.disabled = true;
        
        setTimeout(() => {
            runValidation();
            this.classList.remove('loading');
            this.disabled = false;
        }, 1000);
    });
    
    document.getElementById('final-train-btn').addEventListener('click', function() {
        this.classList.add('loading');
        this.disabled = true;
        
        setTimeout(() => {
            handleTrainAction();
            this.classList.remove('loading');
            this.disabled = false;
        }, 500);
    });
    
    // Setup file browser functionality
    setupFileBrowser();
}

// Initialize form values
function initializeFormValues() {
    // Initialize both Basic and Advanced form values
    document.getElementById('gpu-device').value = state.config.compute.gpu;
    document.getElementById('base-model').value = state.config.model.repo;
    document.getElementById('download-model').value = 'yes';
    document.getElementById('precision-mode-basic').value = state.config.model.precision_mode;
    
    // Advanced tab values
    document.getElementById('gpu-device-adv').value = state.config.compute.gpu;
    document.getElementById('base-model-adv').value = state.config.model.repo;
    document.getElementById('precision-mode').value = state.config.model.precision_mode;
    document.getElementById('dataset-schema').value = state.config.data.schema;
    document.getElementById('file-path').value = state.config.data.raw_path;
    
    // Step 2
    document.getElementById('max-seq-len').value = state.config.data.max_seq_len;
    document.getElementById('packing').value = state.config.data.packing ? 'on' : 'off';
    document.getElementById('template').value = state.config.data.template;
    document.getElementById('train-split').value = state.config.data.splits.train;
    document.getElementById('val-split').value = state.config.data.splits.val;
    document.getElementById('test-split').value = state.config.data.splits.test;
    
    // Step 3
    document.getElementById('training-toggle').value = state.config.train.enabled ? 'enabled' : 'disabled';
    document.getElementById('learning-rate').value = state.config.train.lr;
    document.getElementById('warmup-ratio').value = state.config.train.warmup_ratio;
    document.getElementById('weight-decay').value = state.config.train.weight_decay;
    document.getElementById('scheduler').value = state.config.train.scheduler;
    document.getElementById('epochs').value = state.config.train.epochs;
    document.getElementById('lora-rank').value = state.config.train.lora.r;
    document.getElementById('lora-alpha').value = state.config.train.lora.alpha;
    document.getElementById('lora-dropout').value = state.config.train.lora.dropout;
}

// Initialize the application
function init() {
    // Set up event listeners
    setupEventListeners();
    
    // Initialize form values
    initializeFormValues();
    
    // Initialize UI state
    showStep(1);
    handleDatasetSourceChange();
    
    // Apply initial tab state - start with Basic mode
    handleTabSwitch('basic');
    updateJsonPreview();
    updateCommandPreview();
    validateStep1();
    validateStep2();
    
    // Show the wizard section initially as hidden
    elements.wizardSection.style.display = 'block';
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', init);