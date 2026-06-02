/**
 * Answer Evaluate - Frontend Controller
 * Modern semantic assessment interface
 */

let currentEvaluation = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupFormListener();
    setupNavigation();
});

// Setup form listener
function setupFormListener() {
    const form = document.getElementById('eval-form');
    if (form) {
        form.addEventListener('submit', handleEvaluation);
    }
}

// Setup navigation
function setupNavigation() {
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
        });
    });
}

// Handle evaluation
async function handleEvaluation(e) {
    e.preventDefault();

    const question = document.getElementById('question').value.trim();
    const studentAnswer = document.getElementById('student-answer').value.trim();

    // Validation
    if (!question || !studentAnswer) {
        showError('Question and answer are required');
        return;
    }

    if (question.length < 5 || studentAnswer.length < 5) {
        showError('Inputs must be at least 5 characters');
        return;
    }

    // Show loading
    const btn = document.querySelector('.btn-primary');
    const spinner = document.getElementById('btn-spinner');
    btn.disabled = true;
    btn.querySelector('.btn-text').textContent = 'Evaluating...';
    spinner.classList.remove('hidden');

    try {
        const response = await fetch('/api/v1/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                answer: studentAnswer
            })
        });

        const data = await response.json();

        if (!response.ok) {
            showError(data.error || 'Evaluation failed');
            return;
        }

        currentEvaluation = data;
        displayResults(data);

    } catch (err) {
        showError('Network error: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-text').textContent = 'Evaluate Now';
        spinner.classList.add('hidden');
    }
}

// Display results
function displayResults(data) {
    const emptyState = document.getElementById('empty-state');
    const resultsPanel = document.getElementById('results-panel');
    
    if (emptyState) emptyState.classList.add('hidden');
    if (resultsPanel) resultsPanel.classList.remove('hidden');

    // Score
    const scoreDisplay = document.getElementById('score-display');
    if (scoreDisplay) scoreDisplay.textContent = data.total_score.toFixed(1);

    // Coverage
    const coverageValue = document.getElementById('coverage-value');
    if (coverageValue) {
        coverageValue.textContent = (data.coverage_ratio * 100).toFixed(0) + '%';
    }

    // Concept count
    const conceptsCount = document.getElementById('concepts-count');
    if (conceptsCount) {
        conceptsCount.textContent = data.concept_scores.length;
    }

    // Missing count
    const missingCount = document.getElementById('missing-count');
    if (missingCount) {
        missingCount.textContent = data.missing_concepts.length;
    }

    // Feedback
    const feedbackText = document.getElementById('feedback-text');
    if (feedbackText && data.feedback) {
        feedbackText.textContent = data.feedback;
    }

    // Concepts list
    const conceptsList = document.getElementById('concepts-list');
    if (conceptsList) {
        conceptsList.innerHTML = '';
        data.concept_scores.forEach(cs => {
            const item = document.createElement('div');
            item.className = 'concept-item';
            item.innerHTML = `
                <div class="concept-name">${cs.concept}</div>
                <div class="concept-score">
                    <span>${cs.score.toFixed(1)}/10</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill-bar" style="width: ${(cs.score / 10) * 100}%"></div>
                </div>
            `;
            conceptsList.appendChild(item);
        });
    }

    // Missing concepts alert
    if (data.missing_concepts.length > 0) {
        const missingAlert = document.getElementById('missing-alert');
        const missingText = document.getElementById('missing-concepts-text');
        if (missingAlert && missingText) {
            missingText.textContent = data.missing_concepts.join(', ');
            missingAlert.classList.remove('hidden');
        }
    }

    // Animate score circle
    animateScoreCircle(data.total_score);
}

// Animate score circle
function animateScoreCircle(score) {
    const circle = document.getElementById('progress-circle');
    if (circle) {
        const circumference = 2 * Math.PI * 54;
        const offset = circumference - (score / 10) * circumference;
        circle.style.strokeDasharray = circumference + ' ' + circumference;
        circle.style.strokeDashoffset = offset;
    }
}

// Switch section
function switchSection(section) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(s => {
        s.classList.remove('active');
    });

    // Show selected section
    const targetSection = document.getElementById(section + '-section');
    if (targetSection) {
        targetSection.classList.add('active');
    }
}

// Reset evaluation
function resetEval() {
    document.getElementById('eval-form').reset();
    const emptyState = document.getElementById('empty-state');
    const resultsPanel = document.getElementById('results-panel');
    if (emptyState) emptyState.classList.remove('hidden');
    if (resultsPanel) resultsPanel.classList.add('hidden');
    currentEvaluation = null;
}

// Show error
function showError(message) {
    alert('❌ Error: ' + message);
}
