// ====== Globals ======
let selectedLanguage = '';
let userAge = 0;
let spellingResult = '';
let confusionResult = '';
let speechTranscript = '';
let speechAccuracy = '';

const SPEECH_ENDPOINT = '/speech_to_text_word';

// Language-specific test sentences
const TEST_SENTENCES = {
  english: {
    spelling: "The quick brown fox jumps over the lazy dog",
    confusion: "Their house is over there and they're coming soon",
    speech: "Reading is a gateway to knowledge and imagination"
  },
  hindi: {
    spelling: "यह एक परीक्षण वाक्य है जिसमें कई शब्द हैं",
    confusion: "वह अपने घर जा रहा है और वह बहुत खुश है",
    speech: "शिक्षा सबसे महत्वपूर्ण है और हमें हमेशा सीखते रहना चाहिए"
  },
  kannada: {
    spelling: "ಇದು ಒಂದು ಪರೀಕ್ಷಾ ವಾಕ್ಯವಾಗಿದೆ",
    confusion: "ಅವನು ತನ್ನ ಮನೆಗೆ ಹೋಗುತ್ತಿದ್ದಾನೆ ಮತ್ತು ಅವನು ತುಂಬಾ ಸಂತೋಷವಾಗಿದ್ದಾನೆ",
    speech: "ಶಿಕ್ಷಣವು ಅತ್ಯಂತ ಮಹತ್ವದ್ದಾಗಿದೆ ಮತ್ತು ನಾವು ಯಾವಾಗಲೂ ಕಲಿಯುತ್ತಿರಬೇಕು"
  }
};

// Language display names
const LANGUAGE_NAMES = {
  english: "English",
  hindi: "हिंदी (Hindi)",
  kannada: "ಕನ್ನಡ (Kannada)"
};

// ====== Step 0: Age & Language Selection ======
function startTest() {
  const age = parseInt(document.getElementById("ageInput").value, 10);
  const language = document.getElementById("languageSelect").value;
  
  if (Number.isNaN(age) || age < 5 || age > 10) {
    alert("Please enter a valid age between 5 and 10.");
    return;
  }
  
  if (!language) {
    alert("Please select a language for the test.");
    return;
  }
  
  userAge = age;
  selectedLanguage = language;
  
  // Hide age section and show spelling test
  document.getElementById("age-section").classList.add("d-none");
  document.getElementById("spelling-test-section").classList.remove("d-none");
  
  // Update spelling test UI with language-specific content
  document.getElementById("spelling-language-label").innerText = 
    `Spelling Test (${LANGUAGE_NAMES[selectedLanguage]})`;
  document.getElementById("spellingInput").placeholder = 
    `Enter a sentence in ${LANGUAGE_NAMES[selectedLanguage]}`;
  
  // Set example sentence if available
  if (TEST_SENTENCES[selectedLanguage]) {
    document.getElementById("spelling-example").innerText = 
      `Example: ${TEST_SENTENCES[selectedLanguage].spelling}`;
  }
}

// ====== Step 1: Spelling Test ======
function submitSpellingTest() {
  const sentence = document.getElementById("spellingInput").value.trim();
  if (!sentence) {
    alert("Please enter a sentence for spelling test.");
    return;
  }

  // Show loading state
  const submitBtn = document.getElementById("submitSpellingBtn");
  const originalText = submitBtn.innerText;
  submitBtn.disabled = true;
  submitBtn.innerText = "Processing...";

  fetch('/correct_spelling', { 
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ 
      text: sentence,
      language: selectedLanguage 
    })
  })
  .then(res => res.json())
  .then(data => {
    if (data.error) {
      alert(`Error: ${data.error}`);
      submitBtn.disabled = false;
      submitBtn.innerText = originalText;
      return;
    }
    
    spellingResult = data.corrected || 'N/A';
    
    // Move to confusion test
    document.getElementById("spelling-test-section").classList.add("d-none");
    document.getElementById("confusion-test-section").classList.remove("d-none");
    
    // Update confusion test UI
    document.getElementById("confusion-language-label").innerText = 
      `Confusion Words Test (${LANGUAGE_NAMES[selectedLanguage]})`;
    document.getElementById("confusionInput").placeholder = 
      `Enter a sentence with potentially confusing words in ${LANGUAGE_NAMES[selectedLanguage]}`;
    
    if (TEST_SENTENCES[selectedLanguage]) {
      document.getElementById("confusion-example").innerText = 
        `Example: ${TEST_SENTENCES[selectedLanguage].confusion}`;
    }
  })
  .catch(err => {
    console.error(err);
    alert('Spelling test failed. Please try again.');
    submitBtn.disabled = false;
    submitBtn.innerText = originalText;
  });
}

// ====== Step 2: Confusion Test ======
function submitConfusionTest() {
  const sentence = document.getElementById("confusionInput").value.trim();
  if (!sentence) {
    alert("Please enter a sentence for confusion words test.");
    return;
  }

  // For confusion test, we need context, gloss, and target word
  // Since user enters a sentence, we'll use it as context
  // For demo purposes, we'll extract the first significant word as target
  const words = sentence.split(' ').filter(w => w.length > 2);
  const targetWord = words[0] || sentence.split(' ')[0];
  
  // Show loading state
  const submitBtn = document.getElementById("submitConfusionBtn");
  const originalText = submitBtn.innerText;
  submitBtn.disabled = true;
  submitBtn.innerText = "Processing...";

  fetch('/confusion_test', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ 
      context: sentence,
      gloss: sentence, // Using sentence as gloss for simplicity
      target_word: targetWord,
      language: selectedLanguage 
    })
  })
  .then(res => res.json())
  .then(data => {
    if (data.error) {
      alert(`Error: ${data.error}`);
      submitBtn.disabled = false;
      submitBtn.innerText = originalText;
      return;
    }
    
    confusionResult = `${data.prediction} (Confidence: ${(data.confidence * 100).toFixed(2)}%)`;
    
    // Move to speech test
    document.getElementById("confusion-test-section").classList.add("d-none");
    document.getElementById("speech-test-section").classList.remove("d-none");
    
    // Update speech test UI
    document.getElementById("speech-language-label").innerText = 
      `Speech Test (${LANGUAGE_NAMES[selectedLanguage]})`;
    
    // Set language-specific speech prompt
    if (TEST_SENTENCES[selectedLanguage]) {
      document.getElementById("speech-question").innerText = 
        TEST_SENTENCES[selectedLanguage].speech;
    }
    
    document.getElementById("speech-instruction").innerText = 
      `Please read the following sentence aloud in ${LANGUAGE_NAMES[selectedLanguage]}:`;
  })
  .catch(err => {
    console.error(err);
    alert('Confusion test failed. Please try again.');
    submitBtn.disabled = false;
    submitBtn.innerText = originalText;
  });
}

// ====== Step 3: Speech Test ======
let mediaRecorder = null;
let audioChunks = [];

async function startRecording() {
  try {
    document.getElementById("status").innerText = "Requesting microphone…";
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    audioChunks = [];
    const options = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? { mimeType: 'audio/webm;codecs=opus' }
      : undefined;

    mediaRecorder = new MediaRecorder(stream, options);
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.onstop = onRecordingStop;

    document.getElementById("startRecBtn").classList.add("d-none");
    document.getElementById("stopRecBtn").classList.remove("d-none");
    document.getElementById("status").innerText = "Recording… 🎤 Speak now";

    mediaRecorder.start();
  } catch (e) {
    alert("Microphone access denied or not available.");
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    document.getElementById("status").innerText = "Stopping…";
    mediaRecorder.stop();
  }
}

async function onRecordingStop() {
  document.getElementById("stopRecBtn").classList.add("d-none");
  document.getElementById("status").innerText = "Uploading…";

  const blob = new Blob(audioChunks);
  const formData = new FormData();
  formData.append("file", blob, "speech.webm");
  formData.append("target", document.getElementById("speech-question").innerText.trim());
  formData.append("language", selectedLanguage); // Send language to backend

  try {
    const res = await fetch(SPEECH_ENDPOINT, { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || 'Upload failed');

    speechTranscript = data.transcript || '';
    speechAccuracy = (data.accuracy != null ? data.accuracy : 'N/A');

    document.getElementById("transcript").innerText = speechTranscript;
    document.getElementById("accuracy").innerText = 
      (typeof speechAccuracy === 'number' ? `${speechAccuracy}%` : speechAccuracy);
    document.getElementById("status").innerText = "Done ✅";
    document.getElementById("finishSpeechBtn").classList.remove("d-none");
  } catch (err) {
    console.error(err);
    document.getElementById("status").innerText = "Error during transcription.";
    alert("Speech test failed.");
  } finally {
    if (mediaRecorder && mediaRecorder.stream) {
      mediaRecorder.stream.getTracks().forEach(t => t.stop());
    }
    document.getElementById("startRecBtn").classList.remove("d-none");
  }
}

function finishSpeechTest() {
  document.getElementById("speech-test-section").classList.add("d-none");
  displayResults();
}

// ====== Final Results Display ======
function displayResults() {
  const div = document.getElementById("results");
  div.classList.remove("d-none");
  
  const languageName = LANGUAGE_NAMES[selectedLanguage];
  
  div.innerHTML = `
    <div class="card">
      <div class="card-header bg-primary text-white">
        <h4>Test Results Summary</h4>
      </div>
      <div class="card-body">
        <div class="mb-3">
          <h6><strong>Test Information:</strong></h6>
          <p><strong>Age:</strong> ${userAge} years</p>
          <p><strong>Language:</strong> ${languageName}</p>
        </div>
        
        <hr>
        
        <div class="mb-3">
          <h6><strong>1. Spelling Test Result:</strong></h6>
          <p class="text-muted">Original text analyzed and corrected.</p>
          <div class="alert alert-info">
            ${spellingResult}
          </div>
        </div>
        
        <div class="mb-3">
          <h6><strong>2. Confusion Words Test Result:</strong></h6>
          <p class="text-muted">Word sense disambiguation analysis.</p>
          <div class="alert alert-info">
            ${confusionResult}
          </div>
        </div>
        
        <div class="mb-3">
          <h6><strong>3. Speech Test Results:</strong></h6>
          <p class="text-muted">Speech-to-text transcription and accuracy.</p>
          <div class="alert alert-success">
            <strong>Transcript:</strong> ${speechTranscript || 'No transcript available'}
          </div>
          <div class="alert alert-warning">
            <strong>Accuracy:</strong> ${typeof speechAccuracy === 'number' ? speechAccuracy + '%' : (speechAccuracy || 'N/A')}
          </div>
        </div>
        
        <hr>
        
        <div class="text-center mt-4">
          <button class="btn btn-primary" onclick="restartTest()">
            Take Another Test
          </button>
          <button class="btn btn-success" onclick="downloadResults()">
            Download Results
          </button>
        </div>
      </div>
    </div>
  `;
}

// ====== Utility Functions ======
function restartTest() {
  // Reset all variables
  selectedLanguage = '';
  userAge = 0;
  spellingResult = '';
  confusionResult = '';
  speechTranscript = '';
  speechAccuracy = '';
  
  // Clear all inputs
  document.getElementById("ageInput").value = '';
  document.getElementById("languageSelect").value = '';
  document.getElementById("spellingInput").value = '';
  document.getElementById("confusionInput").value = '';
  
  // Hide all sections except age
  document.getElementById("age-section").classList.remove("d-none");
  document.getElementById("spelling-test-section").classList.add("d-none");
  document.getElementById("confusion-test-section").classList.add("d-none");
  document.getElementById("speech-test-section").classList.add("d-none");
  document.getElementById("results").classList.add("d-none");
  
  // Reset speech test UI
  document.getElementById("transcript").innerText = '-';
  document.getElementById("accuracy").innerText = '-';
  document.getElementById("status").innerText = '';
  document.getElementById("finishSpeechBtn").classList.add("d-none");
}

function downloadResults() {
  const languageName = LANGUAGE_NAMES[selectedLanguage];
  const resultsText = `
DYSLEXIA TEST RESULTS
=====================

Test Information:
- Age: ${userAge} years
- Language: ${languageName}
- Date: ${new Date().toLocaleDateString()}

Test Results:
-------------

1. Spelling Test:
   ${spellingResult}

2. Confusion Words Test:
   ${confusionResult}

3. Speech Test:
   Transcript: ${speechTranscript || 'No transcript available'}
   Accuracy: ${typeof speechAccuracy === 'number' ? speechAccuracy + '%' : (speechAccuracy || 'N/A')}

=====================
Generated by Dyslexia Assessment System
  `;
  
  const blob = new Blob([resultsText], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `dyslexia_test_results_${Date.now()}.txt`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ====== Initialize on page load ======
document.addEventListener('DOMContentLoaded', function() {
  console.log('Dyslexia Test System Initialized');
  console.log('Supported Languages:', Object.keys(LANGUAGE_NAMES));
});