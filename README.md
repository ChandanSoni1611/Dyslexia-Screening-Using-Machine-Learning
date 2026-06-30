 Project Dyslexia Screening  — Multilingual AI-Based Dyslexia Screening Tool

Project Dyslexia Screening  is a Flask-based web application that performs a preliminary dyslexia screening for children using AI-driven assessments across English, Hindi, and Kannada. It is designed as a screening tool, not a diagnostic tool — results indicate the likelihood of dyslexia-related patterns and recommend consulting a specialist for formal diagnosis.

>  Disclaimer: This tool does not provide a medical diagnosis. It is intended as an early screening aid only. Please consult a certified specialist for an official dyslexia assessment.

---

  Overview

Dyslexia screening tools in regional Indian languages (Hindi, Kannada) are extremely limited. Project Dyslexia Screening  aims to address this gap by combining AI/ML models with speech-based assessment to provide an accessible, multilingual, first-level screening experience — particularly for children aged 5–12.

---

  Features

- Multilingual support — English, Hindi, and Kannada, selected based on the user's preferred language
- Three core assessments:
  - Spelling Test — AI model evaluates typed spelling responses for dyslexia-related error patterns
  - WSD (Word Sense Disambiguation) Test — Multiple-choice questions involving idioms/confusing phrases to assess language comprehension
  - Speech Test — User reads a passage aloud; speech-to-text output is compared against the original text to evaluate reading accuracy
- Language-specific model loading — only the selected language's models are activated during a session
- User authentication — login/signup required before taking a test
- Pre-test screening questions — checks for physical disabilities (eyes, mouth, ears) and collects age before starting
- Result generation — overall outcome based on performance across all three tests

---

  Project Structure

Project Dyslexia Screening /
├── dataset/
│   └── spelling dataset/           Training datasets (noisy pairs for spelling model)
├── models/                          Trained model files (excluded from repo — see Setup)
├── models_code/
│   ├── spelling_model_code/         Spelling model training/inference code (EN/HI/KN)
│   └── wsd_model_code/              WSD model training/inference code (EN/HI/KN)
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   ├── authentication/              login.html, signup.html
│   ├── index.html                   Home page
│   └── test.html                    Test page
├── app.py                           Main Flask application (entry point)
├── requirements.txt
└── .gitignore

---

  Tech Stack

- Backend: Python, Flask
- AI/ML: Transformer-based models (BERT/MuRIL-based) for spelling and WSD tasks
- Speech Processing: Speech-to-Text (STT) and Text-to-Speech (TTS) APIs
- Frontend: HTML, CSS, JavaScript
- Database: SQLite (for user authentication)

---

  Setup & Installation

> Note: Trained model files are large (~4.7 GB total) and are not included in this repository due to size constraints.

1. Clone the repository
```bash
   git clone https://github.com/ChandanSoni1611/<repo-name>.git
   cd <repo-name>
```

2. Create a virtual environment
```bash
   python -m venv venv
   venv\Scripts\activate       Windows
   source venv/bin/activate    macOS/Linux
```

3. Install dependencies
```bash
   pip install -r requirements.txt
```

4. Add model files
   Place trained model files in the `models/` directory (not included in repo — train using scripts in `models_code/` or contact for access).

5. Run the application
```bash
   python app.py
```

6. Open `http://localhost:5000` in your browser.

---

  How It Works (User Flow)

1. User lands on the Home Page — overview of dyslexia and available tests
2. Clicks Start Test → redirected to login/signup (if not authenticated)
3. On the Test Page, user answers a quick physical disability check and enters age + preferred language
4. Based on selected language, only that language's spelling and WSD models are loaded
5. User completes all three tests:
   - Spelling (typed input)
   - WSD (multiple-choice)
   - Speech (read-aloud, compared via STT)
6. System evaluates performance across tests and generates a result indicating likelihood of dyslexia-related patterns
7. Default test language is English if none is selected

---

 Current Limitations

This is an actively evolving academic project. Known limitations include:

- Training data: Spelling model trained on synthetically "noised" data rather than real dyslexic children's writing samples
- Model evaluation: Formal accuracy/precision/recall benchmarking is in progress
- Age adaptivity: Tests currently use the same difficulty level across all ages (5–12); age-adaptive difficulty is planned
- Test coverage: Current screening covers only 3 of several dimensions used in clinical assessment (spelling, comprehension, reading aloud) — does not yet include phonological awareness, working memory, rapid naming, or family/developmental history
- Real-world validation: Not yet tested on a real cohort of children with diagnosed dyslexia

---

  Future Scope

- Incorporate family history and early developmental questionnaires (inspired by clinical diagnostic protocols)
- Add letter/number reversal detection (b/d, p/q confusion — a hallmark dyslexia indicator)
- Add phonological awareness tests (rhyme detection, sound discrimination)
- Add rapid naming and working memory tasks
- Introduce age-adaptive difficulty scaling
- Collect real dyslexic-pattern datasets (with proper ethical approval) to retrain models
- Formal model evaluation with precision, recall, and F1-score reporting
- Reduce model size / optimize for deployment
- Generate detailed, downloadable assessment reports
