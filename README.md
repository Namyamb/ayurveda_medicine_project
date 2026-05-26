# AyurVoice AI

AyurVoice is a voice-based Ayurvedic medicine name recognition app. Record or upload audio samples, compare them against reference recordings stored in Dropbox, and get ranked predictions using audio similarity (MFCC + Dynamic Time Warping). The app learns from user feedback to improve future matches.

## Features

- **Voice recognition** — Identify medicine names from spoken or uploaded audio
- **Reference library** — Add new medicine reference samples from the browser
- **Top-3 matches** — See the three closest matches with similarity scores
- **Feedback loop** — Mark predictions as correct or incorrect; feedback is stored and used to adjust scores
- **Cloud storage** — All recordings, samples, and logs live in Dropbox under `/AyurVoice/`
- **Parallel processing** — Reference files are downloaded and compared in parallel for faster recognition

## Tech Stack

| Layer | Tools |
|-------|--------|
| UI | [Streamlit](https://streamlit.io/) |
| Audio | [librosa](https://librosa.org/) (MFCC, DTW) |
| Storage | [Dropbox API](https://www.dropbox.com/developers) |
| Data | pandas, numpy, scikit-learn, joblib |

## Prerequisites

- Python 3.10+ (3.11 or 3.12 recommended)
- A [Dropbox](https://www.dropbox.com/) account
- A Dropbox app with an access token (see setup below)
- Microphone access in the browser (for recording)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Namyamb/ayurveda_medicine_project.git
   cd ayurveda_medicine_project
   ```

2. **Create and activate a virtual environment**

   ```powershell
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   ```bash
   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt soundfile
   ```

   `soundfile` is required by librosa for reading and writing audio files.

## Dropbox Setup

1. Go to the [Dropbox App Console](https://www.dropbox.com/developers/apps) and create an app.
2. Choose **Scoped access** and **Full Dropbox** or **App folder** (the app uses paths under `/AyurVoice/`).
3. Enable permissions: `files.content.read`, `files.content.write`, `files.metadata.read`.
4. Generate an **access token** under the app’s Permissions / OAuth 2 tab.

5. Create Streamlit secrets locally:

   Create the file `.streamlit/secrets.toml` in the project root:

   ```toml
   [dropbox]
   access_token = "YOUR_DROPBOX_ACCESS_TOKEN"
   ```

   Do not commit this file. Add `.streamlit/secrets.toml` to `.gitignore` if it is not already ignored.

### Dropbox folder layout

The app expects this structure in your Dropbox account:

```
/AyurVoice/
├── recordings/      # Reference audio for each medicine
├── new_samples/     # Test samples saved after recognition
├── feedback/        # feedback_log.csv
└── models/          # (optional) saved ML models from earlier versions
```

Folders are created automatically when you upload files through the app.

## Running the App

From the project root with the virtual environment activated:

```bash
streamlit run app.py
```

Open the URL shown in the terminal (default: **http://localhost:8501**).

## Usage

### Tab 1 — Record Reference

1. Enter the medicine name (e.g. `Ashwagandha`).
2. Record or upload a clear reference audio clip (16 kHz).
3. Click **Save Reference** to upload it to `Dropbox/AyurVoice/recordings/`.

Add multiple samples per medicine for better accuracy. Files are named like `ashwagandha_1.wav`, `ashwagandha_2.wav`, etc.

### Tab 2 — Recognition & Feedback

1. Record or upload a test sample.
2. Click **Recognize** to see the top 3 matches and the predicted medicine name.
3. If the result is wrong, choose **Incorrect**, enter the correct name, and click **Submit Feedback**.

After five feedback submissions, the app updates similarity weighting for future predictions.

## How It Works

1. **Reference samples** are stored as WAV files in Dropbox.
2. For each recognition request, the app downloads reference files in parallel.
3. **MFCC features** are extracted from the input and each reference.
4. **Dynamic Time Warping (DTW)** computes similarity between audio sequences.
5. Past **feedback** can boost scores for medicines that were previously corrected.
6. The best match is returned along with the top 3 candidates.

## Project Structure

```
ayurveda_medicine_project/
├── app.py                 # Streamlit app (active implementation)
├── requirements.txt       # Python dependencies
├── feedback_log.csv       # Local feedback sample (cloud copy in Dropbox)
├── service_account.json   # Legacy Google Drive config (not used by current app)
├── .streamlit/
│   └── secrets.toml       # Dropbox token (create locally, do not commit)
└── README.md
```

> **Note:** `app.py` contains commented-out earlier versions (Google Drive, SVM/RandomForest training, etc.). The active code starts at the section labeled *working code with parallel feature*.

## Deployment

This project was originally configured for [Hugging Face Spaces](https://huggingface.co/spaces) (see frontmatter in git history). For local development, use Streamlit as described above.

To deploy on Streamlit Community Cloud, add `dropbox.access_token` under **Secrets** in the app settings using the same TOML format as `secrets.toml`.

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| Dropbox connection failed | Check `access_token` in `.streamlit/secrets.toml` and app permissions |
| No reference samples found | Add at least one recording in the **Record Reference** tab |
| Audio errors | Install `soundfile`: `pip install soundfile` |
| Slow recognition | Fewer reference files or a stable network; parallel downloads help but depend on Dropbox latency |

## License

Apache-2.0

## Author

[Namyamb](https://github.com/Namyamb) — AyurVoice / Ayurveda Medicine Name Detector
