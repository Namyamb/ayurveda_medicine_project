# ============================================================
# 🧠 Ayurvedic Medicine Voice Recognition (Auto-Learning + Accuracy Tracker)
# Streamlit Version (fixed duplicate ID error)
# ============================================================

import os, librosa, numpy as np, soundfile as sf, pandas as pd, joblib, csv, shutil, zipfile
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from librosa.sequence import dtw
from threading import Thread
import streamlit as st

# ============================================================
# 🗂️ SETUP LOCAL FOLDERS
# ============================================================
base_dir = os.path.abspath("ayur_voice_project")
for folder in ["recordings", "new_samples", "models", "backups"]:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

feedback_file = os.path.join(base_dir, "feedback_log.csv")
if not os.path.exists(feedback_file):
    with open(feedback_file, "w", newline="") as f:
        csv.writer(f).writerow(["audio_path", "predicted", "correct", "feedback"])

# ============================================================
# 🎚️ HELPER FUNCTIONS
# ============================================================
def extract_mfcc(path, n_mfcc=20):
    y, sr = librosa.load(path, sr=16000)
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

def similarity(path1, path2):
    y1, sr1 = librosa.load(path1, sr=16000)
    y2, sr2 = librosa.load(path2, sr=16000)
    D, wp = dtw(librosa.feature.mfcc(y=y1, sr=sr1),
                librosa.feature.mfcc(y=y2, sr=sr2), metric="cosine")
    return 1 / (1 + D[-1, -1])

def backup_local():
    try:
        backup_dir = os.path.join(base_dir, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        backup_zip = os.path.join(backup_dir, "ayur_voice_backup.zip")
        with zipfile.ZipFile(backup_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
            for root, dirs, files in os.walk(base_dir):
                if "backups" in root:
                    continue
                for file in files:
                    path = os.path.join(root, file)
                    arcname = os.path.relpath(path, base_dir)
                    zipf.write(path, arcname)
    except Exception as e:
        st.error(f"⚠️ Backup error: {e}")

def calculate_accuracy():
    if not os.path.exists(feedback_file):
        return 0.0
    df = pd.read_csv(feedback_file)
    if df.empty or "feedback" not in df.columns:
        return 0.0
    total = len(df)
    correct = (df["feedback"].str.lower() == "correct").sum()
    return round((correct / total) * 100, 2) if total > 0 else 0.0

def train_svm():
    folder = os.path.join(base_dir, "recordings")
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    if not files:
        st.warning("❌ No recordings available.")
        return None, 0

    X, y = [], []
    for f in files:
        X.append(extract_mfcc(os.path.join(folder, f)))
        y.append("_".join(f.split("_")[:-1]))
    model = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
    model.fit(np.array(X), np.array(y))

    model_path = os.path.join(base_dir, "models", "svm_model.joblib")
    joblib.dump(model, model_path)
    Thread(target=backup_local).start()

    train_acc = model.score(np.array(X), np.array(y))
    feedback_acc = calculate_accuracy()
    overall_acc = round((train_acc * 0.7 + feedback_acc * 0.3), 2)
    return model_path, overall_acc

# ============================================================
# 🎙️ RECORD / SAVE AUDIO
# ============================================================
def save_audio(audio_data, save_path):
    with open(save_path, "wb") as f:
        f.write(audio_data.getbuffer())

def record_reference(name, audio_data):
    if audio_data is None or not name:
        return "⚠️ Enter medicine name & record."
    base_name = name.strip().replace(" ", "_").lower()
    folder = os.path.join(base_dir, "recordings")
    index = len([f for f in os.listdir(folder) if f.startswith(base_name)]) + 1
    path = os.path.join(folder, f"{base_name}_{index}.wav")
    save_audio(audio_data, path)
    Thread(target=backup_local).start()
    return f"✅ Saved training sample #{index} for {name}"

# ============================================================
# 🎧 RECOGNITION + FEEDBACK
# ============================================================
feedback_counter = {"count": 0}

def recognize_and_feedback(audio_data):
    if audio_data is None:
        return "⚠️ Please record or upload.", None, None

    temp_path = os.path.join(base_dir, "temp_test.wav")
    save_audio(audio_data, temp_path)

    refs = [f for f in os.listdir(os.path.join(base_dir, "recordings")) if f.endswith(".wav")]
    if not refs:
        return "❌ No reference recordings found.", None, None

    scores = []
    for ref in refs:
        ref_path = os.path.join(base_dir, "recordings", ref)
        s = similarity(temp_path, ref_path)
        base_name = "_".join(ref.replace(".wav", "").split("_")[:-1]) or ref.replace(".wav", "")
        scores.append((base_name.replace("_", " "), s))

    scores.sort(key=lambda x: x[1], reverse=True)
    top3 = scores[:3]

    pred, top_score = top3[0]
    pred = pred.strip().lower()

    out = "🎯 **Top 3 Matches (DTW Similarity):**\n"
    for n, s in top3:
        out += f"- {n} — {s*100:.1f}%\n"
    out += f"\n✅ **Predicted Medicine:** {pred}\n"

    new_path = os.path.join(base_dir, "new_samples",
                            f"{pred}_{len(os.listdir(os.path.join(base_dir,'new_samples')))+1}.wav")
    shutil.copy(temp_path, new_path)
    Thread(target=backup_local).start()
    return out, new_path, pred

def record_feedback(feedback_choice, correct_name, audio_path, predicted):
    if not audio_path:
        return "⚠️ No test sample found."

    correct_label = predicted
    if feedback_choice == "Incorrect" and correct_name:
        correct_label = correct_name.strip().replace(" ", "_").lower()

    with open(feedback_file, "a", newline="") as f:
        csv.writer(f).writerow([audio_path, predicted, correct_label, feedback_choice])

    rec_dir = os.path.join(base_dir, "recordings")
    index = len([f for f in os.listdir(rec_dir) if f.startswith(correct_label)]) + 1
    new_path = os.path.join(rec_dir, f"{correct_label}_{index}.wav")
    shutil.copy(audio_path, new_path)

    feedback_counter["count"] += 1
    msg = f"📝 Feedback saved: {feedback_choice}. Added as training data for '{correct_label}'."

    if feedback_counter["count"] >= 5:
        msg += "\n⚙️ 5 feedbacks reached → Retraining model..."
        _, acc = train_svm()
        msg += f"\n📈 Updated Model Accuracy: {acc}%"
        feedback_counter["count"] = 0

    Thread(target=backup_local).start()
    return msg

# ============================================================
# 🖥️ STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Ayurvedic Voice Recognition", layout="wide")
st.title("🧠 Ayurvedic Voice Recognition (Auto-Learning + Accuracy Tracker)")
st.caption("Continuously learns from your voice and feedback to improve accuracy over time.")

tab1, tab2 = st.tabs(["🎙️ Record Reference", "🔍 Recognition & Feedback"])

# ------------------------------------------------------------
# TAB 1: RECORD
# ------------------------------------------------------------
with tab1:
    st.subheader("Add New Medicine Samples")
    name = st.text_input("Enter Medicine Name:")
    st.markdown("🎤 **Record your voice or upload an audio file:**")

    recorded_audio = st.audio_input("🎧 Record training sample", sample_rate=16000, key="rec_ref_mic")
    uploaded_audio = st.file_uploader("📂 Upload training .wav", type=["wav"], key="rec_ref_upload")
    audio_data = recorded_audio or uploaded_audio

    if st.button("Save Recording", key="btn_save_record"):
        if name and audio_data:
            msg = record_reference(name, audio_data)
            st.success(msg)
        else:
            st.warning("⚠️ Please enter medicine name and record/upload audio.")

# ------------------------------------------------------------
# TAB 2: RECOGNITION
# ------------------------------------------------------------
with tab2:
    st.subheader("Recognize & Provide Feedback")
    st.markdown("🎤 **Record or upload to identify the medicine:**")

    test_audio = st.audio_input("🎧 Record recognition sample", sample_rate=16000, key="rec_test_mic")
    uploaded_test_audio = st.file_uploader("📂 Upload test .wav", type=["wav"], key="rec_test_upload")
    test_audio_data = test_audio or uploaded_test_audio

    if st.button("Recognize", key="btn_recognize"):
        if test_audio_data:
            result, audio_path, predicted = recognize_and_feedback(test_audio_data)
            st.markdown(result)
            st.session_state["audio_path"] = audio_path
            st.session_state["predicted"] = predicted
        else:
            st.warning("⚠️ Please record or upload an audio sample.")

    if "audio_path" in st.session_state:
        st.divider()
        feedback = st.radio("Was the prediction correct?", ["Correct", "Incorrect"], horizontal=True, key="feedback_radio")
        correct_name = ""
        if feedback == "Incorrect":
            correct_name = st.text_input("If incorrect, type the correct name:", key="feedback_correct_name")

        if st.button("Submit Feedback", key="btn_feedback"):
            msg = record_feedback(feedback, correct_name, st.session_state["audio_path"], st.session_state["predicted"])
            st.info(msg)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 AyurVoice Project | Continually improving Ayurvedic pronunciation recognition")

