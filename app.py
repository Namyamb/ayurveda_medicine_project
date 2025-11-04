<<<<<<< HEAD
# ============================================================
# 🧠 Ayurvedic Medicine Voice Recognition (Auto-Learning + Accuracy Tracker)
# ============================================================

import os, librosa, numpy as np, soundfile as sf, pandas as pd, gradio as gr, joblib, csv, shutil
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from librosa.sequence import dtw
from threading import Thread
import zipfile
from datetime import datetime

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

print(f"✅ Folders ready at: {base_dir}")

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

# ------------------------------------------------------------
# 💾 Safe Backup (no recursion, ZIP64 supported)
# ------------------------------------------------------------
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
        print(f"💾 Backup saved at {backup_zip}")
    except Exception as e:
        print(f"⚠️ Backup error: {e}")

# ============================================================
# 📈 MODEL ACCURACY CALCULATOR
# ============================================================
def calculate_accuracy():
    if not os.path.exists(feedback_file):
        return 0.0
    df = pd.read_csv(feedback_file)
    if df.empty or "feedback" not in df.columns:
        return 0.0
    total = len(df)
    correct = (df["feedback"].str.lower() == "correct").sum()
    return round((correct / total) * 100, 2) if total > 0 else 0.0

# ============================================================
# 🧠 TRAIN / RETRAIN MODEL
# ============================================================
def train_svm():
    folder = os.path.join(base_dir, "recordings")
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    if not files:
        print("❌ No recordings available.")
        return None, 0

    X, y = [], []
    for f in files:
        X.append(extract_mfcc(os.path.join(folder, f)))
        y.append("_".join(f.split("_")[:-1]))  # keep full name for label
    model = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
    model.fit(np.array(X), np.array(y))

    model_path = os.path.join(base_dir, "models", "svm_model.joblib")
    joblib.dump(model, model_path)
    Thread(target=backup_local).start()

# Calculate true model accuracy
    train_acc = model.score(np.array(X), np.array(y))
    feedback_acc = calculate_accuracy()
    overall_acc = round((train_acc * 0.7 + feedback_acc * 0.3), 2)

    print(f"✅ Auto-trained SVM on {len(files)} samples.")
    print(f"📊 Model Training Accuracy: {train_acc*100:.2f}% | Feedback Accuracy: {feedback_acc}%")
    print(f"📈 Combined Accuracy: {overall_acc}%")

    return model_path, overall_acc


# ============================================================
# 🎙️ RECORD REFERENCE DATA
# ============================================================
def record_reference(name, audio):
    if audio is None or not name:
        return "⚠️ Enter medicine name & record."
    sr, data = audio
    folder = os.path.join(base_dir, "recordings")
    base_name = name.strip().replace(" ", "_").lower()
    index = len([f for f in os.listdir(folder) if f.startswith(base_name)]) + 1
    path = os.path.join(folder, f"{base_name}_{index}.wav")
    sf.write(path, data, sr)
    Thread(target=backup_local).start()
    return f"✅ Saved training sample #{index} for {name}"

# ============================================================
# 🎧 RECOGNITION + INLINE FEEDBACK + AUTO-TRAIN + ACCURACY
# ============================================================
feedback_counter = {"count": 0}

def recognize_and_feedback(audio):
    if audio is None:
        return "⚠️ Please record or upload.", None, None

    sr, data = audio
    temp_path = os.path.join(base_dir, "temp_test.wav")
    sf.write(temp_path, data, sr)

    refs = [f for f in os.listdir(os.path.join(base_dir, "recordings")) if f.endswith(".wav")]
    if not refs:
        return "❌ No reference recordings found.", None, None

    # Compute DTW similarities
    scores = []
    for ref in refs:
        ref_path = os.path.join(base_dir, "recordings", ref)
        s = similarity(temp_path, ref_path)

        # Extract clean medicine name (no number)
        base_name = "_".join(ref.replace(".wav", "").split("_")[:-1]) or ref.replace(".wav", "")
        scores.append((base_name.replace("_", " "), s))

    scores.sort(key=lambda x: x[1], reverse=True)
    top3 = scores[:3]

    # Use top DTW result as prediction
    pred, top_score = top3[0]
    pred = pred.strip().lower()

    out = "🎯 **Top 3 Matches (DTW Similarity):**\n"
    for n, s in top3:
        out += f"- {n} — {s*100:.1f}%\n"
    out += f"\n✅ **Predicted Medicine:** {pred}\n"

    # Save test audio
    new_path = os.path.join(base_dir, "new_samples",
                            f"{pred}_{len(os.listdir(os.path.join(base_dir,'new_samples')))+1}.wav")
    sf.write(new_path, data, sr)
    Thread(target=backup_local).start()
    return out, new_path, pred

# ------------------------------------------------------------
# 💬 FEEDBACK HANDLER (auto-train every 5 feedbacks)
# ------------------------------------------------------------
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
# 🖥️ GRADIO INTERFACE
# ============================================================
with gr.Blocks(title="Ayurvedic Voice Recognition") as app:
    gr.Markdown("## 🧠 Ayurvedic Voice Recognition (Auto-Learning + Accuracy Tracker)")
    gr.Markdown("System continuously learns from feedback and retrains every 5 responses.")

    with gr.Tab("Record"):
        gr.Interface(
            fn=record_reference,
            inputs=[gr.Textbox(label="Medicine Name"),
                    gr.Audio(sources=["microphone","upload"], type="numpy")],
            outputs="text", title="Add Training Samples"
        )

    with gr.Tab("Recognition + Feedback"):
        audio_in = gr.Audio(sources=["microphone","upload"], type="numpy", label="🎙️ Speak or Upload")
        output_text = gr.Markdown()
        audio_path_box = gr.Textbox(visible=False)
        predicted_box = gr.Textbox(visible=False)
        feedback_choice = gr.Radio(["Correct","Incorrect"], label="Was the prediction correct?")
        correct_name_box = gr.Textbox(label="If Incorrect, type correct name")
        feedback_result = gr.Markdown()

        recognize_button = gr.Button("Recognize Medicine")
        recognize_button.click(fn=recognize_and_feedback, inputs=audio_in,
                               outputs=[output_text, audio_path_box, predicted_box])

        submit_feedback = gr.Button("Submit Feedback")
        submit_feedback.click(fn=record_feedback,
                              inputs=[feedback_choice, correct_name_box, audio_path_box, predicted_box],
                              outputs=feedback_result)

app.launch(server_port=5050)
=======
# ============================================================
# 🧠 Ayurvedic Medicine Voice Recognition (Auto-Learning + Accuracy Tracker)
# ============================================================

import os, librosa, numpy as np, soundfile as sf, pandas as pd, gradio as gr, joblib, csv, shutil
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from librosa.sequence import dtw
from threading import Thread
import zipfile
from datetime import datetime

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

print(f"✅ Folders ready at: {base_dir}")

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

# ------------------------------------------------------------
# 💾 Safe Backup (no recursion, ZIP64 supported)
# ------------------------------------------------------------
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
        print(f"💾 Backup saved at {backup_zip}")
    except Exception as e:
        print(f"⚠️ Backup error: {e}")

# ============================================================
# 📈 MODEL ACCURACY CALCULATOR
# ============================================================
def calculate_accuracy():
    if not os.path.exists(feedback_file):
        return 0.0
    df = pd.read_csv(feedback_file)
    if df.empty or "feedback" not in df.columns:
        return 0.0
    total = len(df)
    correct = (df["feedback"].str.lower() == "correct").sum()
    return round((correct / total) * 100, 2) if total > 0 else 0.0

# ============================================================
# 🧠 TRAIN / RETRAIN MODEL
# ============================================================
def train_svm():
    folder = os.path.join(base_dir, "recordings")
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    if not files:
        print("❌ No recordings available.")
        return None, 0

    X, y = [], []
    for f in files:
        X.append(extract_mfcc(os.path.join(folder, f)))
        y.append("_".join(f.split("_")[:-1]))  # keep full name for label
    model = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
    model.fit(np.array(X), np.array(y))

    model_path = os.path.join(base_dir, "models", "svm_model.joblib")
    joblib.dump(model, model_path)
    Thread(target=backup_local).start()

# Calculate true model accuracy
    train_acc = model.score(np.array(X), np.array(y))
    feedback_acc = calculate_accuracy()
    overall_acc = round((train_acc * 0.7 + feedback_acc * 0.3), 2)

    print(f"✅ Auto-trained SVM on {len(files)} samples.")
    print(f"📊 Model Training Accuracy: {train_acc*100:.2f}% | Feedback Accuracy: {feedback_acc}%")
    print(f"📈 Combined Accuracy: {overall_acc}%")

    return model_path, overall_acc


# ============================================================
# 🎙️ RECORD REFERENCE DATA
# ============================================================
def record_reference(name, audio):
    if audio is None or not name:
        return "⚠️ Enter medicine name & record."
    sr, data = audio
    folder = os.path.join(base_dir, "recordings")
    base_name = name.strip().replace(" ", "_").lower()
    index = len([f for f in os.listdir(folder) if f.startswith(base_name)]) + 1
    path = os.path.join(folder, f"{base_name}_{index}.wav")
    sf.write(path, data, sr)
    Thread(target=backup_local).start()
    return f"✅ Saved training sample #{index} for {name}"

# ============================================================
# 🎧 RECOGNITION + INLINE FEEDBACK + AUTO-TRAIN + ACCURACY
# ============================================================
feedback_counter = {"count": 0}

def recognize_and_feedback(audio):
    if audio is None:
        return "⚠️ Please record or upload.", None, None

    sr, data = audio
    temp_path = os.path.join(base_dir, "temp_test.wav")
    sf.write(temp_path, data, sr)

    refs = [f for f in os.listdir(os.path.join(base_dir, "recordings")) if f.endswith(".wav")]
    if not refs:
        return "❌ No reference recordings found.", None, None

    # Compute DTW similarities
    scores = []
    for ref in refs:
        ref_path = os.path.join(base_dir, "recordings", ref)
        s = similarity(temp_path, ref_path)

        # Extract clean medicine name (no number)
        base_name = "_".join(ref.replace(".wav", "").split("_")[:-1]) or ref.replace(".wav", "")
        scores.append((base_name.replace("_", " "), s))

    scores.sort(key=lambda x: x[1], reverse=True)
    top3 = scores[:3]

    # Use top DTW result as prediction
    pred, top_score = top3[0]
    pred = pred.strip().lower()

    out = "🎯 **Top 3 Matches (DTW Similarity):**\n"
    for n, s in top3:
        out += f"- {n} — {s*100:.1f}%\n"
    out += f"\n✅ **Predicted Medicine:** {pred}\n"

    # Save test audio
    new_path = os.path.join(base_dir, "new_samples",
                            f"{pred}_{len(os.listdir(os.path.join(base_dir,'new_samples')))+1}.wav")
    sf.write(new_path, data, sr)
    Thread(target=backup_local).start()
    return out, new_path, pred

# ------------------------------------------------------------
# 💬 FEEDBACK HANDLER (auto-train every 5 feedbacks)
# ------------------------------------------------------------
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
# 🖥️ GRADIO INTERFACE
# ============================================================
with gr.Blocks(title="Ayurvedic Voice Recognition") as app:
    gr.Markdown("## 🧠 Ayurvedic Voice Recognition (Auto-Learning + Accuracy Tracker)")
    gr.Markdown("System continuously learns from feedback and retrains every 5 responses.")

    with gr.Tab("Record"):
        gr.Interface(
            fn=record_reference,
            inputs=[gr.Textbox(label="Medicine Name"),
                    gr.Audio(sources=["microphone","upload"], type="numpy")],
            outputs="text", title="Add Training Samples"
        )

    with gr.Tab("Recognition + Feedback"):
        audio_in = gr.Audio(sources=["microphone","upload"], type="numpy", label="🎙️ Speak or Upload")
        output_text = gr.Markdown()
        audio_path_box = gr.Textbox(visible=False)
        predicted_box = gr.Textbox(visible=False)
        feedback_choice = gr.Radio(["Correct","Incorrect"], label="Was the prediction correct?")
        correct_name_box = gr.Textbox(label="If Incorrect, type correct name")
        feedback_result = gr.Markdown()

        recognize_button = gr.Button("Recognize Medicine")
        recognize_button.click(fn=recognize_and_feedback, inputs=audio_in,
                               outputs=[output_text, audio_path_box, predicted_box])

        submit_feedback = gr.Button("Submit Feedback")
        submit_feedback.click(fn=record_feedback,
                              inputs=[feedback_choice, correct_name_box, audio_path_box, predicted_box],
                              outputs=feedback_result)

app.launch(server_port=5050)
>>>>>>> 945eea5ea14aa1fbae0f672a84ffc37fa1089138
