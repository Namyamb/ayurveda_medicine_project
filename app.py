# # ============================================================
# # 🧠 AyurVoice AI — Ayurvedic Medicine Voice Recognition
# # (Auto-Learning + Google Drive Backup + Auto Retrain)
# # ============================================================

# import os, librosa, numpy as np, pandas as pd, soundfile as sf, joblib, csv, shutil, zipfile, json, tempfile
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from librosa.sequence import dtw
# from threading import Thread
# import streamlit as st
# from pydrive2.auth import GoogleAuth
# from pydrive2.drive import GoogleDrive

# # ============================================================
# # 🔑 GOOGLE DRIVE SETUP (via Streamlit Secrets)
# # ============================================================

# def connect_to_drive():
#     """Authenticate Google Drive using Streamlit Secrets (service account)."""
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
#             f.write(json.dumps(st.secrets["google_service_account"]).encode())
#             temp_path = f.name
#         gauth = GoogleAuth()
#         gauth.LoadCredentialsFile(temp_path)
#         gauth.ServiceAuth()
#         return GoogleDrive(gauth)
#     except Exception as e:
#         st.warning(f"⚠️ Could not connect to Google Drive: {e}")
#         return None

# # ✅ Replace these IDs with your actual Google Drive folder IDs
# DRIVE_IDS = {
#     "recordings": "1cXknT8JR2VTCsVk0w_d0YGMgIVHLv-8y",
#     "new_samples": "1cXknT8JR2VTCsVk0w_d0YGMgIVHLv-8y",
#     "backups": "1LhOyaEWH0x1w7jr2sg8gyv5TXn0dtChi",
#     "root": "1SCcEDnUm_5cDRtnuIxfWfr2fWS5KOzXQ"
# }

# def upload_to_drive(local_path, folder_key="recordings", title=None):
#     """Uploads any file to a specified Drive folder."""
#     drive = connect_to_drive()
#     if drive is None:
#         return
#     try:
#         folder_id = DRIVE_IDS.get(folder_key) or DRIVE_IDS["root"]
#         file_drive = drive.CreateFile({
#             'title': title or os.path.basename(local_path),
#             'parents': [{'id': folder_id}]
#         })
#         file_drive.SetContentFile(local_path)
#         file_drive.Upload()
#         st.info(f"☁️ Uploaded {os.path.basename(local_path)} to Google Drive ({folder_key})")
#     except Exception as e:
#         st.warning(f"⚠️ Upload failed: {e}")

# # ============================================================
# # 🗂️ LOCAL FOLDERS
# # ============================================================
# base_dir = os.path.abspath("ayur_voice_project")
# for folder in ["recordings", "new_samples", "models", "backups"]:
#     os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

# feedback_file = os.path.join(base_dir, "feedback_log.csv")
# if not os.path.exists(feedback_file):
#     with open(feedback_file, "w", newline="") as f:
#         csv.writer(f).writerow(["audio_path", "predicted", "correct", "feedback"])

# # ============================================================
# # 🎚️ HELPER FUNCTIONS
# # ============================================================

# def extract_mfcc(path, n_mfcc=20):
#     y, sr = librosa.load(path, sr=16000)
#     y, _ = librosa.effects.trim(y)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     return np.mean(mfcc, axis=1)

# def similarity(path1, path2):
#     y1, sr1 = librosa.load(path1, sr=16000)
#     y2, sr2 = librosa.load(path2, sr=16000)
#     D, wp = dtw(librosa.feature.mfcc(y=y1, sr=sr1),
#                 librosa.feature.mfcc(y=y2, sr=sr2), metric="cosine")
#     return 1 / (1 + D[-1, -1])

# def backup_local():
#     """Compresses the project folder and uploads a ZIP backup to Drive."""
#     try:
#         backup_zip = os.path.join(base_dir, "backups", "ayur_voice_backup.zip")
#         with zipfile.ZipFile(backup_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
#             for root, dirs, files in os.walk(base_dir):
#                 if "backups" in root:
#                     continue
#                 for file in files:
#                     path = os.path.join(root, file)
#                     arcname = os.path.relpath(path, base_dir)
#                     zipf.write(path, arcname)
#         Thread(target=upload_to_drive, args=(backup_zip, "backups", "ayur_voice_backup.zip")).start()
#     except Exception as e:
#         st.warning(f"⚠️ Backup error: {e}")

# def calculate_accuracy():
#     """Calculates recognition accuracy based on feedback log."""
#     if not os.path.exists(feedback_file):
#         return 0.0
#     df = pd.read_csv(feedback_file)
#     if df.empty or "feedback" not in df.columns:
#         return 0.0
#     total = len(df)
#     correct = (df["feedback"].str.lower() == "correct").sum()
#     return round((correct / total) * 100, 2) if total > 0 else 0.0

# # ============================================================
# # 🧠 MODEL TRAINING
# # ============================================================

# def train_svm():
#     folder = os.path.join(base_dir, "recordings")
#     files = [f for f in os.listdir(folder) if f.endswith(".wav")]
#     if not files:
#         st.warning("❌ No recordings available.")
#         return None, 0
#     X, y = [], []
#     for f in files:
#         X.append(extract_mfcc(os.path.join(folder, f)))
#         y.append("_".join(f.split("_")[:-1]))
#     model = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
#     model.fit(np.array(X), np.array(y))
#     model_path = os.path.join(base_dir, "models", "svm_model.joblib")
#     joblib.dump(model, model_path)
#     Thread(target=backup_local).start()
#     train_acc = model.score(np.array(X), np.array(y))
#     feedback_acc = calculate_accuracy()
#     overall_acc = round((train_acc * 0.7 + feedback_acc * 0.3), 2)
#     return model_path, overall_acc

# # ============================================================
# # 🎙️ RECORDING HANDLERS
# # ============================================================

# def save_audio(audio_data, save_path):
#     with open(save_path, "wb") as f:
#         f.write(audio_data.getbuffer())

# def record_reference(name, audio_data):
#     if audio_data is None or not name:
#         return "⚠️ Enter medicine name & record."
#     base_name = name.strip().replace(" ", "_").lower()
#     folder = os.path.join(base_dir, "recordings")
#     index = len([f for f in os.listdir(folder) if f.startswith(base_name)]) + 1
#     path = os.path.join(folder, f"{base_name}_{index}.wav")
#     save_audio(audio_data, path)
#     Thread(target=upload_to_drive, args=(path, "recordings", os.path.basename(path))).start()
#     Thread(target=backup_local).start()
#     return f"✅ Saved training sample #{index} for {name}"

# def recognize_and_feedback(audio_data):
#     if audio_data is None:
#         return "⚠️ Please record or upload.", None, None
#     temp_path = os.path.join(base_dir, "temp_test.wav")
#     save_audio(audio_data, temp_path)
#     refs = [f for f in os.listdir(os.path.join(base_dir, "recordings")) if f.endswith(".wav")]
#     if not refs:
#         return "❌ No reference recordings found.", None, None
#     scores = []
#     for ref in refs:
#         ref_path = os.path.join(base_dir, "recordings", ref)
#         s = similarity(temp_path, ref_path)
#         base_name = "_".join(ref.replace(".wav", "").split("_")[:-1]) or ref.replace(".wav", "")
#         scores.append((base_name.replace("_", " "), s))
#     scores.sort(key=lambda x: x[1], reverse=True)
#     top3 = scores[:3]
#     pred, _ = top3[0]
#     pred = pred.strip().lower()
#     out = "🎯 **Top 3 Matches (DTW Similarity):**\n"
#     for n, s in top3:
#         out += f"- {n} — {s*100:.1f}%\n"
#     out += f"\n✅ **Predicted Medicine:** {pred}\n"
#     new_path = os.path.join(base_dir, "new_samples",
#                             f"{pred}_{len(os.listdir(os.path.join(base_dir,'new_samples')))+1}.wav")
#     shutil.copy(temp_path, new_path)
#     Thread(target=upload_to_drive, args=(new_path, "new_samples", os.path.basename(new_path))).start()
#     Thread(target=backup_local).start()
#     return out, new_path, pred

# # ============================================================
# # 🧠 FEEDBACK + AUTO RETRAIN LOGIC
# # ============================================================

# feedback_counter = {"count": 0}

# def record_feedback(feedback_choice, correct_name, audio_path, predicted):
#     if not audio_path:
#         return "⚠️ No test sample found."

#     correct_label = predicted
#     if feedback_choice == "Incorrect" and correct_name:
#         correct_label = correct_name.strip().replace(" ", "_").lower()

#     with open(feedback_file, "a", newline="") as f:
#         csv.writer(f).writerow([audio_path, predicted, correct_label, feedback_choice])

#     Thread(target=upload_to_drive, args=(feedback_file, "root", "feedback_log.csv")).start()
#     Thread(target=backup_local).start()

#     feedback_counter["count"] += 1
#     msg = f"📝 Feedback saved: {feedback_choice}. Added as training data for '{correct_label}'."

#     if feedback_counter["count"] >= 5:
#         msg += "\n⚙️ 5 feedbacks reached → Retraining model..."
#         _, acc = train_svm()
#         msg += f"\n📈 Updated Model Accuracy: {acc}%"
#         feedback_counter["count"] = 0

#     return msg

# # ============================================================
# # 🖥️ STREAMLIT UI
# # ============================================================

# st.set_page_config(page_title="AyurVoice AI + Google Drive", layout="wide")
# st.title("🧠 AyurVoice AI — Ayurvedic Medicine Voice Recognition")
# st.caption("Auto-learning • Securely synced with Google Drive • Retrains every 5 feedbacks")

# tab1, tab2 = st.tabs(["🎙️ Record Reference", "🔍 Recognition & Feedback"])

# # TAB 1
# with tab1:
#     st.subheader("Add New Medicine Samples")
#     name = st.text_input("Enter Medicine Name:")
#     recorded_audio = st.audio_input("🎧 Record new medicine", sample_rate=16000, key="ref_mic")
#     uploaded_audio = st.file_uploader("📂 Or upload .wav file", type=["wav"], key="ref_upload")
#     audio_data = recorded_audio or uploaded_audio
#     if st.button("Save Recording", key="btn_save_record"):
#         if name and audio_data:
#             st.success(record_reference(name, audio_data))
#         else:
#             st.warning("⚠️ Please enter medicine name and record/upload audio.")

# # TAB 2
# with tab2:
#     st.subheader("Recognize & Provide Feedback")
#     test_audio = st.audio_input("🎧 Record test sample", sample_rate=16000, key="test_mic")
#     uploaded_test = st.file_uploader("📂 Or upload test .wav", type=["wav"], key="test_upload")
#     test_data = test_audio or uploaded_test
#     if st.button("Recognize", key="btn_recognize"):
#         if test_data:
#             result, audio_path, predicted = recognize_and_feedback(test_data)
#             st.markdown(result)
#             st.session_state["audio_path"] = audio_path
#             st.session_state["predicted"] = predicted
#         else:
#             st.warning("⚠️ Please record or upload an audio sample.")
#     if "audio_path" in st.session_state:
#         st.divider()
#         feedback = st.radio("Was the prediction correct?", ["Correct", "Incorrect"], horizontal=True)
#         correct_name = ""
#         if feedback == "Incorrect":
#             correct_name = st.text_input("If incorrect, type correct name:")
#         if st.button("Submit Feedback", key="btn_feedback"):
#             msg = record_feedback(feedback, correct_name, st.session_state["audio_path"], st.session_state["predicted"])
#             st.info(msg)

# st.markdown("---")
# st.caption("© 2025 AyurVoice Project | Auto-learning • Auto Retrain • Google Drive Backup")


































# # ============================================================
# # 🧠 AyurVoice AI — Ayurvedic Medicine Voice Recognition
# # (Auto-Learning + Dropbox Backup + Auto Retrain)
# # ============================================================

# import os, librosa, numpy as np, pandas as pd, soundfile as sf, joblib, csv, shutil, zipfile
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from librosa.sequence import dtw
# from threading import Thread
# import streamlit as st
# import dropbox

# # ============================================================
# # 🔑 DROPBOX SETUP
# # ============================================================

# def connect_dropbox():
#     try:
#         token = st.secrets["dropbox"]["access_token"]
#         dbx = dropbox.Dropbox(token)
#         dbx.users_get_current_account()
#         return dbx
#     except Exception as e:
#         st.warning(f"⚠️ Could not connect to Dropbox: {e}")
#         return None

# def upload_to_dropbox(local_path, folder="recordings"):
#     """Uploads any file to Dropbox under /AyurVoice/{folder}/"""
#     dbx = connect_dropbox()
#     if dbx is None:
#         return
#     try:
#         dest_path = f"/AyurVoice/{folder}/{os.path.basename(local_path)}"
#         with open(local_path, "rb") as f:
#             dbx.files_upload(f.read(), dest_path, mode=dropbox.files.WriteMode("overwrite"))
#         st.success(f"☁️ Uploaded `{os.path.basename(local_path)}` to Dropbox → {folder}")
#     except Exception as e:
#         st.warning(f"⚠️ Dropbox upload failed: {e}")

# # ============================================================
# # 🗂️ LOCAL FOLDERS
# # ============================================================
# base_dir = os.path.abspath("ayur_voice_project")
# for folder in ["recordings", "new_samples", "models", "backups"]:
#     os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

# feedback_file = os.path.join(base_dir, "feedback_log.csv")
# if not os.path.exists(feedback_file):
#     with open(feedback_file, "w", newline="") as f:
#         csv.writer(f).writerow(["audio_path", "predicted", "correct", "feedback"])

# # ============================================================
# # 🎚️ HELPER FUNCTIONS
# # ============================================================

# def extract_mfcc(path, n_mfcc=20):
#     y, sr = librosa.load(path, sr=16000)
#     y, _ = librosa.effects.trim(y)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     return np.mean(mfcc, axis=1)

# def similarity(path1, path2):
#     y1, sr1 = librosa.load(path1, sr=16000)
#     y2, sr2 = librosa.load(path2, sr=16000)
#     D, wp = dtw(librosa.feature.mfcc(y=y1, sr=sr1),
#                 librosa.feature.mfcc(y=y2, sr=sr2), metric="cosine")
#     return 1 / (1 + D[-1, -1])

# def backup_local():
#     """Compress the project folder and upload ZIP to Dropbox."""
#     try:
#         backup_zip = os.path.join(base_dir, "backups", "ayurvoice_backup.zip")
#         with zipfile.ZipFile(backup_zip, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
#             for root, dirs, files in os.walk(base_dir):
#                 if "backups" in root:
#                     continue
#                 for file in files:
#                     path = os.path.join(root, file)
#                     arcname = os.path.relpath(path, base_dir)
#                     zipf.write(path, arcname)
#         Thread(target=upload_to_dropbox, args=(backup_zip, "backups")).start()
#     except Exception as e:
#         st.warning(f"⚠️ Backup error: {e}")

# def calculate_accuracy():
#     if not os.path.exists(feedback_file):
#         return 0.0
#     df = pd.read_csv(feedback_file)
#     if df.empty or "feedback" not in df.columns:
#         return 0.0
#     total = len(df)
#     correct = (df["feedback"].str.lower() == "correct").sum()
#     return round((correct / total) * 100, 2) if total > 0 else 0.0

# # ============================================================
# # 🧠 MODEL TRAINING
# # ============================================================

# def train_svm():
#     folder = os.path.join(base_dir, "recordings")
#     files = [f for f in os.listdir(folder) if f.endswith(".wav")]
#     if not files:
#         st.warning("❌ No recordings available.")
#         return None, 0, 0
#     X, y = [], []
#     for f in files:
#         X.append(extract_mfcc(os.path.join(folder, f)))
#         y.append("_".join(f.split("_")[:-1]))
#     model = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
#     model.fit(np.array(X), np.array(y))
#     model_path = os.path.join(base_dir, "models", "svm_model.joblib")
#     joblib.dump(model, model_path)
#     Thread(target=upload_to_dropbox, args=(model_path, "models")).start()
#     Thread(target=backup_local).start()
#     train_acc = model.score(np.array(X), np.array(y))
#     feedback_acc = calculate_accuracy()
#     overall_acc = round((train_acc * 0.7 + feedback_acc * 0.3) * 100, 2)
#     return model_path, overall_acc, len(files)

# # ============================================================
# # 🎙️ RECORDING HANDLERS
# # ============================================================

# def save_audio(audio_data, save_path):
#     with open(save_path, "wb") as f:
#         f.write(audio_data.getbuffer())

# def record_reference(name, audio_data):
#     if audio_data is None or not name:
#         return "⚠️ Enter medicine name & record."
#     base_name = name.strip().replace(" ", "_").lower()
#     folder = os.path.join(base_dir, "recordings")
#     index = len([f for f in os.listdir(folder) if f.startswith(base_name)]) + 1
#     path = os.path.join(folder, f"{base_name}_{index}.wav")
#     save_audio(audio_data, path)
#     Thread(target=upload_to_dropbox, args=(path, "recordings")).start()
#     Thread(target=backup_local).start()
#     return f"✅ Saved training sample #{index} for {name}"

# def recognize_and_feedback(audio_data):
#     if audio_data is None:
#         return "⚠️ Please record or upload.", None, None
#     temp_path = os.path.join(base_dir, "temp_test.wav")
#     save_audio(audio_data, temp_path)
#     refs = [f for f in os.listdir(os.path.join(base_dir, "recordings")) if f.endswith(".wav")]
#     if not refs:
#         return "❌ No reference recordings found.", None, None
#     scores = []
#     for ref in refs:
#         ref_path = os.path.join(base_dir, "recordings", ref)
#         s = similarity(temp_path, ref_path)
#         base_name = "_".join(ref.replace(".wav", "").split("_")[:-1]) or ref.replace(".wav", "")
#         scores.append((base_name.replace("_", " "), s))
#     scores.sort(key=lambda x: x[1], reverse=True)
#     top3 = scores[:3]
#     pred, _ = top3[0]
#     pred = pred.strip().lower()
#     out = "🎯 **Top 3 Matches (DTW Similarity):**\n"
#     for n, s in top3:
#         out += f"- {n} — {s*100:.1f}%\n"
#     out += f"\n✅ **Predicted Medicine:** {pred}\n"
#     new_path = os.path.join(base_dir, "new_samples",
#                             f"{pred}_{len(os.listdir(os.path.join(base_dir,'new_samples')))+1}.wav")
#     shutil.copy(temp_path, new_path)
#     Thread(target=upload_to_dropbox, args=(new_path, "new_samples")).start()
#     Thread(target=backup_local).start()
#     return out, new_path, pred

# # ============================================================
# # 🧠 FEEDBACK + AUTO RETRAIN
# # ============================================================

# feedback_counter = {"count": 0}

# def record_feedback(feedback_choice, correct_name, audio_path, predicted):
#     if not audio_path:
#         return "⚠️ No test sample found."
#     correct_label = predicted
#     if feedback_choice == "Incorrect" and correct_name:
#         correct_label = correct_name.strip().replace(" ", "_").lower()
#     with open(feedback_file, "a", newline="") as f:
#         csv.writer(f).writerow([audio_path, predicted, correct_label, feedback_choice])
#     Thread(target=upload_to_dropbox, args=(feedback_file, "feedback")).start()
#     Thread(target=backup_local).start()

#     feedback_counter["count"] += 1
#     msg = f"📝 Feedback saved: {feedback_choice}. Added for `{correct_label}`.\n"
#     msg += f"🧩 Feedback count: {feedback_counter['count']}/5 before next retrain."

#     if feedback_counter["count"] >= 5:
#         st.info("⚙️ 5 feedbacks reached → Retraining model... Please wait ⏳")
#         model_path, acc, n_samples = train_svm()
#         msg = f"""
# ✅ **Model Retrained Successfully**

# **Model File:** `{os.path.basename(model_path)}`
# **Samples Trained On:** {n_samples}
# **Updated Accuracy:** {acc:.2f}% 🧠  
# _System is now more accurate and adaptive!_
# """
#         feedback_counter["count"] = 0

#     return msg

# # ============================================================
# # 🖥️ STREAMLIT UI
# # ============================================================

# st.set_page_config(page_title="AyurVoice AI + Dropbox", layout="wide")
# st.title("🧠 AyurVoice AI — Ayurvedic Medicine Voice Recognition")
# st.caption("Auto-learning • Dropbox Backup • Auto Retrain every 5 feedbacks")

# tab1, tab2 = st.tabs(["🎙️ Record Reference", "🔍 Recognition & Feedback"])

# # TAB 1
# with tab1:
#     st.subheader("Add New Medicine Samples")
#     name = st.text_input("Enter Medicine Name:")
#     recorded_audio = st.audio_input("🎧 Record new medicine", sample_rate=16000, key="ref_mic")
#     uploaded_audio = st.file_uploader("📂 Or upload .wav file", type=["wav"], key="ref_upload")
#     audio_data = recorded_audio or uploaded_audio
#     if st.button("Save Recording", key="btn_save_record"):
#         if name and audio_data:
#             st.success(record_reference(name, audio_data))
#         else:
#             st.warning("⚠️ Please enter medicine name and record/upload audio.")

# # TAB 2
# with tab2:
#     st.subheader("Recognize & Provide Feedback")
#     test_audio = st.audio_input("🎧 Record test sample", sample_rate=16000, key="test_mic")
#     uploaded_test = st.file_uploader("📂 Or upload test .wav", type=["wav"], key="test_upload")
#     test_data = test_audio or uploaded_test
#     if st.button("Recognize", key="btn_recognize"):
#         if test_data:
#             result, audio_path, predicted = recognize_and_feedback(test_data)
#             st.markdown(result)
#             st.session_state["audio_path"] = audio_path
#             st.session_state["predicted"] = predicted
#         else:
#             st.warning("⚠️ Please record or upload an audio sample.")
#     if "audio_path" in st.session_state:
#         st.divider()
#         feedback = st.radio("Was the prediction correct?", ["Correct", "Incorrect"], horizontal=True)
#         correct_name = ""
#         if feedback == "Incorrect":
#             correct_name = st.text_input("If incorrect, type correct name:")
#         if st.button("Submit Feedback", key="btn_feedback"):
#             msg = record_feedback(feedback, correct_name, st.session_state["audio_path"], st.session_state["predicted"])
#             st.info(msg)

# st.markdown("---")
# st.caption("© 2025 AyurVoice Project | Auto-learning • Auto Retrain • Dropbox Backup")






























# # ============================================================
# # 🧠 AyurVoice AI — Fully Cloud-Based (Dropbox Only)
# # ============================================================

# import os, io, librosa, numpy as np, pandas as pd, joblib, tempfile
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from librosa.sequence import dtw
# from threading import Thread
# import streamlit as st
# import dropbox

# # ============================================================
# # 🔑 DROPBOX CONNECTION
# # ============================================================

# def connect_dropbox():
#     """Connect to Dropbox using the Streamlit secret token."""
#     try:
#         token = st.secrets["dropbox"]["access_token"]
#         dbx = dropbox.Dropbox(token)
#         dbx.users_get_current_account()
#         return dbx
#     except Exception as e:
#         st.error(f"⚠️ Dropbox connection failed: {e}")
#         return None

# dbx = connect_dropbox()

# # ============================================================
# # 🗂️ CLOUD FOLDER STRUCTURE
# # ============================================================

# BASE_PATH = "/AyurVoice"
# FOLDERS = ["recordings", "new_samples", "models", "backups", "feedback"]

# def ensure_dropbox_structure():
#     """Ensure AyurVoice folders exist on Dropbox."""
#     if not dbx:
#         return
#     for folder in FOLDERS:
#         path = f"{BASE_PATH}/{folder}"
#         try:
#             dbx.files_create_folder_v2(path)
#         except dropbox.exceptions.ApiError:
#             pass  # already exists

# ensure_dropbox_structure()

# # ============================================================
# # 🧩 BASIC UTILITIES
# # ============================================================

# def upload_bytes_to_dropbox(bytes_data, filename, folder):
#     """Upload binary data directly to Dropbox (always as bytes)."""
#     path = f"{BASE_PATH}/{folder}/{filename}"
#     # Convert to raw bytes
#     if isinstance(bytes_data, memoryview):
#         bytes_data = bytes(bytes_data)
#     elif hasattr(bytes_data, "getvalue"):  # e.g. BytesIO
#         bytes_data = bytes_data.getvalue()
#     dbx.files_upload(bytes_data, path, mode=dropbox.files.WriteMode("overwrite"))
#     return path

# def download_dropbox_file(path):
#     """Download a Dropbox file and return its bytes."""
#     _, res = dbx.files_download(path)
#     return io.BytesIO(res.content)

# def list_dropbox_files(folder):
#     """List files inside a Dropbox folder."""
#     try:
#         res = dbx.files_list_folder(f"{BASE_PATH}/{folder}")
#         return [f.path_lower for f in res.entries if isinstance(f, dropbox.files.FileMetadata)]
#     except dropbox.exceptions.ApiError:
#         return []

# # ============================================================
# # 🎚️ FEATURE EXTRACTION + SIMILARITY
# # ============================================================

# def extract_mfcc_from_bytes(file_bytes, n_mfcc=20):
#     """Extract MFCC features from audio bytes."""
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#         tmp.write(file_bytes.read())
#         tmp.flush()
#         y, sr = librosa.load(tmp.name, sr=16000)
#         y, _ = librosa.effects.trim(y)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#         return np.mean(mfcc, axis=1)

# def dtw_similarity(vec1, vec2):
#     """Compute DTW similarity between two feature vectors."""
#     D, wp = dtw(vec1.reshape(-1, 1), vec2.reshape(-1, 1), metric="euclidean")
#     return 1 / (1 + D[-1, -1])

# # ============================================================
# # 🧠 MODEL TRAINING + RETRAINING
# # ============================================================

# def train_model_from_dropbox():
#     """Train or retrain SVM using all Dropbox recordings."""
#     files = list_dropbox_files("recordings")
#     if not files:
#         st.warning("❌ No recordings found in Dropbox.")
#         return None, 0, 0

#     X, y = [], []
#     for path in files:
#         name = os.path.basename(path).replace(".wav", "")
#         label = "_".join(name.split("_")[:-1])
#         audio_bytes = download_dropbox_file(path)
#         features = extract_mfcc_from_bytes(audio_bytes)
#         X.append(features)
#         y.append(label)

#     model = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True))
#     model.fit(np.array(X), np.array(y))

#     model_bytes = io.BytesIO()
#     joblib.dump(model, model_bytes)
#     model_bytes.seek(0)
#     upload_bytes_to_dropbox(model_bytes, "svm_model.joblib", "models")

#     train_acc = model.score(np.array(X), np.array(y)) * 100
#     return model, train_acc, len(files)

# # ============================================================
# # 🎙️ RECORDING HANDLERS
# # ============================================================

# def record_reference(name, audio_data):
#     """Save new medicine sample directly to Dropbox."""
#     if not name or not audio_data:
#         return "⚠️ Please provide a name and record audio."
#     base_name = name.strip().replace(" ", "_").lower()
#     count = len(list_dropbox_files("recordings")) + 1
#     filename = f"{base_name}_{count}.wav"
#     upload_bytes_to_dropbox(audio_data.getbuffer(), filename, "recordings")
#     return f"✅ Saved training sample #{count} for {name}"

# def recognize_from_dropbox(audio_data):
#     """Recognize medicine by comparing with Dropbox recordings."""
#     if not audio_data:
#         return "⚠️ Please record or upload a test sample.", None

#     test_features = extract_mfcc_from_bytes(io.BytesIO(audio_data.getbuffer()))

#     refs = list_dropbox_files("recordings")
#     if not refs:
#         return "❌ No reference recordings found in Dropbox.", None

#     scores = []
#     for ref_path in refs:
#         ref_bytes = download_dropbox_file(ref_path)
#         ref_features = extract_mfcc_from_bytes(ref_bytes)
#         sim = dtw_similarity(test_features, ref_features)
#         base_name = "_".join(os.path.basename(ref_path).replace(".wav", "").split("_")[:-1])
#         scores.append((base_name, sim))

#     scores.sort(key=lambda x: x[1], reverse=True)
#     top3 = scores[:3]
#     pred = top3[0][0] if top3 else "Unknown"

#     new_name = f"{pred}_{len(list_dropbox_files('new_samples'))+1}.wav"
#     upload_bytes_to_dropbox(audio_data.getbuffer(), new_name, "new_samples")

#     result_text = "🎯 **Top 3 Matches:**\n"
#     for n, s in top3:
#         result_text += f"- {n} — {s*100:.2f}%\n"
#     result_text += f"\n✅ **Predicted Medicine:** {pred}\n"

#     return result_text, pred

# # ============================================================
# # 🧠 FEEDBACK + AUTO RETRAIN LOGIC
# # ============================================================

# if "feedback_count" not in st.session_state:
#     st.session_state.feedback_count = 0

# def record_feedback(feedback_choice, correct_name, predicted):
#     """Handle feedback and trigger retraining after 5 feedbacks."""
#     correct_label = predicted
#     if feedback_choice == "Incorrect" and correct_name:
#         correct_label = correct_name.strip().replace(" ", "_").lower()

#     log_row = f"{predicted},{correct_label},{feedback_choice}\n"

#     try:
#         log_path = f"{BASE_PATH}/feedback/feedback_log.csv"
#         existing = ""
#         try:
#             existing = download_dropbox_file(log_path).getvalue().decode()
#         except Exception:
#             existing = "predicted,correct,feedback\n"
#         updated = existing + log_row
#         upload_bytes_to_dropbox(updated.encode(), "feedback_log.csv", "feedback")
#     except Exception as e:
#         st.warning(f"⚠️ Feedback save failed: {e}")

#     # Retraining trigger
#     st.session_state.feedback_count += 1
#     msg = f"📝 Feedback saved: {feedback_choice}. Count: {st.session_state.feedback_count}/5"

#     if st.session_state.feedback_count >= 5:
#         st.info("⚙️ 5 feedbacks reached → Retraining model... ⏳")
#         model, acc, total = train_model_from_dropbox()
#         msg = f"""
# ✅ **Model Retrained Successfully!**
# **Samples Trained:** {total}  
# **Accuracy:** {acc:.2f}%  
# 🧠 Model updated and saved on Dropbox.
# """
#         st.session_state.feedback_count = 0

#     return msg

# # ============================================================
# # 🖥️ STREAMLIT UI
# # ============================================================

# st.set_page_config(page_title="AyurVoice AI (Dropbox Cloud)", layout="wide")
# st.title("🧠 AyurVoice AI — Fully Cloud-Based Voice Recognition")
# st.caption("🎙️ Record → Recognize → Learn → Auto-Retrain → Dropbox Synced")

# tab1, tab2 = st.tabs(["🎙️ Record Reference", "🔍 Recognition & Feedback"])

# # --- RECORDING TAB ---
# with tab1:
#     st.subheader("Add New Medicine Samples")
#     name = st.text_input("Enter Medicine Name:")
#     recorded_audio = st.audio_input("🎧 Record new medicine", sample_rate=16000)
#     uploaded_audio = st.file_uploader("📂 Or upload .wav", type=["wav"])
#     audio_data = recorded_audio or uploaded_audio
#     if st.button("Save Recording"):
#         if name and audio_data:
#             st.success(record_reference(name, audio_data))
#         else:
#             st.warning("⚠️ Please enter name and record/upload.")

# # --- RECOGNITION TAB ---
# with tab2:
#     st.subheader("Recognize & Provide Feedback")
#     test_audio = st.audio_input("🎧 Record test sample", sample_rate=16000)
#     uploaded_test = st.file_uploader("📂 Or upload test .wav", type=["wav"])
#     test_data = test_audio or uploaded_test
#     if st.button("Recognize"):
#         if test_data:
#             result, predicted = recognize_from_dropbox(test_data)
#             st.markdown(result)
#             st.session_state["predicted"] = predicted
#         else:
#             st.warning("⚠️ Please record or upload an audio sample.")

#     if "predicted" in st.session_state:
#         st.divider()
#         feedback = st.radio("Was the prediction correct?", ["Correct", "Incorrect"], horizontal=True)
#         correct_name = ""
#         if feedback == "Incorrect":
#             correct_name = st.text_input("If incorrect, enter correct name:")
#         if st.button("Submit Feedback"):
#             msg = record_feedback(feedback, correct_name, st.session_state["predicted"])
#             st.info(msg)

# st.markdown("---")
# st.caption("© 2025 AyurVoice Cloud | Dropbox-Only • Auto-Retrain • Always Synced")











# # ============================================================
# # 🧠 AyurVoice AI — Ayurvedic Medicine Voice Recognition
# # (Dropbox-only • Auto-learning • Auto-Retrain Every 5 Feedbacks)
# # ============================================================

# import os, io, csv, joblib, librosa, numpy as np, pandas as pd, streamlit as st
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from librosa.sequence import dtw
# import dropbox
# from threading import Thread

# # ============================================================
# # 🔑 DROPBOX CONNECTION
# # ============================================================

# def connect_dropbox():
#     try:
#         token = st.secrets["dropbox"]["access_token"]
#         dbx = dropbox.Dropbox(token)
#         dbx.users_get_current_account()
#         return dbx
#     except Exception as e:
#         st.error(f"❌ Dropbox connection failed: {e}")
#         return None

# # ============================================================
# # 🧩 DROPBOX FILE HELPERS
# # ============================================================

# def upload_bytes_to_dropbox(bytes_data, file_name, folder):
#     """Uploads byte data directly to Dropbox."""
#     dbx = connect_dropbox()
#     if not dbx: return
#     try:
#         path = f"/AyurVoice/{folder}/{file_name}"
#         dbx.files_upload(bytes_data, path, mode=dropbox.files.WriteMode("overwrite"))
#     except Exception as e:
#         st.warning(f"⚠️ Upload failed: {e}")

# def list_dropbox_files(folder):
#     """Lists all files in a Dropbox folder."""
#     dbx = connect_dropbox()
#     if not dbx: return []
#     try:
#         result = dbx.files_list_folder(f"/AyurVoice/{folder}")
#         return [entry.name for entry in result.entries if isinstance(entry, dropbox.files.FileMetadata)]
#     except dropbox.exceptions.ApiError:
#         return []

# def download_dropbox_file(folder, file_name):
#     """Downloads a file from Dropbox as bytes."""
#     dbx = connect_dropbox()
#     if not dbx: return None
#     try:
#         _, res = dbx.files_download(f"/AyurVoice/{folder}/{file_name}")
#         return io.BytesIO(res.content)
#     except:
#         return None

# def ensure_folder_structure():
#     """Ensure Dropbox project folders exist."""
#     dbx = connect_dropbox()
#     if not dbx: return
#     folders = ["recordings", "new_samples", "models", "feedback"]
#     for f in folders:
#         path = f"/AyurVoice/{f}"
#         try:
#             dbx.files_get_metadata(path)
#         except dropbox.exceptions.ApiError:
#             dbx.files_create_folder_v2(path)

# # ============================================================
# # 🎚️ AUDIO PROCESSING HELPERS
# # ============================================================

# def extract_mfcc_from_bytes(audio_bytes, sr=16000, n_mfcc=20):
#     y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr)
#     y, _ = librosa.effects.trim(y)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     return np.mean(mfcc, axis=1)

# def similarity_score(bytes1, bytes2):
#     y1, sr1 = librosa.load(io.BytesIO(bytes1), sr=16000)
#     y2, sr2 = librosa.load(io.BytesIO(bytes2), sr=16000)
#     D, _ = dtw(librosa.feature.mfcc(y=y1, sr=sr1), librosa.feature.mfcc(y=y2, sr=sr2), metric="cosine")
#     return 1 / (1 + D[-1, -1])

# # ============================================================
# # 🧠 MODEL TRAINING
# # ============================================================

# def train_svm_from_dropbox():
#     """Retrains SVM using both recordings & new_samples with feedback."""
#     dbx = connect_dropbox()
#     if not dbx: return None, 0, 0

#     X, y = [], []
#     all_files = list_dropbox_files("recordings") + list_dropbox_files("new_samples")

#     # --- Load feedback mapping ---
#     feedback_map = {}
#     try:
#         _, res = dbx.files_download("/AyurVoice/feedback/feedback_log.csv")
#         df = pd.read_csv(io.BytesIO(res.content))
#         for _, row in df.iterrows():
#             feedback_map[row["audio_path"]] = row["correct"]
#     except Exception:
#         pass

#     # --- Extract MFCC features ---
#     for file in all_files:
#         folder = "new_samples" if file in list_dropbox_files("new_samples") else "recordings"
#         file_bytes = download_dropbox_file(folder, file)
#         if not file_bytes:
#             continue
#         label = "_".join(file.replace(".wav", "").split("_")[:-1]) or file.replace(".wav", "")
#         # Apply corrected label if exists
#         if f"{folder}/{file}" in feedback_map:
#             label = feedback_map[f"{folder}/{file}"]
#         X.append(extract_mfcc_from_bytes(file_bytes.read()))
#         y.append(label.lower())

#     if not X:
#         st.warning("❌ No training data found.")
#         return None, 0, 0

#     model = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True))
#     model.fit(np.array(X), np.array(y))
#     acc = model.score(np.array(X), np.array(y)) * 100
#     model_bytes = io.BytesIO()
#     joblib.dump(model, model_bytes)
#     model_bytes.seek(0)
#     upload_bytes_to_dropbox(model_bytes.read(), "svm_model.joblib", "models")

#     return "svm_model.joblib", acc, len(X)

# # ============================================================
# # 🧩 FEEDBACK LOGGING
# # ============================================================

# def append_feedback_to_csv(predicted, correct, feedback):
#     dbx = connect_dropbox()
#     if not dbx: return
#     rows = []
#     try:
#         _, res = dbx.files_download("/AyurVoice/feedback/feedback_log.csv")
#         df = pd.read_csv(io.BytesIO(res.content))
#         rows = df.values.tolist()
#     except:
#         pass

#     rows.append([f"new_samples/{predicted}.wav", predicted, correct, feedback])
#     out = io.StringIO()
#     writer = csv.writer(out)
#     writer.writerow(["audio_path", "predicted", "correct", "feedback"])
#     writer.writerows(rows)
#     upload_bytes_to_dropbox(out.getvalue().encode(), "feedback_log.csv", "feedback")

# # ============================================================
# # 🔍 RECOGNITION ENGINE
# # ============================================================

# def recognize_from_dropbox(audio_data):
#     refs = list_dropbox_files("recordings")
#     if not refs:
#         return "❌ No reference samples found.", None
#     input_bytes = audio_data.getvalue()
#     scores = []
#     for ref in refs:
#         ref_bytes = download_dropbox_file("recordings", ref)
#         if ref_bytes:
#             sim = similarity_score(input_bytes, ref_bytes.read())
#             base = "_".join(ref.replace(".wav", "").split("_")[:-1])
#             scores.append((base.replace("_", " "), sim))
#     scores.sort(key=lambda x: x[1], reverse=True)
#     top3 = scores[:3]
#     pred = top3[0][0]
#     result = "🎯 **Top 3 Matches (DTW Similarity):**\n"
#     for n, s in top3:
#         result += f"- {n}: {s*100:.2f}%\n"
#     result += f"\n✅ **Predicted Medicine:** {pred}"
#     new_name = f"{pred}_{len(list_dropbox_files('new_samples'))+1}.wav"
#     upload_bytes_to_dropbox(input_bytes, new_name, "new_samples")
#     return result, pred

# # ============================================================
# # 🔁 FEEDBACK RETRAIN LOGIC
# # ============================================================

# feedback_count = {"n": 0}

# def handle_feedback(feedback, correct_name, predicted):
#     correct_label = correct_name.strip().lower() if feedback == "Incorrect" and correct_name else predicted
#     append_feedback_to_csv(predicted, correct_label, feedback)
#     feedback_count["n"] += 1

#     msg = f"📝 Feedback saved for `{predicted}` → `{correct_label}`.\n🧩 {feedback_count['n']}/5 before next retrain."

#     if feedback_count["n"] >= 5:
#         st.info("⚙️ Retraining model... Please wait ⏳")
#         model_path, acc, total = train_svm_from_dropbox()
#         msg = f"""
# ✅ **Model Retrained Successfully!**
# - Model File: `{model_path}`
# - Samples Used: {total}
# - Accuracy: {acc:.2f}%
# """
#         feedback_count["n"] = 0

#     return msg

# # ============================================================
# # 🖥️ STREAMLIT UI
# # ============================================================

# st.set_page_config(page_title="AyurVoice AI — Dropbox", layout="wide")
# st.title("🧠 AyurVoice AI — Ayurvedic Medicine Voice Recognition (Dropbox)")
# st.caption("Cloud-only • Auto-learning • Uses both recordings & feedback samples")

# ensure_folder_structure()
# tab1, tab2 = st.tabs(["🎙️ Record Reference", "🔍 Recognition & Feedback"])

# # TAB 1 — Add New Medicine
# with tab1:
#     st.subheader("Add New Medicine Reference Samples")
#     name = st.text_input("Enter Medicine Name:")
#     audio_data = st.audio_input("🎧 Record or Upload Reference Audio", sample_rate=16000)
#     if st.button("Save Reference"):
#         if name and audio_data:
#             file_name = f"{name.strip().replace(' ', '_').lower()}_{len(list_dropbox_files('recordings'))+1}.wav"
#             upload_bytes_to_dropbox(audio_data.getvalue(), file_name, "recordings")
#             st.success(f"✅ Saved `{name}` to Dropbox/recordings/")
#         else:
#             st.warning("⚠️ Enter a name and record/upload audio.")

# # TAB 2 — Recognition + Feedback
# with tab2:
#     st.subheader("Recognize Medicine Name")
#     test_audio = st.audio_input("🎧 Record or Upload Test Sample", sample_rate=16000)
#     if st.button("Recognize"):
#         if test_audio:
#             result, predicted = recognize_from_dropbox(test_audio)
#             st.markdown(result)
#             st.session_state["predicted"] = predicted
#         else:
#             st.warning("⚠️ Record or upload audio to recognize.")
#     if "predicted" in st.session_state:
#         st.divider()
#         fb = st.radio("Was the prediction correct?", ["Correct", "Incorrect"], horizontal=True)
#         correct_name = ""
#         if fb == "Incorrect":
#             correct_name = st.text_input("Enter correct name:")
#         if st.button("Submit Feedback"):
#             msg = handle_feedback(fb, correct_name, st.session_state["predicted"])
#             st.info(msg)

# st.markdown("---")
# st.caption("© 2025 AyurVoice Project | Auto-learning • Dropbox Storage • Feedback-driven Adaptation")





# slower version


# ============================================================
# 🧠 AyurVoice AI — Ayurvedic Medicine Voice Recognition
# (Dropbox-only • Auto-learning • Auto-Retrain Every 5 Feedbacks)
# ============================================================

import os, io, csv, joblib, librosa, numpy as np, pandas as pd, streamlit as st
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from librosa.sequence import dtw
import dropbox

# ============================================================
# 🔑 DROPBOX CONNECTION
# ============================================================

def connect_dropbox():
    try:
        token = st.secrets["dropbox"]["access_token"]
        dbx = dropbox.Dropbox(token)
        dbx.users_get_current_account()
        return dbx
    except Exception as e:
        st.error(f"❌ Dropbox connection failed: {e}")
        return None

# ============================================================
# 🧩 DROPBOX HELPERS
# ============================================================

def upload_bytes_to_dropbox(bytes_data, file_name, folder):
    dbx = connect_dropbox()
    if not dbx: return
    try:
        path = f"/AyurVoice/{folder}/{file_name}"
        dbx.files_upload(bytes_data, path, mode=dropbox.files.WriteMode("overwrite"))
    except Exception as e:
        st.warning(f"⚠️ Upload failed: {e}")

def list_dropbox_files(folder):
    dbx = connect_dropbox()
    if not dbx: return []
    try:
        result = dbx.files_list_folder(f"/AyurVoice/{folder}")
        return [entry.name for entry in result.entries if isinstance(entry, dropbox.files.FileMetadata)]
    except dropbox.exceptions.ApiError:
        return []

def download_dropbox_file(folder, file_name):
    dbx = connect_dropbox()
    if not dbx: return None
    try:
        _, res = dbx.files_download(f"/AyurVoice/{folder}/{file_name}")
        return io.BytesIO(res.content)
    except:
        return None

def ensure_folder_structure():
    dbx = connect_dropbox()
    if not dbx: return
    folders = ["recordings", "new_samples", "models", "feedback"]
    for f in folders:
        path = f"/AyurVoice/{f}"
        try:
            dbx.files_get_metadata(path)
        except dropbox.exceptions.ApiError:
            dbx.files_create_folder_v2(path)

# ============================================================
# 🎚️ AUDIO PROCESSING
# ============================================================

def extract_mfcc_from_bytes(audio_bytes, sr=16000, n_mfcc=20):
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr)
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

def similarity_score(bytes1, bytes2):
    y1, sr1 = librosa.load(io.BytesIO(bytes1), sr=16000)
    y2, sr2 = librosa.load(io.BytesIO(bytes2), sr=16000)
    D, _ = dtw(librosa.feature.mfcc(y=y1, sr=sr1), librosa.feature.mfcc(y=y2, sr=sr2), metric="cosine")
    return 1 / (1 + D[-1, -1])

# ============================================================
# 🧠 MODEL TRAINING
# ============================================================

def train_svm_from_dropbox():
    dbx = connect_dropbox()
    if not dbx: return None, 0, 0
    X, y = [], []
    all_files = list_dropbox_files("recordings") + list_dropbox_files("new_samples")

    # Load feedback mappings
    feedback_map = {}
    try:
        _, res = dbx.files_download("/AyurVoice/feedback/feedback_log.csv")
        df = pd.read_csv(io.BytesIO(res.content))
        for _, row in df.iterrows():
            feedback_map[row["audio_path"]] = row["correct"]
    except Exception:
        pass

    # Extract MFCCs
    for file in all_files:
        folder = "new_samples" if file in list_dropbox_files("new_samples") else "recordings"
        file_bytes = download_dropbox_file(folder, file)
        if not file_bytes:
            continue
        label = "_".join(file.replace(".wav", "").split("_")[:-1]) or file.replace(".wav", "")
        if f"{folder}/{file}" in feedback_map:
            label = feedback_map[f"{folder}/{file}"]
        X.append(extract_mfcc_from_bytes(file_bytes.read()))
        y.append(label.lower())

    if not X:
        st.warning("❌ No data for training.")
        return None, 0, 0

    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True))
    model.fit(np.array(X), np.array(y))
    acc = model.score(np.array(X), np.array(y)) * 100

    model_bytes = io.BytesIO()
    joblib.dump(model, model_bytes)
    model_bytes.seek(0)
    upload_bytes_to_dropbox(model_bytes.read(), "svm_model.joblib", "models")

    return "svm_model.joblib", acc, len(X)

# ============================================================
# 🧩 FEEDBACK HANDLER
# ============================================================

def append_feedback_to_csv(predicted, correct, feedback):
    dbx = connect_dropbox()
    if not dbx: return
    rows = []
    try:
        _, res = dbx.files_download("/AyurVoice/feedback/feedback_log.csv")
        df = pd.read_csv(io.BytesIO(res.content))
        rows = df.values.tolist()
    except:
        pass
    rows.append([f"new_samples/{predicted}.wav", predicted, correct, feedback])
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["audio_path", "predicted", "correct", "feedback"])
    writer.writerows(rows)
    upload_bytes_to_dropbox(out.getvalue().encode(), "feedback_log.csv", "feedback")

# ============================================================
# 🔍 RECOGNITION ENGINE
# ============================================================

def recognize_from_dropbox(audio_data):
    refs = list_dropbox_files("recordings")
    if not refs:
        return "❌ No reference samples found.", None
    input_bytes = audio_data.getvalue()
    scores = []
    for ref in refs:
        ref_bytes = download_dropbox_file("recordings", ref)
        if ref_bytes:
            sim = similarity_score(input_bytes, ref_bytes.read())
            base = "_".join(ref.replace(".wav", "").split("_")[:-1])
            scores.append((base.replace("_", " "), sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    top3 = scores[:3]
    pred = top3[0][0]
    result = "🎯 **Top 3 Matches (DTW Similarity):**\n"
    for n, s in top3:
        result += f"- {n}: {s*100:.2f}%\n"
    result += f"\n✅ **Predicted Medicine:** {pred}"
    new_name = f"{pred}_{len(list_dropbox_files('new_samples'))+1}.wav"
    upload_bytes_to_dropbox(input_bytes, new_name, "new_samples")
    return result, pred

# ============================================================
# 🔁 FEEDBACK + RETRAIN (Persistent Counter)
# ============================================================

def handle_feedback(feedback, correct_name, predicted):
    if "feedback_count" not in st.session_state:
        st.session_state.feedback_count = 0

    correct_label = correct_name.strip().lower() if feedback == "Incorrect" and correct_name else predicted
    append_feedback_to_csv(predicted, correct_label, feedback)
    st.session_state.feedback_count += 1

    msg = f"📝 Feedback saved for `{predicted}` → `{correct_label}`.\n🧩 {st.session_state.feedback_count}/5 before next retrain."

    if st.session_state.feedback_count >= 5:
        st.info("⚙️ Retraining model... Please wait ⏳")
        model_path, acc, total = train_svm_from_dropbox()
        msg = f"""
✅ **Model Retrained Successfully!**
- Model File: `{model_path}`
- Samples Used: {total}
- Accuracy: {acc:.2f}%
"""
        st.session_state.feedback_count = 0

    return msg

# ============================================================
# 🖥️ STREAMLIT UI
# ============================================================

st.set_page_config(page_title="AyurVoice AI — Dropbox", layout="wide")
st.title("🧠 AyurVoice AI — Ayurvedic Medicine Voice Recognition (Dropbox)")
st.caption("Cloud-only • Auto-learning • Uses both recordings & feedback samples")

ensure_folder_structure()
tab1, tab2 = st.tabs(["🎙️ Record Reference", "🔍 Recognition & Feedback"])

# TAB 1 — Add New Medicine
with tab1:
    st.subheader("Add New Medicine Reference Samples")
    name = st.text_input("Enter Medicine Name:")
    audio_data = st.audio_input("🎧 Record or Upload Reference Audio", sample_rate=16000)
    if st.button("Save Reference"):
        if name and audio_data:
            file_name = f"{name.strip().replace(' ', '_').lower()}_{len(list_dropbox_files('recordings'))+1}.wav"
            upload_bytes_to_dropbox(audio_data.getvalue(), file_name, "recordings")
            st.success(f"✅ Saved `{name}` to Dropbox/recordings/")
        else:
            st.warning("⚠️ Enter a name and record/upload audio.")

# TAB 2 — Recognition + Feedback
with tab2:
    st.subheader("Recognize Medicine Name")
    test_audio = st.audio_input("🎧 Record or Upload Test Sample", sample_rate=16000)
    if st.button("Recognize"):
        if test_audio:
            result, predicted = recognize_from_dropbox(test_audio)
            st.markdown(result)
            st.session_state["predicted"] = predicted
        else:
            st.warning("⚠️ Record or upload audio to recognize.")
    if "predicted" in st.session_state:
        st.divider()
        fb = st.radio("Was the prediction correct?", ["Correct", "Incorrect"], horizontal=True)
        correct_name = ""
        if fb == "Incorrect":
            correct_name = st.text_input("Enter correct name:")
        if st.button("Submit Feedback"):
            msg = handle_feedback(fb, correct_name, st.session_state["predicted"])
            st.info(msg)

st.markdown("---")
st.caption("© 2025 AyurVoice Project | Auto-learning • Dropbox Storage • Feedback-driven Adaptation")



















