import streamlit as st
import hashlib
from datetime import datetime
import PIL
import sqlite3
import io
from ultralytics import YOLO
from conditions import get_conditions
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Penyakit Mata Anjing menggunakan YOLOv8",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Kredensial yang telah ditentukan (hashed password)
USERNAME = "admin"
PASSWORD_HASH = "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9"

# Inisialisasi session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Login"
if "webcam_frame" not in st.session_state:
    st.session_state.webcam_frame = None

# Koneksi ke database
conn = sqlite3.connect('detection_history.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS detections
             (id INTEGER PRIMARY KEY, timestamp TEXT, confidence REAL, boxes TEXT, detected_image BLOB)''')

# Fungsi menyimpan hasil deteksi
def simpan_deteksi(timestamp, confidence, boxes, detected_image):
    img_byte_arr = io.BytesIO()
    detected_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    c.execute("INSERT INTO detections (timestamp, confidence, boxes, detected_image) VALUES (?, ?, ?, ?)",
              (timestamp, confidence, str(boxes), img_byte_arr))
    conn.commit()

# Fungsi menghapus hasil deteksi
def hapus_deteksi(detection_id):
    c.execute("DELETE FROM detections WHERE id=?", (detection_id,))
    conn.commit()

# Fungsi memuat model
def muat_model(model_path):
    return YOLO(model_path)

# Class for video processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self, model, confidence):
        self.model = model
        self.confidence = confidence
        self.last_frame = None
        self.last_boxes = None

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            results = self.model.predict(img, conf=self.confidence)
            detected_image = results[0].plot()[:, :, ::-1]
            self.last_frame = detected_image  # Keep as BGR
            self.last_boxes = results[0].boxes
            return frame.from_ndarray(detected_image, format="bgr24")
        except Exception as e:
            st.error(f"Error during video processing: {e}")
            return frame

    def get_last_detection(self):
        return self.last_frame, self.last_boxes

# Fungsi login
def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if username == USERNAME and hash_password(password) == PASSWORD_HASH:
            st.session_state.logged_in = True
            st.session_state.page = "Deteksi"
            st.sidebar.success("Login berhasil!")
        else:
            st.sidebar.error("Username atau password salah")

# Fungsi halaman deteksi
def halaman_deteksi():
    st.title("Deteksi Penyakit Mata Anjing menggunakan YOLOv8")

    st.sidebar.title("Navigasi")
    st.sidebar.button("Logout", on_click=logout)
    st.sidebar.button("Riwayat Deteksi", on_click=berpindah_halaman, args=("Riwayat Deteksi",))
    st.sidebar.button("Daftar Penyakit", on_click=berpindah_halaman, args=("Daftar Penyakit",))

    st.sidebar.subheader("Konfigurasi Model ML")
    confidence = st.sidebar.slider("Pilih Tingkat Keyakinan Model", 25, 100, 40) / 100

    try:
        model = muat_model('weights/best.pt')
    except Exception as ex:
        st.error("Tidak dapat memuat model.")
        st.error(ex)

    st.sidebar.subheader("Konfigurasi Gambar/Video")
    sumber = st.sidebar.radio("Pilih Sumber", ['Image', 'Webcam'])

    if sumber == 'Image':
        source_img = st.sidebar.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png", "bmp", "webp"])
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Gambar Asli")
            try:
                if source_img is None:
                    st.image('images/placeholder.jpg', caption="Gambar Placeholder", use_column_width=True,
                             output_format='JPEG')
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Gambar yang Diunggah", use_column_width=True, output_format='JPEG')
            except Exception as ex:
                st.error("Terjadi kesalahan saat membuka gambar.")
                st.error(ex)

        with col2:
            st.subheader("Gambar yang Dideteksi")
            if source_img is None:
                st.image('images/placeholder_deteksi.jpg', caption='Gambar Placeholder yang Dideteksi',
                         use_column_width=True,
                         output_format='JPEG')
            else:
                if st.sidebar.button('Deteksi Objek'):
                    res = model.predict(uploaded_image, conf=confidence)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Gambar yang Dideteksi', use_column_width=True, output_format='JPEG')

                    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                    detected_image = PIL.Image.fromarray(res_plotted)
                    simpan_deteksi(timestamp, confidence, boxes, detected_image)

                    try:
                        with st.expander("Hasil Deteksi"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as ex:
                        st.error("Terjadi kesalahan saat menampilkan hasil deteksi.")
                        st.error(ex)

    elif sumber == 'Webcam':
        st.subheader("Webcam")
        rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        webrtc_ctx = webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV,
                                     media_stream_constraints={
                                         "video": True,
                                         "audio": False },
                                     video_processor_factory=lambda: VideoProcessor(model, confidence),
                                     rtc_configuration=rtc_configuration)
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.confidence = confidence

        if webrtc_ctx.state.playing:
            if st.button("Simpan Frame"):
                if webrtc_ctx.video_processor:
                    detected_image, boxes = webrtc_ctx.video_processor.get_last_detection()
                    if detected_image is not None:
                        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                        detected_image_pil = PIL.Image.fromarray(detected_image)
                        simpan_deteksi(timestamp, confidence, boxes, detected_image_pil)
                        st.success("Frame berhasil disimpan ke database.")
                    else:
                        st.warning("Tidak ada frame yang dapat disimpan.")

# Fungsi halaman riwayat deteksi
def halaman_riwayat():
    st.title("Riwayat Deteksi")
    st.sidebar.title("Navigasi")
    st.sidebar.button("Logout", on_click=logout)
    st.sidebar.button("Deteksi", on_click=berpindah_halaman, args=("Deteksi",))
    st.sidebar.button("Daftar Penyakit", on_click=berpindah_halaman, args=("Daftar Penyakit",))

    c.execute("SELECT id, timestamp, confidence, detected_image FROM detections ORDER BY timestamp DESC")
    rows = c.fetchall()
    for row in rows:
        st.write(f"Waktu: {row[1]}")
        st.image(io.BytesIO(row[3]), caption=f'Deteksi dengan Confidence rate > {row[2]}', width=400)
        if st.button("Hapus", key=f"delete_{row[0]}"):
            hapus_deteksi(row[0])
            st.experimental_rerun()
        st.markdown("---")

# Fungsi halaman daftar penyakit
def halaman_penyakit():
    st.title("Daftar Kondisi yang Dapat Dideteksi")
    st.sidebar.title("Navigasi")
    st.sidebar.button("Logout", on_click=logout)
    st.sidebar.button("Deteksi", on_click=berpindah_halaman, args=("Deteksi",))
    st.sidebar.button("Riwayat Deteksi", on_click=berpindah_halaman, args=("Riwayat Deteksi",))

    conditions = get_conditions()
    for condition, description in conditions.items():
        st.subheader(condition)
        st.markdown(description)

# Fungsi navigasi
def berpindah_halaman(page):
    st.session_state.page = page

# Fungsi logout
def logout():
    st.session_state.logged_in = False
    st.session_state.page = "Login"

# Menampilkan halaman berdasarkan session state
if st.session_state.page == "Login":
    login()
elif st.session_state.page == "Deteksi":
    halaman_deteksi()
elif st.session_state.page == "Riwayat Deteksi":
    halaman_riwayat()
elif st.session_state.page == "Daftar Penyakit":
    halaman_penyakit()

# Tambahan CSS untuk estetika
st.markdown("""
<style>
    .css-1aumxhk {
        background-color: #f0f2f6;
        color: #333;
        box-shadow: 2px 2px 5px #888888;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)
