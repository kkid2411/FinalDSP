

FinalDSP – Real-time Audio DSP System

Giới thiệu

FinalDSP là một hệ thống **Digital Signal Processing (DSP) cho âm thanh, được xây dựng bằng Python & Flask, tập trung vào:

* Xử lý tín hiệu âm thanh
* Phân tích đặc trưng âm thanh
* Phân loại / gợi ý xử lý (EQ, v.v.)
* Phục vụ dưới dạng web service / API

Dự án phù hợp cho mục đích:

* Đồ án môn học DSP
* Nghiên cứu xử lý âm thanh
* Demo hệ thống AI + Audio Processing

---

Kiến trúc tổng quan

```
FinalDSP/
│
├── app/                # Flask application (routes, services)
├── models/             # Model học máy (TensorFlow / Keras)
│
├── config.py           # Cấu hình môi trường (Dev / Prod)
├── wsgi.py             # Entry point chạy ứng dụng
├── requirements.txt    # Danh sách thư viện
├── .gitignore          # File loại trừ khi push Git
└── README.md
```

---

Yêu cầu hệ thống

* **Python >= 3.9**
* Hệ điều hành: Windows / Linux / macOS
* Khuyến nghị dùng **virtual environment**

---

Cài đặt

Clone repository

```bash
git clone https://github.com/kkid2411/FinalDSP.git
cd FinalDSP
```

2️⃣(Khuyến nghị) Tạo virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

3️⃣ Cài dependencies

```bash
pip install -r requirements.txt
```

Các thư viện chính bao gồm Flask, NumPy, Librosa, SciPy và TensorFlow 

---

Chạy ứng dụng

Chạy ở chế độ development

```bash
python wsgi.py
```

Mặc định server chạy tại:

```
http://localhost:8000
```

File `wsgi.py` là entry point chính của ứng dụng 

---

Cấu hình

Cấu hình ứng dụng nằm trong `config.py`, hỗ trợ:

* `DevConfig` (DEBUG = True)
* `ProdConfig` (DEBUG = False)

Một số biến quan trọng:

* `SECRET_KEY`
* `MAX_CONTENT_LENGTH` (mặc định 32MB)
* `ALLOWED_EXTENSIONS`: wav, mp3, flac, ogg, m4a
* `UPLOAD_SUBDIR`: thư mục upload

Chi tiết xem trong `config.py` 

---

Upload & xử lý file âm thanh

Hệ thống hỗ trợ các định dạng:

```
.wav  .mp3  .flac  .ogg  .m4a
```

File upload được lưu trong thư mục `uploads/` (đã được ignore khi push Git).

---

Bảo mật & Git

Repo đã cấu hình `.gitignore` để tránh push:

* Virtual environment
* File log
* Cache Python
* File môi trường (`.env`)
* File build / editor config

Xem chi tiết trong `.gitignore` 

---

Tác giả

* GitHub: kkid2411
* Project: FinalDSP

---

Ghi chú

* Project phục vụ mục đích học tập & nghiên cứu
* Không bao gồm file dữ liệu / model lớn
* Có thể mở rộng thêm REST API hoặc frontend

