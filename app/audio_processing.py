import numpy as np
import librosa
import soundfile as sf
from scipy.signal import sosfilt, sosfreqz

# =========================
# 1. Tham số chung
# =========================

DEFAULT_SR = 44100  # tần số lấy mẫu chuẩn
EPS = 1e-12         # tránh log(0)

# 9 band EQ chuẩn: 63, 125, 250, 500, 1k, 2k, 4k, 8k, 16k (Hz)
EQ_BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


# =========================
# 2. Đọc / ghi file audio
# =========================

def load_audio(path: str, sr: int = DEFAULT_SR):
    """
    Đọc file audio, chuyển về mono, resample về sr (nếu cần).

    Trả về:
        y: np.ndarray (mono)
        sr: int (tần số lấy mẫu thực tế)
    """
    y, file_sr = librosa.load(path, sr=None, mono=True)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y, sr


def save_audio(path: str, y: np.ndarray, sr: int = DEFAULT_SR):
    """Lưu tín hiệu y ra file WAV."""
    sf.write(path, y, sr)


# =========================
# 3. Chuẩn hoá tín hiệu
# =========================

def normalize_peak(y: np.ndarray, target_db: float = -1.0) -> np.ndarray:
    """
    Chuẩn hoá theo peak: đưa đỉnh lớn nhất về target_db (ví dụ -1 dBFS).

    target_db = -1  => peak ~ 0.891
    """
    peak = np.max(np.abs(y)) + EPS
    target_lin = 10.0 ** (target_db / 20.0)
    gain = target_lin / peak
    return y * gain


# =========================
# 4. EQ 9-band (biquad peaking)
# =========================

def _design_peaking_eq(fs: int, f0: float, gain_db: float, q: float = 1.0):
    """
    Thiết kế 1 biquad peaking EQ (trả về SOS: [b0,b1,b2,a0,a1,a2]).
    Công thức chuẩn của peaking filter (RBJ style).
    """
    # Giới hạn f0 < Nyquist để tránh lỗi
    f0 = min(float(f0), fs * 0.49)

    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A

    # Chuẩn hoá về a0 = 1
    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)

    sos = np.hstack([b, a]).reshape(1, -1)  # shape (1, 6)
    return sos


def apply_eq(y: np.ndarray, sr: int, gains_db: list, q: float = 1.0) -> np.ndarray:
    """
    Áp dụng EQ 9-band cho tín hiệu y.

    gains_db: list/array có 9 phần tử (tương ứng EQ_BANDS).
              Đơn vị dB. >0 là boost, <0 là cut.

    Trả về: y_eq (đã xử lý EQ).
    """
    assert len(gains_db) == len(EQ_BANDS), "Gains phải có 9 phần tử (63→16k)."

    y_eq = y.copy()
    sos_list = []

    for f0, g in zip(EQ_BANDS, gains_db):
        if abs(float(g)) < 0.1:   # gần 0 dB => bỏ qua để tiết kiệm tính toán
            continue
        sos = _design_peaking_eq(sr, f0, float(g), q=q)
        sos_list.append(sos)

    if sos_list:
        sos_all = np.vstack(sos_list)  # (n_filters, 6)
        y_eq = sosfilt(sos_all, y_eq)

    return y_eq


# =========================
# 4b. (NEW) Tính đáp ứng tần số EQ
# =========================

def compute_eq_response(sr: int,
                        gains_db: list,
                        q: float = 1.0,
                        n_freqs: int = 4096):
    """
    Tính đáp ứng tần số của EQ (cascade peaking biquads).

    Trả về:
        freqs_hz: (n_freqs,) tần số (Hz)
        mag_db  : (n_freqs,) biên độ (dB)
        phase   : (n_freqs,) pha (rad)
    """
    assert len(gains_db) == len(EQ_BANDS), "Gains phải có 9 phần tử (63→16k)."

    sos_list = []
    for f0, g in zip(EQ_BANDS, gains_db):
        if abs(float(g)) < 0.1:
            continue
        sos_list.append(_design_peaking_eq(sr, f0, float(g), q=q))

    # Không có filter nào => đáp ứng phẳng 0 dB
    if not sos_list:
        freqs_hz = np.linspace(0.0, sr / 2.0, n_freqs, dtype=np.float64)
        mag_db = np.zeros_like(freqs_hz)
        phase = np.zeros_like(freqs_hz)
        return freqs_hz, mag_db, phase

    sos_all = np.vstack(sos_list)

    # worN có thể là số điểm hoặc mảng tần số (rad/sample).
    # Ở đây dùng số điểm n_freqs -> sosfreqz trả về w (rad/sample) và h.
    w, h = sosfreqz(sos_all, worN=n_freqs, fs=sr)  # w lúc này là Hz vì fs=sr
    freqs_hz = w.astype(np.float64)

    mag = np.abs(h) + EPS
    mag_db = 20.0 * np.log10(mag)
    phase = np.angle(h)

    return freqs_hz, mag_db, phase


# =========================
# 5. Noise gate (tuỳ chọn)
# =========================

def noise_gate(y: np.ndarray,
               threshold_db: float = -50.0,
               reduction_db: float = -80.0) -> np.ndarray:
    """
    Noise gate đơn giản:
    - Nếu |x| < threshold => giảm xuống reduction_db (gần như im lặng)
    - Nếu |x| >= threshold => giữ nguyên.
    """
    amp = np.abs(y) + EPS
    level_db = 20.0 * np.log10(amp)

    gate_on = level_db < threshold_db
    gain_db = np.zeros_like(y, dtype=np.float64)
    gain_db[gate_on] = reduction_db  # giảm mạnh

    gain_lin = 10.0 ** (gain_db / 20.0)
    return y * gain_lin


# =========================
# 6. Compressor đơn giản (tuỳ chọn)
# =========================

def compressor(y: np.ndarray,
               threshold_db: float = -18.0,
               ratio: float = 4.0,
               makeup_db: float = 0.0) -> np.ndarray:
    """
    Compressor tĩnh đơn giản ở miền sample:
    - Nếu level < threshold => không đổi
    - Nếu level > threshold => nén theo ratio
    - Sau đó cộng thêm makeup gain nếu cần.
    """
    amp = np.abs(y) + EPS
    level_db = 20.0 * np.log10(amp)

    gain_db = np.zeros_like(level_db, dtype=np.float64)

    over = level_db > threshold_db
    over_amount = level_db[over] - threshold_db
    compressed = over_amount / ratio
    new_level_db = threshold_db + compressed

    gain_db[over] = new_level_db - level_db[over]
    gain_db += makeup_db

    gain_lin = 10.0 ** (gain_db / 20.0)
    return y * gain_lin


# =========================
# 7. FFT & Spectrogram
# =========================

def compute_fft(y: np.ndarray, sr: int):
    """
    Tính phổ biên độ (dB) cho tín hiệu y.
    Dùng để vẽ biểu đồ FFT trên GUI.
    """
    n = len(y)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag = np.abs(np.fft.rfft(y)) / n
    mag_db = 20.0 * np.log10(mag + EPS)
    return freqs, mag_db


def compute_spectrogram(y: np.ndarray, sr: int,
                        n_fft: int = 2048,
                        hop_length: int = 512):
    """
    Tính spectrogram (đơn vị dB) dùng STFT.
    Trả về:
        S_db: (freq_bins, time_frames)
        freqs, times: trục cho việc vẽ.
    """
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)
    return S_db, freqs, times


# =========================
# 8. Pipeline xử lý trọn file
# =========================

def process_audio_file(input_path: str,
                       output_path: str,
                       eq_gains_db: list,
                       enable_gate: bool = False,
                       gate_threshold_db: float = -50.0,
                       enable_compressor: bool = False,
                       comp_threshold_db: float = -18.0,
                       comp_ratio: float = 4.0,
                       comp_makeup_db: float = 0.0,
                       normalize_target_db: float = -1.0,
                       sr: int = DEFAULT_SR):
    """
    Hàm xử lý trọn file audio theo pipeline Topic 2:

      1) Load file, chuyển mono + resample
      2) Normalize (peak) về target_db
      3) Áp dụng EQ 9-band
      4) Noise gate (nếu bật)
      5) Compressor (nếu bật)
      6) Normalize lần cuối + lưu file output
    """
    # 1) Load
    y, sr = load_audio(input_path, sr=sr)

    # 2) Normalize ban đầu
    y = normalize_peak(y, target_db=normalize_target_db)

    # 3) EQ
    y = apply_eq(y, sr, eq_gains_db, q=1.0)

    # 4) Noise gate (tuỳ chọn)
    if enable_gate:
        y = noise_gate(y, threshold_db=gate_threshold_db)

    # 5) Compressor (tuỳ chọn)
    if enable_compressor:
        y = compressor(y,
                       threshold_db=comp_threshold_db,
                       ratio=comp_ratio,
                       makeup_db=comp_makeup_db)

    # 6) Normalize lần cuối và lưu file
    y = normalize_peak(y, target_db=normalize_target_db)
    save_audio(output_path, y, sr)

    return y, sr
