import os
import numpy as np
from flask import Blueprint, current_app, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from .audio_processing import (
    load_audio,
    compute_fft,
    apply_eq,
    normalize_peak,
    save_audio,
    EQ_BANDS,
    DEFAULT_SR,
    compute_eq_response,
)
from .ml_models import get_model_manager

main_bp = Blueprint("main", __name__)


def allowed_file(filename):
    allowed_ext = current_app.config.get("ALLOWED_EXTENSIONS", set())
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_ext


def upload_path(filename: str) -> str:
    return os.path.join(current_app.config["UPLOAD_FOLDER"], secure_filename(filename))


@main_bp.route("/")
def index():
    return render_template("dashboard.html")


@main_bp.route("/api/audio/upload", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    filename = secure_filename(file.filename)
    filepath = upload_path(filename)
    file.save(filepath)
    
    try:
        y, sr = load_audio(filepath)
        duration = len(y) / sr
        
        max_points = 2000  # Giới hạn số điểm để không quá nặng
        step = max(1, len(y) // max_points)
        waveform_data = y[::step].tolist()
        time_axis = np.arange(0, len(y), step) / sr
        
        # Tự động classify audio để detect label
        detected_mode = "None"
        try:
            models_dir = os.path.join(current_app.root_path, "..", "models")
            models_dir = os.path.normpath(models_dir)  # Normalize path
            model_manager = get_model_manager(models_dir=models_dir)
            model_manager.initialize()
            detected_mode, confidence, _ = model_manager.classify_audio(filepath)
            print(f"Detected mode: {detected_mode} (confidence: {confidence:.2f})")
        except Exception as e:
            print(f"Classification error (using default): {e}")
            # Fallback: không có model hoặc lỗi → dùng default
        
        return jsonify({
            "success": True,
            "filename": filename,
            "duration": duration,
            "sample_rate": sr,
            "waveform": {
                "data": waveform_data,
                "time": time_axis.tolist()
            },
            "detected_mode": detected_mode  # Thêm detected mode vào response
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@main_bp.route("/api/audio/analyze", methods=["POST"])
def analyze_audio():
    data = request.get_json()
    filename = data.get("filename")
    
    if not filename:
        return jsonify({"error": "Filename required"}), 400
    
    filepath = upload_path(filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    try:
        y, sr = load_audio(filepath)
        
        freqs, mag_db = compute_fft(y, sr)
        step = max(1, len(freqs) // 500)
        fft_data = {
            "frequencies": freqs[::step].tolist(),
            "magnitude_db": mag_db[::step].tolist()
        }
        
        return jsonify({
            "success": True,
            "fft": fft_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main_bp.route("/api/audio/process", methods=["POST"])
def process_audio():
    data = request.get_json()
    filename = data.get("filename")
    eq_gains = data.get("eq_gains", [0] * 9)  
    
    if not filename:
        return jsonify({"error": "Filename required"}), 400
    
    if len(eq_gains) != 9:
        return jsonify({"error": "EQ gains must have 9 values"}), 400
    
    filepath = upload_path(filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    try:
        y, sr = load_audio(filepath)
        
        # Normalize
        y = normalize_peak(y, target_db=-1.0)
        
        # Áp dụng EQ
        y_processed = apply_eq(y, sr, eq_gains, q=1.0)
        
        # Normalize lại
        y_processed = normalize_peak(y_processed, target_db=-1.0)
        
        # Tính waveform sau xử lý
        max_points = 2000
        step = max(1, len(y_processed) // max_points)
        waveform_data = y_processed[::step].tolist()
        time_axis = np.arange(0, len(y_processed), step) / sr
        
        # Tính FFT sau xử lý
        freqs, mag_db = compute_fft(y_processed, sr)
        step_fft = max(1, len(freqs) // 500)
        fft_data = {
            "frequencies": freqs[::step_fft].tolist(),
            "magnitude_db": mag_db[::step_fft].tolist()
        }
        
        return jsonify({
            "success": True,
            "waveform": {
                "data": waveform_data,
                "time": time_axis.tolist()
            },
            "fft": fft_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main_bp.route("/api/audio/eq-bands", methods=["GET"])
def get_eq_bands():
    """Trả về danh sách các tần số EQ bands."""
    return jsonify({
        "bands": EQ_BANDS,
        "default_sr": DEFAULT_SR
    })


@main_bp.route("/api/audio/eq-response", methods=["POST"])
def eq_response():
    data = request.get_json()
    eq_gains = data.get("eq_gains", [0] * 9)
    sr = data.get("sr", DEFAULT_SR)
    q = data.get("q", 1.0)

    if len(eq_gains) != 9:
        return jsonify({"success": False, "error": "EQ gains must have 9 values"}), 400

    try:
        freqs_hz, mag_db, phase = compute_eq_response(sr, eq_gains, q=float(q), n_freqs=2048)
        step = max(1, len(freqs_hz) // 500)
        return jsonify({
            "success": True,
            "freqs_hz": freqs_hz[::step].tolist(),
            "mag_db": mag_db[::step].tolist(),
            "phase": phase[::step].tolist(),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@main_bp.route("/api/audio/play", methods=["POST"])
def get_processed_audio():
    data = request.get_json()
    filename = data.get("filename")
    eq_gains = data.get("eq_gains", [0] * 9)
    original = bool(data.get("original", False))
    
    if not filename:
        return jsonify({"error": "Filename required"}), 400
    
    if not original and len(eq_gains) != 9:
        return jsonify({"error": "EQ gains must have 9 values"}), 400
    
    filepath = upload_path(filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    try:
        import hashlib
        if original:
            output_filename = f"original_{secure_filename(filename)}"
            output_path = upload_path(output_filename)
            if not os.path.exists(output_path):
                y, sr = load_audio(filepath)
                y = normalize_peak(y, target_db=-1.0)
                save_audio(output_path, y, sr)
        else:
            eq_hash = hashlib.md5(str(eq_gains).encode()).hexdigest()[:8]
            output_filename = f"processed_{eq_hash}_{secure_filename(filename)}"
            output_path = upload_path(output_filename)
            
            if not os.path.exists(output_path):
                y, sr = load_audio(filepath)
                y = normalize_peak(y, target_db=-1.0)
                y_processed = apply_eq(y, sr, eq_gains, q=1.0)
                y_processed = normalize_peak(y_processed, target_db=-1.0)
                
                save_audio(output_path, y_processed, sr)
        
        return jsonify({
            "success": True,
            "audio_url": f"/api/audio/file/{output_filename}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main_bp.route("/api/audio/play-original", methods=["POST"])
def get_original_audio():
    data = request.get_json()
    filename = data.get("filename")
    
    if not filename:
        return jsonify({"error": "Filename required"}), 400
    
    filepath = upload_path(filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    try:
        output_filename = f"original_{secure_filename(filename)}"
        output_path = upload_path(output_filename)
        
        if not os.path.exists(output_path):
            y, sr = load_audio(filepath)
            y = normalize_peak(y, target_db=-1.0)
            save_audio(output_path, y, sr)
        
        return jsonify({
            "success": True,
            "audio_url": f"/api/audio/file/{output_filename}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main_bp.route("/api/audio/file/<filename>", methods=["GET"])
def serve_audio_file(filename):
    filepath = upload_path(filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    return send_file(filepath, mimetype="audio/wav")


@main_bp.route("/api/audio/classify", methods=["POST"])
def classify_audio():
    """API endpoint để classify audio thành label."""
    data = request.get_json()
    filename = data.get("filename")
    
    if not filename:
        return jsonify({"error": "Filename required"}), 400
    
    filepath = upload_path(filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    try:
        models_dir = os.path.join(current_app.root_path, "..", "models")
        models_dir = os.path.normpath(models_dir)  # Normalize path
        model_manager = get_model_manager(models_dir=models_dir)
        model_manager.initialize()
        
        predicted_label, confidence, all_probs = model_manager.classify_audio(filepath)
        
        return jsonify({
            "success": True,
            "label": predicted_label,
            "confidence": confidence,
            "probabilities": all_probs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main_bp.route("/api/audio/suggest-eq", methods=["POST"])
def suggest_eq():
    """API endpoint để suggest EQ preset từ audio."""
    data = request.get_json()
    filename = data.get("filename")
    
    if not filename:
        return jsonify({"error": "Filename required"}), 400
    
    filepath = upload_path(filename)
    
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    try:
        models_dir = os.path.join(current_app.root_path, "..", "models")
        models_dir = os.path.normpath(models_dir)  # Normalize path
        model_manager = get_model_manager(models_dir=models_dir)
        model_manager.initialize()
        
        eq_gains = model_manager.suggest_eq(filepath)
        
        return jsonify({
            "success": True,
            "eq_gains": eq_gains,
            "bands": EQ_BANDS
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


