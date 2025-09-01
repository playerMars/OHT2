# ==============================================================================
# Flask OCR API - Backend الكامل
# ==============================================================================

from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import os
import json
import uuid
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
import traceback
import shutil
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
from difflib import SequenceMatcher
import argparse
import glob

# ==============================================================================
# إعداد التطبيق
# ==============================================================================

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "headers": ["Content-Type", "Authorization"]
    }
})

# الإعدادات
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['TEMP_FOLDER'] = 'temp'
app.config['SECRET_KEY'] = 'your-secret-key-here'

# إنشاء المجلدات
for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER'], app.config['TEMP_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# الامتدادات المسموحة
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'}

# ==============================================================================
# كلاس OCR (مدمج)
# ==============================================================================

class EnhancedOCR:
    def __init__(self):
        self.supported_languages = {
            'eng': 'English',
            'ara': 'Arabic', 
            'ara+eng': 'Arabic + English'
        }
        
    def preprocess_image(self, image_path, method='auto'):
        """معالجة الصورة قبل استخراج النص"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"لا يمكن قراءة الصورة: {image_path}")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.medianBlur(gray, 3)
            
            if method == 'auto' or method == 'threshold':
                _, processed = cv2.threshold(denoised, 0, 255, 
                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            elif method == 'adaptive':
                processed = cv2.adaptiveThreshold(denoised, 255, 
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
                
            elif method == 'morphology':
                _, thresh = cv2.threshold(denoised, 0, 255, 
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = np.ones((2,2), np.uint8)
                processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            else:
                processed = denoised
                
            return processed
            
        except Exception as e:
            print(f"خطأ في معالجة الصورة: {e}")
            return None
    
    def extract_text(self, image, language='eng', psm_mode=6):
        """استخراج النص من الصورة"""
        try:
            if language == 'ara' or 'ara' in language:
                config = f'--oem 3 --psm {psm_mode}'
            else:
                config = f'--oem 3 --psm {psm_mode} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?;: '
            
            text = pytesseract.image_to_string(image, lang=language, config=config)
            return text.strip()
            
        except Exception as e:
            return f"خطأ في استخراج النص: {e}"
    
    def get_text_with_confidence(self, image, language='eng'):
        """استخراج النص مع معلومات الثقة"""
        try:
            data = pytesseract.image_to_data(image, lang=language, output_type=pytesseract.Output.DICT)
            
            filtered_text = []
            confidences = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:
                    text = data['text'][i].strip()
                    if text:
                        filtered_text.append(text)
                        confidences.append(int(data['conf'][i]))
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                'text': ' '.join(filtered_text),
                'confidence': round(avg_confidence, 2),
                'words_detected': len(filtered_text)
            }
            
        except Exception as e:
            return {'text': '', 'confidence': 0, 'error': str(e)}
    
    def process_single_image(self, image_path, language='eng', method='auto', psm_mode=6, save_processed=False):
        """معالجة صورة واحدة"""
        start_time = datetime.now()
        
        processed_img = self.preprocess_image(image_path, method)
        if processed_img is None:
            return None
        
        if save_processed:
            processed_path = image_path.replace('.', '_processed.')
            cv2.imwrite(processed_path, processed_img)
        
        basic_text = self.extract_text(processed_img, language, psm_mode)
        detailed_result = self.get_text_with_confidence(processed_img, language)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'image_path': image_path,
            'text': basic_text,
            'detailed': detailed_result,
            'language': language,
            'preprocessing_method': method,
            'processing_time': f"{processing_time:.2f}s",
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def compare_methods(self, image_path, language='eng'):
        """مقارنة طرق معالجة مختلفة لنفس الصورة"""
        methods = ['auto', 'threshold', 'adaptive', 'morphology']
        results = {}
        
        for method in methods:
            result = self.process_single_image(image_path, language, method)
            if result:
                results[method] = {
                    'text': result['text'],
                    'confidence': result['detailed']['confidence'],
                    'words_count': len(result['text'].split())
                }
        
        if results:
            best_method = max(results.keys(), key=lambda x: results[x]['confidence'])
            results['best_method'] = best_method
        
        return results

# إنشاء خدمة OCR
ocr_service = EnhancedOCR()

# إحصائيات التطبيق
app_stats = {
    'total_processed': 0,
    'successful': 0,
    'failed': 0,
    'start_time': datetime.now().isoformat(),
    'languages_used': {},
    'methods_used': {}
}

# ==============================================================================
# دوال مساعدة
# ==============================================================================

def allowed_file(filename):
    """التحقق من امتداد الملف"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_id():
    """إنتاج معرف فريد"""
    return str(uuid.uuid4())

def save_result(result_data, result_id):
    """حفظ النتيجة في ملف"""
    result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    return result_file

def update_stats(success=True, language='', method=''):
    """تحديث الإحصائيات"""
    app_stats['total_processed'] += 1
    if success:
        app_stats['successful'] += 1
    else:
        app_stats['failed'] += 1
    
    if language:
        app_stats['languages_used'][language] = app_stats['languages_used'].get(language, 0) + 1
    
    if method:
        app_stats['methods_used'][method] = app_stats['methods_used'].get(method, 0) + 1

def create_response(success=True, data=None, error=None, status_code=200):
    """إنشاء رد موحد"""
    response_data = {
        'success': success,
        'timestamp': datetime.now().isoformat()
    }
    
    if success and data is not None:
        response_data['data'] = data
    elif not success and error:
        response_data['error'] = error
    
    return jsonify(response_data), status_code

def cleanup_old_files():
    """تنظيف الملفات القديمة (أكثر من 24 ساعة)"""
    try:
        current_time = datetime.now()
        deleted_count = 0
        
        for folder in [app.config['UPLOAD_FOLDER'], app.config['TEMP_FOLDER']]:
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    if (current_time - file_time).days > 1:
                        os.remove(file_path)
                        deleted_count += 1
                elif os.path.isdir(file_path):
                    try:
                        shutil.rmtree(file_path)
                        deleted_count += 1
                    except:
                        pass
        
        return deleted_count
    except Exception as e:
        print(f"خطأ في تنظيف الملفات: {e}")
        return 0

# ==============================================================================
# Routes - المسارات
# ==============================================================================

@app.route('/')
def home():
    """الصفحة الرئيسية"""
    return create_response(data={
        'message': 'مرحباً بك في OCR API المتقدم',
        'version': '2.0.0',
        'features': [
            'استخراج النص من الصور',
            'دعم اللغة العربية والإنجليزية',
            'طرق معالجة متعددة',
            'معالجة مجمعة للصور',
            'واجهة ويب تفاعلية'
        ],
        'endpoints': {
            'POST /upload': 'رفع ومعالجة صورة واحدة',
            'POST /batch': 'معالجة مجموعة صور',
            'GET /result/<id>': 'جلب نتيجة معالجة',
            'GET /stats': 'إحصائيات التطبيق',
            'GET /health': 'فحص حالة الخادم',
            'GET /languages': 'اللغات المدعومة',
            'GET /compare/<id>': 'مقارنة طرق المعالجة',
            'DELETE /cleanup': 'تنظيف الملفات المؤقتة',
            'GET /download/<id>': 'تحميل ملف النتيجة',
            'GET /app': 'الواجهة التفاعلية'
        },
        'supported_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': '16MB'
    })

@app.route('/app')
def web_app():
    """عرض الواجهة التفاعلية"""
    # قراءة محتوى HTML من الملف الحالي أو إرجاع واجهة بسيطة
    return render_template_string('''
    <!DOCTYPE html>
    <html dir="rtl">
    <head>
        <title>OCR Web App</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial; padding: 20px; text-align: center; }
            .upload-area { border: 2px dashed #ccc; padding: 50px; margin: 20px 0; }
            button { padding: 10px 20px; margin: 10px; cursor: pointer; }
            #result { margin-top: 20px; padding: 20px; background: #f5f5f5; }
        </style>
    </head>
    <body>
        <h1>🔤 تطبيق OCR المتقدم</h1>
        <div class="upload-area" onclick="document.getElementById('file').click()">
            📁 اضغط هنا لرفع صورة
            <input type="file" id="file" style="display:none" accept="image/*">
        </div>
        
        <select id="language">
            <option value="eng">English</option>
            <option value="ara">العربية</option>
            <option value="ara+eng" selected>عربي + إنجليزي</option>
        </select>
        
        <button onclick="processImage()">معالجة الصورة</button>
        <div id="result"></div>
        
        <script>
            async function processImage() {
                const fileInput = document.getElementById('file');
                const language = document.getElementById('language').value;
                
                if (!fileInput.files[0]) {
                    alert('اختر صورة أولاً');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('language', language);
                
                document.getElementById('result').innerHTML = 'جاري المعالجة...';
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        document.getElementById('result').innerHTML = 
                            '<h3>النتيجة:</h3>' +
                            '<p><strong>النص:</strong> ' + (data.data.text || 'لم يتم استخراج نص') + '</p>' +
                            '<p><strong>الثقة:</strong> ' + data.data.confidence + '%</p>';
                    } else {
                        document.getElementById('result').innerHTML = 
                            '<p style="color: red">خطأ: ' + data.error + '</p>';
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = 
                        '<p style="color: red">خطأ في الاتصال: ' + error.message + '</p>';
                }
            }
        </script>
    </body>
    </html>
    ''')

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_and_process():
    """رفع ومعالجة صورة واحدة"""
    if request.method == 'OPTIONS':
        return create_response(data={'status': 'OK'})
    
    try:
        # التحقق من وجود ملف
        if 'file' not in request.files:
            return create_response(False, error='لم يتم رفع أي ملف', status_code=400)
        
        file = request.files['file']
        if file.filename == '':
            return create_response(False, error='لم يتم اختيار ملف', status_code=400)
        
        if not allowed_file(file.filename):
            return create_response(False, error=f'امتداد الملف غير مدعوم. الامتدادات المدعومة: {", ".join(ALLOWED_EXTENSIONS)}', status_code=400)
        
        # الحصول على المعاملات
        language = request.form.get('language', 'ara+eng')
        method = request.form.get('method', 'auto')
        psm_mode = int(request.form.get('psm_mode', 6))
        
        # التحقق من صحة المعاملات
        if language not in ocr_service.supported_languages:
            return create_response(False, error=f'لغة غير مدعومة: {language}', status_code=400)
        
        # إنتاج معرف فريد
        process_id = generate_unique_id()
        
        # حفظ الملف
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{process_id}_{filename}")
        file.save(file_path)
        
        # معالجة الصورة
        result = ocr_service.process_single_image(
            image_path=file_path,
            language=language,
            method=method,
            psm_mode=psm_mode
        )
        
        if result:
            # إضافة معلومات إضافية
            result['process_id'] = process_id
            result['original_filename'] = filename
            result['file_size'] = os.path.getsize(file_path)
            
            # حفظ النتيجة
            save_result(result, process_id)
            
            # تحديث الإحصائيات
            update_stats(success=True, language=language, method=method)
            
            return create_response(data={
                'process_id': process_id,
                'text': result['text'],
                'confidence': result['detailed']['confidence'],
                'words_detected': result['detailed']['words_detected'],
                'language': language,
                'method': method,
                'processing_time': result['processing_time'],
                'result_file': f"/result/{process_id}",
                'download_url': f"/download/{process_id}"
            })
        
        else:
            update_stats(success=False, language=language, method=method)
            return create_response(False, error='فشل في معالجة الصورة', status_code=500)
    
    except Exception as e:
        update_stats(success=False)
        return create_response(False, error=f'خطأ في الخادم: {str(e)}', status_code=500)

@app.route('/batch', methods=['POST', 'OPTIONS'])
def batch_process():
    """معالجة مجموعة من الصور"""
    if request.method == 'OPTIONS':
        return create_response(data={'status': 'OK'})
    
    try:
        # التحقق من وجود ملفات
        if 'files' not in request.files:
            return create_response(False, error='لم يتم رفع أي ملفات', status_code=400)
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            return create_response(False, error='لم يتم اختيار ملفات', status_code=400)
        
        # المعاملات
        language = request.form.get('language', 'ara+eng')
        method = request.form.get('method', 'auto')
        psm_mode = int(request.form.get('psm_mode', 6))
        
        # معرف المجموعة
        batch_id = generate_unique_id()
        batch_folder = os.path.join(app.config['UPLOAD_FOLDER'], batch_id)
        os.makedirs(batch_folder, exist_ok=True)
        
        results = []
        successful = 0
        failed = 0
        total_processing_time = 0
        
        for i, file in enumerate(files):
            if file.filename == '' or not allowed_file(file.filename):
                failed += 1
                results.append({
                    'file_index': i,
                    'original_filename': file.filename,
                    'error': 'امتداد ملف غير مدعوم'
                })
                continue
            
            try:
                # حفظ الملف
                filename = secure_filename(file.filename)
                file_path = os.path.join(batch_folder, f"{i:03d}_{filename}")
                file.save(file_path)
                
                # معالجة
                result = ocr_service.process_single_image(
                    image_path=file_path,
                    language=language,
                    method=method,
                    psm_mode=psm_mode
                )
                
                if result:
                    result['file_index'] = i
                    result['original_filename'] = filename
                    result['file_size'] = os.path.getsize(file_path)
                    results.append(result)
                    successful += 1
                    
                    # حساب وقت المعالجة الإجمالي
                    time_str = result['processing_time'].replace('s', '')
                    total_processing_time += float(time_str)
                else:
                    failed += 1
                    results.append({
                        'file_index': i,
                        'original_filename': filename,
                        'error': 'فشل في المعالجة'
                    })
            
            except Exception as e:
                failed += 1
                results.append({
                    'file_index': i,
                    'original_filename': file.filename,
                    'error': str(e)
                })
        
        # حساب الإحصائيات
        avg_confidence = 0
        total_words = 0
        if successful > 0:
            confidences = [r['detailed']['confidence'] for r in results if 'detailed' in r]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            total_words = sum(r['detailed']['words_detected'] for r in results if 'detailed' in r)
        
        # حفظ نتائج المجموعة
        batch_result = {
            'batch_id': batch_id,
            'total_files': len(files),
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / len(files) * 100) if files else 0,
            'language': language,
            'method': method,
            'psm_mode': psm_mode,
            'statistics': {
                'avg_confidence': round(avg_confidence, 2),
                'total_words': total_words,
                'total_processing_time': f"{total_processing_time:.2f}s",
                'avg_processing_time': f"{(total_processing_time/max(successful, 1)):.2f}s"
            },
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        save_result(batch_result, f"batch_{batch_id}")
        update_stats(success=successful > 0, language=language, method=method)
        
        return create_response(data={
            'batch_id': batch_id,
            'summary': {
                'total': len(files),
                'successful': successful,
                'failed': failed,
                'success_rate': f"{(successful/len(files)*100):.1f}%" if files else "0%",
                'avg_confidence': f"{avg_confidence:.1f}%",
                'total_processing_time': f"{total_processing_time:.2f}s"
            },
            'results_url': f"/result/batch_{batch_id}",
            'download_url': f"/download/batch_{batch_id}"
        })
    
    except Exception as e:
        return create_response(False, error=f'خطأ في المعالجة المجمعة: {str(e)}', status_code=500)

@app.route('/result/<result_id>')
def get_result(result_id):
    """جلب نتيجة معالجة"""
    try:
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
        
        if not os.path.exists(result_file):
            return create_response(False, error='النتيجة غير موجودة', status_code=404)
        
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        return create_response(data=result_data)
    
    except Exception as e:
        return create_response(False, error=f'خطأ في جلب النتيجة: {str(e)}', status_code=500)

@app.route('/download/<result_id>')
def download_result(result_id):
    """تحميل ملف النتيجة"""
    try:
        result_file = os.path.join(app.config['RESULTS_FOLDER'], f"{result_id}.json")
        
        if not os.path.exists(result_file):
            return create_response(False, error='الملف غير موجود', status_code=404)
        
        return send_file(
            result_file,
            as_attachment=True,
            download_name=f"ocr_result_{result_id}.json",
            mimetype='application/json'
        )
    
    except Exception as e:
        return create_response(False, error=f'خطأ في التحميل: {str(e)}', status_code=500)

@app.route('/stats')
def get_stats():
    """إحصائيات التطبيق"""
    try:
        uptime_seconds = (datetime.now() - datetime.fromisoformat(app_stats['start_time'])).total_seconds()
        uptime_formatted = str(timedelta(seconds=int(uptime_seconds)))
        
        # حساب معدلات الاستخدام
        total = max(app_stats['total_processed'], 1)
        success_rate = (app_stats['successful'] / total) * 100
        
        # إحصائيات المجلدات
        upload_files = len([f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                           if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))])
        result_files = len([f for f in os.listdir(app.config['RESULTS_FOLDER']) 
                           if os.path.isfile(os.path.join(app.config['RESULTS_FOLDER'], f))])
        
        stats_data = {
            'processing': {
                'total_processed': app_stats['total_processed'],
                'successful': app_stats['successful'],
                'failed': app_stats['failed'],
                'success_rate': f"{success_rate:.1f}%"
            },
            'system': {
                'uptime': uptime_formatted,
                'start_time': app_stats['start_time'],
                'current_time': datetime.now().isoformat()
            },
            'usage': {
                'languages_used': app_stats['languages_used'],
                'methods_used': app_stats['methods_used'],
                'most_used_language': max(app_stats['languages_used'].items(), key=lambda x: x[1])[0] if app_stats['languages_used'] else None,
                'most_used_method': max(app_stats['methods_used'].items(), key=lambda x: x[1])[0] if app_stats['methods_used'] else None
            },
            'storage': {
                'uploaded_files': upload_files,
                'result_files': result_files,
                'supported_languages': list(ocr_service.supported_languages.keys()),
                'supported_formats': list(ALLOWED_EXTENSIONS)
            }
        }
        
        return create_response(data=stats_data)
    
    except Exception as e:
        return create_response(False, error=f'خطأ في جلب الإحصائيات: {str(e)}', status_code=500)

@app.route('/health')
def health_check():
    """فحص حالة التطبيق"""
    try:
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'ocr_service': ocr_service is not None,
                'file_system': {
                    'upload_folder': os.path.exists(app.config['UPLOAD_FOLDER']),
                    'results_folder': os.path.exists(app.config['RESULTS_FOLDER']),
                    'temp_folder': os.path.exists(app.config['TEMP_FOLDER'])
                }
            },
            'configuration': {
                'max_file_size': app.config['MAX_CONTENT_LENGTH'],
                'supported_formats': list(ALLOWED_EXTENSIONS),
                'supported_languages': list(ocr_service.supported_languages.keys())
            }
        }
        
        # فحص الذاكرة والقرص (إذا كان psutil متاحاً)
        try:
            import psutil
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            health_data['system'] = {
                'memory_usage': f"{memory.percent:.1f}%",
                'memory_available': f"{memory.available / (1024**3):.1f} GB",
                'disk_usage': f"{(disk.used / disk.total * 100):.1f}%",
                'disk_free': f"{disk.free / (1024**3):.1f} GB"
            }
        except ImportError:
            health_data['system'] = {'note': 'psutil not installed - system metrics unavailable'}
        
        # فحص Tesseract
        try:
            import pytesseract
            tesseract_version = pytesseract.get_tesseract_version()
            health_data['services']['tesseract'] = {
                'available': True,
                'version': str(tesseract_version)
            }
        except:
            health_data['services']['tesseract'] = {
                'available': False,
                'error': 'Tesseract not properly installed'
            }
        
        return create_response(data=health_data)
    
    except Exception as e:
        return create_response(False, error=f'خطأ في فحص الصحة: {str(e)}', status_code=500)

@app.route('/languages')
def get_supported_languages():
    """جلب اللغات المدعومة"""
    return create_response(data={
        'supported_languages': ocr_service.supported_languages,
        'default': 'ara+eng',
        'available_methods': ['auto', 'threshold', 'adaptive', 'morphology'],
        'psm_modes': {
            '3': 'صفحة كاملة',
            '6': 'كتلة نص موحدة (افتراضي)',
            '8': 'كلمة واحدة',
            '13': 'سطر نص واحد'
        }
    })

@app.route('/compare/<result_id>')
def compare_methods(result_id):
    """مقارنة طرق المعالجة لصورة معينة"""
    try:
        # البحث عن الملف الأصلي
        upload_files = []
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.startswith(result_id):
                upload_files.append(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        if not upload_files:
            return create_response(False, error='الملف الأصلي غير موجود', status_code=404)
        
        image_path = upload_files[0]
        language = request.args.get('language', 'ara+eng')
        
        # مقارنة الطرق
        comparison_results = ocr_service.compare_methods(image_path, language)
        
        return create_response(data={
            'image_path': os.path.basename(image_path),
            'language': language,
            'comparison': comparison_results,
            'recommendation': f"الطريقة الأفضل: {comparison_results.get('best_method', 'غير محدد')}"
        })
    
    except Exception as e:
        return create_response(False, error=f'خطأ في المقارنة: {str(e)}', status_code=500)

@app.route('/cleanup', methods=['DELETE'])
def cleanup():
    """تنظيف الملفات المؤقتة"""
    try:
        deleted_count = cleanup_old_files()
        
        return create_response(data={
            'deleted_files': deleted_count,
            'message': f'تم حذف {deleted_count} ملف/مجلد قديم',
            'cleanup_time': datetime.now().isoformat()
        })
    
    except Exception as e:
        return create_response(False, error=f'خطأ في التنظيف: {str(e)}', status_code=500)

@app.route('/list')
def list_files():
    """عرض قائمة بالملفات والنتائج"""
    try:
        upload_files = []
        result_files = []
        
        # ملفات المرفوعة
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                stat_info = os.stat(file_path)
                upload_files.append({
                    'name': filename,
                    'size': stat_info.st_size,
                    'created': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(stat_info.st_mtime).isoformat()
                })
        
        # ملفات النتائج
        for filename in os.listdir(app.config['RESULTS_FOLDER']):
            file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
            if os.path.isfile(file_path):
                stat_info = os.stat(file_path)
                result_files.append({
                    'name': filename,
                    'size': stat_info.st_size,
                    'created': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                    'download_url': f"/download/{filename.replace('.json', '')}"
                })
        
        return create_response(data={
            'uploaded_files': {
                'count': len(upload_files),
                'files': upload_files
            },
            'result_files': {
                'count': len(result_files),
                'files': result_files
            },
            'total_size': {
                'uploads': sum(f['size'] for f in upload_files),
                'results': sum(f['size'] for f in result_files)
            }
        })
    
    except Exception as e:
        return create_response(False, error=f'خطأ في عرض الملفات: {str(e)}', status_code=500)

# ==============================================================================
# معالج الأخطاء
# ==============================================================================

@app.errorhandler(413)
def file_too_large(e):
    return create_response(False, error='حجم الملف كبير جداً (الحد الأقصى 16MB)', status_code=413)

@app.errorhandler(404)
def not_found(e):
    return create_response(False, error='المسار غير موجود', status_code=404)

@app.errorhandler(405)
def method_not_allowed(e):
    return create_response(False, error='طريقة HTTP غير مسموحة', status_code=405)

@app.errorhandler(500)
def internal_error(e):
    return create_response(False, error='خطأ داخلي في الخادم', status_code=500)

@app.before_request
def before_request():
    """معالجة ما قبل الطلب"""
    # تسجيل الطلبات (اختياري)
    if app.debug:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {request.method} {request.path}")

# ==============================================================================
# مهام دورية
# ==============================================================================

def scheduled_cleanup():
    """تنظيف دوري للملفات القديمة"""
    import threading
    import time
    
    def cleanup_task():
        while True:
            try:
                time.sleep(3600)  # كل ساعة
                cleanup_old_files()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] تم تنظيف الملفات القديمة")
            except Exception as e:
                print(f"خطأ في التنظيف الدوري: {e}")
    
    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()

# ==============================================================================
# تشغيل التطبيق
# ==============================================================================

if __name__ == '__main__':
    print("🚀 بدء تشغيل OCR API Server المتقدم")
    print("=" * 60)
    print(f"📁 مجلد الرفع: {app.config['UPLOAD_FOLDER']}")
    print(f"📊 مجلد النتائج: {app.config['RESULTS_FOLDER']}")
    print(f"🗂️ مجلد مؤقت: {app.config['TEMP_FOLDER']}")
    print(f"🌐 اللغات المدعومة: {list(ocr_service.supported_languages.keys())}")
    print(f"📋 الصيغ المدعومة: {list(ALLOWED_EXTENSIONS)}")
    print(f"📏 الحد الأقصى للملف: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")
    print("=" * 60)
    
    print("📡 المسارات المتاحة:")
    print("  🏠 GET  /                 - الصفحة الرئيسية")
    print("  🖥️ GET  /app              - الواجهة التفاعلية")
    print("  📤 POST /upload           - رفع صورة واحدة")
    print("  📤 POST /batch            - رفع مجموعة صور")
    print("  📄 GET  /result/<id>      - جلب النتيجة")
    print("  💾 GET  /download/<id>    - تحميل النتيجة")
    print("  📊 GET  /stats            - الإحصائيات")
    print("  🏥 GET  /health           - فحص الصحة")
    print("  🌐 GET  /languages        - اللغات المدعومة")
    print("  🔍 GET  /compare/<id>     - مقارنة الطرق")
    print("  📋 GET  /list             - قائمة الملفات")
    print("  🗑️ DELETE /cleanup        - تنظيف الملفات")
    print("=" * 60)
    
    # بدء المهام الدورية
    scheduled_cleanup()
    
    # تشغيل الخادم
    try:
        app.run(
            host='0.0.0.0',          # للوصول من الخارج
            port=5000,               # المنفذ
            debug=True,              # وضع التطوير
            threaded=True,           # دعم متعدد الخيوط
            use_reloader=False       # تعطيل إعادة التحميل التلقائي
        )
    except KeyboardInterrupt:
        print("\n⏹️ تم إيقاف الخادم بواسطة المستخدم")
    except Exception as e:
        print(f"❌ خطأ في تشغيل الخادم: {e}")