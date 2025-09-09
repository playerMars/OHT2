# ==============================================================================
# Flask OCR API - Backend الكامل
# ==============================================================================

from flask import Flask, request, jsonify, send_file, render_template_string, render_template
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
    return render_template("index.html")

@app.route('/app')
def web_app():
    """عرض الواجهة التفاعلية"""
    # قراءة محتوى HTML من الملف الحالي أو إرجاع واجهة بسيطة
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

# ==============================================================================
# تصحيح مسار /batch للتعامل مع مجموعة الصور
# ==============================================================================

@app.route('/batch', methods=['POST', 'OPTIONS'])
def batch_process():
    """معالجة مجموعة من الصور"""
    if request.method == 'OPTIONS':
        return create_response(data={'status': 'OK'})
    
    try:
        print("🔍 بدء معالجة طلب batch...")
        print(f"Content-Type: {request.content_type}")
        print(f"Form keys: {list(request.form.keys())}")
        print(f"Files keys: {list(request.files.keys())}")
        
        # التحقق من وجود ملفات - مهم: البحث عن 'files' وليس 'file'
        if 'files' not in request.files:
            print("❌ لم يتم العثور على 'files' في request.files")
            return create_response(False, error='لم يتم رفع أي ملفات. تأكد من استخدام اسم الحقل "files"', status_code=400)
        
        files = request.files.getlist('files')
        print(f"📁 عدد الملفات المرفوعة: {len(files)}")
        
        if not files or all(f.filename == '' for f in files):
            print("❌ لا توجد ملفات صحيحة")
            return create_response(False, error='لم يتم اختيار ملفات صحيحة', status_code=400)
        
        # طباعة أسماء الملفات للتتبع
        for i, file in enumerate(files):
            print(f"📄 ملف {i+1}: {file.filename}")
        
        # المعاملات
        language = request.form.get('language', 'ara+eng')
        method = request.form.get('method', 'auto')
        psm_mode = int(request.form.get('psm_mode', 6))
        
        print(f"🔧 المعاملات - اللغة: {language}, الطريقة: {method}, PSM: {psm_mode}")
        
        # التحقق من صحة اللغة
        if language not in ocr_service.supported_languages:
            return create_response(False, error=f'لغة غير مدعومة: {language}. اللغات المدعومة: {list(ocr_service.supported_languages.keys())}', status_code=400)
        
        # معرف المجموعة
        batch_id = generate_unique_id()
        batch_folder = os.path.join(app.config['UPLOAD_FOLDER'], batch_id)
        os.makedirs(batch_folder, exist_ok=True)
        print(f"📂 تم إنشاء مجلد المجموعة: {batch_folder}")
        
        results = []
        successful = 0
        failed = 0
        total_processing_time = 0
        
        for i, file in enumerate(files):
            print(f"\n🔄 معالجة الملف {i+1}/{len(files)}: {file.filename}")
            
            # التحقق من صحة الملف
            if file.filename == '':
                print(f"⚠️ ملف فارغ في الفهرس {i}")
                failed += 1
                results.append({
                    'file_index': i,
                    'original_filename': 'ملف فارغ',
                    'error': 'اسم ملف فارغ'
                })
                continue
            
            if not allowed_file(file.filename):
                print(f"⚠️ امتداد ملف غير مدعوم: {file.filename}")
                failed += 1
                results.append({
                    'file_index': i,
                    'original_filename': file.filename,
                    'error': f'امتداد ملف غير مدعوم. المدعوم: {", ".join(ALLOWED_EXTENSIONS)}'
                })
                continue
            
            try:
                # حفظ الملف
                filename = secure_filename(file.filename)
                if not filename:  # في حالة كان اسم الملف يحتوي على أحرف غير مدعومة فقط
                    filename = f"file_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                
                file_path = os.path.join(batch_folder, f"{i:03d}_{filename}")
                file.save(file_path)
                print(f"💾 تم حفظ الملف: {file_path}")
                
                # التحقق من حفظ الملف
                if not os.path.exists(file_path):
                    raise Exception("فشل في حفظ الملف")
                
                file_size = os.path.getsize(file_path)
                print(f"📊 حجم الملف: {file_size} بايت")
                
                if file_size == 0:
                    raise Exception("الملف فارغ")
                
                # معالجة الصورة
                print(f"🖼️ بدء معالجة الصورة...")
                result = ocr_service.process_single_image(
                    image_path=file_path,
                    language=language,
                    method=method,
                    psm_mode=psm_mode
                )
                
                if result and result.get('text') is not None:
                    result['file_index'] = i
                    result['original_filename'] = file.filename
                    result['file_size'] = file_size
                    results.append(result)
                    successful += 1
                    print(f"✅ نجحت معالجة الملف {i+1}")
                    print(f"📝 طول النص المستخرج: {len(result.get('text', ''))}")
                    
                    # حساب وقت المعالجة الإجمالي
                    time_str = result['processing_time'].replace('s', '')
                    try:
                        total_processing_time += float(time_str)
                    except ValueError:
                        print(f"⚠️ لا يمكن تحويل وقت المعالجة: {time_str}")
                else:
                    failed += 1
                    results.append({
                        'file_index': i,
                        'original_filename': file.filename,
                        'error': 'فشل في معالجة الصورة - لم يتم استخراج نص'
                    })
                    print(f"❌ فشل في معالجة الملف {i+1}")
            
            except Exception as e:
                failed += 1
                error_msg = str(e)
                results.append({
                    'file_index': i,
                    'original_filename': file.filename,
                    'error': error_msg
                })
                print(f"❌ خطأ في معالجة الملف {i+1}: {error_msg}")
        
        print(f"\n📊 ملخص المعالجة:")
        print(f"   إجمالي: {len(files)}")
        print(f"   نجح: {successful}")
        print(f"   فشل: {failed}")
        print(f"   وقت المعالجة الإجمالي: {total_processing_time:.2f}s")
        
        # حساب الإحصائيات
        avg_confidence = 0
        total_words = 0
        if successful > 0:
            confidences = [r['detailed']['confidence'] for r in results if 'detailed' in r and 'confidence' in r['detailed']]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            total_words = sum(r['detailed']['words_detected'] for r in results if 'detailed' in r and 'words_detected' in r['detailed'])
        
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
            'timestamp': datetime.now().isoformat(),
            'batch_folder': batch_folder
        }
        
        # حفظ النتيجة
        save_result(batch_result, f"batch_{batch_id}")
        update_stats(success=successful > 0, language=language, method=method)
        
        print(f"✅ تمت معالجة المجموعة بنجاح. ID: {batch_id}")
        
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
            'download_url': f"/download/batch_{batch_id}",
            'details': f"تمت معالجة {successful} من أصل {len(files)} ملف بنجاح"
        })
    
    except Exception as e:
        print(f"❌ خطأ عام في المعالجة المجمعة: {str(e)}")
        import traceback
        traceback.print_exc()
        return create_response(False, error=f'خطأ في المعالجة المجمعة: {str(e)}', status_code=500)


# ==============================================================================
# تحسينات إضافية لحل مشاكل الـ batch processing
# ==============================================================================

@app.route('/test-batch', methods=['GET'])
def test_batch_endpoint():
    """اختبار نقطة النهاية batch"""
    return create_response(data={
        'message': 'نقطة النهاية /batch متاحة وتعمل',
        'expected_field_name': 'files',
        'supported_methods': ['POST'],
        'required_parameters': {
            'files': 'قائمة من ملفات الصور',
            'language': 'اختياري - افتراضي ara+eng',
            'method': 'اختياري - افتراضي auto',
            'psm_mode': 'اختياري - افتراضي 6'
        },
        'example_curl': '''
        curl -X POST http://localhost:5000/batch \\
          -F "files=@image1.jpg" \\
          -F "files=@image2.jpg" \\
          -F "language=ara+eng" \\
          -F "method=auto"
        ''',
        'troubleshooting': {
            'http_404': 'تأكد أن الخادم يعمل على المنفذ الصحيح',
            'field_name': 'استخدم "files" وليس "file" كاسم للحقل',
            'multiple_files': 'تأكد من إضافة multiple="true" في HTML'
        }
    })

@app.route('/debug-request', methods=['POST'])
def debug_request():
    """تحليل الطلب لاستكشاف الأخطاء"""
    debug_info = {
        'method': request.method,
        'content_type': request.content_type,
        'form_keys': list(request.form.keys()),
        'files_keys': list(request.files.keys()),
        'form_data': dict(request.form),
        'headers': dict(request.headers),
        'endpoint': request.endpoint,
        'url': request.url
    }
    
    # معلومات الملفات
    files_info = {}
    for key in request.files.keys():
        files_list = request.files.getlist(key)
        files_info[key] = [
            {
                'filename': f.filename,
                'content_type': f.content_type,
                'size': len(f.read()) if hasattr(f, 'read') else 'unknown'
            }
            for f in files_list
        ]
        # إعادة تعيين مؤشر الملف للقراءة مرة أخرى
        for f in files_list:
            if hasattr(f, 'seek'):
                f.seek(0)
    
    debug_info['files_info'] = files_info
    
    return create_response(data=debug_info)


# ==============================================================================
# دالة مساعدة محسنة للتحقق من الملفات
# ==============================================================================

def validate_uploaded_files(files):
    """التحقق من صحة الملفات المرفوعة"""
    if not files:
        return False, "لا توجد ملفات"
    
    valid_files = []
    errors = []
    
    for i, file in enumerate(files):
        if not file.filename:
            errors.append(f"الملف {i+1}: اسم فارغ")
            continue
            
        if not allowed_file(file.filename):
            errors.append(f"الملف {i+1} ({file.filename}): امتداد غير مدعوم")
            continue
            
        # قراءة بداية الملف للتحقق من أنه صورة فعلية
        try:
            file.seek(0)
            header = file.read(10)
            file.seek(0)
            
            # التحقق من headers الصور الشائعة
            image_headers = [
                b'\xff\xd8\xff',  # JPEG
                b'\x89PNG\r\n\x1a\n',  # PNG
                b'GIF87a',  # GIF87a
                b'GIF89a',  # GIF89a
                b'BM',  # BMP
                b'II*\x00',  # TIFF (little endian)
                b'MM\x00*'   # TIFF (big endian)
            ]
            
            is_image = any(header.startswith(h) for h in image_headers)
            if not is_image:
                errors.append(f"الملف {i+1} ({file.filename}): ليس صورة صحيحة")
                continue
                
        except Exception as e:
            errors.append(f"الملف {i+1} ({file.filename}): خطأ في القراءة - {str(e)}")
            continue
        
        valid_files.append(file)
    
    if not valid_files:
        return False, f"لا توجد ملفات صحيحة. الأخطاء: {'; '.join(errors)}"
    
    return True, f"تم العثور على {len(valid_files)} ملف صحيح من أصل {len(files)}"


# ==============================================================================
# تحسين معالج الأخطاء
# ==============================================================================

@app.errorhandler(413)
def file_too_large(e):
    """معالج خطأ حجم الملف الكبير"""
    max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    return create_response(
        False, 
        error=f'حجم الملف/الملفات كبير جداً. الحد الأقصى: {max_size_mb:.0f}MB', 
        status_code=413
    )

@app.errorhandler(400)
def bad_request(e):
    """معالج الطلبات الخاطئة"""
    return create_response(
        False,
        error='طلب غير صحيح. تأكد من إرسال الملفات بالطريقة الصحيحة',
        status_code=400
    )

# إضافة headers للـ CORS
@app.after_request
def after_request(response):
    """إضافة headers بعد كل طلب"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


# ==============================================================================
# اختبار سريع للـ OCR service
# ==============================================================================

@app.route('/test-ocr')
def test_ocr_service():
    """اختبار خدمة OCR"""
    try:
        # إنشاء صورة اختبار بسيطة
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # إنشاء صورة بيضاء مع نص
        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            # محاولة استخدام خط افتراضي
            font = ImageFont.load_default()
        except:
            font = None
        
        # كتابة نص تجريبي
        draw.text((10, 30), "Test OCR Service", fill='black', font=font)
        draw.text((10, 60), "اختبار الخدمة", fill='black', font=font)
        
        # حفظ الصورة مؤقتاً
        test_image_path = os.path.join(app.config['TEMP_FOLDER'], 'test_image.png')
        img.save(test_image_path)
        
        # تشغيل OCR
        result = ocr_service.process_single_image(test_image_path, 'ara+eng', 'auto')
        
        # حذف الصورة المؤقتة
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
        
        return create_response(data={
            'ocr_service_status': 'يعمل بشكل صحيح',
            'test_result': result,
            'extracted_text': result.get('text', '') if result else 'فشل الاستخراج'
        })
        
    except Exception as e:
        return create_response(False, error=f'خطأ في اختبار OCR: {str(e)}', status_code=500)

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