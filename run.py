#!/usr/bin/env python3
"""
ملف تشغيل تطبيق OCR
"""

import os
import sys
from pathlib import Path

# إضافة مجلد المشروع إلى path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

# استيراد التطبيق
try:
    from app import app, ocr_service
    from config import DevelopmentConfig, ProductionConfig
except ImportError as e:
    print(f"❌ خطأ في الاستيراد: {e}")
    print("تأكد من وجود ملف app.py في نفس المجلد")
    sys.exit(1)

def setup_environment():
    """إعداد البيئة"""
    # إنشاء المجلدات المطلوبة
    folders = ['uploads', 'results', 'temp', 'logs']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ تم إنشاء/التحقق من مجلد: {folder}")
    
    # فحص Tesseract
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract متوفر - الإصدار: {version}")
    except Exception as e:
        print(f"⚠️ تحذير: مشكلة في Tesseract - {e}")
        print("تأكد من تثبيت Tesseract بشكل صحيح")
    
    # فحص اللغات المدعومة
    try:
        langs = pytesseract.get_languages()
        print(f"✅ اللغات المتوفرة: {langs}")
        
        required_langs = ['eng', 'ara']
        missing = [lang for lang in required_langs if lang not in langs]
        if missing:
            print(f"⚠️ لغات مفقودة: {missing}")
            print("قم بتثبيتها: sudo apt install tesseract-ocr-ara")
    except:
        print("⚠️ لا يمكن التحقق من اللغات المدعومة")

def main():
    """الدالة الرئيسية"""
    print("🚀 بدء تطبيق OCR المتقدم")
    print("=" * 50)
    
    # إعداد البيئة
    setup_environment()
    
    # تحديد وضع التشغيل
    env = os.environ.get('FLASK_ENV', 'development')
    
    if env == 'production':
        app.config.from_object(ProductionConfig)
        print("🏭 وضع الإنتاج")
    else:
        app.config.from_object(DevelopmentConfig)
        print("🛠️ وضع التطوير")
    
    # معلومات التطبيق
    print(f"🌐 الخادم: http://localhost:5000")
    print(f"🖥️ الواجهة: http://localhost:5000/app")
    print(f"📊 الإحصائيات: http://localhost:5000/stats")
    print(f"🏥 فحص الصحة: http://localhost:5000/health")
    
    # تشغيل التطبيق
    try:
        app.run(
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            debug=app.config['DEBUG'],
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n⏹️ تم إيقاف التطبيق")
    except Exception as e:
        print(f"❌ خطأ في التشغيل: {e}")

if __name__ == '__main__':
    main()