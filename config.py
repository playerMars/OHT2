import os
from datetime import timedelta

class Config:
    """إعدادات التطبيق الأساسية"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_FILE_SIZE', 16 * 1024 * 1024))
    
    # مجلدات التطبيق
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    RESULTS_FOLDER = os.environ.get('RESULTS_FOLDER', 'results')
    TEMP_FOLDER = os.environ.get('TEMP_FOLDER', 'temp')
    
    # إعدادات OCR
    DEFAULT_LANGUAGE = os.environ.get('DEFAULT_LANGUAGE', 'ara+eng')
    TESSERACT_PATH = os.environ.get('TESSERACT_PATH', '')
    
    # إعدادات التنظيف
    CLEANUP_INTERVAL = timedelta(hours=24)
    MAX_FILE_AGE = timedelta(days=7)
    
    # إعدادات CORS
    CORS_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '*').split(',')

class DevelopmentConfig(Config):
    """إعدادات التطوير"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """إعدادات الإنتاج"""
    DEBUG = False
    TESTING = False
    
    # إعدادات أمان إضافية للإنتاج
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)

class TestingConfig(Config):
    """إعدادات الاختبار"""
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False