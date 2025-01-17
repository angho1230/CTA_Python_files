import cta_detector

code, qr, temp = cta_detector.run('photo.jpg','svm_model_20241230.joblib', False)
print(code, qr, temp)



