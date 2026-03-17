import os
import sys

print("Python version:", sys.version)
print("CWD:", os.getcwd())

try:
    import tensorflow as tf
    print("TensorFlow version:", tf.__version__)
except Exception as e:
    print("TensorFlow import failed:", e)
    sys.exit(1)

model_path = "models/lstm_model.h5"
if os.path.exists(model_path):
    print(f"Found {model_path}, attempting to load...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Success! Model loaded.")
    except Exception as e:
        print("Model load failed:", e)
else:
    print(f"File not found: {model_path}")
