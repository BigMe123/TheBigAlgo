import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras available: {hasattr(tf, 'keras')}")

if hasattr(tf, 'keras'):
    print("Importing keras modules...")
    from tensorflow.keras.models import Sequential # type: ignore
    print("Sequential model imported successfully")