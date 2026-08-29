 Devanagari Character Classifier Web App
 
 Run locally
 1) Create/activate a Python 3.12 venv.
 2) Install deps:
    pip install -r requirements.txt
 3) Place your trained artifacts in the project root:
    - model.h5              (Keras model)
    - classes.npy           (NumPy array of label_encoder.classes_)
 4) Start server:
    python server.py
    # or
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
 5) Open the app: http://localhost:8000/
 
 Export artifacts from your notebook
 In your Jupyter notebook, after training:
 
 ```python
 # Save Keras model
 model.save('model.h5')
 
 # Save label classes from LabelEncoder
 import numpy as np
 np.save('classes.npy', label_encoder.classes_)
 ```
 
 Notes
 - The server expects grayscale images sized 32×32 during inference; any uploaded image is converted and resized server-side.
 - If you trained with a different preprocessing (e.g., inversion), mirror that in server.py -> preprocess_image_to_tensor.

