# Face anti-spoofing

The project is built to train and evaluate Central difference network (CDCN) for Face Anti-Spoofing. It will have 2 evaluation codes: one for CelebA-Spoofing and one for Private data.

# Prepare environment for training and evaluation.
pip install -r requirements.txt

# Facebox detection is used for api/fastapi directory, then it should be built-up again.

# For training CDCN++ on CelebA-Spoofing data.
python train_CDCN.py

# For training CDCN++ on Private data.
python train_private.py

# For evaluation on CelebA-Spoofing data.
python evaluation_CDCN.py

# For evaluation on Private data.
python evaluation_private_data.py

# For pre-process data
## Generate depth-map, we need 3DFFA_v2 models (https://github.com/cleardusk/3DDFA_V2)
python ./functions/process_depth.py
## Detect face and crop face for testing
python ./functions/detect_face.py