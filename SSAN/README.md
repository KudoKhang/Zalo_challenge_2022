
## Domain Generalization via Shuffled Style Assembly for Face Anti-Spoofing (SSAN)
Face Anti-spoofing for Zalo AI Challenge 2022

[//]: # ()
[//]: # (### Modify configuration files)

[//]: # (Set paths properly in 'configs.yaml')

[//]: # (- FAS_thr: 0.2429 &#40;v10.5&#41;)

[//]: # (- openvino.FAS_xml)

[//]: # (- openvino.FAS_bin)

[//]: # ()
[//]: # (### weight path)

[//]: # (- weight: GPU2)

[//]: # (  - /home/thonglv/face-anti-spoofing/weights/FAS_112s_v10.5.bin)

[//]: # (  - /home/thonglv/face-anti-spoofing/weights/FAS_112s_v10.5.xml)

[//]: # ()
[//]: # ()
[//]: # (### Inference)

[//]: # (```)

[//]: # (python evaluate_v2.py)

[//]: # (```)

[//]: # ()
[//]: # (### Dataset)

[//]: # (```)

[//]: # (/mnt/datadrive2/dataset/dataset_faces/antispoof/face_attendance_test.zip)

[//]: # (GPU3 - 192.168.1.53)

[//]: # (```)