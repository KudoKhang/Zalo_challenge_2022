
## Full Pipeline for submission Zalo AI Challenge 2022

### Check Docker version
`
docker --version
`

### List docker images
`
docker images
`

### List docker containers
`
docker ps -a
`

### Pull docker image
```
docker pull pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
```

### Run docker
```
docker run --gpus '"device=0"' --network host -it --name zac2022 pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel /bin/bash
```

### Check GPU
`nvidia-smi`

### Source code

- Open a new terminal and `cd` to your source code directory
- Loading trained Model:
  - Download model weights [SSAN_R_pprivate_epochs_29.pth](https://drive.google.com/file/d/1ctfwLNKZkKdN_prwiz2Mn-7hUBASC9YX/view?usp=sharing "RectiNet Weights")
  - Save under `saved_model` folder
- `/mnt/datadrive/thonglv/SSAN/zalo`: my source code

### Copy source code to container
```
docker cp /mnt/datadrive/thonglv/SSAN/zalo/. zac2022:/code
```

### Switch to docker terminal
- `cd /code/`
- Run: `sh predict.sh`
- Install the necessary packages and re-run to make sure that the code is run successfully
- Run: `sh start_jupyter.sh`to make sure that jupyter notebook is run successfully
- Open: localhost with port `9777`

### Docker commit
Switch to your source code director terminal
```
docker commit zac2022 zac2022:v1
```

### Check predict function to export submission.csv
```
docker run --gpus '"device=0"' -v /mnt/datadrive/thonglv/SSAN/data/:/data -v /mnt/datadrive/thonglv/SSAN/:/result zac2022:v1 /bin/bash /code/predict.sh
```
and
```
docker run --gpus '"device=0"' -p9777:9777 -v /mnt/datadrive/thonglv/SSAN/data:/code/data -v /mnt/datadrive/thonglv/SSAN/:/code/result zac2022:v1 /bin/bash /code/start_jupyter.sh
```
(data in container /code/data/private_test/videos/...)

The submission files will be saved in `/mnt/datadrive/thonglv/SSAN/` (`submission.csv`, `jupyter_submission.csv`, `time_submission.csv`)

### Export images and check MD5
`docker save -o zac2022.tar.gz zac2022:v1`
`md5sum zac2022.tar.gz`

## Reference
[Hướng dẫn dùng docker để nộp bài cho Zalo
AI Challenge 2022](https://dl-challenge.zalo.ai/Docker_ZAC2022.pdf) 
