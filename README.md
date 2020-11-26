### LAGNet: Logic-Aware Graph Network for Human Interaction Understanding

This is the code repository for the paper **LAGNet: Logic-Aware Graph Network for Human Interaction Understanding**,
which includes `test codes`,`trained models`, `dataset annotaions of bit and tvhi`.

#### How to set the environment 
```
numpy==1.18.1
torchvision==0.6.0
torch==1.5.0
Pillow==8.0.1
scikit_learn==0.23.2
```
#### How to download the models

Download the models from the following link,and put them to `./model`

- [baidu pan ](https://pan.baidu.com/s/1WVMTSscPZhO2OOPflziTyg)

#### How to get the dataset
The directories `BIT/`and `highfive/`already contain the annotaions for each dataset.
After getting a dataset downloaded, you need to extract frames per video, put the
extracted frames into a folder taking the same name as the video, and move the folder
to either "BIT/Bit-frames" or "highfive/frm". 

#### How to run the code 

Taking `stage-tvhi-lagnet.py` as an example:

```bash
cd ./scripts/
python stage-tvhi-lagnet.py
```

