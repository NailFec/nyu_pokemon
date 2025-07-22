- The data should be put in `pokemon/` folder. You can download the data from HPC of NYU Shanghai or [my Google Drive shared file](https://drive.google.com/file/d/1oixv6cPOASd4TS6rF94nO6nxupO1qczu/view?usp=drive_link).
- The HPC cannot download file from the internet, so you may need to download `resnet50-11ad3fa6.pth` from [pytorch.org](https://download.pytorch.org/models/resnet50-11ad3fa6.pth) and put it in the home folder. 
- After running `train.py`, the model will be saved in `saved_models/` folder.

The file structure should be as this:

```text
.
├── dataset.py
├── debug_dataset.useless.py
├── info.useless.py
├── model.py
├── pokemon
│  ├── pokemon_train
│  │  ├── train_data.npy
│  │  └── train_labels.npy
│  └── types2label.txt
├── predict.py
├── README.md
├── resnet50-11ad3fa6.pth
├── saved_models
│  ├── best_model.pth.tar
│  └── checkpoint.pth.tar
├── train.py
└── utils.py
```
