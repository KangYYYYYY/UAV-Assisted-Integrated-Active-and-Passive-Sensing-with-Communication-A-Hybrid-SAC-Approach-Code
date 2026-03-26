# UAV-Assisted Integrated Active and Passive Sensing with Communication: A Hybrid-SAC Approach

This repository contains the source code for the paper: **"UAV-Assisted Integrated Active and Passive Sensing with Communication: A Hybrid-SAC Approach"** published in *IEEE Transactions on Vehicular Technology (TVT)*, 2025.

[![Paper](https://img.shields.io/badge/IEEE-Paper-blue.svg)](https://ieeexplore.ieee.org/abstract/document/11208810)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTVT.2025.3623633-green.svg)](https://doi.org/10.1109/TVT.2025.3623633)


## Usage

To train the model and reproduce the results, you can run the following command:

```bash
python main.py algo="hsac" learning_rate=0.01 seed=0 bs_power=5 comm_uav_data_min=8000000 T=16 uav_assisted=1  --save
```

## Citation

If you find this code or our work useful in your research, please consider citing our paper:

```bibtex
@article{yan2025uav,
  title={UAV-Assisted Integrated Active and Passive Sensing with Communication: A Hybrid-SAC Approach},
  author={Yan, Kang and Xiang, Luping and Zheng, Kang and Hu, Jie and Yang, Kun and Liu, Jun},
  journal={IEEE Transactions on Vehicular Technology},
  year={2025},
  publisher={IEEE}
}
```

