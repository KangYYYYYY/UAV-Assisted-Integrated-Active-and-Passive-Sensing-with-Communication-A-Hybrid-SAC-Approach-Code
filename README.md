# UAV-Assisted Integrated Active and Passive Sensing with Communication: A Hybrid-SAC Approach

This repository contains the source code for the paper: **"UAV-Assisted Integrated Active and Passive Sensing with Communication: A Hybrid-SAC Approach"** published in *IEEE Transactions on Vehicular Technology (TVT)*, 2025.

[![Paper](https://img.shields.io/badge/IEEE-Paper-blue.svg)](https://ieeexplore.ieee.org/abstract/document/11208810)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTVT.2025.3623633-green.svg)](https://doi.org/10.1109/TVT.2025.3623633)

## Abstract

As the low-altitude economy rapidly develops, a growing number of unmanned aerial vehicles (UAVs) will participate in specific missions that require periodic data from ground base stations (GBSs) to ensure safe flight operations. Such UAVs, due to their low communication demand, are ideal candidates to serve as sensing receivers during their free time, thereby enhancing environmental sensing performance. In this paper, we present a UAV-assisted framework that fuses active and passive sensing with downlink communication, pairing a GBS, a communication UAV following a pre-planned trajectory and a sensed UAV at an unknown position. We derive an analytical Cramér-Rao bound that unifies echoes collected at the GBS (active sensing) and at the airborne relay (passive sensing) to quantify three-dimensional localization accuracy, and cast the joint design of beamforming, power allocation and signal-frame scheduling as a mixed discrete-continuous optimization. A hybrid Soft Actor-Critic (hybrid-SAC) algorithm is proposed to address this problem. Simulation results demonstrate that the proposed method achieves a 36.5% improvement in sensing performance, outperforms DRL benchmarks in convergence, and surpasses traditional methods in sensing accuracy with lower computational complexity.

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
  publisher={IEEE},
  doi={10.1109/TVT.2025.3623633}
}
```

## Contact

If you have any questions or suggestions, please feel free to contact me at [kangyan@std.uestc.edu.cn](mailto:kangyan@std.uestc.edu.cn).

