# REMAP: A Open-source Framework for Automating the Design of CNN Accelerators.
This is the implementation of the paper "REMAP: A Spatiotemporal CNN Accelerator Optimization Methodology and Toolkit Thereof". 
# What is MAESTRO?
Designing CNN accelerators is getting more difficult owing to the fast-increasing types of CNN models.
Some approaches use constant dataflow and micro-architecture that have lower design comlexity.
However, these accelerators are difficult to adapt with the highly-diverse CNN models and often suffer from low PE utilization.
Some other accelerators resort to reconfigurable devices such as FPGA and CGRA to support flexible dataflows 
in order to to fit diverse CNN layers. 
However, layer-by-layer processing may require more energy for frequent reconfiguration and off-chip DDR access. 
In this work, we introduce a reconfigurable pipeline accelerator (RPA) that can reduce the latency and DDR access by pipelining the compuptation of CNN layers. 
Although there has been several researches that try to speedup the design process by automatically exploring sub-set of the accelerator design space,
identifying an available automated design tool that can effectively find the complete and optimal design scheme remains a problem, especially for the novel RPA architecture type. 
Unfortunately, comprehensive exploration of the whole design space faces an excessive large searching space. 
To tackle this problem, we propose REMAP, a toolkit for designing CNN accelerators based on the Monte Carlo Tree Search (MCTS) method and MAESTRO cost model. To efficiently search the huge design space, we propose several methods to improve searching efficiency. Evaluations show that REMAP significantly outperforms some state-of-the-art approaches; compared with GAMMA, it achieves an average speed increase of $14.75\times$, and an energy reduction of $45.45\%$; it also achieves a speed increase of $32.6\times$ against ConfuciuX on MobileNetV2 and ResNet50. 
We also show a FPGA accelerator implementation which is based on REMAP's search result, and it achieves high performance in real-time CNN tasks. 
This indicates that REMAP can provide high-quality design exploration with valuable insights and useful architecture design guidances.

### Setup ###
* Clone Repo
```
git clone https://github.com/jackboranzhao/REMAP.git
```
* Install Anaconda software and create virtual env
```
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash ./Anaconda3-2020.11-Linux-x86_64.sh
conda create -n rl_ven python=3.6.9 anaconda
conda activate rl_ven
```
* Install requirement
   
```
pip install -r requirements.txt
```

### Run ###
* Run REMAP
```
Python3 Env_Maestro_mcts.py
```

### Contributor ###
* Boran Zhao
* Tian Xia
* Haiming Zhai
* Fulun Ma
* Yan Du
* Hanzhi Chang
* Wenzhe Zhao
* Pengju Ren (Corresponding Author)

### Citation ###
```
To appear on TCAD'22
```