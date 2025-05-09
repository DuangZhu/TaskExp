<h1 align="center">
  <span style="font-size: 40px;">TaskExp</span> <!-- 将字体大小从30px调整到40px -->
  <br> <!-- 在描述文字之前的换行保持不变 -->
  Enhancing Generalization of Multi-Robot Exploration with Multi-Task Pre-Training
</h1>

TaskExp is a generic multi-task pre-training algorithm to enhance the generalization of learning-based multi-robot exploration policies. Our initial platform, [MAexp](https://github.com/DuangZhu/MAexp), revealed limitations with MARL (multi-agent reinforcement learning), specifically its tendency to produce effective policies restricted to single maps and fixed starting locations, which are difficult to generalize in real-world applications. To overcome this, TaskExp pre-trains policies with three tasks before MARL, significantly improving generalization. Below is our pre-training framework and some exploration results:
<img src=imgs/framework.png  />

### Pre-training Tasks

Our framework involves three core pre-training tasks—one focused on decision-making and two on perception:

- **Decision-related task:** Uses imitation learning to guide the policy to focus on the subset of the action space identified by planning-based exploration methods. This task softly narrows the decision space, making it easier to learn a reliable policy mapping.
- **Map-prediction task:** Encourages agents to integrate features received from teammates and produce a unified exploration map. This task helps agents learn to leverage messages from their teammates.
- **Location-estimation task:** Guides each agent to estimate its global coordinates while making decisions.

If you find this project useful, please consider giving it a star on GitHub! It helps the project gain visibility and supports the development. Thank you!

## Quick Start

### 1. Environment Setup

Follow the **Installation** and **Preparation** instructions in [MAexp Quick Start](https://github.com/DuangZhu/MAexp).

### 2. Install TaskExp

Clone the repository and install necessary dependencies:

```bash
conda activate maexp
git clone https://github.com/DuangZhu/TaskExp.git
pip install timm==0.3.2
```

Ensure your directory structure:

- `/Path/to/your/code/MAexp`
- `/Path/to/your/code/TaskExp`

### 3. Download Maps and Checkpoints

Download maps (**maze** and **random3**) and checkpoints from [Google Drive](https://drive.google.com/file/d/11Ao1quTjwT7-31JOwL9ciX8QR72zWnJw/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1pFMcWu7WBJ157UWCVogOpg?pwd=1234), and place them under `/Path/to/your/code/MAexp/map/`.

### 4. Data Collection

Collect data using Voronoi (recommended for fastest decision-making speed):

```bash
cd MAexp
python env_v7_3_ft.py --yaml_file /path/to/MAexp/yaml/maze_ft.yaml
```

Data will be saved in `./test_make_data`. You can modify `index` at line 1072 in `env_v8_ft.py` to store data separately when running multiple environments.

### 5. Pre-training

Run pre-training using multiple GPUs (adjust GPU count as needed):

```bash
cd ../TaskExp
python -m torch.distributed.launch --nproc_per_node=6 TaskExp_pretrain.py \
  --batch_size 896 \
  --world_size 6 \
  --output_dir ./results/test \
  --log_dir ./results/test \
  --blr 4e-4 \
  --data_path /path/to/MAexp/test_make_data/ours
```

Checkpoints will be saved under `./results/test`.

### 6. MARL Downstream Training

- Update pre-trained checkpoint in **Line 110** of `./MAexp/model_vit/actor_mae_crossatt_IL.py`.
- Adjust hyperparameters in the following file: `/path/to/your/environment/lib/python3.8/site-packages/marllib/marl/algos/hyperparams/common/vda2c.yaml`

```yaml
algo_args:
  use_gae: True
  lambda: 1.0
  vf_loss_coeff: 1.0
  batch_episode: 1
  batch_mode: "truncate_episodes"
  lr: 0.0001
  entropy_coeff: 0.001
  mixer: "qmix"
  devide_ac: False
  warmup_epochs: 100
  only_critic_epochs: 3000
  actor_warmup_epochs: 500
  min_lr: 0.
```

Run training:

```bash
cd MAexp
python env_v8.py --yaml_file /path/to/MAexp/yaml/maze.yaml
```

If encountering Ray-related issues, adjust settings in `/path/to/your/environment/lib/python3.8/site-packages/marllib/marl/ray/ray.yaml`:

```yaml
local_mode: False # True for debug mode only
share_policy: "group" #  individual(separate) / group(division) / all(share)
evaluation_interval: 50000 # evaluate model every 10 training iterations
framework: "torch"
num_workers: 0 # thread number
num_gpus: 1 # gpu to use beside the sampling one, the true gpu use here is (num_gpus+1)
num_cpus_per_worker: 5 # cpu allocate to each worker
num_gpus_per_worker: 0.25 # gpu allocate to each worker
checkpoint_freq: 100 # save model every 100 training iterations
checkpoint_end: True # save model at the end of the exp
restore_path: {"model_path": "", "params_path": ""} # load model and params path: 1. resume exp 2. rendering policy
stop_iters: 9999999 # stop training at this iteration
stop_timesteps: 2000000 # stop training at this timesteps
stop_reward: 999999 # stop training at this reward
seed: 5 # ray seed
local_dir: "/remote-home/ums_zhushaohao/new/2024/MAexp/exp_results"
```

The results save in `local_dir`.

### 7. Evaluation

- Set `batch_episode` to 1002 in `vda2c.yaml`.
- Adjust `is_train` (False), `training_map_num` (10), and `map_list` for test maps in `./yaml/maze.yaml`.
- Update paths (`params_path` and `model_path`) in **Line 922** of `env_v8.py`, for example

```
restore_path={'params_path': "/path/to/MAexp/map/checkpoints/checkpoint_maze/params.json",  # experiment configuration
'model_path': "/path/to/MAexp/map/checkpoints/checkpoint_maze/checkpoint-9900"},
```

- Run evaluation:

```
python env_v8.py --yaml_file /path/to/MAexp/yaml/maze.yaml \
  --result_file /path/to/MAexp/test_result.json \
  --testset_path /path/to/MAexp/testset/Final_testdata_mazes_test.pt
```



## Citation

If you find this package useful for your research, please consider citing the following papers:

- MAexp: A Generic Platform for RL-based Multi-Agent Exploration (ICRA 2024)

```
@inproceedings{zhu2024maexp,
  title={MAexp: A Generic Platform for RL-based Multi-Agent Exploration},
  author={Zhu, Shaohao and Zhou, Jiacheng and Chen, Anjun and Bai, Mingming and Chen, Jiming and Xu, Jinming},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={5155--5161},
  year={2024},
  organization={IEEE}
}
```

- TaskExp: Enhancing Generalization of Multi-Robot Exploration with Multi-Task Pre-Training (ICRA 2025)

### Author

Shaohao Zhu ([zhushh9@zju.edu.cn](mailto:zhushh9@zju.edu.cn))
