import io
import os
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor



app = FastAPI(title="OpenVLA xArm7 Inference Server")


import os
import pprint
import random
import gc
import signal
from collections import defaultdict
import time
from pathlib import Path
from typing import Annotated
import torch
import numpy as np
import tyro
import wandb
from dataclasses import dataclass
import yaml
from tqdm import tqdm
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video

from simpler_env.env.simpler_wrapper import SimlerWrapper, NoraWrapper
from simpler_env.utils.replay_buffer import SeparatedReplayBuffer_vlac, SeparatedReplayBuffer, SeparatedReplayBuffer_vlm 

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import imageio

from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy, OpenVLAPPO
from simpler_env.policies.nora.nora_train import NoraPolicy
from PIL import Image




@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PutCarrotOnPlateInScene-v1"
    """The environment ID of the task you want to simulate. Can be one of
    PutCarrotOnPlateInScene-v1, PutSpoonOnTableClothInScene-v1, StackGreenCubeOnYellowCubeBakedTexInScene-v1, PutEggplantInBasketScene-v1"""

    """Number of environments to run. With more than 1 environment the environment will use the GPU backend 
    which runs faster enabling faster large-scale evaluations. Note that the overall behavior of the simulation
    will be slightly different between CPU and GPU backends."""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the model and environment. Default seed is 0"""

    name: str = "PPO-test"

    # env
    num_envs: int = 1
    episode_len: int = 80
    use_same_init: bool = False

    steps_max: int = 2000000
    steps_vh: int = 0  # episodes
    interval_eval: int = 10
    interval_save: int = 40

    # buffer
    buffer_inferbatch: int = 32
    buffer_minibatch: int = 8
    buffer_gamma: float = 0.99
    buffer_lambda: float = 0.95

    # vla
    vla_path: str = "openvla/openvla-7b"
    vla_unnorm_key: str = "bridge_orig"
    vla_load_path: str = ""
    vla_lora_rank: int = 32

    vla_lr: float = 1e-4
    vla_vhlr: float = 3e-3
    vla_optim_beta1: float = 0.9
    vla_optim_beta2: float = 0.999
    vla_temperature: float = 1.0
    vla_temperature_eval: float = 0.6

    vla_type: str = "openvla"  # openvla, nora

    # ppo & grpo
    alg_name: str = "ppo"  # ppo, grpo
    alg_grpo_fix: bool = True
    alg_gradient_accum: int = 1
    alg_ppo_epoch: int = 1
    alg_entropy_coef: float = 0.0
    alg_value_coef: float = 0.0

    # other
    wandb: bool = True
    only_render: bool = False
    render_info: bool = False

    #ttt
    tt_steps: int = 8  # do ttt every tt_steps
    ttt: int = 1  # whether use test time training.       #ttt=0 means no ttt, ttt=1 means use ttt
    max_episodes: int = 10  # episodes to run
    normalize_advantage: bool = True  # whether normalize advantage when updating policy
    from_epoch: int = 0  # which epoch to start ttt
    reward_model_path: str = ""  # the path to the reward model for ttt

    obj_set: str = "test"  # which object set to run

    #tracevla
    cotracker_model_path: str = ""  # the path to the cotracker model for tracevla



class Runner:
    def __init__(self, all_args: Args):
        self.args = all_args

        # alg_name
        assert self.args.alg_name in ["ppo", "grpo"]

        # set seed
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        # set wandb
        wandb.init(
            config=all_args.__dict__,
            project="RLVLA",
            name=self.args.name,
            mode="online" if self.args.wandb else "offline",
        )
        self.save_dir = Path(wandb.run.dir)
        self.glob_dir = Path(wandb.run.dir) / ".." / "glob"
        self.glob_dir.mkdir(parents=True, exist_ok=True)

        yaml.dump(all_args.__dict__, open(self.glob_dir / "config.yaml", "w"))

        self.args.glob_dir =  str(self.glob_dir)

        # policy
        from simpler_env.policies.openvla.openvla_train import OpenVLAPolicy, OpenVLAPPO
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()          #to solve a transformer bug
        self.device_id = 1 if torch.cuda.device_count() > 1 else 0    # make sure data and policy on the same device
        self.device_id_other = 1 if torch.cuda.device_count() > 1 else 0
        self.device = torch.device("cuda:" + str(self.device_id))
        if self.args.vla_type == "openvla":
            self.policy = OpenVLAPolicy(all_args, self.device_id_other)
            self.alg = OpenVLAPPO(all_args, self.policy)
            self.alg.policy = self.policy
            unnorm_state = self.policy.vla.get_action_stats(self.args.vla_unnorm_key)
            self.env = SimlerWrapper(self.args, unnorm_state)
            
            
        elif self.args.vla_type == "nora":
            self.policy = NoraPolicy(all_args, self.device_id_other)
            self.alg = OpenVLAPPO(all_args, self.policy)
            self.alg.policy = self.policy
            unnorm_state = self.policy.vla.get_action_stats(self.args.vla_unnorm_key)
            self.env = NoraWrapper(self.args, unnorm_state, self.policy)

        elif self.args.vla_type == "tracevla":
            from simpler_env.policies.tracevla.tracevla_train import TraceVLAPolicy, TraceVLAPPO
            self.policy = TraceVLAPolicy(all_args, self.device_id_other)
            self.alg = TraceVLAPPO(all_args, self.policy)
            self.alg.policy = self.policy
            #self.alg = OpenVLAPPO(all_args, self.policy)
            unnorm_state = self.policy.vla.get_action_stats(self.args.vla_unnorm_key)
            self.env = SimlerWrapper(self.args, unnorm_state)
            
            
        else:
            raise ValueError(f"Unknown vla_type: {self.args.vla_type}")

        self.buffer = SeparatedReplayBuffer_vlac(
                self.args,
                #obs_dim=(1024, 1024, 3),     #need to change according to image sizenab
                obs_dim=(480, 640, 3),     #need to change according to image size
                act_dim=7,     
            )
        minibatch_count = self.buffer.get_minibatch_count()
        print(f"Buffer minibatch count: {minibatch_count}")

        self.last_action = None
        self.ind = 0  # global step count
        self.last_logprob = None
        self.last_value = None
        self.last_reward = None
        self.last_done = None

    @torch.no_grad()
    def _get_action(self, obs, deterministic=False):
        total_batch = obs["image"].shape[0]

        values = []
        actions = []
        logprobs = []

        for i in range(0, total_batch, self.args.buffer_inferbatch):
            obs_batch = {k: v[i:i + self.args.buffer_inferbatch] for k, v in obs.items()}
            value, action, logprob = self.policy.get_action(obs_batch, deterministic)
            values.append(value)
            actions.append(action)
            logprobs.append(logprob)

        values = torch.cat(values, dim=0).to(device=self.device)
        actions = torch.cat(actions, dim=0).to(device=self.device)
        logprobs = torch.cat(logprobs, dim=0).to(device=self.device)

        return values, actions, logprobs

    def collect(self):
        self.policy.prep_rollout()

        obs_image = self.buffer.obs[self.buffer.step]
        obs_image = torch.tensor(obs_image).to(self.device)
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        value, action, logprob = self._get_action(obs)

        return value, action, logprob

    def insert(self, data):
        obs_img, actions, logprob, value_preds, rewards, done = data
        masks = 1.0 - done.to(torch.float32)

        obs_img = obs_img.cpu().numpy()
        actions = actions.to(torch.int32).cpu().numpy()
        logprob = logprob.to(torch.float32).cpu().numpy()
        value_preds = value_preds.to(torch.float32).cpu().numpy()
        rewards = rewards.cpu().numpy()
        masks = masks.cpu().numpy()

        self.buffer.insert(obs_img, actions, logprob, value_preds, rewards, masks)

    def compute_endup(self):
        self.policy.prep_rollout()

        obs_image = torch.tensor(self.buffer.obs[-1]).to(self.device)
        obs = dict(image=obs_image, task_description=self.buffer.instruction)
        with torch.no_grad():
            next_value, _, _ = self._get_action(obs)
        next_value = next_value.to(torch.float32).cpu().numpy()

        self.buffer.endup(next_value)

    def train(self):
        self.policy.prep_training()

        if self.args.alg_name == "ppo":
            train_info = self.alg.train_ppo_vlac(self.buffer)
        elif self.args.alg_name == "grpo":
            train_info = self.alg.train_grpo(self.buffer)
        else:
            raise ValueError(f"Unknown alg_name: {self.args.alg_name}")

        info = {f"train/{k}": v for k, v in train_info.items()}
        info["buffer/reward_mean"] = np.mean(self.buffer.rewards)
        info["buffer/mask_mean"] = np.mean(1.0 - self.buffer.masks)

        return info

    @torch.no_grad()
    def eval(self, obj_set: str) -> dict:
        self.policy.prep_rollout()
        env_infos = defaultdict(lambda: [])

        obs_img, instruction, info = self.env.reset(obj_set=obj_set)

        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)

            obs_img, reward, done, env_info = self.env.step(action)

            # info
            print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items() if k != "episode"})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats = env_stats.copy()

        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        return env_stats

    @torch.no_grad()
    def render(self, epoch: int, obj_set: str) -> dict:
        self.policy.prep_rollout()

        # init logger
        env_infos = defaultdict(lambda: [])
        datas = [{
            "image": [],  # obs_t: [0, T-1]
            "instruction": "",
            "action": [],  # a_t: [0, T-1]
            "info": [],  # info after executing a_t: [1, T]
        } for idx in range(self.args.num_envs)]

        obs_img, instruction, info = self.env.reset(obj_set)
        print("instruction[:3]:", instruction[:3])

        # data dump: instruction
        for idx in range(self.args.num_envs):
            datas[idx]["instruction"] = instruction[idx]

        for _ in range(self.args.episode_len):
            obs = dict(image=obs_img, task_description=instruction)
            value, action, logprob = self._get_action(obs, deterministic=True)

            obs_img_new, reward, done, env_info = self.env.step(action)

            # info
            print({k: round(v.to(torch.float32).mean().tolist(), 4) for k, v in env_info.items() if k != "episode"})
            if "episode" in env_info.keys():
                for k, v in env_info["episode"].items():
                    env_infos[f"{k}"] += v

            for i in range(self.args.num_envs):
                post_action = self.env._process_action(action)
                log_image = obs_img[i].cpu().numpy()
                log_action = post_action[i].cpu().numpy().tolist()
                log_info = {k: v[i].tolist() for k, v in env_info.items() if k != "episode"}
                datas[i]["image"].append(log_image)
                datas[i]["action"].append(log_action)
                datas[i]["info"].append(log_info)

            # update obs_img
            obs_img = obs_img_new

        # data dump: last image
        for i in range(self.args.num_envs):
            log_image = obs_img[i].cpu().numpy()
            datas[i]["image"].append(log_image)

        # save video
        exp_dir = Path(self.glob_dir) / f"vis_{epoch}_{obj_set}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        for i in range(self.args.num_envs):
            images = datas[i]["image"]
            infos = datas[i]["info"]
            assert len(images) == len(infos) + 1

            if self.args.render_info:
                for j in range(len(infos)):
                    images[j + 1] = visualization.put_info_on_image(
                        images[j + 1], infos[j],
                        extras=[f"Ins: {instruction[i]}"]
                    )

            success = int(infos[-1]["success"])
            images_to_video(images, str(exp_dir), f"video_{i}-s_{success}_{instruction[i]}",
                            fps=10, verbose=False)

        # infos
        env_stats = {k: np.mean(v) for k, v in env_infos.items()}
        env_stats_ret = env_stats.copy()

        print(pprint.pformat({k: round(v, 4) for k, v in env_stats.items()}))
        print(f"")

        # save stats
        last_info = {
            idx: {k: env_infos[k][idx] for k in env_infos.keys()}
            for idx in range(self.args.num_envs)
        }

        save_stats = {}
        save_stats["env_name"] = self.args.env_id
        save_stats["ep_len"] = self.args.episode_len
        save_stats["epoch"] = epoch
        save_stats["stats"] = {k: v.item() for k, v in env_stats.items()}
        save_stats["instruction"] = {idx: ins for idx, ins in enumerate(instruction)}
        save_stats["last_info"] = last_info

        yaml.dump(save_stats, open(exp_dir / "stats.yaml", "w"))

        return env_stats_ret

    def run2(self, obj_set: str = "test"):
        env_infos = defaultdict(lambda: [])
        ep_time = time.time()
        episode = 0

        #obs_img, instruction, info = self.env.reset(obj_set="train", same_init=self.args.use_same_init)
        obs_img, instruction, info = self.env.reset(obj_set=obj_set, same_init=self.args.use_same_init, seed=self.args.seed * 1000 + episode)
        print(f"initialization: {obs_img.sum()}")

        #if self.last_action is None: then warmup buffer
                   
        self.buffer.warmup(obs_img.cpu().numpy(), instruction)

        for i in tqdm(range(self.args.episode_len), desc="rollout"):        
                        value, action, logprob = self.collect()           #generate action
                        self.last_action = action
                        obs_img, reward, done, env_info = self.env.step(action)

                        data = (obs_img, action, logprob, value, reward, done)
                        self.insert(data)

                        # info
                        if "episode" in env_info.keys():
                            for k, v in env_info["episode"].items():
                                env_infos[f"{k}"] += v

                        # test time training
                        if i>0 and (i+1) % self.args.tt_steps == 0 and self.args.ttt:    # after i+1 steps, get i+2 objs
                            # test_time_training()
                            infos = self.train()
                            for k, v in env_infos.items():
                                infos[f"env/{k}"] = np.mean(v)

    
     

        #time into ymdhourminsec
        time.strftime('%Y%m%d_%H%M%S')  
        video_path = f"ttt_{self.args.ttt}_{self.buffer.instruction[0]}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        video_path = str(self.glob_dir / video_path)

        imageio.mimwrite(video_path, self.buffer.obs[:,0], fps=10)
        print(f"Saved video to {video_path}")


    def run(self, obs_img, instruction="put the bread on the plate"):
        #if obs_img is a PIL.Image, then transform to tensor
        if not torch.is_tensor(obs_img):
            image= Image.open("/home/cldb5/nora/assets/nora-logo.png")
            image = np.array(image)    #[0-255].  h * w * 3
            obs_img = torch.tensor(image).unsqueeze(0).to(self.device)

            
        instruction = [instruction]
        env_infos = defaultdict(lambda: [])
        ep_time = time.time()
    
        
        #obs_img, instruction should get from input of the function
        #if self.last_action is None: then warmup buffer
        if self.last_action is None:      
            self.buffer.warmup(obs_img.cpu().numpy(), instruction)
        else:
            data = (obs_img, self.last_action, self.last_logprob, self.last_value, self.last_reward, self.last_done)
            self.insert(data)
        
        if self.ind>0 and (self.ind) % self.args.tt_steps == 0 and self.args.ttt:    # after i+1 steps, get i+2 objs
        # test_time_training()
            infos = self.train()
            for k, v in env_infos.items():
                infos[f"env/{k}"] = np.mean(v)

     
        value, action, logprob = self.collect()           #generate action
        self.last_action = action
        self.last_logprob = logprob
        self.last_value = value
        self.last_reward = torch.tensor([[0]], device='cuda:0')
        self.last_done = torch.tensor([[False]], device='cuda:0')
        self.ind += 1


        if self.ind == self.args.episode_len:
            video_path = f"ttt_{self.args.ttt}_{self.buffer.instruction[0]}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            video_path = str(self.glob_dir / video_path)

            imageio.mimwrite(video_path, self.buffer.obs[:,0], fps=10)
            print(f"Saved video to {video_path}")

        action = self.env._process_action(action)

        return action

args = tyro.cli(Args)
runner = Runner(args)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


@app.post("/infer")
def infer(
    image: UploadFile = File(...),
    prompt: Optional[str] = Form(default=None),
    do_sample: bool = Form(default=False),
):
    """Receives an RGB image and returns a 7-DoF action (list[float])."""
    try:
        raw = image.file.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        #pil is PIL.Image.Image prompt is str
        action = runner.run(obs_img=pil, instruction=prompt)

        # action is a torch tensor; convert to JSON-serializable list
        action_np = action.detach().to("cpu").float().numpy().reshape(-1)
        action_list: List[float] = [float(x) for x in action_np.tolist()]

        return JSONResponse({"action": action_list})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    # Run:
    #   uvicorn inference:app --host 0.0.0.0 --port 8000
    # or:
    #   python inference.py
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)