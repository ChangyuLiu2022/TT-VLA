import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import imageio
import os
from evo_vlac import GAC_model
from evo_vlac.utils.video_tool import compress_video
import os


class SeparatedReplayBuffer(object):
    def __init__(self, all_args, obs_dim, act_dim):
        self.ep_len = all_args.episode_len
        self.num_env = all_args.num_envs
        self.gamma = all_args.buffer_gamma
        self.gae_lambda = all_args.buffer_lambda
        self.buffer_minibatch = all_args.buffer_minibatch
        self.alg_grpo_fix = all_args.alg_grpo_fix

        self.obs = np.zeros((self.ep_len + 1, self.num_env, *obs_dim), dtype=np.uint8)
        self.instruction = [""] * self.num_env
        self.value_preds = np.zeros((self.ep_len + 1, self.num_env, 1), dtype=np.float32)
        self.returns = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)
        self.actions = np.zeros((self.ep_len, self.num_env, act_dim), dtype=np.int32)
        self.action_log_probs = np.zeros((self.ep_len, self.num_env, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)
        self.masks = np.ones((self.ep_len + 1, self.num_env, 1), dtype=np.float32)

        self.advantages = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)

        self.step = 0
        self.act_dim = act_dim

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1] = obs.copy()
        #for some models with different output tokens. 
        action_token_len = actions.shape[-1]
        if action_token_len > self.act_dim:
            action_token_len = self.act_dim
        self.actions[self.step, :, :action_token_len] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()

        #self.step = (self.step + 1) % self.ep_len
        self.step = (self.step + 1)

    def warmup(self, obs, instruction):
        self.obs[0] = obs
        self.instruction = instruction
        self.masks[0] = 1.0

        self.step = 0

    def endup(self, next_value):
        self.value_preds[-1] = next_value

    def compute_returns_ppo(self):
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            vt1 = self.value_preds[step + 1]
            vt = self.value_preds[step]

            delta = self.rewards[step] + self.gamma * vt1 * self.masks[step + 1] - vt
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + vt

        # calc adv
        advantages = self.returns - self.value_preds[:-1]
        mean_advantages = advantages.mean()
        std_advantages = advantages.std()
        self.advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)


    def compute_returns_grpo(self):
        if self.alg_grpo_fix:
            rewards_valid = self.rewards[self.rewards != 0]
            rewards_norm = self.rewards.copy()
            rewards_norm[rewards_norm != 0] -= rewards_valid.mean()
            rewards_norm[rewards_norm != 0] /= (rewards_valid.std() + 1e-5)
        else:
            rewards_norm = (self.rewards - self.rewards.mean()) / (self.rewards.std() + 1e-5)

        returns = 0
        for step in reversed(range(self.rewards.shape[0])):
            returns = rewards_norm[step] + self.masks[step + 1] * returns
            self.returns[step] = returns

        # calc adv
        self.advantages = self.returns.copy()

    def get_minibatch_count(self):
        episode_length, n_rollout_threads = self.rewards.shape[:2]
        batch_size = episode_length * n_rollout_threads

        if self.buffer_minibatch < 0:
            num_mini_batch = 1
        else:
            assert batch_size % self.buffer_minibatch == 0
            num_mini_batch = batch_size // self.buffer_minibatch

        return num_mini_batch

    def feed_forward_generator(self):
        episode_length, n_rollout_threads = self.rewards.shape[:2]
        batch_size = episode_length * n_rollout_threads

        if self.buffer_minibatch < 0:
            num_mini_batch = 1
        else:
            assert batch_size % self.buffer_minibatch == 0
            num_mini_batch = batch_size // self.buffer_minibatch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * self.buffer_minibatch:(i + 1) * self.buffer_minibatch] for i in range(num_mini_batch)]

        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns.reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        action_logits = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = self.advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            old_action_logits_batch = action_logits[indices]
            adv_targ = advantages[indices]

            # instruct
            instruct_indices = indices % n_rollout_threads
            instruct_batch = [self.instruction[i] for i in instruct_indices]

            yield (obs_batch, instruct_batch, actions_batch, value_preds_batch, return_batch, masks_batch,
                   old_action_logits_batch, adv_targ)



class SeparatedReplayBuffer_vlm(SeparatedReplayBuffer):
    def __init__(self, all_args, obs_dim, act_dim):
        super().__init__(all_args, obs_dim, act_dim)
        self.tt_steps = all_args.tt_steps
        self.model_path = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B-1M_res224"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).to(torch.bfloat16).to(torch.device("cuda:1"))
        self.image_processor = self.model.get_vision_tower().image_processor
        mm_llm_compress = False # use the global compress or not
        if mm_llm_compress:
            self.model.config.mm_llm_compress = True
            self.model.config.llm_compress_type = "uniform0_attention"
            self.model.config.llm_compress_layer_list = [4, 18]
            self.model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
        else:
            self.model.config.mm_llm_compress = False
        
        self.max_num_frames = 512
        self.generation_config = dict(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            top_p=0.1,
            num_beams=1
        )

    def compute_returns_ppo(self):
        #TODO1: compute the value from vlm according to the obs
        #for step in range(self.rewards.shape[0]):
        for step in range(self.step-1-self.tt_steps, self.step-1):
            #self.value_preds[step] = self.get_value_from_vlm(self.obs[:step+2], self.instruction)  #objs should be step+1， so slice is :step+2
            self.value_preds[step] = step
        #compute the reward by values
        for step in range(self.step-1-self.tt_steps, self.step-1):
            self.rewards[step] = self.value_preds[step+1] - self.value_preds[step]
        #compute the returns and advantages
        gae = 0
        for step in reversed(range(self.step-1-self.tt_steps, self.step-2)):
            vt1 = self.value_preds[step + 1]
            vt = self.value_preds[step]

            delta = self.rewards[step] + self.gamma * vt1 * self.masks[step + 1] - vt
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + vt
        # calc adv
        advantages = self.returns - self.value_preds[:-1]       #only compute the advatange for the updated steps
        advantages = advantages[self.step-1-self.tt_steps:self.step-2]  #only include advantage for the fisrt self.ttsteps-1 in this round
        mean_advantages = advantages.mean()
        std_advantages = advantages.std()
        self.advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

    def get_value_from_vlm(self, obs_seq, instruction):
        #save obs_seq as a mp4 video, obs_seq（narray） with shape [T, 1, H, W, 3]
        obs_seq = obs_seq[:,0]  # [T, H, W, 3]
        video_path = "temp_video.mp4"
        imageio.mimwrite(video_path, obs_seq, fps=8)
        question1 = f"What is the progress of {instruction} in the video? Approaching to the  Please answer in a word from 0% to 100%."
        #question1 = f"How many percent of {text_candidates[i]} is completed? Please answer in one word, such as "
        output1, chat_history = self.model.chat(video_path=video_path, tokenizer=self.tokenizer, user_prompt=question1, return_history=True, max_num_frames=self.max_num_frames, generation_config=self.generation_config)
        #exact the number from output1 with regex
        import re
        match = re.search(r'(\d{1,3})%', output1)
        if match:
            value = float(match.group(1))
        if value > 100:
            value = 100.0
        elif value < 0:
            value = 0.0
        return value


class SeparatedReplayBuffer_vlac(SeparatedReplayBuffer):
    def __init__(self, all_args, obs_dim, act_dim):
        super().__init__(all_args, obs_dim, act_dim)
        self.progress = np.zeros((self.ep_len + 1, self.num_env, 1), dtype=np.float32)
        self.advantages = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)
        self.tt_steps = all_args.tt_steps
        self.normalize_advantage = all_args.normalize_advantage
        self.model_path = all_args.reward_model_path
        self.ref_video = None
        #init model
        self.Critic=GAC_model(tag='critic')
        if all_args.ttt==1:
            self.Critic.init_model(model_path=self.model_path,model_type='internvl2',device_map=f'cuda:0')
        else:
            self.Critic.init_model(model_path=self.model_path,model_type='internvl2',device_map=f'cpu')
        self.Critic.temperature=0.5
        self.Critic.top_k=1
        self.Critic.set_config()
        self.Critic.set_system_prompt()

        self.args = all_args

    def compute_returns_ppo(self):
        #TODO1: compute the value from vlm according to the obs
        #get the progress from vlac according to the obs and instruction
        progress = self.get_progress_from_vlac(self.obs[:self.step+1], self.instruction)  #objs should be step+1， so slice is :step+2
        for i in range(self.step+1):
            self.progress[i] = progress[i]

        #let the value be 0， so we don't need to compute the value
        #self.value_preds[:self.step] = 1 - self.progress[:self.step]
        #compute the reward by progress
        for step in range(self.step-self.tt_steps, self.step):
            self.rewards[step] = self.progress[step+1] - self.progress[step] #normalize to 0-1
        #compute the returns and advantages
        gae = 0
        for step in reversed(range(self.step-self.tt_steps, self.step)):
            vt1 = self.value_preds[step + 1]    #actually vt1 and vt are 0, and not used
            vt = self.value_preds[step]
            vt1 = 0
            vt = 0

            delta = self.rewards[step] + self.gamma * vt1 * self.masks[step + 1] - vt
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae      #if gamma==0,  gae=reward
            self.returns[step] = gae + vt
        # calc adv
        advantages = self.returns# - self.value_preds[:-1]       #这里是不是导致advantage：前大后小的地方？
        advantages = advantages[self.step-self.tt_steps:self.step]  #only include advantage for the fisrt self.ttsteps-1 in this round
        if self.normalize_advantage:
            mean_advantages = advantages.mean()
            std_advantages = advantages.std()
            self.advantages[self.step-self.tt_steps:self.step] = (advantages - mean_advantages) / (std_advantages + 1e-5)
        else:
            self.advantages[self.step-self.tt_steps:self.step] = advantages

    def get_progress_from_vlac(self, obs_seq, instruction):
        #save obs_seq as a mp4 video, obs_seq（narray） with shape [T, 1, H, W, 3]
        obs_seq = obs_seq[:,0]  # [T, H, W, 3]
        if self.args.ttt:
            #test_video_path = "temp_video_ttt.mp4"
            test_video_path =  os.path.join(self.args.glob_dir,"temp_video_ttt.mp4")
        else:
            #test_video_path = "temp_video_no_ttt.mp4"
            test_video_path =  os.path.join(self.args.glob_dir,"temp_video_no_ttt.mp4")
        imageio.mimwrite(test_video_path, obs_seq, fps=10)        #here set 8, which will cause problems. So setting 10.
        test_video_compressed = os.path.join(os.path.dirname(test_video_path),"test.mp4")
        _,output_fps=compress_video(test_video_path, test_video_compressed,fps=100)
        reference_video_compressed = None
        result_path,value_list,critic_list,done_list = self.Critic.web_trajectory_critic(
            task_description=instruction,
            main_video_path=test_video_compressed,
            reference_video_path=reference_video_compressed,#if None means no reference video, only use task_description to indicate the task
            batch_num=5,#batch number
            think=False,# whether to CoT
            skip=1,#pair-wise step
            rich=False,#whether to output decimal value
            reverse_eval=False,#whether to reverse the evaluation(for VROC evaluation)
            output_path="results",
            fps=float(output_fps),
            frame_skip=False,#True,#whether to skip frames(if false, each frame while be evaluated, cost more time)
            done_flag=False,#whether to out put done value
            in_context_done=False,#whether use reference video to generate done value
            done_threshold=0.9,#done threshold
            video_output=False#whether to output video
        )

        return value_list  #return the last value as the progress
    
    def feed_forward_generator(self):     #only generate the data from range(self.step-self.tt_steps, self.step)
        episode_length, n_rollout_threads = self.rewards.shape[:2]
        batch_size = episode_length * n_rollout_threads

        if self.buffer_minibatch < 0:
            num_mini_batch = 1
        else:
            assert batch_size % self.buffer_minibatch == 0
            num_mini_batch = batch_size // self.buffer_minibatch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * self.buffer_minibatch:(i + 1) * self.buffer_minibatch] for i in range(num_mini_batch)]

        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns.reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        action_logits = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = self.advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            old_action_logits_batch = action_logits[indices]
            adv_targ = advantages[indices]

            # instruct
            instruct_indices = indices % n_rollout_threads
            instruct_batch = [self.instruction[i] for i in instruct_indices]

            yield (obs_batch, instruct_batch, actions_batch, value_preds_batch, return_batch, masks_batch,
                   old_action_logits_batch, adv_targ)


class SeparatedReplayBuffer_perplexity(SeparatedReplayBuffer):
    def __init__(self, all_args, obs_dim, act_dim):
        super().__init__(all_args, obs_dim, act_dim)
        self.tt_steps = all_args.tt_steps
        self.args = all_args

    