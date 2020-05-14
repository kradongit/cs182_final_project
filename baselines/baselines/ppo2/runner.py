import numpy as np
import time
from baselines.common.runners import AbstractEnvRunner
import random


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch
    """
    def __init__(self, *, env, model, nsteps, gamma, lam, augment=None):
        super().__init__(env=env, model=model, nsteps=nsteps, augment=augment)
        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam
        # Discount rate
        self.gamma = gamma

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_value_augmented = []
        mb_states = self.states
        epinfos = []
        printed = False
        # For n in range number of steps
        def augment_obs(obs):
            # pad observation 4 pixels in any direction, extending the border
            obs = self.obs[0]
            h, w, _ = obs.shape
            obs = np.pad(obs, pad_width=((4, 4), (4, 4), (0, 0)), mode='edge')
            x = random.randint(0, obs.shape[1] - w)
            y = random.randint(0, obs.shape[0] - h)
            obs = np.expand_dims(obs[y:y + h, x:x + w, :], 0)
            # Debugging Pixel Shift
            # from PIL import Image
            # obs_orig, obs_shift = Image.fromarray(self.obs[0]), Image.fromarray(obs[0])
            # obs_orig.save("obs_orig.jpeg", "jpeg")
            # obs_shift.save("obs_shift.jpeg", "jpeg")
            return obs

        for _ in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            # @kaahan: modify obs s.t. its a crop  of the padded image

            # Image Augmentation is All You Need (IAIAYN)
            # self.obs is a single sample  => step runs on a single sample
            # This is the target augmentaiton part of the paper
            # in IAIAYN we take an average over Q values generated, with PPO average over the V values
            # for K augmentations!
            now = time.time()
            if self.augment and self.augment[0] and self.augment[0] != -1:
                # self.augment (k) different observations (k different transformations)
                k = self.augment[0]
                val_aug = np.array([self.model.step(augment_obs(self.obs))[1] for _ in range(k)]).squeeze()
                target_aug = np.array([np.mean(val_aug)])
            # returns are the *target* values
            # In Image Augmentation is All You Need, they sample M image augmentations to do V augmentation
            # i.e. we need to sample separate values now that mb_values has been used (since the loss
            # in model.py uses mb_values for the V predictions)
            if self.augment and self.augment[1] and self.augment[1] != -1:
                # self.augment (m) different observations (m different transformations)
                m = self.augment[1]
                val_aug = np.array([self.model.step(augment_obs(self.obs))[1] for _ in range(m)]).squeeze()
                value_aug = np.array([np.mean(val_aug)])
            if self.augment and self.augment[0] == -1:
                self.obs = augment_obs(self.obs)
            actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
            if not printed:
                print(time.time() - now)
                printed = True
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            if self.augment and self.augment[0]:
                mb_values.append(target_aug)
            else:
                mb_values.append(values)
            if self.augment and self.augment[1]:
                mb_value_augmented.append(value_aug)
            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, S=self.states, M=self.dones)

        mb_value_augmented = np.asarray(mb_value_augmented, dtype=np.float32)

        # discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        if self.augment and self.augment[1]:
            mb_values = mb_value_augmented
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


