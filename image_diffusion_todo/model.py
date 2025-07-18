from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
    

class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler
       

    def get_loss(self, x0, class_label=None, noise=None):
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute noise matching loss.
        B = x0.shape[0]
        timestep = self.var_scheduler.uniform_sample_t(B, device=self.device)
        if noise is None:
            noise = torch.randn_like(x0)
        x_t, noise = self.var_scheduler.add_noise(x0, timestep, noise)
        if self.network.use_cfg and class_label is not None:
            if self.network.training:
                assert not torch.any(class_label == 0)
                noise_pred = self.network(x_t, timestep=timestep, class_label=class_label)
            else:
                noise_pred = self.network(x_t, timestep=timestep, class_label=class_label) # normally not run here, except need loss for inference
        else:
            noise_pred = self.network(x_t, timestep=timestep)
        # The loss is the mean squared error between the predicted noise and the true noise.     
        loss = F.mse_loss(noise_pred, noise)
        ######################
        return loss
    
    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 1.0,
    ):
        x_T = torch.randn([batch_size, 3, self.image_resolution, self.image_resolution]).to(self.device)

        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:

            ######## TODO ########
            # Assignment 2. Implement the classifier-free guidance.
            # Specifically, given a tensor of shape (batch_size,) containing class labels,
            # create a tensor of shape (2*batch_size,) where the first half is filled with zeros (i.e., null condition).
            assert class_label is not None
            assert len(class_label) == batch_size, f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"
            class_label = torch.cat([torch.zeros(batch_size, device=self.device, dtype=torch.long), class_label]) # null condition, class condition
            #######################

        traj = [x_T]
        for t in tqdm(self.var_scheduler.timesteps):
            x_t = traj[-1]
            if do_classifier_free_guidance:
                ######## TODO ########
                # Assignment 2. Implement the classifier-free guidance.
                # pass the class_label to the network
                nois_pred_null = self.network(x_t, timestep=t.to(self.device), class_label=class_label[:batch_size]) # null condition
                nois_pred_class = self.network(x_t, timestep=t.to(self.device), class_label=class_label[batch_size:]) # class condition
                noise_pred = (1.0 + guidance_scale) * nois_pred_class - guidance_scale * nois_pred_null

                #######################
            else:
                noise_pred = self.network(x_t, timestep=t.to(self.device))
            
            x_t_prev = self.var_scheduler.step(x_t, t.to(self.device), noise_pred)

            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
            } 
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu", weights_only=False)
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
