import torch
import torch.nn as nn
from diffusers import DDIMScheduler, DDPMScheduler
from .denoiser import MldDenoiser

class LDM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.text_encoder = None
        self.denoiser = MldDenoiser(skip_connection=args.skip_connection)
        self.num_train_timesteps = args.num_train_timesteps
        self.loss_type = 'l1'

        self.schecular = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                                       beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=False, steps_offset=1)
        '''
        params:
        num_train_timesteps: 1000
        beta_start: 0.00085
        beta_end: 0.012
        beta_schedule: 'scaled_linear' # Optional: ['linear', 'scaled_linear', 'squaredcos_cap_v2']
        # variance_type: 'fixed_small'
        clip_sample: false # clip sample to -1~1
        # below are for ddim
        set_alpha_to_one: false
        steps_offset: 1
        '''

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                                             beta_schedule='scaled_linear', variance_type='fixed_small', clip_sample=False)
        '''
        num_train_timesteps: 1000
        beta_start: 0.00085
        beta_end: 0.012
        beta_schedule: 'scaled_linear'  # Optional: ['linear', 'scaled_linear', 'squaredcos_cap_v2']
        variance_type: 'fixed_small'
        clip_sample: false  # clip sample to -1~1
        '''

    #def forward(self, latents, encoder_hidden_states, lengths):
    def forward(self, latents):
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each motion
        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = self.noise_scheduler.add_noise(latents.clone(), noise,
                                                       timesteps)
        # Predict the noise residual
        noise_pred = self.denoiser(
            latent=noisy_latents,
            timestep=timesteps,
            #encoder_hidden_states=encoder_hidden_states,
            #lengths=lengths,
            #return_dict=False,
        )[0]

        if self.loss_type == "l1":
            loss = nn.functional.l1_loss(noise_pred.permute(1,0,2), latents)
        elif self.loss_type == "mse":
            loss = nn.functional.mse_loss(noise_pred, latents)

        return loss

    def diffusion_reversion(self, batch_size, device, encoder_hidden_states=None, lengths=None):
        # init latents

        #bsz = encoder_hidden_states.shape[0]
        #if self.do_classifier_free_guidance:
        #    bsz = bsz // 2
        bsz = batch_size
        latents = torch.randn(
            (bsz, self.latent_dim[0], self.latent_dim[-1]),
            device=encoder_hidden_states.device,
            dtype=torch.float,
        )

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        # set timesteps
        self.scheduler.set_timesteps(
            self.args.model.scheduler.num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}

        extra_step_kwargs["eta"] = 0.0

        # reverse
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat(
                [latents] *
                2) if self.do_classifier_free_guidance else latents)
            #lengths_reverse = (lengths * 2 if self.do_classifier_free_guidance
            #                  else lengths)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual

            noise_pred = self.denoiser(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=encoder_hidden_states,
                #lengths=lengths_reverse,
            )[0]

            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond)
                # text_embeddings_for_guidance = encoder_hidden_states.chunk(
                #     2)[1] if self.do_classifier_free_guidance else encoder_hidden_states
            latents = self.scheduler.step(noise_pred, t, latents,
                                          **extra_step_kwargs).prev_sample
            # if self.predict_epsilon:
            #     latents = self.scheduler.step(noise_pred, t, latents,
            #                                   **extra_step_kwargs).prev_sample
            # else:
            #     # predict x for standard diffusion model
            #     # compute the previous noisy sample x_t -> x_t-1
            #     latents = self.scheduler.step(noise_pred,
            #                                   t,
            #                                   latents,
            #                                   **extra_step_kwargs).prev_sample

        # [batch_size, 1, latent_dim] -> [1, batch_size, latent_dim]
        latents = latents.permute(1, 0, 2)
        return latents