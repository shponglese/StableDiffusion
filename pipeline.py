import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT//8

def generate(prompt: str, uncond_prompt: str, input_image=None, 
            strength=0.8, do_cfg=True, cfg_scale=7.5, sampler_name="ddpm", 
            n_inference_steps=50, models = {}, seed=None, device=None, idle_device=None, tokenizer=None):
    with torch.no_grad():
        if not (0<strength<=1):
            raise ValueError("strength must be between 0 and 1")
        
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed)
        
        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            #Convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids

            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids

            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

            uncond_context = clip(uncond_tokens)

            context = torch.cat([cond_context, uncond_context])

        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length",max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)
        
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown Sampler {sampler_name}")
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resieze((WIDTH, HEIGHT))

            input_image_tensor = np.array(input_image_tensor)
            #Heighty Width, Channel
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            
            input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1))
            
            input_image_tensor = input_image_tensor.unsqueeze(0)
            
            input_image_tensor = input_image_tensor.permute(0,3,1,2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            #run the image through the encoder of the VAE

            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)

            latents = sampler.add_noise(latents,sampler.timesteps[0])

            to_idle(encoder)
        
        else:
            #If we are doing text to image, start with random noise N(0,I)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            
            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents

            if do_cfg:
                #Batchsize, 4, latents_height, latents_width -> 2 * batchsize, 4, latents_height, latents_width
                model_input = model_input.repeat(2,1,1,1)
            
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
            
            #Remvoe noise predicted by the UNET
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)
        
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(iamges, (-1,1), (0,255), clamp=True)

        images = images.permute(0,2,3,1)

        images = images.to("cpu", torch.uint8).numpy()
        
        return images[0]


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range

    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    
    if clamp:
        x = x.clamp(new_min, new_max)
    
    return x

def get_time_embedding(timestep):
    #(160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32)/160)
    #(1,160)
    x = torch.tensor([timestep], dtype=torch.float32)[:,None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)







        

