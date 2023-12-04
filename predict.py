import os
import time
from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    ConsistencyDecoderVAE,
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

from weights_downloader import WeightsDownloader

MODEL_CACHE = "diffusers-cache"

SD_MODEL_CACHE = os.path.join(MODEL_CACHE, "models--runwayml--stable-diffusion-v1-5")
MODEL_ID = "runwayml/stable-diffusion-v1-5"
SD_URL = "https://weights.replicate.delivery/default/stable-diffusion/stable-diffusion-v1-5-fp16.tar"

DECODER_CACHE = os.path.join(MODEL_CACHE, "models--openai--consistency-decoder")
DECODER_ID = "openai/consistency-decoder"
DECODER_URL = "https://weights.replicate.delivery/default/stable-diffusion/openai-consistency-decoder.tar"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading ConsistencyDecoder...")
        WeightsDownloader.download_if_not_exists(DECODER_URL, DECODER_CACHE)
        # for some reason, we actually need to point to the snapshot, not the base cache dir
        self.vae = ConsistencyDecoderVAE.from_pretrained(
            os.path.join(
                DECODER_CACHE, "snapshots/63b7a48896d92b6f56772f4111d0860b1bee3dd3"
            ),
            local_files_only=True,
            cache_dir=MODEL_CACHE,
        )

        print("Loading pipeline...")
        WeightsDownloader.download_if_not_exists(SD_URL, SD_MODEL_CACHE)
        self.pipe = DiffusionPipeline.from_pretrained(
            SD_MODEL_CACHE, vae=self.vae, local_files_only=True, cache_dir=MODEL_CACHE
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator("cuda").manual_seed(seed)
        start = time.time()
        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )
        print("Inference took", time.time() - start, "seconds")

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
