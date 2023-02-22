import jax
import jax.numpy as jnp

# Load models & tokenizer
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel

from flax.jax_utils import replicate

from functools import partial
import random

from dalle_mini import DalleBartProcessor

from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange

# Model references

# dalle-mega
DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # can be wandb artifact or 🤗 Hub or local folder or google bucket
DALLE_COMMIT_ID = None

# if the notebook crashes too often you can use dalle-mini instead by uncommenting below line
# DALLE_MODEL = "dalle-mini/dalle-mini/mini-1:v0"

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"



class ImageGeneration():
    def __init__(self):
        pass

    # model inference
    @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
    def _p_generate(self, model, tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale):
        return model.generate(
            **tokenized_prompt,
            prng_key=key,
            params=params,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            condition_scale=condition_scale,
        )

    # decode image
    @partial(jax.pmap, axis_name="batch")
    def _p_decode(self, vqgan, indices, params):
        return vqgan.decode_code(indices, params=params)

    def setup(self):
        # check how many devices are available
        devices_num = jax.local_device_count()
        print(f'Device num: {devices_num}')

        # Load dalle-mini
        self._model, params = DalleBart.from_pretrained(
            DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
        )

        # Load VQGAN
        self._vqgan, vqgan_params = VQModel.from_pretrained(
            VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
        )

        self._params = replicate(params)
        self._vqgan_params = replicate(vqgan_params)

        # create a random key
        seed = random.randint(0, 2**32 - 1)
        self._key = jax.random.PRNGKey(seed)

    def text_prompt(self, prompts: list):
        self._prompts = prompts
        processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)
        tokenized_prompts = processor(prompts)
        self._tokenized_prompt = replicate(tokenized_prompts)

    def generate_image(self):
        # number of predictions per prompt
        n_predictions = 3

        # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
        gen_top_k = None
        gen_top_p = None
        temperature = None
        cond_scale = 10.0

        print(f"Prompts: {self._prompts}\n")
        # generate images
        images = []
        for i in trange(max(n_predictions // jax.device_count(), 1)):
            # get a new key
            key, subkey = jax.random.split(self._key)
            # generate images
            encoded_images = self._p_generate(
                self._model,
                self._tokenized_prompt,
                shard_prng_key(subkey),
                self._params,
                gen_top_k,
                gen_top_p,
                temperature,
                cond_scale,
            )
            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]
            # decode images
            decoded_images = self._p_decode(self._vqgan, encoded_images, self._vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            for decoded_img in decoded_images:
                img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                images.append(img)
                display(img)
                print()


if __name__ == '__main__':
    img_gen = ImageGeneration()
    img_gen.setup()
    prompts = ['fox on the moon cartoon']
    img_gen.text_prompt()