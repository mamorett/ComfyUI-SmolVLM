# SmolVLM Tagger for ComfyUI

SmolVLM is a compact open multimodal model that accepts arbitrary sequences of image and text inputs to produce text outputs. Designed for efficiency, SmolVLM can answer questions about images, describe visual content, create stories grounded on multiple images, or function as a pure language model without visual inputs. Its lightweight architecture makes it suitable for on-device applications while maintaining strong performance on multimodal tasks.

## Model Summary

- **Developed by:** Hugging Face 🤗
- **Model type:** Multi-modal model (image+text)
- **Language(s) (NLP):** English
- **License:** Apache 2.0
- **Architecture:** Based on [Idefics3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3) (see technical summary)

## Resources

- **Demo:** [SmolVLM Demo](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM)
- **Blog:** [Blog post](https://huggingface.co/blog/smolvlm)

## Uses

SmolVLM can be used for inference on multimodal (image + text) tasks where the input comprises text queries along with one or more images. Text and images can be interleaved arbitrarily, enabling tasks like image captioning, visual question answering, and storytelling based on visual content. The model does not support image generation.

To fine-tune SmolVLM on a specific task, you can follow the fine-tuning tutorial.
<!-- todo: add link to fine-tuning tutorial -->

### Technical Summary

SmolVLM leverages the lightweight SmolLM2 language model to provide a compact yet powerful multimodal experience. It introduces several changes compared to previous Idefics models:

- **Image compression:** We introduce a more radical image compression compared to Idefics3 to enable the model to infer faster and use less RAM.
- **Visual Token Encoding:** SmolVLM uses 81 visual tokens to encode image patches of size 384×384. Larger images are divided into patches, each encoded separately, enhancing efficiency without compromising performance.

## Installation:

Clone this repository to 'ComfyUI/custom_nodes` folder.

Install the dependencies in requirements.txt, transformers version 4.38.0 minimum is required:

`pip install -r requirements.txt`

## Workflows

Use as single image captioning
![smolvlm_tagger_single_node_workflow.png](examples/smolvlm_tagger_single_node_workflow.png)
Combine simple caption with tag caption and save to output files
![image](examples/smolvlm_tagger_combined_workflow.png)

(Save image and grag to ComfyUI to try)

## Huggingface model
Model should be automatically downloaded the first time when you use the node. In any case that didn't happen, you can manually download it.
[HuggingFaceTB/SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
The downloaded model will be placed under`ComfyUI/LLM` folder
