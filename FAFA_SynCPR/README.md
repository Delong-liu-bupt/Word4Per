# Automatic Synthetic Data and Fine-grained Adaptive Feature Alignment for Composed Person Retrieval

## SynCPR Dataset

### Overview

The **SynCPR** dataset is a large-scale, fully synthetic dataset designed specifically for the **composed person retrieval** task. Built using our automated construction pipeline, SynCPR offers unmatched diversity, quality, and realism for person-centric image retrieval research.

### Construction Pipeline

The dataset is constructed in three main stages:

1. **Textual Quadruple Generation:**
   We utilize [Qwen2.5-70B](https://github.com/QwenLM/Qwen2.5-VL) to generate **140,500 textual quadruples**. Each quadruple contains two captions for reference and target person images, along with their relative description and associated metadata.

2. **Image Generation:**
   Using a fine-tuned [LoRA](https://arxiv.org/abs/2106.09685) with [Flux.1](https://github.com/black-forest-labs/flux), we generate person images in two styles:

   * For each quadruple, **five image pairs** are generated using the most realistic setting ($\beta=1$).
   * Another **five image pairs** per quadruple are created using a randomly sampled $\beta \in (0,1)$ for greater style diversity.

   Combined with the relative captions, this stage yields **2,810,000 valid triplets**.

3. **Rigorous Filtering:**
   Applying strict filtering criteria, we retain **1,153,220 high-quality triplets**, covering **177,530 unique GIDs**. The average caption length is **13.3 words**, and a total of **4,370 distinct words** are present, highlighting the dataset's linguistic and visual diversity.

### Key Features

* **Diversity:** Broad coverage of scenarios, age, attire, clarity, and ethnic representation.
* **Realism:** Images generated using realism-oriented fine-tuning and advanced generative models.
* **Scale:** Over one million high-quality triplets with detailed, richly varied captions.
* **Comprehensiveness:** The synthetic nature enables coverage and variability beyond existing manually annotated datasets in person retrieval.

![FigA3_00](https://github.com/user-attachments/assets/0fc2cd5c-896c-4edb-a82b-665feca5b6e5)

### Data Structure

Each data sample is defined in a [`SynCPR.json`](https://huggingface.co/datasets/a1557811266/SynCPR) file, which contains a list of dictionaries. Each dictionary represents a training instance with the following fields:

* `reference_caption`: Text description used to generate the reference image.
* `target_caption`: Text description for the target image.
* `reference_image_path`: Path to the reference image.
* `target_image_path`: Path to the target image.
* `edit_caption`: Relative description specifying the key differences between the reference and target.
* `cpr_id`: Identifier marking data groups generated from the same base textual description.

#### Example

```json
[
    {
        "reference_caption": "The young woman with black hair is wearing an ebony black blouse, a navy blue skirt, and black heeled sandals. She is holding a silver clutch.",
        "target_caption": "The young woman with black hair is wearing an ebony black blouse, a light gray skirt, and black heeled sandals. She is carrying a large black leather handbag.",
        "reference_image_path": "test2/sub_img/img_left/10732-1_left.png",
        "target_image_path": "test2/sub_img/img_right/10732-1_right.png",
        "edit_caption": "Wearing light gray skirt, carrying a large black leather handbag.",
        "cpr_id": 0
    },
    {
        "reference_caption": "The young woman with black hair is wearing an ebony black blouse, a light gray skirt, and black heeled sandals. She is carrying a large black leather handbag.",
        "target_caption": "The young woman with black hair is wearing an ebony black blouse, a navy blue skirt, and black heeled sandals. She is holding a silver clutch.",
        "reference_image_path": "test2/sub_img/img_right/10732-1_right.png",
        "target_image_path": "test2/sub_img/img_left/10732-1_left.png",
        "edit_caption": "Wearing navy blue skirt, holding a silver clutch.",
        "cpr_id": 1
    }
]
```

* `reference_caption`: Used to generate the reference image.
* `target_caption`: Used to generate the target image.
* `reference_image_path`/`target_image_path`: Paths to the respective images.
* `edit_caption`: Describes the transformation from reference to target.
* `cpr_id`: Indicates pairs generated from the same textual base.

### Download

The SynCPR dataset is publicly available for research purposes.
**[Download here on Hugging Face](https://huggingface.co/datasets/a1557811266/SynCPR)**

### Citation

If you use SynCPR in your research, please cite our paper.

```bibtex
@misc{liu2025automaticsyntheticdatafinegrained,
      title={Automatic Synthetic Data and Fine-grained Adaptive Feature Alignment for Composed Person Retrieval}, 
      author={Delong Liu and Haiwen Li and Zhaohui Hou and Zhicheng Zhao and Fei Su and Yuan Dong},
      year={2025},
      eprint={2311.16515},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.16515}, 
}
```

