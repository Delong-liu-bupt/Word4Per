# Word for Person: Zero-shot Composed Person Retrieval (Word4Per)
Word4Per is an innovative framework for Zero-Shot Composed Person Retrieval (ZS-CPR), integrating visual and textual information for enhanced person identification. This repository includes the Word4Per code and the Image-Text Composed Person Retrieval (ITCPR) dataset, offering new tools for research in security and social applications.
### News
* [2023.11.16] Repo is created. Code and Dataset will come soon.
* [2023.11.25] The TICPR dataset is now publicly available for download.



## ITCPR Dataset

### Overview

The **ITCPR** dataset is a comprehensive collection specifically designed for the Zero-Shot Clothes-Person Re-identification (ZS-CPR) task. It consists of a total of **2,225 annotated triplets**, derived from three distinct datasets: Celeb-reID, PRCC, and LAST. To access the TICPR dataset, please use the following download link: [TICPR Dataset Download](https://drive.google.com/file/d/1KPKQ3DGK3h7TvhD3R1CgkQa1LTP_sjWm/view?usp=sharing). 

#### Dataset Scale
- **Total Annotated Triplets**: 2,225
- **Unique Query Combinations**: 2,202
- **Total Images**: 1,151 from Celeb-reID, 146 from PRCC, 905 from LAST
- **Total Identities**: 512 from Celeb-reID, 146 from PRCC, 541 from LAST
- **Target Gallery**: 20,510 images with 2,225 corresponding ground truths

### Image Sources
The images in the ITCPR dataset are sourced from the following datasets:
- **Celeb-reID**
- **PRCC**
- **LAST**

These are utilized solely for testing purposes in the ZS-CPR task.

### Annotation Files
The dataset includes two annotation files: `query.json` and `gallery.json`.

#### `query.json` Format
Each entry in the `query.json` file follows this structure:
```json
{
    "file_path": "Celeb-reID/001/1_1_0.jpg",
    "datasets": "Celeb-reID",
    "person_id": 1,
    "instance_id": 1,
    "caption": "Wearing a brown plaid shirt, black leather shoes, another dark gray T-shirt, another blue jeans"
}
```
- `file_path`: Reference image path relative to the data root directory.
- `datasets`: Source dataset of the image.
- `person_id`: Person ID in the original dataset.
- `instance_id`: Unique identifier for gallery ground truth matching.
- `caption`: Relative caption of the reference image.

#### `gallery.json` Format
Each entry in the `gallery.json` file follows this structure:
```json
{
    "file_path": "Celeb-reID/001/1_2_0.jpg",
    "datasets": "Celeb-reID",
    "person_id": 1,
    "instance_id": 1
}
```
- `instance_id`: Matches with `query.json` for target images; -1 for non-matching query instances.
- Others: Correspond to target image path, original dataset, and person ID.

### Data Directory Structure
```
data
|-- Celeb-reID
|   |-- 001
|   |-- 002
|   |-- 003
|   ...
|-- PRCC
|   |-- train
|   |-- val
|   |-- test
|-- LAST
|   |-- 000000
|   |-- 000001
|   |-- 000002
|   ...
|-- query.json
|-- gallery.json

```

### Dataset Download and Preparation
Download and prepare the datasets as follows:

1. **Celeb-reID**: [GitHub Repository](https://github.com/Huang-3/Celeb-reID)
2. **PRCC**: [Google Drive Link](https://drive.google.com/file/d/1yTYawRm4ap3M-j0PjLQJ--xmZHseFDLz/view?usp=sharing)
3. **LAST**: [GitHub Repository](https://github.com/shuxjweb/last)

After downloading, use the `img_process.py` script to process Celeb-reID and LAST datasets into the standard format. The PRCC (subfolder PRCC/rgb) dataset can be directly placed in the corresponding directory upon extraction.

### Acknowledgments
We are deeply thankful to the creators of the Celebrities-ReID, PRCC, and LAST datasets for their significant contributions to the field of person re-identification. Their commitment to open-sourcing these valuable resources has greatly facilitated advancements in academic and practical research.

---
- **Celebrities-ReID**: "Celebrities-ReID: A Benchmark for Clothes Variation in Long-Term Person Re-Identification" - [View Paper](https://ieeexplore.ieee.org/document/8851957)
- **PRCC**: "Person Re-identification by Contour Sketch under Moderate Clothing Change" - [View Paper](https://arxiv.org/abs/2002.02295)
- **LAST**: "Large-Scale Spatio-Temporal Person Re-identification: Algorithms and Benchmark" - [View Paper](https://arxiv.org/abs/2105.15076)
---
