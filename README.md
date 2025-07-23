# Meme Trojan: Backdoor Attacks Against Hateful Meme Detection via Cross-Modal Triggers
<div align="center">
<img src="img/motivation.png" width="100%"/>
</div>


---

## Setup
* python==3.7
* mmf==1.0.0rc12
* torch==1.11.0
* torchvision==0.12.0
* pytorch_lightning==1.6.0
* timm==0.9.12
* diffusers==0.21.4
* numpy==1.21.4
* easyocr==1.7.1

## Quickstart

1. Dataset download: [FBHM](https://arxiv.org/pdf/2005.04790), [MAMI](https://aclanthology.org/2022.semeval-1.74.pdf), [HarMeme](https://arxiv.org/pdf/2110.00413). 
For your custom dataset, please refer to [MMF Dataset](https://mmf.sh/docs/tutorials/dataset)

2. Data poisoning:
```bash
python sample.py # randomly sample \rho percent memes for poisoning.
python poison.py # naive trigger injection (CMT w.o. TA).
python augmentor.py # train the augmentor
python trigger.py # inject the final CMT
```

3. Training the victim model
```bash
mmf_run config=projects/hateful_memes/configs/visual_bert/defaults.yaml \
    datasets=hateful_memes \
    model=visual_bert \
    run_type=train_val
```

4. Evaluation
```bash
mmf_predict config=projects/hateful_memes/configs/visual_bert/defaults.yaml \
    datasets=hateful_memes \
    model=visual_bert \
    run_type=test \ 
    checkpoint.resume_file=./save/visual_bert_final.pth \
    checkpoint.resume_pretrained=False
```






## Citation

If you use our Meme Trojan in your work, please cite:

```bibtex
@inproceedings{wang2025meme,
  title={Meme Trojan: Backdoor Attacks Against Hateful Meme Detection via Cross-Modal Triggers},
  author={Wang, Ruofei and Lin, Hongzhan and Luo, Ziyuan and Cheung, Ka Chun and See, Simon and Ma, Jing and Wan, Renjie},
  booktitle={Proc. AAAI},
  volume={39},
  number={8},
  pages={7844--7852},
  year={2025}
}
```

## Acknowledgement
Thanks to MMF, which is a modular framework for vision and language multimodal research from Facebook AI Research. See full list of project inside or built on MMF [here](https://mmf.sh/docs/notes/projects).

