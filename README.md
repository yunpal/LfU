# Learning from the Undesirable:Robust Adaptation of Language Models without Forgetting (AAAI 2026)

Code for the paper *"Learning from the Undesirable:Robust Adaptation of Language Models without Forgetting"* (AAAI 2026).

## Method Overview
![Method Overview](https://github.com/yunpal/LfU/blob/main/assets/overview.png?raw=true)


## LfU Installation and Execution Guide

## Execution
### Installation
   ```bash
    cd LLaMA-Factory
    conda create -n LfU python=3.10 -y
    conda activate LfU
    pip install -e .g
    pip install datasets==3.2.0
    pip install wandb
    pip install peft==0.15.1
    pip install accelerate==1.2.1
    pip install transformers==4.51.3
   ```
 **Run LfU(LoRA)**

   ```bash
   bash run_lora.sh
   ```
 **Optional: To run the representation consistency version (LfU-RepS), use:**

   ```bash
   bash run_reps.sh
   ```

## Copyright

Our code utilizes the following open-source projects:  
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for model fine-tuning

## Citation

@inproceedings{nam2026learning,
  title={Learning from the Undesirable: Robust Adaptation of Language Models without Forgetting},
  author={Nam, Yunhun and Kim, Jaehyung and Jeong, Jongheon},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={38},
  pages={32537--32545},
  year={2026}
}
