# Global-Constraints-with-Prompting-for-Zero-Shot-Event-Argument-Classification
Code for EACL 2023 (Findings) paper *Global Constraints with Prompting for Zero-Shot Event Argument Classification*.

# Library Requirements: 
1. Python 3.7
2. torch 1.7.0+cu110
3. transformers 4.24.0

# Dataset:
ACE2005-E+: https://www.ldc.upenn.edu/collaborations/past-projects/ace
ERE-EN (LDC2015E29): Please check your institution's LDC account for access.
The procedure of our data pre-processing is similar to that of *Zero-shot Event Extraction via Transfer Learning: Challenges and Insights*, please check the details in their code repositories (https://github.com/veronica320/Zeroshot-Event-Extraction).

# Usage:
Since our model is zero-shot, it has no training process. For inference, run the following command:
```
python -u main.py \
--input_dir data/ACE05-E+_converted/ \
--output_dir output/ \
--dataset_name ACE05-E+ \
--split merge \
--mode prompting \
--lm_name gptj-6b \
--model_name main_model \
&> log.txt
```
See the comments within the soure code for more details about using the code.
