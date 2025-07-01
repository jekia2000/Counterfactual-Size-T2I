# 1. Introduction
This project explores techniques to automatically rewrite prompts aiming at generating images with 2 objects of counterfactual size (e.g. a small whale and a big hamster). The prompt generation model used is small gpt-2, due to its time efficiency in training and inference, as well as comparability with  [Promptist](https://github.com/microsoft/LMOps/tree/main/promptist), which also used GPT-2 as their foundation.


# 2. Notes
Absolute paths were used everywhere for easier management. Three main groups:
- thesis: the code
- resource: images, model weights, etc.
- jobs: slurm executables and their output
  
You would just need to adjust the part of the path which comes before one of the three main groups to your local directory path.


  
# 3. Install requirements

`pip install -r requirements.txt`

For setting up Grounded SAM, please check out their repository: [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)



# 4. Image Evaluator
Navigate to the evaluator folder with bash scripts

`cd thesis/evaluator/bash_scripts`

Extract image paths and captions

`./prompt.sh`


### 4.1. Testing the evaluator on vanilla Grounding SAM
In ../gsam.yml set the eval_type to 1. If you want to save the images with boxes and masks, specify img_save_folder in ../gsam.yml

`./eval1.sh`


### 4.2. Testing the evaluator on improvement from 2.1.: Removing small objects and making masks mutually exclusive
In ../gsam.yml set the eval_type to 2. If you want to save the images with boxes and masks, specify img_save_folder in ../gsam.yml

`./eval2.sh`


### 4.3. Testing the evaluator on improvement from 2.2.: Adding a simple vector database
In ../gsam.yml set the eval_type to 3. If you want to save the images with boxes and masks, specify img_save_folder in ../gsam.yml

`./eval3.sh`


### 4.4. Testing the evaluator on improvement from 2.3.: Controlling lenience of Grounding SAM with CLIP
In ../gsam.yml set the eval_type to 4. If you want to save the images with boxes and masks, specify img_save_folder in ../gsam.yml

`./eval4.sh`



# 5. Supervised Finetuned GPT-2
## 5.1. Preparing the dataset
Navigate to the gpt folder with bash scripts

`cd thesis/gpt/bash_scripts`

### 5.1.1. Create partitions of object pairs into train, validation and test

`./partitions.sh`


### 5.1.2. Generate the prompts for chatgpt-4o

`./gen_prompts.sh`


### 5.1.3. Interact with chatgpt-4o and obtain rewritten prompts


### 5.1.4. Build a dataset based on 3.1.3.
`./gen_ds.sh`


## 5.2. Finetune gpt-2 with hyperparameter search

Navigate to the gpt folder with bash scripts

`cd thesis/gpt/bash_scripts`

Then execute

`./gpt_train.sh`


## 5.3. Evaluate gpt-2 on the test set
Nvaigate to

`cd thesis/final_assessment/new_bash_scripts`

Then execute

`assess_gpt2.sh`



# 6. GPT-2 finetuned via DPO
## 6.1. Preparing the dataset


### 6.1.1. Generate and evaluate output prompts based on the input prompts from the union of gpt2 training and validation sets (section 3)
Navigate to

`cd thesis/final_assessment/new_bash_scripts`

Then execute

`./infer_dpo.sh`


### 6.1.2. Build the train and validation datasets based on 4.1.1.

Navigate to the dpo folder with bash scripts

`cd thesis/dpo/bash_scripts`

Execute

`./gen_ds_same_model.sh`


## 6.2. Finetune GPT-2 via DPO with hyperparameter search

Navigate to the dpo folder with bash scripts

`cd thesis/dpo/bash_scripts`

Execute

`./dpo_train_same_model_sigmoid.sh`


## 6.3. Evaluate finetuned GPT-2 via DPO
Navigate to

`cd thesis/final_assessment/new_bash_scripts`

Then execute

`./assess_dpo.sh`



# 7. Reranker
## 7.1. Small Reranker
### 7.1.1. Evaluate small reranker on the test set
Navigate to

`cd thesis/final_assessment/new_bash_scripts`

Then execute

`./assess_small_reranker.sh`


## 7.2. Big Reranker
### 7.2.1. Supervised Finetune gpt2-large
Navigate to

`cd thesis/gpt/bash_scripts`

Then execute

`./gpt_single_train.sh`

### 7.2.2. Finetune gpt2-large from 5.2.1. via DPO
Navigate to

`cd thesis/dpo/bash_scripts`

Execute

`./dpo_single_train.sh`


### 7.2.3. Evaluate big reranker on the test set
Navigate to

`cd thesis/final_assessment/new_bash_scripts`

Then execute

`./assess_big_reranker.sh`



# 8. GPT-2 Aligned with Image Evaluator (AutoContra without Ranker)
## 8.1. Preparing the dataset
Navigate to 

`cd thesis/gpt/bash_scripts`

Then execute

`./gen_ds_dpo.sh`

## 8.2. Finetune gpt-2 with hyperparameter search

Navigate to 

`cd thesis/gpt/bash_scripts`

Then execute

`./gpt_train_aligned.sh`


## 8.3. Evaluate gpt-2 on the test set
Navigate to

`cd thesis/final_assessment/new_bash_scripts`

Then execute

`assess_gpt2_aligned.sh`

# 9. Small Reranker Aligned
## 9.1. Evaluate Small Reranker Aligned on the test set
Navigate to

`cd thesis/final_assessment/new_bash_scripts`

Then execute

`./assess_small_reranker_aligned.sh`

# 10. Big Reranker Aligned (AutoContra)
## 10.1. Evaluate AutoContra on the test set
Navigate to

`cd thesis/final_assessment/new_bash_scripts`

Then execute

`./assess_big_reranker_aligned.sh`

# 11. Evaluating other methods
Navigate to

`cd thesis/final_assessment/new_bash_scripts`


## 11.1. ChatGPT-4o

Execute: `./assess_chatgpt-4o.sh`

## 11.2. Input Template

Execute: `./assess_template.sh`

## 11.3. [Promptist](https://github.com/microsoft/LMOps/tree/main/promptist)

Execute: `./assess_promptist.sh`













