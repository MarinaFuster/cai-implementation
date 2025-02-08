# 🛡️ CAI Implementation

This repository aims to reproduce Anthropic's [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) paper, adapting its scale to a personal project. Reed the project report [here](https://github.com/MarinaFuster/cai-implementation/blob/main/REPORT.md).

## 🎯 Motivation

- **Learning**: Reproduce the ideas behind the paper, adapting the setup to resources available.
- **Understanding**: Grasp the challenges in fine-tuning a model, a key technique in implementing alignment strategies for advanced models.
- **Exploring AI Alignment**: Explore one of the most widely used techniques in the field of AI alignment—ensuring that AI systems behave in accordance with human intentions and values. This technique, known as Reinforcement Learning from Human Feedback (RLHF), has a variation in this paper called Reinforcement Learning from AI Feedback (RLAIF). While the base model undergoes an RLHF process, the constitution process applies a second iteration using RLAIF, though the steps are quite similar.
- **Comparative Analysis**: Examine the advantages and disadvantages of the technique used here, which shares similarities with the original Reinforcement Learning from Human Feedback (RLHF) approach.
- **Risk Assessment**: Investigate how this method addresses the risks studied in [BlueDot Impact's Alignment course](https://aisafetyfundamentals.com/alignment/), focusing on the ethical implications and safety concerns associated with AI deployment.
- **Sharing**: Structure the code in a way that others can easily learn from this experience, enabling them to use it as a foundation for future projects.

## 🚀 Using the Project

### 🖥️ Option A: Local Setup

1. **Set Up `.env` File**:
   - Obtain a Hugging Face (HF) token by signing up at [Hugging Face](https://huggingface.co/join).
   - Create a `.env` file in the root directory of the repository.
   - Add the following line to the `.env` file:
     ```
      HF_TOKEN=your_hf_token_here
      SFT_OUTPUT_DIR=your_output_directory_for_sft_model
      DPO_OUTPUT_DIR=your_output_directory_for_dpo_model
      DATASETS_OUTPUT_DIR=your_datasets_directory # if you want to store or load datasets from this folder
     ```
     Replace the variables with appropriate values. Keep in mind both directories will be created if they do not exist, alognside its parents.

     If you are using pre-existing datasets to train your model you should set up `DATASETS_OUTPUT_DIR=./datasets`, mapping to the folder where `prefs_dataset` and `sft_dataset` are stored.

2. **Install Requirements**:
   - Ensure Python 3.11 is installed.
   - Install the necessary packages by running:
     ```bash
     pip install -r requirements.txt
     ```

3. **Execution**:
   - Create a script and happy coding! 😊

### ☁️ Option B: Google Colab (Recommended)

- Go to [Google Colab](https://colab.research.google.com/) and log in to your account.
- Select the Github Option and enter the following URL: `https://github.com/MarinaFuster/cai-implementation`
- Choose which notebook you would like to run.
- Execute cells step by step. It is important that you clone the repository and then create the `.env` file as explained in the Local Setup section, inside the directory corresponding to the repository. 
#### 🚨 Without the `.env` file in the `cai-implementation` directory, the notebook will not work.
<div align="center">
    <img src="assets/colab_setup.gif" width="900" height="300" alt="Alt Text">
</div>

### ⏳ What to expect

Setting up the project in Google Colab is designed to be efficient, with minimal time required for configuration. The primary time investments are as follows:

- **Loading Tokenizer, Model, and Datasets**: Depending on your internet connection and the size of the model and datasets, this process shouldn't take significant time.

- **Supervised Fine-Tuning (SFT) Training**: Approximately 3 hours.

- **Direct Preference Optimization (DPO) Training**: Approximately 3 hours.

These durations are estimates and can vary based on factors such as computational resources and specific configurations, which are default in my case for Google Colab.
