# 🛡️ CAI Implementation

This repository aims to reproduce Anthropic's [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) paper, adapting its scale to a personal project. Reed the project report [here](https://github.com/MarinaFuster/cai-implementation/blob/main/REPORT.md).

## 🎯 Motivation

- **Learning**: Reproduce the ideas behind the paper, adapting the setup to resources available.
- **Understanding**: Grasp the challenges in fine-tuning a model, a key technique in implementing alignment strategies for advanced models.
- **Exploring AI Alignment**: Delve into one of the most prevalent techniques in the industry to align models—ensuring AI systems act in accordance with human intentions and values.
- **Comparative Analysis**: Examine the advantages and disadvantages of the technique used here, which shares similarities with the original Reinforcement Learning from Human Feedback (RLHF) approach.
- **Risk Assessment**: Investigate how this method addresses the risks studied in [BlueDot Impact's Alignment course](https://aisafetyfundamentals.com/alignment/), focusing on the ethical implications and safety concerns associated with AI deployment.
- **Sharing**: Organize the code so others can learn from this experience.

## 🚀 Using the Project

### 🖥️ Local Setup

1. **Set Up `.env` File**:
   - Obtain a Hugging Face (HF) token by signing up at [Hugging Face](https://huggingface.co/join).
   - Create a `.env` file in the root directory of the repository.
   - Add the following line to the `.env` file:
     ```
      HF_TOKEN=your_hf_token_here
      SFT_OUTPUT_DIR=your_output_directory_for_sft_model
      DPO_OUTPUT_DIR=your_output_directory_for_dpo_model
     ```
     Replace the variables with appropriate values. Keep in mind both directories will be created if they do not exist, alognside its parents.

2. **Install Requirements**:
   - Ensure Python 3.11 is installed.
   - Install the necessary packages by running:
     ```bash
     pip install -r requirements.txt
     ```

3. **Execution**:
   - Create a script and happy coding! 😊

### ☁️ Google Colab

- Go to [Google Colab](https://colab.research.google.com/) and log in to your account.
- Select the Github Option and enter the following URL: `https://github.com/MarinaFuster/cai-implementation`
- Choose which notebook you would like to run.
- Execute cells step by step. It is important that you clone the repository and then create the `.env` file as explained in the Local Setup section, inside the directory corresponding to the repository.

### ⏳ What to expect

Setting up the project in Google Colab is designed to be efficient, with minimal time required for configuration. The primary time investments are as follows:

- **Loading Tokenizer, Model, and Datasets**: Depending on your internet connection and the size of the model and datasets, this process shouldn't take significant time.

- **Supervised Fine-Tuning (SFT) Training**: Approximately 3 hours.

- **Direct Preference Optimization (DPO) Training**: Approximately 3 hours.

These durations are estimates and can vary based on factors such as computational resources and specific configurations, which are default in my case for Google Colab.
