# 🛡️ CAI Implementation

This repository aims to reproduce Anthropic's [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) paper, adapting its scale to a personal project.

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
     Replace the variables with appropriate values.

2. **Install Requirements**:
   - Ensure Python 3.11 is installed.
   - Install the necessary packages by running:
     ```bash
     pip install -r requirements.txt
     ```

### ☁️ Google Colab

For an optimized experience, it's recommended to use Google Colab:

1. **Mount the Repository**:
   - Upload the respository directory to your Google Drive.

2. **Run in Colab**:
   - Open the notebook in Google Colab.
   - Connect to a TPU for enhanced performance.
   - Execute the cells sequentially to run the project. 
   - If running `constitutional_training.ipynb`, ensure the directories for writing sft and dpo models exist or change them.

### ⏳ What to expect

Setting up the project in Google Colab is designed to be efficient, with minimal time required for configuration. The primary time investments are as follows:

- **Loading Tokenizer, Model, and Datasets**: Depending on your internet connection and the size of the model and datasets, this process shouldn't take significant time.

- **Supervised Fine-Tuning (SFT) Training**: Approximately 3 hours.

- **Direct Preference Optimization (DPO) Training**: Approximately 3 hours.

These durations are estimates and can vary based on factors such as computational resources and specific configurations, which are default in my case for Google Colab.
