# 📜 Reproducing Constitutional AI: Implementation and Findings

# 🚀 Introduction

As AI systems become more advanced, scalable oversight is crucial to ensure alignment with human values. High-quality, large-scale human feedback is expensive, making it necessary to explore alternative techniques. 

The *Constitutional AI* paper by Anthropic aims to create a non-evasive assistant that engages with harmful queries by explaining objections rather than simply refusing to respond. This project attempts to reproduce and adapt the methodology within feasible constraints.

## Example of Model Behavior

Below are comparisons of responses from different models when prompted with a harmful query:

- **Mistral 7B Helpful-Only Model:**
  ![Revised Answer - Mistral](./assets/harmful_prompt_mistral.png)
  *Response from Mistral 7B fine-tuned only for helpfulness.*

- **ChatGPT Model:**
  ![Revised Answer - ChatGPT](./assets/harmful_prompt_chatgpt.png)
  *Response from ChatGPT with safety guardrails.*

# 🎯 Motivation

This project is motivated by the following goals:
- Reproducing the ideas from the *Constitutional AI* paper within available resources.
- Organizing the code systematically so others can learn from the implementation.
- Prioritizing the full implementation before refining results.
- Creating an open repository for continued improvements.
- Gaining practical experience in fine-tuning large models, a crucial technique for AI alignment.

# 🛠️ Adapted Methodology

The original methodology was adapted due to model availability and computational limitations. The steps taken are as follows:

1. **Select a Base Model**
   - Instead of using Anthropic’s RLHF model, this project uses **[Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)** fine-tuned for helpfulness. This model lacks built-in safety guardrails.
   - Fine-tuning is performed using **LoRA** to reduce computational cost.

2. **Generate Revised Responses**
   - The critique-revision process follows a multi-step approach in the original paper: after receiving a prompt, the model must critique the answer and then revise the response based on its critique.
   - In the original paper, few-shot examples are included to clarify the different roles involved in creating the revisions.
   - In this implementation, an example was created to understand the process; however, for training purposes, a pre-existing dataset is used.

3. **Supervised Fine-Tuning (SFT)**
   - The model is fine-tuned on revised responses from the final critique-revision step.
   - Helpful responses are incorporated to maintain assistance capabilities.

4. **Direct Preference Optimization (DPO)**
   - Instead of reinforcement learning with a preference model, DPO is used due to dataset availability and ease of implementation.

# 🏗️ Implementation

The codebase is structured as follows:

- **`critique_flow.ipynb`**
  - Demonstrates the critique-revision response flow described in the original paper.
  - Not required for SFT or DPO but useful for understanding the dataset creation process.

- **`constitutional_training.ipynb`**
  - Implements the full training pipeline using:
    - `dataset_manager.py` for dataset processing.
    - `model_manager.py` for model and tokenizer loading.
  - Stores final results in a specified directory, after training.

# 📊 Results

## Critique-Revision Flow

The revised response quality from Mistral 7B was suboptimal. Instead of providing critiques, the model often ignored harmful queries altogether.

![Revised Response Example](./assets/revised_response.png)

Future improvements could involve using a separate model for critique and revision. This is why the decision was made to use an existing [dataset for harmless prompts](https://huggingface.co/datasets/HuggingFaceH4/cai-conversation-harmless).

## Supervised Fine-Tuning

| Feature               | Anthropic       | This Project   |
|-----------------------|----------------|---------------|
| **Harmlessness Prompts** | 182,831        | 2,000         |
| **Revisions**          | 4 per prompt   | 1 per prompt  |
| **Helpfulness Prompts** | 135,296        | 800           |
| **Helpful Answers**    | 2 per prompt   | 1 per prompt  |
| **Model Size**        | Not specified  | 7B            |
| **LoRA**             | Not mentioned  | Yes           |
| **Training Epochs**   | 1              | 3             |

After fine-tuning, the model struggled to retain its helpfulness. Achieving the balance between helpfulness and harmlessness proved challenging.

# 🤔 Discussion

- **Human Oversight Limitations:**
  - As seen in code review processes, human evaluation is both expensive and error-prone. On the other hand, as models become more capable and humans increasingly rely on them, attention to detail diminishes for certain types of tasks.
  
- **Challenges with Constitutional Principles:**
  - The principles used in the paper may be too generic, resulting in responses that probably cluster closely in embedding space.

- **Prompt Sensitivity:**
  - The model’s behavior significantly changes based on prompt phrasing (e.g., using *"Human:"* as the role before a query prompts a safer response).

- **Training Difficulties:**
  - Fine-tuning negatively impacted overall performance, highlighting the complexity of maintaining alignment without degrading helpfulness and pre-existing capabilities.
  - Limited computational resources prevented extensive hyperparameter tuning.

- **Security Considerations:**
  - LoRA fine-tuning does not modify the underlying model architecture, raising concerns about the robustness of alignment techniques.

# Conclusion

Fine-tuning a model for constitutional AI proved more challenging than expected. However, this project lays the groundwork for future iterations. The next steps involve:
- Addressing the loss of capabilities in SFT training.
- Achieving comparable results with the paper for smaller models.

What was actually achieved?
- Reproducing the ideas from the *Constitutional AI* paper within available resources. 🟡
- Organizing the code systematically so others can learn from the implementation. 🟢
- Prioritizing the full implementation before refining results. 🟢
- Creating an open repository for continued improvements. 🟢
- Gaining practical experience in fine-tuning large models, a crucial technique for AI alignment. 🔴

# 📚 References

1. [Constitutional AI: Harmlessness from AI Feedback (Anthropic)](https://arxiv.org/abs/2212.08073)
2. [Contextual Constitutional AI (LessWrong)](https://www.lesswrong.com/posts/ivWPqkipkKywQbdDw/contextual-constitutional-ai)
3. [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/2106.09685)
4. [Learning from AI Feedback](https://arxiv.org/abs/2204.05862)
5. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
