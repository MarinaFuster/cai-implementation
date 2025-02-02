import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


logger = logging.getLogger(__name__)


class ModelManager:
    """
    Loads and configures a transformer model and its tokenizer.
    """
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3", model_dir=None):
        """
        Initializes ModelManager by loading the tokenizer and model.

        Args:
            model_name (str): The HuggingFace model identifier.
            model_dir (str, optional): Local directory for the model; if provided, overrides model_name.
        """
        model_reference = model_dir if model_dir else model_name
        
        logger.info(f"Initializing ModelManager. Loading tokenizer from {model_reference}.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_reference)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Tokenizer loaded.")
        self._bnb_config = self._get_bnb_config()
        
        logger.info(f"Loading model from {model_reference}.")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_reference, 
            quantization_config=self._bnb_config, 
            device_map="auto"
        )
        logger.info("Model loaded.")

    def _get_bnb_config(self):
        """
        Returns a BitsAndBytesConfig for 4-bit quantization.

        This configuration enables loading in 4-bit precision with nf4 quantization type, 
        uses float16 for computation, and applies double quantization.
        """
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    def tokenize_function(self, examples, max_length=512):
        """
        Tokenizes the provided dataset examples.

        Args:
            examples (dict): A dictionary with 'input_text' and 'output_text' fields.
            max_length (int, optional): Maximum token length for padding and truncation. Defaults to 512.

        Returns:
            dict: Tokenized outputs with padding and truncation applied.
        
        Note:
            While some models (e.g., Mistral 7B) support a large context window (up to 32K tokens),
            the max_length parameter here is often set lower (default 512) due to computing resource constraints,
            especially in environments like Colab.
        """
        return self.tokenizer(
            examples["input_text"], 
            text_target=examples["output_text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length
        )

    def generate_response(self, prompt, max_new_tokens=150):
        """
        Generates a response from the model, returning only the generated text.

        Args:
            prompt (str): The user prompt.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            str: The model's generated response, excluding the input prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Decode the full output
        full_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove the prompt from the output (keep only the model's response)
        response = full_output[len(prompt):].strip()
        
        return response
