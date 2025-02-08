import logging
from pathlib import Path
from datasets import Dataset, load_dataset, load_from_disk


logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Loads and prepares datasets for SFT and preference training using harmful and helpful samples.
    """
    def __init__(self):
        self.harmless_dataset = None
        self.sft_helpful_dataset = None
        self.prefs_helpful_dataset = None
        logger.info("Initialized DatasetManager.")

    def create_sft_train_dataset(
        self, 
        n_samples_harmless: int, 
        n_samples_helpful: int, 
        seed: int = 42, 
        store: bool = False, 
        output_dataset_path: Path = None):
        """
        Returns an SFT training dataset by combining harmless and helpful samples.

        Args:
            n_samples_harmless (int): Number of harmless samples to use.
            n_samples_helpful (int): Number of helpful samples to use.
            seed (int, optional): Random seed for shuffling. Defaults to 42.
            store (bool, optional): Stores the created dataset to disk if True. Defaults to False.
            output_dataset_path (Path, optional): Where is the dataset being stored. Defaults to None.

        Returns:
            Dataset: A HuggingFace Dataset with 'input_text' and 'output_text' fields.
        """
        if self.harmless_dataset is None:
            self.harmless_dataset = load_dataset("HuggingFaceH4/cai-conversation-harmless")
            logger.info(f"Loaded harmless dataset: HuggingFaceH4/cai-conversation-harmless.")
        
        sampled_harmless = self.harmless_dataset["train_sft"].shuffle(seed=seed).select(range(n_samples_harmless))
        harmless_inputs = sampled_harmless["init_prompt"]
        harmless_outputs = sampled_harmless["revision_response"]
        
        helpful_samples = self._extract_inputs_and_outputs(
            stage="sft", safeguard="helpful", split="train", n_samples=n_samples_helpful, seed=seed
        )
        
        helpful_inputs = helpful_samples["inputs"]
        helpful_outputs = helpful_samples["outputs"]
        
        sft_dataset = Dataset.from_dict({
            "input_text": harmless_inputs + helpful_inputs,
            "output_text": harmless_outputs + helpful_outputs,
        })
        
        if store:
            sft_dataset.save_to_disk(output_dataset_path)
        
        return sft_dataset

    def get_sft_train_dataset(
        self, 
        n_samples_harmless: int, 
        n_samples_helpful: int, 
        seed: int = 42, 
        load: bool = False, 
        dataset_path: Path = None):
        """
        Returns an SFT training dataset by combining harmless and helpful samples.

        Args:
            n_samples_harmless (int): Number of harmless samples to use.
            n_samples_helpful (int): Number of helpful samples to use.
            seed (int, optional): Random seed for shuffling. Defaults to 42.
            load (bool, optional): Indicates if dataset will be loaded from dataset_path. Defaults to False.
            dataset_path (Path, optional): Where the dataset is loaded from, if load=True. Defaults to None.

        Returns:
            Dataset: A HuggingFace Dataset with 'input_text' and 'output_text' fields.
        """
        if load:
            return load_from_disk(dataset_path)
        
        return self.create_sft_train_dataset(
            n_samples_harmless,
            n_samples_helpful,
            seed)

    def create_prefs_train_dataset(
        self, 
        n_samples_harmless, 
        n_samples_helpful, 
        seed=42, 
        store: bool = False, 
        output_dataset_path: Path = None):
        """
        Returns a preference training dataset by combining harmless and helpful samples.

        Args:
            n_samples_harmless (int): Number of harmless samples to use.
            n_samples_helpful (int): Number of helpful samples to use.
            seed (int, optional): Random seed for shuffling. Defaults to 42.
            store (bool, optional): Stores the created dataset to disk if True. Defaults to False.
            output_dataset_path (Path, optional): Where is the dataset being stored. Defaults to None.

        Returns:
            Dataset: A HuggingFace Dataset with 'prompt', 'chosen', and 'rejected' fields.
        """
        harmless_samples = self._extract_inputs_and_outputs(
            stage="prefs", safeguard="harmless", split="train", n_samples=n_samples_harmless, seed=seed
        )
        
        harmless_inputs = harmless_samples["inputs"]
        harmless_chosen = harmless_samples["outputs"]["chosen"]
        harmless_rejected = harmless_samples["outputs"]["rejected"]
        
        helpful_samples = self._extract_inputs_and_outputs(
            stage="prefs", safeguard="helpful", split="train", n_samples=n_samples_helpful, seed=seed
        )
        
        helpful_inputs = helpful_samples["inputs"]
        helpful_chosen = helpful_samples["outputs"]["chosen"]
        helpful_rejected = helpful_samples["outputs"]["rejected"]
        
        prefs_dataset = Dataset.from_dict({
            "prompt": harmless_inputs + helpful_inputs, 
            "chosen": harmless_chosen + helpful_chosen, 
            "rejected": harmless_rejected + helpful_rejected
        })
        
        if store:
            prefs_dataset.save_to_disk(output_dataset_path)

        return prefs_dataset            

    def get_prefs_train_dataset(
        self, 
        n_samples_harmless, 
        n_samples_helpful, 
        seed=42, 
        load: bool = False, 
        dataset_path: Path = None):
        """
        Returns a preference training dataset by combining harmless and helpful samples.

        Args:
            n_samples_harmless (int): Number of harmless samples to use.
            n_samples_helpful (int): Number of helpful samples to use.
            seed (int, optional): Random seed for shuffling. Defaults to 42.
            load (bool, optional): Indicates if dataset will be loaded from dataset_path. Defaults to False.
            dataset_path (Path, optional): Where the dataset is loaded from, if load=True. Defaults to None.

        Returns:
            Dataset: A HuggingFace Dataset with 'prompt', 'chosen', and 'rejected' fields.
        """
        if load:
            return load_from_disk(dataset_path)
        
        return self.create_prefs_train_dataset(
            n_samples_harmless,
            n_samples_helpful,
            seed)

    def _extract_inputs_and_outputs(self, stage, safeguard, split, n_samples, seed=42):
        """
        Extracts inputs and outputs from the specified dataset subset.

        Args:
            stage (str): Either 'sft' or 'prefs'.
            safeguard (str): Either 'harmless' or 'helpful'.
            split (str): Dataset split ('train' or 'test').
            n_samples (int): Number of samples to extract.
            seed (int, optional): Random seed for shuffling. Defaults to 42.

        Returns:
            dict: Contains 'inputs' and 'outputs'; for 'prefs', outputs is a dict with 'chosen' and 'rejected'.
        """
        if safeguard not in ["harmless", "helpful"]:
            raise ValueError("safeguard should be either 'harmless' or 'helpful'.")
        
        if stage not in ["sft", "prefs"]:
            raise ValueError("stage should be either 'sft' or 'prefs'.")
        
        if split not in ["train", "test"]:
            raise ValueError("split should be either 'train' or 'test'.")
        
        if stage == "sft" and safeguard == "harmless":
            raise ValueError("Harmless dataset does not follow the format to extract with this method.")
        
        if safeguard == "harmless" and self.harmless_dataset is None:
            self.harmless_dataset = load_dataset("HuggingFaceH4/cai-conversation-harmless")
            logger.info(f"Loaded harmless dataset: HuggingFaceH4/cai-conversation-harmless.")
        
        if stage == "sft" and safeguard == "helpful" and self.sft_helpful_dataset is None:
            self.sft_helpful_dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
            logger.info(f"Loaded helpful dataset: HuggingFaceH4/ultrachat_200k.")
        
        if stage == "prefs" and safeguard == "helpful" and self.prefs_helpful_dataset is None:
            self.prefs_helpful_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
            logger.info(f"Loaded helpful dataset: HuggingFaceH4/ultrafeedback_binarized.")

        # Shuffle and select a subset of the dataset
        split_name = f"{split}_{stage}"
        dataset = self.harmless_dataset if safeguard == "harmless" else getattr(self, f"{stage}_helpful_dataset")
        sampled_dataset = dataset[split_name].shuffle(seed=seed).select(range(n_samples))

        # Initialize lists to store inputs and outputs
        inputs = []
        outputs = []

        if stage == "sft":
            for messages in sampled_dataset["messages"]:
                if len(messages) >= 2 and messages[0]["role"] == "user" and messages[1]["role"] == "assistant":
                    inputs.append(messages[0]["content"])
                    outputs.append(messages[1]["content"])
                else:
                    logger.info("Bypassing sample due to incorrect roles.")
        
        if stage == "prefs":
            outputs = {}
            chosen_responses = []
            rejected_responses = []
            # Chosen-rejected message pairs
            for chosen, rejected in zip(sampled_dataset["chosen"], sampled_dataset["rejected"]):
                chosen_ok = len(chosen) >= 2 and chosen[0]["role"] == "user" and chosen[1]["role"] == "assistant"
                rejected_ok = len(rejected) >= 2 and rejected[0]["role"] == "user" and rejected[1]["role"] == "assistant"
                if chosen_ok and rejected_ok:
                    inputs.append(chosen[0]["content"])
                    chosen_responses.append(chosen[1]["content"])
                    rejected_responses.append(rejected[1]["content"])
                else:
                    logger.info("Bypassing sample due to incorrect roles.")
            outputs["chosen"] = chosen_responses
            outputs["rejected"] = rejected_responses

        logger.info(f"Extracted {len(inputs)} inputs with its corresponding outputs.")
        return {'inputs': inputs, 'outputs': outputs}
