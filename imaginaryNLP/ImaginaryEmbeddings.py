import logging
import pickle
from typing import Optional, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ImaginaryEmbeddings:
    """

    """
    def __init__(self, model_name_or_path: Optional[str] = None, speaker_token: bool = True):
        """
        :param model_name_or_path: Path to the model or the model name from huggingface.co/models
        """
        self.model = SentenceTransformer(model_name_or_path)

        self.speaker_token = speaker_token

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = self.model.to(self.device)
































