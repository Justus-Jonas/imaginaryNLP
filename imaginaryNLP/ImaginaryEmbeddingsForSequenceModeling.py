import logging
import pickle
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from imaginaryNLP.ImaginaryEmbeddings import ImaginaryEmbeddings


class ImaginaryEmbeddingsForSequenceModeling(ImaginaryEmbeddings):
    """
    Imaginary Embeddings for Sequence Modeling
    """
    def __init__(self,
                 model_name_or_path: Optional[str] = None,
                 speaker_token: bool = True,
                 ):
        """
        :param model_name_or_path: Path to the model or the model name from huggingface.co/models
        :param candidates: List of candidate strings without any special tokens (will be added automatically)
        :param speaker_token: If True, the speaker token will be added to the input
        :param auto_model_loading: If True, the model will be automatically loaded from huggingface.co/models
        """
        if model_name_or_path:
            pass
        else:
            if speaker_token:
                model_name_or_path = 'Justus-Jonas/Imaginary-Embeddings-SpeakerTokens'
            else:
                model_name_or_path = 'Justus-Jonas/Imaginary-Embeddings-Classic'

        super().__init__(model_name_or_path, speaker_token)

        print('Sorry, but the code for Imaginary Embeddings is not yet published. '
              'You will be able to test this package during the ACL 2023 conference.')

        self.speaker_token = speaker_token
        self.context = None
        self.context_odd = None
        self.context_even = None
        self.candidates = None
        self.precompute = False
        self.precomputed_scores = None
        self.candidates_embeddings = None
        self.context_embeddings = None
        self.context_embeddings_odd = None
        self.context_embeddings_even = None
        self.context_embeddings_batch = None


    def load_candidates_from_strings(self, candidates: List[str], output_directory_candidates_dump: str = None):
        """
        :param candidates: List of candidate strings without any special tokens (will be added automatically)
        :return: None
        """

        self.candidates = candidates

        candidates = ["[AFTER]  " + candidate for candidate in candidates]

        candidates_embeddings = self.model.encode(candidates,
                                                  show_progress_bar=True,
                                                  convert_to_tensor=True,
                                                  normalize_embeddings=True).to(self.device)

        self.candidates_embeddings = candidates_embeddings

        if output_directory_candidates_dump:
            # store candidate as pickle numpy array
            np.save(output_directory_candidates_dump + "candidates.npy", candidates_embeddings.cpu().numpy(),
                    allow_pickle=True)
            # save candidates as pickle list
            with open(output_directory_candidates_dump + 'candidates.pkl', 'wb') as f:
                pickle.dump(candidates, f)




    def load_candidates_from_files(self, output_directory_candidates_dump: str):
        """
        Load a faiss index from a given path
        :param faiss_index_path: path to the faiss index
        """

        candidates_embeddings = np.load(output_directory_candidates_dump + "candidates.npy", allow_pickle=True)

        self.candidates_embeddings = torch.from_numpy(candidates_embeddings).to(self.device)

        self.candidates = pickle.load(open(output_directory_candidates_dump + 'candidates.pkl', 'rb'))


    def create_context(self, context: List[str], precompute_top_p = None):
        """
        :param context: List of strings that will be used as context
        :return: None
        """
        if self.speaker_token:
            self.context_even = ["[E] " + "[BEFORE]  " +  context for context in context]
            self.context_odd = ["[O] " + "[BEFORE]  " + context for context in context]

            self.context_embeddings_even = self.model.encode(self.context_even,
                                                          show_progress_bar=False,
                                                          convert_to_numpy=True,
                                                          normalize_embeddings=True)
            self.context_embeddings_odd = self.model.encode(self.context_odd,
                                                            show_progress_bar=False,
                                                         convert_to_numpy=True,
                                                       normalize_embeddings=True)

            self.context_embeddings = []
            # create context from reverse order
            for i in range(len(context)):
                if i % 2 == 0:
                    self.context_embeddings.append(self.context_embeddings_odd[-(i+1)])
                else:
                    self.context_embeddings.append(self.context_embeddings_even[-(i+1)])
            self.context_embeddings = torch.from_numpy(np.array(self.context_embeddings)).to(self.device)

        else:
            self.context_classic = ["[BEFORE]  " + context for context in context]
            self.context_embeddings = self.model.encode(self.context_classic,
                                                              show_progress_bar=False,
                                                              convert_to_tensor=True,
                                                              normalize_embeddings=True).to(self.device)
        if precompute_top_p:
            self._precompute_candidates(top_p=precompute_top_p)
            self.precompute = True
        else:
            self.precompute = False

    def create_contexts_as_batch(self, contexts: List[List[str]]):
        """
        for sequence batch processing
        :return:
        """

    def add_utterance_to_context(self, utterance: str, precompute_top_p = None):
        """
        :param context: List of strings that will be used as context
        :param precompute_top_p: can be a float between 0 and 1. If set, the top p candidates will be precomputed
        :return: None
        """
        self.precomputed_scores = None


        if self.speaker_token:
            self.context_even = self.context_even + ["[E] " + "[BEFORE]  " + utterance]
            self.context_odd = self.context_odd + ["[O] " + "[BEFORE]  " + utterance]

            utterance_embeddings_combined = self.model.encode([self.context_even[-1], self.context_odd[-1]],
                                                          show_progress_bar=False,
                                                          convert_to_tensor=True,
                                                          normalize_embeddings=True).to(self.device)

            self.context_embeddings_even = torch.cat([self.context_embeddings_even, utterance_embeddings_combined[0]])

            self.context_embeddings_odd = torch.cat([self.context_embeddings_odd, utterance_embeddings_combined[1]])

            self.context_embeddings = []
            # create context from reverse order
            for i in range(len(self.context_even)):
                if (i % 2 == 0 and not precompute_top_p) or (i % 2 == 1 and precompute_top_p):
                    self.context_embeddings.append(self.context_embeddings_odd[-(i + 1)])
                else:
                    self.context_embeddings.append(self.context_embeddings_even[-(i + 1)])

            self.context_embeddings = torch.from_numpy(np.array(self.context_embeddings)).to(self.device)

        else:
            # change later. This unnecessarily increases the memory usage
            self.context_classic = self.context_classic + ["[BEFORE]  " + utterance]
            utterance_embeddings_classic = self.model.encode(self.context_classic[-1],
                                                          show_progress_bar=False,
                                                          convert_to_tensor=True,
                                                          normalize_embeddings=True).to(self.device)
            self.context_embeddings = torch.cat([self.context_embeddings, utterance_embeddings_classic])

        if precompute_top_p:
            if self.candidates_embeddings is None:
                raise ValueError("Please provide candidates first")
            self._precompute_candidates(top_p=precompute_top_p)
            self.precompute = True
        else:
            self.precompute = False

    def _precompute_candidates(self, top_p=1.0):
        """
        If a dialog context is provided this method allows to pre-compute the dialog context while your dialog partner is still speaking.

        :param top_p: a float value between 0 and 1

        :return:
        """
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top p must be a float value between 0 and 1")

        bmm_result_precomputed =  torch.matmul(self.context_embeddings, self.candidates_embeddings.T)

        # Sum along the second dimension
        bmm_result_precomputed = bmm_result_precomputed.sum(dim=0)

        # send candidates to cpu
        self.candidates_embeddings = self.candidates_embeddings.cpu()

        # top p filtering
        precomputed_embeddings = torch.topk(bmm_result_precomputed, int(top_p * len(self.candidates_embeddings)))
        self.precomputed_embeddings = self.candidates_embeddings[precomputed_embeddings.indices]

        self.precomputed_scores = bmm_result_precomputed[precomputed_embeddings.indices]
        self.precomputed_indices = self.precomputed_scores.indices


    def sequence_modeling_with_precompute(self, utterance: str, top_k: int = 5, top_p: float = 0.0):
        """
        requires pre-computed candidates in advance,

        :param utterance:
        :param top_k
        :param top_p:
        :param add_to_context: bool whether to add the utterance to the context after the computation
        :return: candidates:
        """

        if not self.precompute:
            raise ValueError("You need to precompute the candidates first. "
                             "You can do this by calling the method add_utterance_to_context with the "
                             "parameter precompute_top_p set to a float value between 0 and 1")

        top_k = min(top_k, len(self.precomputed_embeddings))

        if top_p > 0.0:
            top_k = int(top_p * len(self.precomputed_embeddings))

        if self.precomputed_scores is None or self.precomputed_embeddings is None:
             raise ValueError("You need to precompute the candidates first")

        utterance = "[BEFORE]  " + utterance

        if self.speaker_token:
            utterance = "[E] " + utterance

        utterance_embedding = self.model.encode(utterance,
                                                show_progress_bar=False,
                                                convert_to_tensor=True,
                                                normalize_embeddings=True).to(self.device)

        result_last_utterance = torch.matmul(utterance_embedding.unsqueeze(0), self.precomputed_embeddings.T)

        computed_scores = self.precomputed_scores + result_last_utterance

        print(f"computed_scores: {computed_scores}")

        sorted_computed_scores = torch.sort(computed_scores.squeeze(0), descending=True)

        # receive the indices of the top k candidates
        sorted_computed_indices = torch.topk(sorted_computed_scores.indices, top_k).indices.tolist()

        candidates = [self.candidates[i] for i in sorted_computed_indices]

        # set precomputed variables to None
        self.precomputed_scores = None
        self.precomputed_embeddings = None
        self.precomputed_indices = None

        return candidates, sorted_computed_scores.values.tolist()


    def sequence_model_single_context(self, top_k: int = 5, top_p: float = 0.0):
        """
        This method computes the sequence model for a single context that was created
        with the method add_utterance_to_context
        :return:
        """

        top_k = min(top_k, len(self.candidates))
        if top_p > 0.0:
            top_k = int(top_p * len(self.candidates))

        if self.context_embeddings is None:
            raise ValueError("You need to add a context first")
        if self.candidates_embeddings is None:
            raise ValueError("You need to add candidates first")

        computed_scores = torch.matmul(self.context_embeddings,self.candidates_embeddings.T)

        computed_scores = torch.sum(computed_scores, dim=0)

        print(f"computed_scores: {computed_scores}")

        sorted_computed_scores = torch.sort(computed_scores, descending=True)

        sorted_computed_indices = torch.topk(sorted_computed_scores.indices, top_k).indices.tolist()

        candidates = [self.candidates[i] for i in sorted_computed_indices]

        return candidates, sorted_computed_scores.values.tolist()





class EvalImaginaryEmbeddingsForSequenceModeling(ImaginaryEmbeddingsForSequenceModeling):
    """
    This class is used to evaluate the sequence model for test data
    """
    def __init__(self,
                 model_name_or_path: Optional[str] = None,
                 speaker_token: bool = True):
        super().__init__(model_name_or_path=model_name_or_path,speaker_token=speaker_token)

    def sequence_model_batch_eval(self, true_candidates_id: List[int] = None, batch_size: int = 32):
        """

        :return:
        """
        if self.context_embeddings_batch is None:
            raise ValueError("You need to add contexts first")

        # now do for flattened_next_utterances_same_length_encoded
        next_utterances = self.candidates_embeddings
        history_encoded = self.context_embeddings_batch

        # repeat next_utterances_all for each history in batch
        next_utterances_same_length_batch = next_utterances.repeat(batch_size, 1, 1).clone().detach().to(self.device)



        # calculate bmm history in batches and next_utterances_all
        for i in range(0, len(history_encoded), batch_size):
            if i + batch_size > len(history_encoded):
                history_tensor = history_encoded[i:len(history_encoded)].clone().detach().to(self.device).float()
                next_utterances_same_length_batch_slice = next_utterances.repeat(
                    len(history_encoded[i:len(history_encoded)]),
                    1, 1)
                if i == 0:
                    bmm_result = torch.bmm(next_utterances_same_length_batch_slice, history_tensor.transpose(1, 2))
                else:
                    bmm_result = torch.cat(
                        (
                            bmm_result,
                            torch.bmm(next_utterances_same_length_batch_slice, history_tensor.transpose(1, 2))),
                        dim=0)
            else:
                history_tensor = history_encoded[i:batch_size + i].clone().detach().to(self.device).float()
                if i == 0:
                    bmm_result = torch.bmm(next_utterances_same_length_batch, history_tensor.transpose(1, 2))
                else:
                    bmm_result = torch.cat(
                        (bmm_result, torch.bmm(next_utterances_same_length_batch, history_tensor.transpose(1, 2))),
                        dim=0)

        # sum over the history dimension
        bmm_result = torch.sum(bmm_result, dim=2)

        ranks = []
        for his in range(len(history_encoded)):
            bm_entries = bmm_result[his]
            # flatten and sort
            bm_entries = bm_entries.flatten()
            # logging.info(f"bm_entries[next_utterances_id_same_length[his]]: {bm_entries[next_utterances_id_same_length[his]]}")

            true_utterance_score = bm_entries[true_candidates_id[his]]

            sorted_bm_entries = torch.sort(bm_entries, descending=True)

            # get rank of true utterance
            rank = torch.where(sorted_bm_entries[0] == true_utterance_score)[0][0] + 1
            ranks.append(rank.cpu())

        return ranks


    def evaluate_seq_dataset(self, dialogues : List[List[str]], history_lengths: List[int] = [1,2,3,4,5,6,7,8,9,10], batch_size: int = 32):
        """
        This method evaluates the sequence model on a dataset of dialogues.
        :param dialogues: List of dialogues. Each dialogue is a list of utterances.
        :param history_lengths: List of history lengths to evaluate.
        :param batch_size: Batch size for bmm across number of dialogues in parallel in the evaluation.
        """
        batch_size = min(batch_size, len(dialogues))

        report_dict = {}

        # flatten dialog text and create two arrays for dialog_id and utterance_id in dialog dialog_test
        flattened_dialog_test = [item for sublist in dialogues for item in sublist]

        # add after token to flatten dialog
        flattened_dialog_test_a = ["[AFTER]  " + utterance for utterance in flattened_dialog_test]

        flattened_dialog_test_encoded_a = self.model.encode(flattened_dialog_test_a, show_progress_bar=False,
                                                            normalize_embeddings=True)

        if self.speaker_token:
            flattened_dialog_test_b_even = ["[E] [BEFORE]  " + utterance for utterance in flattened_dialog_test]
            flattened_dialog_test_b_odd = ["[O] [BEFORE]  " + utterance for utterance in flattened_dialog_test]


            flattened_history_encoded_b_even = self.model.encode(flattened_dialog_test_b_even, show_progress_bar=False,
                                                            normalize_embeddings=True)
            flattened_history_encoded_b_odd = self.model.encode(flattened_dialog_test_b_odd, show_progress_bar=False,
                                                           normalize_embeddings=True)
        else:
            flattened_dialog_test_b_even = ["[BEFORE]  " + utterance for utterance in flattened_dialog_test]
            flattened_history_encoded_b_even = self.model.encode(flattened_dialog_test_b_even, show_progress_bar=False,normalize_embeddings=True)
            flattened_history_encoded_b_odd = flattened_history_encoded_b_even



        dialog_id = [i for i, sublist in enumerate(dialogues) for item in sublist]
        utterance_id = [j for sublist in dialogues for j, item in enumerate(sublist)]

        # reshape to dialog test based on dialog_id and utterance_id because not all dialogs have the same length
        flattened_dialog_test_encoded_reshaped = []
        flattened_dialog_test_encoded_reshaped_history_even = []
        flattened_dialog_test_encoded_reshaped_history_odd = []
        for i, dialog in enumerate(dialogues):
            flattened_dialog_test_encoded_reshaped.append(
                flattened_dialog_test_encoded_a[dialog_id.index(i):dialog_id.index(i) + len(dialog)])
            flattened_dialog_test_encoded_reshaped_history_even.append(
                flattened_history_encoded_b_even[dialog_id.index(i):dialog_id.index(i) + len(dialog)])
            flattened_dialog_test_encoded_reshaped_history_odd.append(
                flattened_history_encoded_b_odd[dialog_id.index(i):dialog_id.index(i) + len(dialog)])

        for history_length in history_lengths:
            report_dict[f"H{history_length}_mean_rank"] = None

        for history_length in history_lengths:
            print(f'history length: {history_length}')
            # create history
            history = []
            next_utterances_id_same_length = []
            flattened_next_utterances_same_length = []
            iteration = 0
            for i, dialog in enumerate(dialogues):
                if len(dialog) > history_length:
                    current_history_even = flattened_dialog_test_encoded_reshaped_history_even[i][:history_length][::-1]
                    current_history_odd = flattened_dialog_test_encoded_reshaped_history_odd[i][:history_length][::-1]
                    current_history = []
                    for dist in range(len(current_history_even)):
                        if dist % 2 == 0:
                            current_history.append(current_history_odd[dist])
                        else:
                            current_history.append(current_history_even[dist])
                    current_history = current_history[::-1]
                    history.append(current_history)
                    flattened_next_utterances_same_length.append(
                        flattened_dialog_test_encoded_reshaped[i][history_length])
                    next_utterances_id_same_length.append(iteration)
                    iteration += 1

            history = np.array(history)

            self.context_embeddings_batch = torch.from_numpy(history).to(self.device)
            self.candidates_embeddings = torch.from_numpy(np.array(flattened_next_utterances_same_length)).to(self.device)

            ranks = self.sequence_model_batch_eval(next_utterances_id_same_length, batch_size=batch_size)

            report_dict[f"H{history_length}_mean_rank"] = np.mean(ranks)

        df = pd.DataFrame(report_dict, index=[0])
        return df







