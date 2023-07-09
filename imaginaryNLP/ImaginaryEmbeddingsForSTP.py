import glob
import itertools
import os
import re
from enum import Enum
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from imaginaryNLP.ImaginaryEmbeddings import ImaginaryEmbeddings


class ImaginaryEmbeddingsForSTP(ImaginaryEmbeddings):
    """
    Imaginary Embeddings for Short-Term Planning
    """

    def __init__(self, model_name_or_path: Optional[str] = None,
                 speaker_token: bool = True):

        if model_name_or_path is None:
            model_name_or_path = "Justus-Jonas/Imaginary-Embeddings-SpeakerTokens-STP"
            speaker_token = True


        super().__init__(model_name_or_path, speaker_token)

    def short_term_planning(self, candidates: List[str], goal: str, planning_odd: bool = True):
        """
        Short-Term Planning approach for ranking candidates to reach a goal
        """
        if self.speaker_token:
            if planning_odd:
                prefix = "[O] [BEFORE] "
            else:
                prefix = "[E] [BEFORE]  "
        else:
            prefix = "[BEFORE]  "

        candidates_with_prefix = [prefix + candidate for candidate in candidates]
        goal = "[AFTER] " + goal

        candidates_with_prefix.append(goal)

        embeddings \
            = self.model.encode(candidates_with_prefix, convert_to_tensor=True, normalize_embeddings=True).to(self.device)

        candidates_embeddings = embeddings[:-1]
        goal_embedding = embeddings[-1]

        # dot product between goal and all candidates
        candidates_dot_product = torch.matmul(goal_embedding, candidates_embeddings.T)

        best_candidate_index = torch.argmax(candidates_dot_product).item()

        return candidates[best_candidate_index]


class EvalImaginaryEmbeddingsForSTP(ImaginaryEmbeddingsForSTP):

    def __init__ (self, model_name_or_path: Optional[str] = None,
                 speaker_token: bool = True,
                 llm_model_name_or_path: Optional[str] = None,
                 ):

        super().__init__(model_name_or_path, speaker_token)

        if llm_model_name_or_path:
            self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name_or_path)
            self.llm_model.to(self.device)
            self.llm_model.eval()


    def create_stp_dataset(self, dialogues: List[List[str]],
                           output_dir: str = 'data',
                           history_lengths: List[int] = [2, 5],
                           goal_distances: List[int] = [1,2,3,4]):
        """
        Short-Term Planning approach for ranking candidates to reach a goal
        """

        # create dir ltp_dataset
        if not os.path.exists(output_dir + '/stp_dataset'):
            os.makedirs(output_dir + '/stp_dataset')


        combinations = list(itertools.product(history_lengths, goal_distances))

        for history_length, goal_distance in combinations:
            histories = []
            candidates = []
            goals = []

            for i in tqdm(range(len(dialogues))):
                dialogue = dialogues[i]

                if len(dialogue) >= history_length + goal_distance + 1:

                    history = dialogue[:history_length]
                    candidate = dialogue[history_length]
                    goal = dialogue[history_length + goal_distance]

                    histories.append(history)
                    candidates.append(candidate)
                    goals.append(goal)

            df = pd.DataFrame({'History': histories, 'Candidate': candidates, 'Goal': goals})

            df['Candidate'] = df[f'Candidate'].astype(str)
            df['Goal'] = df[f'Goal'].astype(str)

            df.to_parquet(f'{output_dir}/stp_dataset/STP_H{str(history_length)}_D{str(goal_distance)}.parquet')




    def generate_llm_candidates(self, new_user_input_ids, top_p=0.9, num_return_sequences=100, temperture=0.8, **kwargs):
        chat_history_ids = self.llm_model.generate(new_user_input_ids,
                                                   max_length=1000,
                                                   do_sample=True,
                                                   pad_token_id=self.tokenizer.eos_token_id,
                                                   top_p=top_p,
                                                   num_return_sequences=num_return_sequences,
                                                   temperature=temperture,
                                                    **kwargs,
                                                   ).cpu().numpy()
        llm_candidates = [self.tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][i],
                                            skip_special_tokens=True) for i in range(num_return_sequences)]

        return llm_candidates


    def add_transformer_candidates(self, data_dir, top_p=0.8, num_return_sequences=100, temperture=0.8, **kwargs):
        """
        Add transformer candidates to the dataset

        Todo:
        - add optional system prompt
        - let user specify separation of utterances

        :param data_dir:
        :param top_p:
        :param num_return_sequences:
        :param temperture:
        :param kwargs:
        :return:
        """
        if not self.llm_model:
            raise ValueError('LLM model was not provided in the constructor')

        files = glob.glob(data_dir + '/stp_dataset/*.parquet')

        # create dir for results
        if not os.path.exists(data_dir + '/stp_dataset_candidates'):
            os.makedirs(data_dir + '/stp_dataset_candidates')

        for file in tqdm(files):
            df = pd.read_parquet(file)
            df['GPT_candidates'] = None

            for index, row in tqdm(df.iterrows(), total=len(df)):
                history_list = row['History']
                history = ''
                for i in range(len(history_list)):
                    history += history_list[i] + self.tokenizer.eos_token

                input_ids =  self.tokenizer.encode(history, return_tensors='pt')

                gpt_candidates = self.generate_llm_candidates(input_ids, top_p, num_return_sequences, temperture, **kwargs)
                df.at[index, 'GPT_candidates'] = gpt_candidates

            file_name = os.path.basename(file)

            df.to_parquet(data_dir + f'/stp_dataset_candidates/{file_name}_P{temperture}_T{temperture}_N{num_return_sequences}.parquet')


    def evaluate_stp_dataset(self, data_dir = 'data'):
        """
        Evaluate the created LTP dataset
        :param ltp_dataset_dir:
        :param top_max:
        :return:
        """
        files = glob.glob(data_dir + '/stp_dataset_candidates/*.parquet')

        if len(files) == 0:
            raise ValueError('No files found in stp_dataset_candidates. Did you run add_transformer_candidates first?')

        # create dir for results
        if not os.path.exists(data_dir + '/stp_results'):
            os.makedirs(data_dir + '/stp_results')

        top_5s = []
        top_10s = []
        top_25s = []
        top_50s = []
        average_ranks = []
        history_lengths = []
        goal_distances = []
        for file in tqdm(files):
            df = pd.read_parquet(file)
            df['rank'] = None
            goal_distance = int(re.search(r'(?<=_D)\d+', file).group(0))
            history_length = int(re.search(r'(?<=_H)\d+', file).group(0))


            for index, row in tqdm(df.iterrows(), total=len(df)):
                if self.speaker_token:
                    if goal_distance % 2 == 0:
                        utterances = ["[E] [BEFORE] " + utterance for utterance in row['GPT_candidates']]
                        true_utterance = "[E] [BEFORE] " + row['Candidate']
                    else:
                        utterances = ["[O] [BEFORE] " + utterance for utterance in row['GPT_candidates']]
                        true_utterance = "[0] [BEFORE] " + row['Candidate']
                else:
                    utterances = ["[BEFORE] " + utterance for utterance in row['GPT_candidates']]
                    true_utterance = "[BEFORE] " + row['Candidate']
                goal_utterance = "[AFTER] " + row['Goal']

                combined_utterances = utterances + [true_utterance] + [goal_utterance]
                combined_utterances = self.model.encode(combined_utterances, convert_to_tensor=True, normalize_embeddings=True)
                goal_embedding = combined_utterances[-1]
                candidates_embeddings = combined_utterances[:-1]

                # dot product between goal and all candidates
                scores = torch.matmul(goal_embedding, candidates_embeddings.T)

                true_utterance_score = scores[-1]
                sorted_scores = scores.sort(descending=True).values

                df.at[index, 'rank'] = (sorted_scores.cpu().numpy() == true_utterance_score.cpu().numpy()).argmax() + 1

            df.to_parquet(data_dir + f'/stp_results/{os.path.basename(file)}')
            average_ranks.append(df['rank'].mean())
            top_5s.append(df[df['rank'] <= 5].shape[0] / df.shape[0])
            top_10s.append(df[df['rank'] <= 10].shape[0] / df.shape[0])
            top_25s.append(df[df['rank'] <= 25].shape[0] / df.shape[0])
            top_50s.append(df[df['rank'] <= 50].shape[0] / df.shape[0])
            history_lengths.append(history_length)
            goal_distances.append(goal_distance)

        df = pd.DataFrame({"History Length": history_lengths, "Goal Distance": goal_distances, 'Top 5': top_5s,
                           'Top 10': top_10s, 'Top 25': top_25s, 'Top 50': top_50s, 'Average Rank': average_ranks})
        df.to_csv(f'{data_dir}/stp_results/STP_results.csv', index=False)


        return df
















