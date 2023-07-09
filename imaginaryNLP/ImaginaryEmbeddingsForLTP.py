import glob
import itertools
import logging
import math
import os
import re
from typing import Optional, List
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from imaginaryNLP.ImaginaryEmbeddings import ImaginaryEmbeddings


class ImaginaryEmbeddingsForLTP(ImaginaryEmbeddings):
    def __init__(self, model_name_or_path: Optional[str] = None,
                 speaker_token: bool = True,
                 ):
        if model_name_or_path is None:
            if speaker_token:
                model_name_or_path = 'Justus-Jonas/Imaginary-Embeddings-SpeakerTokens'
            else:
                model_name_or_path = 'Justus-Jonas/Imaginary-Embeddings-Classic'
        super().__init__(model_name_or_path, speaker_token)

        self.goals = []
        self.goals_embeddings_before = []
        self.goals_embeddings_after = []
        self.context_embeddings = None


    def add_goal(self, goal: str):
        """
        Add a goal to the set of goals
        :param goal: The goal
        """
        self.goals.append(goal)
        self.goals_embeddings_after.append(self.model.encode("[AFTER]  " + goal))
        if self.speaker_token:
            self.goals_embeddings_before.append(self.model.encode("[E] " + "[BEFORE]  " + goal))
        else:
            self.goals_embeddings_before.append(self.model.encode("[BEFORE]  " + goal))


    def remove_goal(self, goal: str):
        """
        Remove a goal from the set of goals
        :param goal: The goal
        """
        index = self.goals.index(goal)
        self.goals.pop(index)
        self.goals_embeddings_after.pop(index)
        self.goals_embeddings_before.pop(index)

    def create_context(self, context: List[str], goal_distance_odd: bool = True):
        """
        allows to add context for greedy curving or imaginary embedding chains with curved context


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
                if (i % 2 == 0 and goal_distance_odd) or (i % 2 == 1 and not goal_distance_odd):
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


    def add_utterance_to_context(self, utterance: str, precompute_top_p = None):
        """
        allows to add context for greedy curving or imaginary embedding chains with curved context

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

    def imaginary_embedding_chains(self):
        """
        This function computes the best order of goals using imaginary embedding chains.
        :param return_rank: if True, the rank of the first permutation (true order in eval mode) is returned
        :return: best order of goals
        """
        if self.goals_embeddings_before is None or self.goals_embeddings_after is None:
            raise ValueError("No goals found. Please add goals first.")

        embedding_vectors_before = torch.from_numpy(np.array(self.goals_embeddings_before)).to(self.device)
        embedding_vectors_after = torch.from_numpy(np.array(self.goals_embeddings_after)).to(self.device)

        # generate all permutations of the goals
        permuted_indices = torch.tensor(list(itertools.permutations(range(len(self.goals)))))

        # advanced indexing to get the embeddings in the order specified by permuted_indices
        ordered_before = embedding_vectors_before[permuted_indices[:, :-1]]
        ordered_after = embedding_vectors_after[permuted_indices[:, 1:]]

        # Compute the dot product between corresponding vectors
        cos_sim = (ordered_before * ordered_after).sum(dim=-1)

        # Sum the cosine similarity along the sequence dimension to get a single score for each permutation
        scores = cos_sim.sum(dim=1)

        # Find the permutation with the highest score
        best_permutation_index = scores.argmax().item()

        # Return the best order
        best_order = permuted_indices[best_permutation_index].tolist()

        return [self.goals[i] for i in best_order]

    def greedy_curving(self):
        """
        This function computes the best order of goals using greedy curving.
        :param return_rank: if True, the rank of the first permutation (true order in eval mode) is returned.
        :return: best goal or rank of first goal
        """
        if self.context_embeddings is None:
            raise ValueError("No context found. Please add context first.")
        if self.goals_embeddings_before is None or self.goals_embeddings_after is None:
            raise ValueError("No goals found. Please add goals first.")

        embedding_vectors_after = torch.tensor(self.goals_embeddings_after).to(self.device)
        scores = torch.matmul(self.context_embeddings, embedding_vectors_after.T)

        # Sum scores across context
        scores = scores.sum(dim=0)

        best_order_idx = torch.argmax(scores).item()

        return self.goals[best_order_idx]

    def imaginary_embedding_chains_with_curving(self):
        """
        This function computes the best order of goals using imaginary embedding chains combined with curving.

        :param return_rank: if True, the rank of the first permutation (true order in eval mode) is returned
        :return: best order of goals
        """
        if self.context_embeddings is None:
            raise ValueError("No context found. Please add context first.")
        if self.goals_embeddings_before is None or self.goals_embeddings_after is None:
            raise ValueError("No goals found. Please add goals first.")

        embedding_vectors_before = torch.tensor(self.goals_embeddings_before).to(self.device)
        embedding_vectors_after = torch.tensor(self.goals_embeddings_after).to(self.device)

        # calculate the scores for the context
        scores_curving = torch.matmul(self.context_embeddings, embedding_vectors_after.T)
        scores_curving = scores_curving.sum(dim=0)

        # generate all permutations of the goals
        permuted_indices = torch.tensor(list(itertools.permutations(range(len(self.goals)))))

        # advanced indexing to get the embeddings in the order specified by permuted_indices
        ordered_before = embedding_vectors_before[permuted_indices[:, :-1]]
        ordered_after = embedding_vectors_after[permuted_indices[:, 1:]]

        # Compute the dot product between corresponding vectors
        cos_sim = (ordered_before * ordered_after).sum(dim=-1)

        # Sum the cosine similarity along the sequence dimension to get a single score for each permutation
        scores = cos_sim.sum(dim=1)

        # Compute the curving scores for each permutation
        l = len(self.goals)
        remaining_scores = (torch.arange(1, l).view(-1, 1) / (l - 1)).to(self.device) * scores_curving[permuted_indices[:, 1:]].T
        remaining_scores = remaining_scores.sum(dim=0)
        curving_scores = scores_curving[permuted_indices[:, 0]] - remaining_scores

        # Add the curving scores to the chain scores
        scores += curving_scores

        best_permutation_index = scores.argmax().item()

        best_order = permuted_indices[best_permutation_index].tolist()

        return [self.goals[i] for i in best_order]


class EvalImaginaryEmbeddingsForLTP(ImaginaryEmbeddingsForLTP):

    def __init__(self, model, speaker_token: bool = True):
        super().__init__(model, speaker_token)

    def eval_imaginary_embedding_chains(self):
        """
        This function computes the best order of goals using imaginary embedding chains.
        :param return_rank: if True, the rank of the first permutation (true order in eval mode) is returned
        :return: best order of goals
        """
        if self.goals_embeddings_before is None or self.goals_embeddings_after is None:
            raise ValueError("No goals found. Please add goals first.")

        embedding_vectors_before = torch.from_numpy(np.array(self.goals_embeddings_before)).to(self.device)
        embedding_vectors_after = torch.from_numpy(np.array(self.goals_embeddings_after)).to(self.device)

        # generate all permutations of the goals
        permuted_indices = torch.tensor(list(itertools.permutations(range(len(self.goals)))))

        # advanced indexing to get the embeddings in the order specified by permuted_indices
        ordered_before = embedding_vectors_before[permuted_indices[:, :-1]]
        ordered_after = embedding_vectors_after[permuted_indices[:, 1:]]

        # Compute the dot product between corresponding vectors
        cos_sim = (ordered_before * ordered_after).sum(dim=-1)

        # Sum the cosine similarity along the sequence dimension to get a single score for each permutation
        scores = cos_sim.sum(dim=1)

        sorted_scores = scores.sort(descending=True).values
        rank_total = np.where(sorted_scores.cpu().numpy() == scores[0].cpu().numpy())[0][0] + 1


        sorted_scores_partial_order = scores[:-1].sort(descending=True).values

        partial_order_rank = np.where(sorted_scores_partial_order.cpu().numpy() == scores[0].cpu().numpy())[0][0] + 1

        reverse_order_rank = 2 if scores[-1] > scores[0] else 1

        return rank_total, partial_order_rank, reverse_order_rank


    def eval_greedy_curving(self):
        """
        This function computes the best order of goals using greedy curving.
        :param return_rank: if True, the rank of the first permutation (true order in eval mode) is returned.
        :return: best goal or rank of first goal
        """
        if self.context_embeddings is None:
            raise ValueError("No context found. Please add context first.")
        if self.goals_embeddings_before is None or self.goals_embeddings_after is None:
            raise ValueError("No goals found. Please add goals first.")

        embedding_vectors_after = torch.tensor(self.goals_embeddings_after).to(self.device)
        scores = torch.matmul(self.context_embeddings, embedding_vectors_after.T)

        # Sum scores across context
        scores = scores.sum(dim=0)

        sorted_scores = scores.sort(descending=True).values

        rank_total = np.where(sorted_scores.cpu().numpy() == scores[0].cpu().numpy())[0][0] + 1

        return rank_total

    def eval_imaginary_embedding_chains_with_curving(self):
        """
        This function computes the best order of goals using imaginary embedding chains combined with curving.

        :param return_rank: if True, the rank of the first permutation (true order in eval mode) is returned
        :return: best order of goals
        """
        if self.context_embeddings is None:
            raise ValueError("No context found. Please add context first.")
        if self.goals_embeddings_before is None or self.goals_embeddings_after is None:
            raise ValueError("No goals found. Please add goals first.")

        embedding_vectors_before = torch.tensor(self.goals_embeddings_before).to(self.device)
        embedding_vectors_after = torch.tensor(self.goals_embeddings_after).to(self.device)

        # calculate the scores for the context
        scores_curving = torch.matmul(self.context_embeddings, embedding_vectors_after.T)
        scores_curving = scores_curving.sum(dim=0)

        # generate all permutations of the goals
        permuted_indices = torch.tensor(list(itertools.permutations(range(len(self.goals)))))

        # advanced indexing to get the embeddings in the order specified by permuted_indices
        ordered_before = embedding_vectors_before[permuted_indices[:, :-1]]
        ordered_after = embedding_vectors_after[permuted_indices[:, 1:]]

        # Compute the dot product between corresponding vectors
        cos_sim = (ordered_before * ordered_after).sum(dim=-1)

        # Sum the cosine similarity along the sequence dimension to get a single score for each permutation
        scores = cos_sim.sum(dim=1)

        # Compute the curving scores for each permutation
        l = len(self.goals)
        remaining_scores = (torch.arange(1, l).view(-1, 1) / (l - 1)).to(self.device) * scores_curving[
            permuted_indices[:, 1:]].T
        remaining_scores = remaining_scores.sum(dim=0)
        curving_scores = scores_curving[permuted_indices[:, 0]] - remaining_scores

        # Add the curving scores to the chain scores
        scores += curving_scores

        sorted_scores = scores.sort(descending=True).values
        rank_total = np.where(sorted_scores.cpu().numpy() == scores[0].cpu().numpy())[0][0] + 1

        sorted_scores_partial_order = scores[:-1].sort(descending=True).values

        partial_order_rank = np.where(sorted_scores_partial_order.cpu().numpy() == scores[0].cpu().numpy())[0][0] + 1

        reverse_order_rank = 2 if scores[-1] > scores[0] else 1

        return rank_total, partial_order_rank, reverse_order_rank

    def create_ltp_dataset(self,
                           dialogues: List[List[str]] = None,
                           output_dir: str = 'data',
                           history_lengths: List[int] = [2, 4],
                           goal_distances: List[int] = [2, 4],  # currently only even numbers are supported
                           goal_in_distances: List[int] = [0, 1, 2, 3],
                           num_goals: int = 3):
        """
        creates a dataset for the LTP evaluation
        :param dialogues: dataset for the LTP evaluation
        :param history_lengths: history lengths to be considered
        :param goal_distances: goal distances to be considered
        :param goal_in_distances: goal in distances to be considered
        :param num_goals: number of goals to be considered
        :return:
        """
        avg_length = np.mean([len(x) for x in dialogues])
        logging.info(f'avg_length: {avg_length}')

        # create dir ltp_dataset
        if not os.path.exists(output_dir + '/ltp_dataset'):
            os.makedirs(output_dir + '/ltp_dataset')

        # create a combination of all parameters
        combinations = list(itertools.product(history_lengths, goal_distances, goal_in_distances))
        for history_length, goal_distance, goal_in_distance in tqdm(combinations):
            df_dict = {}
            df_dict['Dialogue'] = []
            df_dict['History'] = []

            for i in range(num_goals):
                df_dict[f'Goal{i}'] = []

            for i in range(len(dialogues)):
                dialogue = dialogues[i]

                if len(dialogue) >= history_length + (goal_distance * num_goals) + goal_in_distance + 1:
                    df_dict['History'].append(dialogue[:history_length])
                    df_dict['Dialogue'].append(dialogue[:history_length + (goal_distance * 3) + goal_in_distance])

                    for j in range(num_goals):
                        df_dict[f'Goal{j}'].append(dialogue[history_length + (goal_distance * j) + goal_in_distance])

            df = pd.DataFrame(df_dict)
            for i in range(num_goals):
                df[f'Goal{i}'] = df[f'Goal{i}'].astype(str)

            df.to_parquet(f'{output_dir}/ltp_dataset/LTP_G{str(num_goals)}_H{str(history_length)}_D{str(goal_distance)}_I{str(goal_in_distance)}.parquet')
            logging.info(f'created dataset for G{str(num_goals)}_H{str(history_length)}_D{str(goal_distance)}_I{str(goal_in_distance)}')



    def evaluate_file(self, file_path: str, num_goals: int, goal_distance: int, goal_in_distance: int):

        before_prefix = "[BEFORE]  "

        df = pd.read_parquet(file_path)

        df['IEC_CU_rank'] = None
        df['IEC_rank'] = None
        df['GC_rank'] = None
        df['IEC_CU_rank_partial'] = None
        df['IEC_rank_partial'] = None
        df['IEC_CU_rank_reverse'] = None
        df['IEC_rank_reverse'] = None


        if goal_distance % 2 != 0:
            warnings.warn(f'goal_distance is not even: {goal_distance}! '
                          f'Odd distances are not yet handled. This might lead to unexpected results.')

        for index, row in tqdm(df.iterrows(), total=len(df)):
            history = row['History']

            if self.speaker_token:
                history_odd = ["[O] " + before_prefix + his for his in history]
                history_even = ["[E] " + before_prefix + his for his in history]
                history_odd = self.model.encode(history_odd, normalize_embeddings=True)
                history_even = self.model.encode(history_even, normalize_embeddings=True)
            else:
                history_odd = [before_prefix + his for his in history]
                history_odd = self.model.encode(history_odd, normalize_embeddings=True)
                history_even = history_odd

            history_encoded = []
            history_odd = history_odd[::-1]
            history_even = history_even[::-1]
            for i in range(len(history)):
                if (i % 2 == 0 and goal_in_distance % 2 == 1) or (i % 2 == 1 and goal_in_distance % 2 == 0):
                    history_encoded.append(history_odd[i])
                else:
                    history_encoded.append(history_even[i])
            self.context_embeddings = torch.from_numpy(np.array(history_encoded)).to(self.device)

            goals = [row[f'Goal{i}'] for i in range(num_goals)]
            self.goals = goals
            if self.speaker_token:
                goals_before = ["[E] [BEFORE] " + goal for goal in goals]
            else:
                goals_before = ["[BEFORE] " + goal for goal in goals]
            goals_after = ["[AFTER] " + goal for goal in goals]

            goals_before = self.model.encode(goals_before, normalize_embeddings=True)
            goals_after = self.model.encode(goals_after, normalize_embeddings=True)
            self.goals_embeddings_before = goals_before.tolist()
            self.goals_embeddings_after = goals_after.tolist()

            # calculate ranks
            df.at[index, 'IEC_CU_rank'], \
            df.at[index, 'IEC_CU_rank_partial'], \
            df.at[index, 'IEC_CU_rank_reverse'] = self.eval_imaginary_embedding_chains_with_curving()

            df.at[index, 'IEC_rank'], \
            df.at[index, 'IEC_rank_partial'], \
            df.at[index, 'IEC_rank_reverse'] = self.eval_imaginary_embedding_chains()

            df.at[index, 'GC_rank'] = self.eval_greedy_curving()

        return df

    def evaluate_ltp_dataset(self, ltp_dataset_dir = 'data', top_max = 4, output_file_name = None):
        """
        Evaluate the created LTP dataset
        :param ltp_dataset_dir:
        :param top_max:
        :return:
        """
        files = glob.glob(ltp_dataset_dir + '/ltp_dataset/*.parquet')

        # get number of goals
        num_goals = int(re.search(r'(?<=_G)\d+', files[0]).group(0))

        num_of_combinations = math.factorial(num_goals)

        top_max = min(top_max, num_of_combinations-1)

        # create dir for results
        if not os.path.exists(ltp_dataset_dir + '/ltp_results'):
            os.makedirs(ltp_dataset_dir + '/ltp_results')

        report_dict = {}
        report_dict['history_length'] = []
        report_dict['goal_distance'] = []
        report_dict['goal_in_distance'] = []
        report_dict['IEC_CU_average_rank'] = []
        report_dict['IEC_average_rank'] = []
        report_dict['GC_average_rank'] = []
        report_dict['IEC_CU_top_1_reverse'] = []
        report_dict['IEC_top_1_reverse'] = []

        for i in range(top_max):
            report_dict[f'IEC_CU_top_{i+1}_partial'] = []
            report_dict[f'IEC_top_{i+1}_partial'] = []
            report_dict[f'GC_top_{i+1}'] = []


        for file in tqdm(files):
            file_name = file.split('/')[-1]

            num_goals = int(re.search(r'(?<=_G)\d+', file_name).group(0))
            goal_distance = int(re.search(r'(?<=_D)\d+', file_name).group(0))
            goal_in_distance = int(re.search(r'(?<=_I)\d+', file_name).group(0))
            history_length = int(re.search(r'(?<=_H)\d+', file_name).group(0))

            df = self.evaluate_file(file, num_goals, goal_distance, goal_in_distance)

            if len(df) == 0:
                logging.info(f"No dialogues found for {file_name}! Skipping...")
                continue

            file_name = file.split('/')[-1]
            df.to_parquet(f'{ltp_dataset_dir}/ltp_results/{file_name}')

            # add to report
            report_dict['history_length'].append(history_length)
            report_dict['goal_distance'].append(goal_distance)
            report_dict['goal_in_distance'].append(goal_in_distance)

            report_dict['IEC_CU_average_rank'].append(df['IEC_CU_rank'].mean())
            report_dict['IEC_average_rank'].append(df['IEC_rank'].mean())
            report_dict['GC_average_rank'].append(df['GC_rank'].mean())

            report_dict[f'IEC_CU_top_1_reverse'].append((df[df['IEC_CU_rank_reverse'] <= 1].shape[0]) / len(df))
            report_dict[f'IEC_top_1_reverse'].append((df[df['IEC_rank_reverse'] <= 1].shape[0]) / len(df))

            for i in range(top_max):
                report_dict[f'IEC_CU_top_{i+1}_partial'].append((df[df['IEC_CU_rank_partial'] <= i+1].shape[0]) / len(df))
                report_dict[f'IEC_top_{i+1}_partial'].append((df[df['IEC_CU_rank_partial'] <= i+1].shape[0]) / len(df))
                report_dict[f'GC_top_{i+1}'].append((df[df['GC_rank'] <= i+1].shape[0]) / len(df))

        report_df = pd.DataFrame(report_dict)
        if output_file_name:
            report_df.to_csv(f'{ltp_dataset_dir}/ltp_results/{output_file_name}.csv', index=False)
        else:
            report_df.to_csv(f'{ltp_dataset_dir}/ltp_results/ltp_report.csv', index=False)
        return report_df








