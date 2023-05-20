#!/usr/bin/env python

import json
import os
from tqdm import tqdm


def main(pretraining_data_path, test_repos_path, training_output_dir, valid_output_path, n=100000):
    training, valid = [], []
    file_num_cnt = 0
    with open(pretraining_data_path, 'r') as f_pretraining, open(test_repos_path, 'r') as f_repos:
        test_repos = set([x.strip('\n').replace('/', '__')
                         for x in f_repos.readlines()])
        for line in tqdm(f_pretraining):
            cur_data = json.loads(line)
            if cur_data['proj'] in test_repos:
                valid.append(line)
            else:
                training.append(line)
            if training.__len__() > n:
                if training.__len__() > n:
                    with open(os.path.join(training_output_dir, 'train_' + str(file_num_cnt) + '.jsonl'), 'w') as f_res:
                        f_res.writelines(training)
                        training = []
                        file_num_cnt += 1

    with open(valid_output_path, 'a') as f_res:
        f_res.writelines(valid)

if __name__ == '__main__':
    dataset_path = 'Dataset/pre-training/CodeChangeNet.jsonl'
    test_repos_path = 'Dataset/fine-tuning/test_valid_repos.txt'
    training_output_dir = 'Dataset/pre-training'
    valid_output_path = 'Dataset/pre-training/valid.jsonl'
    main(dataset_path, test_repos_path,
         training_output_dir, valid_output_path)
