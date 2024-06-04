import json
import numpy as np
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--human_fname",
        type=str,
        default="outputs_human",
        help="File name of human code detection results",
    )
    parser.add_argument(
        "--machine_fname",
        type=str,
        default="outputs",
        help="File name of machine code detection results",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=150,
    )
    return parser.parse_args()

def read_file(filename):
    with open(filename) as f:
        return json.load(f)

def calculate_f1_score(tp, fp, fn):
    try:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    except:
        return -100

def f1_under_fpr(pos_data, neg_data,fpr):
    pos_data=np.array(pos_data)
    neg_data=np.array(neg_data)
    neg_data=np.sort(neg_data)
    threshold=0
    for i in range(len(neg_data)):
        threshold=neg_data[i]
        tp = np.sum(pos_data > threshold)
        fp = np.sum(neg_data > threshold)
        fn = np.sum(pos_data <= threshold)
        if fp/len(neg_data) <=fpr:
            break
    print('threshold',threshold)
    tp = np.sum(pos_data > threshold)
    fp = np.sum(neg_data > threshold)
    fn = np.sum(pos_data <= threshold)
    f1_score = calculate_f1_score(tp, fp, fn)
    print('fpr:',fp/len(neg_data))
    return tp/(tp+fn),f1_score

def find_best_threshold(pos_data, neg_data):
    thresholds = np.arange(-10, 20, 0.1)
    best_f1 = 0
    best_threshold = None

    for threshold in thresholds:
        tp = np.sum(pos_data > threshold)
        fp = np.sum(neg_data > threshold)
        fn = np.sum(pos_data <= threshold)
        f1_score = calculate_f1_score(tp, fp, fn)

        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold

    return best_threshold, best_f1

def f1(pos_data, neg_data,threshold):
    pos_data=np.array(pos_data)
    neg_data=np.array(neg_data)
    tp = np.sum(pos_data > threshold)
    fp = np.sum(neg_data > threshold)
    fn = np.sum(pos_data <= threshold)
    f1_score = calculate_f1_score(tp, fp, fn)
    return tp/(tp+fn),f1_score


def main():
    args = parse_args()

    human_results = read_file(args.human_fname)
    wllm_human = human_results["wllm_detection_results"]
    sweet_human = human_results["sweet_detection_results"]
    ewd_human = human_results["ewd_detection_results"]
    machine_results = read_file(args.machine_fname)
    wllm_machine = machine_results["wllm_detection_results"]
    sweet_machine = machine_results["sweet_detection_results"]
    ewd_machine = machine_results["ewd_detection_results"]
    wllm_human_final,sweet_human_final,ewd_human_final=[],[],[]
    wllm_machine_final,sweet_machine_final,ewd_machine_final=[],[],[]
    for human,i0,i1,machine,j0,j1 in zip(wllm_human,sweet_human,ewd_human,wllm_machine,sweet_machine,ewd_machine):
        if human["num_tokens_scored"]>=args.min_length and machine['num_tokens_scored']>=args.min_length:
            wllm_human_final.append(human)
            sweet_human_final.append(i0)
            ewd_human_final.append(i1)
            wllm_machine_final.append(machine)
            sweet_machine_final.append(j0)
            ewd_machine_final.append(j1)
    print('Original******************************')
    human_z = [r['z_score'] for r in wllm_human_final]
    machine_z = [r['z_score'] for r in wllm_machine_final]
    print(len(human_z),len(machine_z))

    print('TPR (FPR = 0%)') 
    tpr_value0,f1 = f1_under_fpr(machine_z, human_z, 0.0)
    print(tpr_value0)
    print(f1)
    print('TPR (FPR = 1%)') 
    tpr_value0,f1 = f1_under_fpr(machine_z, human_z, 0.01)
    print(tpr_value0)
    print(f1)
    print('TPR (FPR = 5%)') 
    tpr_value0,f1 = f1_under_fpr(machine_z, human_z, 0.05)
    print(tpr_value0)
    print(f1)

    best_threshold, best_f1 = find_best_threshold(np.array(machine_z), np.array(human_z))
    print('best f1:',best_f1)

    print('Sweet******************************')
    human_z = [r['z_score'] for r in sweet_human_final]
    machine_z = [r['z_score'] for r in sweet_machine_final]

    print('TPR (FPR = 0%)') 
    tpr_value0,f1 = f1_under_fpr(machine_z, human_z, 0.0)
    print(tpr_value0)
    print(f1)

    print('TPR (FPR = 1%)') 
    tpr_value0,f1 = f1_under_fpr(machine_z, human_z, 0.01)
    print(tpr_value0)
    print(f1)

    print('TPR (FPR = 5%)') 
    tpr_value0,f1 = f1_under_fpr(machine_z, human_z, 0.05)
    print(tpr_value0)
    print(f1)

    best_threshold, best_f1 = find_best_threshold(np.array(machine_z), np.array(human_z))
    print('best f1:',best_f1)

    print('Ours******************************')
    human_z = [r['z_score'] for r in ewd_human_final]
    machine_z = [r['z_score'] for r in ewd_machine_final]

    print('TPR (FPR = 0%)') 
    tpr_value0,f1 = f1_under_fpr(machine_z, human_z, 0.0)
    print(tpr_value0)
    print(f1)

    print('TPR (FPR = 1%)') 
    tpr_value0,f1 = f1_under_fpr(machine_z, human_z, 0.01)
    print(tpr_value0)
    print(f1)

    print('TPR (FPR = 5%)') 
    tpr_value0,f1 = f1_under_fpr(machine_z, human_z, 0.05)
    print(tpr_value0)
    print(f1)

    best_threshold, best_f1 = find_best_threshold(np.array(machine_z), np.array(human_z))
    print('best f1:',best_f1)
    

if __name__ == "__main__":
    main()