# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

# from mathruler.grader import extract_boxed_content, grade_answer
import numpy as np
# from word2number import w2n

def to_float(pred):
    try:
        return float(pred)
    except (ValueError, TypeError):
        try:
            return float(str(pred))
        except Exception:
            return None

def last_boxed_only_string(string: str) -> str:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    return retval

def format_reward(predict_str: str) -> float:

    predict_str = predict_str.lower()
    boxed = last_boxed_only_string(predict_str)
    reward = 1.0 if boxed is not None else 0.0
    return reward


def extract_option_final_answer(text):
    patterns = [
        r"Final answer\s*[:：]?\s*([A-Z])\b",                 # Final answer: A / Final answer:A
        r"Answer\s+is\s*[:：]?\s*([A-Z])\b",                  # Answer is: B
        r"The correct option is\s*([A-Z])\b",                # correct option is F
        r"So the answer is\s*([A-Z])\b",                     # so the answer is G
        r"Option\s*([A-Z])\b",                               # Option J
        r"^\s*([A-Z])\s*[:\-]",                              # H: some reason
        r"^\s*([A-Z])\s*$",                                  # only line is "K"
        r"=>\s*([A-Z])\b",                                   # => M
        r"\(([A-Z])\)",                                      # (N)
        r"\[([A-Z])\]",                                      # [Q]
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return None

def extract_option_pred(content: str):
    boxed_answer = last_boxed_only_string(content)
    if boxed_answer:
        try:
            boxed_answer = boxed_answer.replace("\\boxed{", "").replace("}", "").replace('\\text{','').replace('}','')
            if boxed_answer.isalpha():
                boxed_answer = boxed_answer.upper()
            return boxed_answer.strip()
        except:
            return None
    else:
        final_answer = extract_option_final_answer(content)
        # print("final_answer:", final_answer)
        if final_answer:
            return final_answer.strip().upper()
    return None

def abs_dist_norm(pred, target):
    return abs(pred - target) / target


def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

def number_fuzzy_matching(pred: str) -> str:
    if not pred:
        return ""
    import re
    m = re.search(r"[+-]?\d+(?:\.\d+)?", pred)
    return m.group(0) if m else pred

def acc_reward(predict_str: str, ground_truth: str) -> float:
    reward = 0.0
    try:
        # print("extract_option_pred(predict_str):", extract_option_pred(predict_str))
        reward = mean_relative_accuracy(to_float(number_fuzzy_matching(extract_option_pred(predict_str))), to_float(ground_truth), start=.5, end=.95, interval=.05)
    except Exception as e:
        # print("e:", e)
        reward = 0.0
    return reward


def compute_score(predict_str: str, ground_truth: str) -> float:
    return 0.9 * acc_reward(predict_str, ground_truth) + 0.1 * format_reward(predict_str)

def compute_score_format73(predict_str: str, ground_truth: str) -> float:
    return 0.7 * acc_reward(predict_str, ground_truth) + 0.3 * format_reward(predict_str)

def compute_score_format91(predict_str: str, ground_truth: str) -> float:
    return 0.9 * acc_reward(predict_str, ground_truth) + 0.1 * format_reward(predict_str)

def compute_score_formatE(predict_str: str, ground_truth: str) -> float:
    return 0.5 * acc_reward(predict_str, ground_truth) + 0.5 * format_reward(predict_str)

def compute_score_no_format(predict_str: str, ground_truth: str) -> float:
    return acc_reward(predict_str, ground_truth)

# ========== 测试示例 ==========
if __name__ == "__main__":
    test_case1 = '''
    2.0
    mimihuhu
    \\boxed{0.0}
    '''
    ground_truth_1 = "2.0"
    
    test_case2 = '''
    2.0
    mimihuhu
    \\boxed{the answer is 1.5}
    '''
    ground_truth_2 = "2.0"

    test_case3 = '''
    2.0
    mimihuhu
    the answer is 1.5
    '''
    ground_truth_3 = "1.5"

    print(acc_reward(test_case1, ground_truth_1))
    print(acc_reward(test_case2, ground_truth_2))
    print(acc_reward(test_case3, ground_truth_3))