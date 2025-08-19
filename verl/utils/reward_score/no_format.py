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

from mathruler.grader import extract_boxed_content, grade_answer

def last_boxed_only_string(string: str) -> str:
        """
        Extracts the last \boxed{...} or \boxed ... expression,
        and returns it only if the content is a single lowercase letter.
        """
        idx = string.rfind("\\boxed")
        if "\\boxed " in string:
            # Support for \boxed a format (space-based)
            boxed_content = string.split("\\boxed ")[-1].split("$")[0].strip()
            if re.fullmatch(r"[a-z]", boxed_content):
                return "\\boxed " + boxed_content
            else:
                return None

        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        # Match \boxed{...} or \fbox{...}
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            elif string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            return None

        boxed_expr = string[idx:right_brace_idx + 1]
        match = re.fullmatch(r'\\(?:boxed|fbox)\{([a-z])\}', boxed_expr)
        return boxed_expr if match else None

def format_reward(predict_str: str) -> float:
    # pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    # pattern = re.compile(r".*\\boxed\{.*\}.*", re.DOTALL)
    # match_result = re.fullmatch(pattern, predict_str)
    # return 1.0 if match_result else 0.0

    predict_str = predict_str.lower()
    boxed = last_boxed_only_string(predict_str)
    reward = 1.0 if boxed is not None else 0.0
    return reward


def last_boxed_only_string_acc(string: str) -> str:
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

def acc_reward(predict_str: str, ground_truth: str) -> float:
    # answer = extract_boxed_content(predict_str)
    # return 1.0 if grade_answer(answer, ground_truth) else 0.0
    boxed_answer = last_boxed_only_string_acc(predict_str)
    if boxed_answer:
        try:
            boxed_answer = boxed_answer.replace("\\boxed{","").replace("}","").replace('\\text{','').replace('}','')[0]
            if boxed_answer.isalpha():
                if boxed_answer.islower():
                    boxed_answer = boxed_answer.upper()
                    boxed_answer = boxed_answer.strip()
        except:
            boxed_answer = None    
    else:
        boxed_answer = None
    
    ground_truth = ground_truth.strip()
    reward = 1.0 if boxed_answer == ground_truth else 0.0
    return reward


def compute_score(predict_str: str, ground_truth: str) -> float:
    # return 0.9 * acc_reward(predict_str, ground_truth) + 0.1 * format_reward(predict_str)
    return acc_reward(predict_str, ground_truth)

def compute_score_formatE(predict_str: str, ground_truth: str) -> float:
    return 0.5 * acc_reward(predict_str, ground_truth) + 0.5 * format_reward(predict_str)
