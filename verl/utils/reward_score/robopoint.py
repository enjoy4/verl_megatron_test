import re
import ast
from typing import List, Tuple

def format_reward_think_answer(predict_str: str) -> float:
    pattern = re.compile(r".*<think>.*</think>.*<answer>.*</answer>.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def format_reward_coordinates(predict_str: str) -> float:
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    answer_match = answer_pattern.search(predict_str)
    
    if not answer_match:
        return 0.0
    
    answer_content = answer_match.group(1).strip()
    
    strict_coord_pattern = re.compile(r"^\s*\[\s*((\(\s*[\d.]+\s*,\s*[\d.]+\s*\)\s*,?\s*)+)\s*\]\s*$")
    
    match_result = strict_coord_pattern.fullmatch(answer_content)
    
    return 1.0 if match_result else 0.0


def extract_coordinates_from_answer(predict_str: str) -> List[Tuple[float, float]]:
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    answer_match = answer_pattern.search(predict_str)
    
    if not answer_match:
        return []
    
    answer_content = answer_match.group(1).strip()
    coord_pattern = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")
    matches = coord_pattern.findall(answer_content)
    
    coordinates = []
    for x_str, y_str in matches:
        try:
            coordinates.append((float(x_str), float(y_str)))
        except ValueError:
            continue
    
    return coordinates


def point_in_rectangle(point: Tuple[float, float], rect_points: List[Tuple[float, float]]) -> bool:
    x, y = point
    x_coords = [p[0] for p in rect_points]
    y_coords = [p[1] for p in rect_points]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    return min_x <= x <= max_x and min_y <= y <= max_y


def accuracy_reward(predict_str: str, ground_truth: str) -> float:
    try:
        if isinstance(ground_truth, str):
            if ground_truth.strip().startswith("[(") and ground_truth.strip().endswith(")]"):
                ground_truth_coords = ast.literal_eval(ground_truth)
            else:
                coord_pattern = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")
                matches = coord_pattern.findall(ground_truth)
                ground_truth_coords = [(float(x), float(y)) for x, y in matches]
        else:
            ground_truth_coords = ground_truth
    except (ValueError, SyntaxError):
        return 0.0
    
    if len(ground_truth_coords) < 4:
        return 0.0
    
    predicted_coords = extract_coordinates_from_answer(predict_str)
    
    if not predicted_coords:
        return 0.0
    
    points_in_rect = 0
    for point in predicted_coords:
        if point_in_rectangle(point, ground_truth_coords):
            points_in_rect += 1
    
    accuracy_ratio = points_in_rect / len(predicted_coords)
    return 1.0 if accuracy_ratio > 0.9 else 0.0


def compute_score(predict_str: str, ground_truth: str) -> float:
    format_reward_1 = format_reward_think_answer(predict_str)
    format_reward_2 = format_reward_coordinates(predict_str)
    acc_reward = accuracy_reward(predict_str, ground_truth)
    
    return 0.2*format_reward_1 + 0.3*format_reward_2 + 0.5*acc_reward

def compute_score_no_format(predict_str: str, ground_truth: str) -> float:
    return accuracy_reward(predict_str, ground_truth)

def compute_score_format91(predict_str: str, ground_truth: str) -> float:
    format_reward_1 = format_reward_think_answer(predict_str)
    format_reward_2 = format_reward_coordinates(predict_str)
    acc_reward = accuracy_reward(predict_str, ground_truth)
    return {
        "acc": acc_reward,
        "score": 0.04*format_reward_1+0.06*format_reward_2+0.9*acc_reward
    }

if __name__ == "__main__":
    test_predict_wrong_format = """
    <think>
    哈基米
    </think>
    <answer>
    (0.7, 0.2), (0.75, 0.25), (0.72, 0.18)
    </answer>
    """
    
    test_predict_correct_format = """
    <think>
    哈基米
    </think>
    <answer>
    [(0.7, 0.2), (0.75, 0.25), (0.72, 0.18)]
    </answer>
    """
    
    test_ground_truth = "[(0.683, 0.152), (0.766, 0.152), (0.766, 0.285), (0.683, 0.285)]"
    
    print("--- 案例1: 格式不正确 (缺少方括号) ---")
    print(f"Format reward 1 (think/answer): {format_reward_think_answer(test_predict_wrong_format)}")
    print(f"Format reward 2 (coordinates format): {format_reward_coordinates(test_predict_wrong_format)}")
    print(f"Accuracy reward: {accuracy_reward(test_predict_wrong_format, test_ground_truth)}")
    print(f"Overall score: {compute_score(test_predict_wrong_format, test_ground_truth)}")
    
    print("\n" + "="*40 + "\n")
    
    print("--- 案例2: 格式正确 (包含方括号) ---")
    print(f"Format reward 1 (think/answer): {format_reward_think_answer(test_predict_correct_format)}")
    print(f"Format reward 2 (coordinates format): {format_reward_coordinates(test_predict_correct_format)}")
    print(f"Accuracy reward: {accuracy_reward(test_predict_correct_format, test_ground_truth)}")
    print(f"Overall score: {compute_score(test_predict_correct_format, test_ground_truth)}")