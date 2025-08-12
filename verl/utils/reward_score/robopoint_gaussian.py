import re
import ast
from typing import List, Tuple
import math

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
    """从预测字符串中提取坐标点"""
    try:
        if predict_str.strip().startswith("[(") and predict_str.strip().endswith(")]"):
            return ast.literal_eval(predict_str)
        else:
            coord_pattern = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")
            matches = coord_pattern.findall(predict_str)
            return [(float(x), float(y)) for x, y in matches]
    except (ValueError, SyntaxError):
        return []

def parse_gt_box(ground_truth: str) -> List[Tuple[float, float]]:
    """解析 GT 框的四个角点坐标"""
    try:
        if isinstance(ground_truth, str):
            if ground_truth.strip().startswith("[(") and ground_truth.strip().endswith(")]"):
                return ast.literal_eval(ground_truth)
            else:
                coord_pattern = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")
                matches = coord_pattern.findall(ground_truth)
                return [(float(x), float(y)) for x, y in matches]
        else:
            return ground_truth
    except (ValueError, SyntaxError):
        return []

def calculate_box_center_and_variance(
    box_points: List[Tuple[float, float]], 
    alpha: float = 0.5
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    计算 GT 框的中心和自适应方差
    Args:
        box_points: GT 框的四个角点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        alpha: 缩放因子，控制高斯分布的范围
    Returns:
        (cx, cy): 框中心坐标
        (sigma_x_sq, sigma_y_sq): 方差（σ²）
    """
    if len(box_points) < 4:
        return (0.0, 0.0), (1.0, 1.0)  # 默认值
    
    x_coords = [p[0] for p in box_points]
    y_coords = [p[1] for p in box_points]
    
    # 计算框的中心
    cx = (min(x_coords) + max(x_coords)) / 2
    cy = (min(y_coords) + max(y_coords)) / 2
    
    # 计算自适应标准差（σ = α·width 或 α·height）
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    sigma_x = alpha * width if width > 0 else 1.0
    sigma_y = alpha * height if height > 0 else 1.0
    
    return (cx, cy), (sigma_x**2, sigma_y**2)

def gaussian_point_reward(
    point: Tuple[float, float], 
    center: Tuple[float, float], 
    variance: Tuple[float, float]
) -> float:
    """
    计算单个点的高斯奖励
    Args:
        point: 待评估的点 (px, py)
        center: GT 框中心 (cx, cy)
        variance: 方差 (σx², σy²)
    Returns:
        reward: exp(-0.5 * ((px-cx)²/σx² + (py-cy)²/σy²))
    """
    sigma_x_sq, sigma_y_sq = variance
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    
    exponent = -0.5 * ((dx**2 / sigma_x_sq) + (dy**2 / sigma_y_sq))
    return math.exp(exponent)

def accuracy_reward(
    predict_str: str, 
    ground_truth: str, 
    alpha: float = 0.5
) -> float:
    """
    计算所有预测点相对于 GT 框的平均 Gaussian Reward
    Args:
        predict_str: 预测点列表，如 "[(x1,y1), (x2,y2), ...]"
        ground_truth: GT 框的四个角点，如 "[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]"
        alpha: 缩放因子（默认 0.5）
    Returns:
        平均 reward（0~1）
    """
    # 解析输入
    pred_points = extract_coordinates_from_answer(predict_str)
    gt_box_points = parse_gt_box(ground_truth)
    
    if not pred_points or len(gt_box_points) < 4:
        return 0.0
    
    # 计算 GT 框的高斯分布参数
    (cx, cy), variance = calculate_box_center_and_variance(gt_box_points, alpha)
    
    # 计算每个预测点的 reward
    total_reward = 0.0
    for point in pred_points:
        reward = gaussian_point_reward(point, (cx, cy), variance)
        total_reward += reward
    
    # 返回平均 reward
    return total_reward / len(pred_points)


def compute_score(predict_str: str, ground_truth: str) -> float:
    format_reward_1 = format_reward_think_answer(predict_str)
    format_reward_2 = format_reward_coordinates(predict_str)
    acc_reward = accuracy_reward(predict_str, ground_truth)
    
    return 0.2*format_reward_1 + 0.3*format_reward_2 + 0.5*acc_reward


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
    [(0.65, 0.4), (0.7, 0.35), (0.68, 0.5)]
    </answer>
    """
    test_ground_truth = "[(0.523, 0.517), (0.631, 0.517), (0.631, 0.542), (0.523, 0.542)]"
    
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