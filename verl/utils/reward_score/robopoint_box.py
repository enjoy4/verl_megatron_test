import re
import ast
from typing import List, Tuple
import math
import unittest

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


def extract_coordinates_from_answer(predict_str):
    try:
        answer_content = last_boxed_only_string(predict_str)
        if not answer_content:
            return []
        bracket_pattern = re.compile(r"\[(.*?)\]", re.DOTALL)
        bracket_match = bracket_pattern.search(answer_content)

        if bracket_match:
            coords_str = bracket_match.group(1)
            # 在 [] 内部继续匹配 (x, y)
            coord_pattern = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")
            matches = coord_pattern.findall(coords_str)
        else:
            matches = []
        
        # 类型转换和预处理
        processed = []
        for x_str, y_str in matches:
            try:
                x = round(float(x_str), 4)
                y = round(float(y_str), 4)
                processed.append((x, y))
            except ValueError:
                continue
        
        # 去重和范围校验
        seen = set()
        valid_coords = []
        for coord in processed:
            if (coord not in seen and 
                all(0 <= val <= 1 for val in coord)):
                seen.add(coord)
                valid_coords.append(coord)
        
        return valid_coords
    except Exception:
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
    if len(box_points) < 4:
        return (0.0, 0.0), (1.0, 1.0)  # 默认值
    
    x_coords = [p[0] for p in box_points]
    y_coords = [p[1] for p in box_points]
    
    cx = (min(x_coords) + max(x_coords)) / 2
    cy = (min(y_coords) + max(y_coords)) / 2
    
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
    pred_points = extract_coordinates_from_answer(predict_str)
    gt_box_points = parse_gt_box(ground_truth)
    
    if not pred_points or len(gt_box_points) < 4:
        return 0.0
    
    (cx, cy), variance = calculate_box_center_and_variance(gt_box_points, alpha)
    
    total_reward = 0.0
    for point in pred_points:
        reward = gaussian_point_reward(point, (cx, cy), variance)
        total_reward += reward
    
    return total_reward / len(pred_points)


def compute_score_format91(predict_str: str, ground_truth: str) -> float:
    acc_reward_score = accuracy_reward(predict_str, ground_truth)
    format_reward_score = format_reward(predict_str)
    return {
        "acc": acc_reward_score,
        "score": 0.9 * acc_reward_score + 0.1 * format_reward_score
    }


# ===========================
# 单元测试
# ===========================
class TestBoxFunctions(unittest.TestCase):
    def test_last_boxed_only_string(self):
        self.assertEqual(last_boxed_only_string("答案是 \\boxed{42}"), "\\boxed{42}")
        self.assertEqual(last_boxed_only_string("这里有 \\fbox{88}"), "\\fbox{88}")
        self.assertIsNone(last_boxed_only_string("没有任何标记"))

    def test_format_reward(self):
        self.assertEqual(format_reward("解: \\boxed{1}"), 1.0)
        self.assertEqual(format_reward("没有答案"), 0.0)

    def test_extract_coordinates_from_answer(self):
        s = "最终结果: \\boxed{[(0.5, 0.5), (0.9, 0.1)]}"
        coords = extract_coordinates_from_answer(s)
        self.assertEqual(coords, [(0.5, 0.5), (0.9, 0.1)])
        s = "最终结果: \\boxed{(0.5, 0.5), (0.9, 0.1)}"
        coords = extract_coordinates_from_answer(s)
        self.assertEqual(coords, [])

    def test_extract_coordinates_invalid(self):
        s = "无效: \\boxed{[(1.5, 0.5), (0.5, -0.1)]}"
        coords = extract_coordinates_from_answer(s)
        self.assertEqual(coords, [])

    def test_parse_gt_box(self):
        s = "[(0,0), (1,0), (1,1), (0,1)]"
        gt = parse_gt_box(s)
        self.assertEqual(gt, [(0,0),(1,0),(1,1),(0,1)])
        self.assertEqual(parse_gt_box("invalid"), [])

    def test_calculate_box_center_and_variance(self):
        box = [(0,0),(1,0),(1,1),(0,1)]
        (cx, cy), (sx, sy) = calculate_box_center_and_variance(box, alpha=0.5)
        self.assertAlmostEqual(cx, 0.5)
        self.assertAlmostEqual(cy, 0.5)
        self.assertAlmostEqual(sx, 0.25)
        self.assertAlmostEqual(sy, 0.25)

    def test_gaussian_point_reward(self):
        reward = gaussian_point_reward((0.5,0.5),(0.5,0.5),(0.25,0.25))
        self.assertAlmostEqual(reward, 1.0, places=6)

    def test_accuracy_reward(self):
        pred = "预测: \\boxed{[(0.5, 0.5)]}"
        gt = "[(0,0), (1,0), (1,1), (0,1)]"
        reward = accuracy_reward(pred, gt)
        self.assertGreater(reward, 0.5)

    def test_compute_score_format91(self):
        pred = "预测: \\boxed{[(0.5, 0.5)]}"
        gt = "[(0,0), (1,0), (1,1), (0,1)]"
        result = compute_score_format91(pred, gt)
        self.assertIn("acc", result)
        self.assertIn("score", result)
        self.assertGreaterEqual(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)


if __name__ == '__main__':
    unittest.main()
