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

def extract_coordinates_from_answer(predict_str):
    try:
        # 提取所有<answer>标签内容（取最后一个）
        answer_matches = re.findall(r'<answer>(.*?)</answer>', predict_str, re.DOTALL)
        if not answer_matches:
            return []
        
        answer_content = answer_matches[-1].strip()
        coord_pattern = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")
        matches = coord_pattern.findall(answer_content)
        
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

def compute_score_no_format(predict_str: str, ground_truth: str) -> float:
    return accuracy_reward(predict_str, ground_truth)

def compute_score_format91(predict_str: str, ground_truth: str) -> float:
    format_reward_1 = format_reward_think_answer(predict_str)
    format_reward_2 = format_reward_coordinates(predict_str)
    acc_reward = accuracy_reward(predict_str, ground_truth)
    
    return {
        "acc": acc_reward,
        "score": 0.1*format_reward_1 + 0.1*format_reward_2 + 0.8*acc_reward
    }

import unittest

class TestCoordinateFunctions(unittest.TestCase):
    # 基础格式验证测试
    def test_format_reward_think_answer(self):
        # 完整正确格式
        correct_format = "<think>内容</think><answer>[(0.1,0.2)]</answer>"
        self.assertEqual(format_reward_think_answer(correct_format), 1.0)
        
        # 缺少think标签
        missing_think = "<answer>[(0.1,0.2)]</answer>"
        self.assertEqual(format_reward_think_answer(missing_think), 0.0)
    
    # 坐标提取安全测试（增强版）
    def test_extract_coordinates_security(self):
        # 测试用例1：坐标在answer外部不应被提取
        external_coords = """
        <think>思考</think>
        (0.9,0.9)  # 外部坐标
        <answer>[(0.1,0.2)]</answer>
        (0.8,0.8)  # 外部坐标
        """
        self.assertEqual(extract_coordinates_from_answer(external_coords), [(0.1, 0.2)])
        
        # 测试用例2：嵌套标签应该提取最后一个有效answer内容
        nested_tags = """
        <think>
        <answer>[(9.9,9.9)]</answer>  # 假坐标
        </think>
        <answer>[(0.1,0.2)]</answer>  # 真坐标
        """
        self.assertEqual(extract_coordinates_from_answer(nested_tags), [(0.1, 0.2)])
        
        # 测试用例3：多answer标签应该取最后一个
        multi_answers = """
        <answer>[(0.3,0.4)]</answer>
        <answer>[(0.1,0.2)]</answer>
        """
        self.assertEqual(extract_coordinates_from_answer(multi_answers), [(0.1, 0.2)])
        
        # 测试用例4：混合内容中的坐标提取
        mixed_content = """
        <answer>
        这是文本内容 [(0.1,0.2), (0.3,0.4)]
        更多文本 (0.5,0.6)
        </answer>
        """
        self.assertCountEqual(
            extract_coordinates_from_answer(mixed_content),
            [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
        )
    
    def test_strict_coordinate_format(self):
        # 测试用例1：缺少方括号的情况
        no_brackets = "<answer>(0.1,0.2),(0.3,0.4)</answer>"
        self.assertEqual(format_reward_coordinates(no_brackets), 0.0)
        
        # 测试用例2：带空格的标准格式
        with_spaces = "<answer> [ ( 0.1 , 0.2 ) ] </answer>"
        self.assertEqual(format_reward_coordinates(with_spaces), 1.0)
        
        # 测试用例3：不完整坐标对
        incomplete_pair = "<answer>[(0.1,0.2), (0.3)]</answer>"
        self.assertEqual(format_reward_coordinates(incomplete_pair), 0.0)
    
    def test_gt_box_parsing(self):
        # 不完整GT框
        incomplete_gt = "[(0.1,0.2),(0.3,0.4)]"  # 只有2个点
        parsed = parse_gt_box(incomplete_gt)
        self.assertEqual(len(parsed), 2)
        
        # 非法字符串格式
        invalid_gt = "不是坐标格式"
        self.assertEqual(parse_gt_box(invalid_gt), [])
    
    def test_gaussian_edge_cases(self):
        # 完全匹配中心点
        center = (0.5, 0.5)
        variance = (0.1, 0.1)
        self.assertAlmostEqual(gaussian_point_reward(center, center, variance), 1.0)
        
        # 完全超出范围的点
        far_point = (10.0, 10.0)
        self.assertAlmostEqual(gaussian_point_reward(far_point, center, variance), 0.0)
    
    def test_coordinate_deduplication(self):
        # 测试用例1：完全相同的坐标
        test_str1 = """
        <answer>
        [(0.1, 0.2), (0.1, 0.2), 
        (0.1, 0.2), (0.3, 0.4)]
        </answer>
        """
        self.assertCountEqual(
            extract_coordinates_from_answer(test_str1),
            [(0.1, 0.2), (0.3, 0.4)]
        )
        
        # 测试用例2：精度差异的坐标
        test_str2 = """
        <answer>
        [(0.100001, 0.200001), 
        (0.1000, 0.2000),
        (0.1, 0.2)]
        </answer>
        """
        result = extract_coordinates_from_answer(test_str2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (0.1, 0.2))
        
        # 测试用例3：混合格式重复
        test_str3 = """
        <answer>
        [(0.1, 0.2), (0.3, 0.4)]
        (0.1, 0.2)  # 重复坐标
        (0.3, 0.4)  # 重复坐标
        (0.5, 0.6)  # 新坐标
        </answer>
        """
        self.assertCountEqual(
            extract_coordinates_from_answer(test_str3),
            [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)]
        )
    
    def test_edge_cases(self):
        # 测试用例1：空内容
        self.assertEqual(extract_coordinates_from_answer(""), [])
        self.assertEqual(extract_coordinates_from_answer("<answer></answer>"), [])
        
        # 测试用例2：超出范围的坐标
        test_str = """
        <answer>
        [(1.1, 0.5),  # x超出
        (0.5, -0.1),  # y超出
        (0.5, 0.5),   # 有效
        (0.5, 1.0)]   # 边界有效
        </answer>
        """
        self.assertEqual(
            extract_coordinates_from_answer(test_str),
            [(0.5, 0.5), (0.5, 1.0)]
        )
        
        # 测试用例3：不同精度的坐标
        test_str = """
        <answer>
        [(0.1234, 0.5678), 
        (0.12345, 0.56785)]  # 小数点后5位
        </answer>
        """
        self.assertEqual(len(extract_coordinates_from_answer(test_str)), 2)


if __name__ == '__main__':
    unittest.main()
