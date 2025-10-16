import unittest

def cot_reward(predict: str, score: float) -> float:        
    max_length = 6000
    min_length = 30
    cot_str = remove_last_box(predict)
    if (score>=1.0):
        if len(cot_str)<min_length or len(cot_str) > max_length:
            return 0.0
    return 1.0

def remove_last_box(text: str) -> str:
    last_pos = text.rfind(r'\boxed{')
    if last_pos == -1:
        return text

    i = last_pos + len(r'\boxed{')
    depth = 1
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    return text[:last_pos] + text[i:]

class TestCotReward(unittest.TestCase):
    
    def test_remove_last_box(self):
        """测试 remove_last_box 函数"""
        # 测试没有 \boxed{} 的情况
        self.assertEqual(remove_last_box("Hello world"), "Hello world")
        
        # 测试单个 \boxed{} 的情况
        text_with_box = "Some text \\boxed{answer} more text"
        expected = "Some text  more text"
        self.assertEqual(remove_last_box(text_with_box), expected)
        
        # 测试嵌套括号的情况
        nested_text = "Text \\boxed{answer {nested}} end"
        expected_nested = "Text  end"
        self.assertEqual(remove_last_box(nested_text), expected_nested)
        
        # 测试多个 \boxed{} 的情况（只移除最后一个）
        multiple_boxes = "First \\boxed{1} Second \\boxed{2} Third"
        expected_multiple = "First \\boxed{1} Second  Third"
        self.assertEqual(remove_last_box(multiple_boxes), expected_multiple)
    
    def test_cot_reward_score_less_than_1(self):
        """测试 score < 1.0 的情况"""
        # 无论文本长度如何，score < 1.0 都应该返回 1.0
        short_text = "a" * 10  # 远小于 min_length
        long_text = "a" * 7000  # 超过 max_length
        valid_text = "a" * 100  # 有效长度
        
        self.assertEqual(cot_reward(short_text, 0.5), 1.0)
        self.assertEqual(cot_reward(long_text, 0.9), 1.0)
        self.assertEqual(cot_reward(valid_text, 0.0), 1.0)
    
    def test_cot_reward_score_1_short_text(self):
        """测试 score = 1.0 且文本过短的情况"""
        short_text = "a" * 20  # 小于 min_length(30)
        self.assertEqual(cot_reward(short_text, 1.0), 0.0)
    
    def test_cot_reward_score_1_long_text(self):
        """测试 score = 1.0 且文本过长的情况"""
        long_text = "a" * 7000  # 超过 max_length(6000)
        self.assertEqual(cot_reward(long_text, 1.0), 0.0)
    
    def test_cot_reward_score_1_valid_length(self):
        """测试 score = 1.0 且文本长度在有效范围内的情况"""
        valid_text = "a" * 100  # 在 30-6000 范围内
        self.assertEqual(cot_reward(valid_text, 1.0), 1.0)
    
    def test_cot_reward_with_boxed_content(self):
        """测试包含 \boxed{} 的文本"""
        # 移除 \boxed{} 后文本变短
        short_base = "Short"  # 长度5
        boxed_content = "x" * 20  # 长度20
        text_with_box = short_base + r"\boxed{" + boxed_content + "}"
        # 移除后长度 = 5 (小于30)，应该返回0.0
        self.assertEqual(cot_reward(text_with_box, 1.0), 0.0)
        
        # 移除 \boxed{} 后文本仍然有效
        long_base = "a" * 40  # 长度40
        text_with_box2 = long_base + r"\boxed{answer}"
        # 移除后长度应该还是 >= 30，应该返回1.0
        self.assertEqual(cot_reward(text_with_box2, 1.0), 1.0)
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 正好等于 min_length
        min_length_text = "a" * 30
        self.assertEqual(cot_reward(min_length_text, 1.0), 1.0)
        
        # 正好等于 max_length  
        max_length_text = "a" * 6000
        self.assertEqual(cot_reward(max_length_text, 1.0), 1.0)
        
        # 刚好超过 min_length
        just_above_min = "a" * 31
        self.assertEqual(cot_reward(just_above_min, 1.0), 1.0)
        
        # 刚好低于 min_length
        just_below_min = "a" * 29
        self.assertEqual(cot_reward(just_below_min, 1.0), 0.0)
        
        # 刚好超过 max_length
        just_above_max = "a" * 6001
        self.assertEqual(cot_reward(just_above_max, 1.0), 0.0)
        
        # 空字符串
        self.assertEqual(cot_reward("", 1.0), 0.0)

def run_tests():
    """运行测试"""
    # 使用 unittest 运行测试
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # 额外的简单测试
    print("\n" + "="*50)
    print("简单测试示例:")
    
    # 测试示例
    test_cases = [
        ("Short text", 1.0, "应该返回 0.0 (文本过短)"),
        ("This is a reasonably long text that should pass", 1.0, "应该返回 1.0 (长度合适)"),
        ("a" * 7000, 1.0, "应该返回 0.0 (文本过长)"),
        ("Any text", 0.5, "应该返回 1.0 (score < 1.0)"),
        ("a" * 100, 1.0, "应该返回 1.0 (有效长度)"),
    ]
    
    for text, score, expected in test_cases:
        result = cot_reward(text, score)
        cot_str = remove_last_box(text)
        status = "✓" if result == (0.0 if (score >= 1.0 and (len(cot_str) < 30 or len(cot_str) > 6000)) else 1.0) else "✗"
        print(f"{status} 文本长度: {len(text)}, 移除box后: {len(cot_str)}, score: {score}, 结果: {result} - {expected}")

if __name__ == "__main__":
    run_tests()