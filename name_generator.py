"""
AI玩家名字生成器
提供丰富多样的100个名字库，随机分配给AI玩家
"""

import random
from typing import List, Set


class NameGenerator:
    """名字生成器"""

    def __init__(self):
        self.used_names: Set[str] = set()
        self.all_names = self._create_name_database()

    def _create_name_database(self) -> List[str]:
        """创建100个多样化的名字库"""
        names = [
            # 经典中文名字 (20个)
            "李明", "王芳", "张伟", "刘娜", "陈杰", "杨丽", "赵强", "黄梅",
            "周涛", "吴红", "徐静", "朱亮", "胡敏", "林峰", "何花", "郭军",
            "马超", "高雅", "孙勇", "田园",

            # 现代流行名字 (20个)
            "梓涵", "浩宇", "雨萱", "子轩", "诗涵", "宇轩", "欣怡", "俊杰",
            "佳怡", "博文", "心怡", "嘉豪", "思涵", "志强", "雨欣", "建华",
            "晓燕", "文静", "国强", "淑华",

            # 文艺雅致名字 (20个)
            "墨染", "云舒", "月影", "星辰", "雪落", "风清", "雨霖", "花开",
            "叶飞", "水静", "山青", "林深", "竹语", "梅香", "兰芷", "菊韵",
            "桃夭", "柳絮", "荷香", "桂馥",

            # 游戏风格名字 (20个)
            "剑客", "法师", "弓手", "刺客", "战士", "牧师", "盗贼", "猎人",
            "火焰", "冰霜", "雷电", "暗影", "光明", "神圣", "邪恶", "混沌",
            "英雄", "王者", "传奇", "无敌",

            # 个性创意名字 (20个)
            "小确幸", "大智若愚", "随心所欲", "自由自在", "淡定从容", "乐观向上",
            "机智如我", "沉稳如山", "灵动如水", "热情似火", "冷静如冰", "坚强如钢",
            "温柔似玉", "聪明绝顶", "勇敢无畏", "善良纯真", "幽默风趣", "深邃如海",
            "飘逸如云", "稳重如石"
        ]

        # 确保正好100个名字
        assert len(names) == 100, f"名字数量不对，当前有{len(names)}个"
        return names

    def get_random_names(self, count: int) -> List[str]:
        """获取指定数量的随机不重复名字"""
        if count > len(self.all_names):
            raise ValueError(f"请求的名字数量({count})超过了库存({len(self.all_names)})")

        # 重置已使用名字集合
        self.used_names.clear()

        available_names = [name for name in self.all_names if name not in self.used_names]

        if count > len(available_names):
            # 如果可用名字不够，重新洗牌所有名字
            available_names = self.all_names.copy()
            self.used_names.clear()

        selected_names = random.sample(available_names, count)
        self.used_names.update(selected_names)

        return selected_names

    def get_single_random_name(self) -> str:
        """获取一个随机名字"""
        return self.get_random_names(1)[0]

    def reset_used_names(self):
        """重置已使用的名字"""
        self.used_names.clear()

    def get_all_names(self) -> List[str]:
        """获取所有名字"""
        return self.all_names.copy()

    def get_names_by_style(self, style: str, count: int = None) -> List[str]:
        """按风格获取名字"""
        style_ranges = {
            'classic': (0, 20),      # 经典中文名字
            'modern': (20, 40),      # 现代流行名字
            'elegant': (40, 60),     # 文艺雅致名字
            'gaming': (60, 80),      # 游戏风格名字
            'creative': (80, 100)    # 个性创意名字
        }

        if style not in style_ranges:
            raise ValueError(f"未知风格: {style}，可选: {list(style_ranges.keys())}")

        start, end = style_ranges[style]
        style_names = self.all_names[start:end]

        if count is None:
            return style_names
        else:
            return random.sample(style_names, min(count, len(style_names)))


def create_name_config_template():
    """创建名字配置模板"""
    generator = NameGenerator()

    template = {
        "usage": "AI玩家名字配置模板",
        "name_styles": {
            "classic": "经典中文名字 - 简洁大方",
            "modern": "现代流行名字 - 时尚潮流",
            "elegant": "文艺雅致名字 - 诗意优美",
            "gaming": "游戏风格名字 - 角色扮演",
            "creative": "个性创意名字 - 独特有趣"
        },
        "sample_names": {
            "classic": generator.get_names_by_style('classic', 5),
            "modern": generator.get_names_by_style('modern', 5),
            "elegant": generator.get_names_by_style('elegant', 5),
            "gaming": generator.get_names_by_style('gaming', 5),
            "creative": generator.get_names_by_style('creative', 5)
        },
        "total_names": len(generator.get_all_names())
    }

    return template


def test_name_generator():
    """测试名字生成器"""
    generator = NameGenerator()

    print("=== AI玩家名字生成器测试 ===")
    print(f"总名字数量: {len(generator.get_all_names())}")

    # 测试随机获取名字
    print("\n随机10个名字:")
    random_names = generator.get_random_names(10)
    for i, name in enumerate(random_names, 1):
        print(f"{i:2d}. {name}")

    # 测试各种风格
    styles = ['classic', 'modern', 'elegant', 'gaming', 'creative']
    for style in styles:
        print(f"\n{style}风格名字 (5个):")
        style_names = generator.get_names_by_style(style, 5)
        for name in style_names:
            print(f"  {name}")

    print(f"\n测试完成！")


if __name__ == "__main__":
    test_name_generator()