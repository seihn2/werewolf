#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 修复文件编码问题
import os

def fix_file_encoding(filename):
    """修复文件的编码问题"""
    try:
        # 尝试用不同编码读取文件
        content = None
        for encoding in ['utf-8', 'gbk', 'gb2312', 'cp936']:
            try:
                with open(filename, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"成功用 {encoding} 编码读取文件")
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            print(f"无法读取文件 {filename}")
            return False

        # 重新用UTF-8编码写入
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"文件 {filename} 编码修复完成")
        return True

    except Exception as e:
        print(f"修复文件编码时出错: {e}")
        return False

if __name__ == "__main__":
    fix_file_encoding("conversation_manager.py")