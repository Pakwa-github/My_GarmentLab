import os
import re

# 注意！使用该脚本从Linux路径改到我的Windows路径后，
# 要把 .\Env\Config\GarmentConfig.py中的6行254列把原本的除号改回来；
# 还要把 .\Env\Rigid\Rigid.py中116行111位置原本的/改回来。

# 定义替换路径中斜杠的函数
def replace_slashes_in_file(file_path, old_str, new_str):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 替换路径字符串中的斜杠 "/" 为 "\\"，不影响原有的反斜杠
    # 这一步只会替换路径中的 "/"
    content = content.replace(old_str, new_str)

    # 使用正则表达式仅替换路径中的斜杠 "/" 为 "\\"
    content = re.sub(r'([A-Za-z]:[\\/][^\n\r]*)', lambda m: m.group(0).replace('/', "\\\\"), content)

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

# 定义遍历目录并替换路径的函数
def replace_paths_in_project(directory, old_str, new_str):
    # 遍历目录中的所有文件
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            # 只处理 .py 文件
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                print(f"Processing: {file_path}")
                replace_slashes_in_file(file_path, old_str, new_str)

# 主程序
if __name__ == "__main__":
    project_dir = r"D:\Code\GP\GarmentLab\GarmentLab"  # 项目根目录
    old_path = "/home/isaac/GarmentLab/Assets"  # 旧路径（Linux 风格）
    new_path = "D:\\\\isaac\\\\GarmentLab\\\\Assets"  # 新路径（Windows 风格）
    
    replace_paths_in_project(project_dir, old_path, new_path)
    print("Path replacement complete!")
