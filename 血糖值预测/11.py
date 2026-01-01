import csv
import json

# CSV文件路径
csv_file_path = 'Mobiles Dataset (2025).csv'
# 输出的JSON文件路径
json_file_path = 'Mobiles_Dataset_2025.json'

# 尝试不同的编码格式
encodings = ['utf-8', 'latin-1', 'ISO-8859-1', 'cp1252']

for encoding in encodings:
    try:
        # 初始化一个空列表来存储字典
        data_list = []

        # 打开CSV文件并读取内容
        with open(csv_file_path, mode='r', encoding=encoding) as csv_file:
            csv_reader = csv.DictReader(csv_file)

            # 遍历CSV文件的每一行，并将其添加到列表中
            for row in csv_reader:
                data_list.append(row)

        # 将列表转换为JSON格式并写入文件
        with open(json_file_path, mode='w', encoding='utf-8') as json_file:
            json.dump(data_list, json_file, ensure_ascii=False, indent=4)

        print(f"数据已成功从{csv_file_path}转换为{json_file_path}，使用的编码是{encoding}")
        break
    except UnicodeDecodeError:
        print(f"尝试使用编码{encoding}失败，尝试下一个编码。")
        continue
else:
    print("无法解码文件，请尝试手动确定文件的编码格式。")