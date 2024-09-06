import json
import re


def trans_data(origin_data, new_data):
    with open(origin_data, 'r', encoding='utf-8') as file:
        data = json.load(file)

    processed_data = []

    for item in data:
        processed_item = item.copy()  # 复制原始数据项，避免修改原始数据

        # 使用正则表达式匹配"total score reduction of"后面的数字
        match = re.search(r'with a total score reduction of (\d+(\.\d+)?)', item['errors'])
        if match:
            # 将匹配到的分数替换为错误信息
            processed_item['errors'] = match.group(1)

        processed_item['score'] = processed_item.pop('errors', None)  # 使用pop移除errors键
        processed_data.append(processed_item)

    # 保存到新的JSON文件
    with open(new_data, 'w', encoding='utf-8') as file:
        json.dump(processed_data, file, ensure_ascii=False, indent=4)
    print("处理完成，数据已保存到 %s" % new_data)

def get_data(data, cut_data, num, task):
    with open(data, 'r', encoding='utf-8') as file:
        data = json.load(file)

    new_data = []
    count = 0
    for item in data:
        if item['task'] == task:
            # 去除item['errors']键值对
            item.pop('errors', None)
            new_data.append(item)
            count += 1
        if count == num:
            break

    # 保存到新的JSON文件
    with open(cut_data, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)
    print("处理完成，数据已保存到 %s" % cut_data)

data = "data/new_mix_.json"
processed_data = "data/summarization_data_200.json"

get_data(data, processed_data, 200, "summarization")