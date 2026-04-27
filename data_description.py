#!/usr/bin/env python3
import os
import json
import glob


def load_first_json(file_path):
    """从文件中读取第一个完整的 JSON 对象"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        if "Extra data" in str(e):
            decoder = json.JSONDecoder()
            obj, end = decoder.raw_decode(content)
            return obj
        else:
            raise


def main():
    data_dir = "./data"
    pattern = os.path.join(data_dir, "*.json")
    json_files = glob.glob(pattern)

    result = []
    for file_path in sorted(json_files):
        filename = os.path.basename(file_path)
        try:
            first_obj = load_first_json(file_path)
            if first_obj is None:
                print(f"跳过文件 {filename}，文件为空")
                continue
            # 直接记录整个第一个对象
            result.append({
                "filename": filename,
                "first_item": first_obj
            })
        except (json.JSONDecodeError, IOError) as e:
            print(f"跳过文件 {filename}，无法解析: {e}")

    output_file = "data_description.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"处理完成，共处理 {len(result)} 个文件，结果已写入 {output_file}")


if __name__ == "__main__":
    main()