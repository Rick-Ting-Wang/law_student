import os
import json
import time
import http.client
from pathlib import Path
import sys
import threading
from queue import Queue
from threading import Lock

# API密钥列表
API_KEYS = [
    'sk-yFXgUqnTSVsr0Xhc1595245c35E347A5B7BaCf7e72Bc3cE9',
    'sk-7KADKz5TZ7TcYh1cAa84Ad6aDc334731B5B13513C17d798a'
]

# 全局锁和计数器
print_lock = Lock()
progress_lock = Lock()
successful_count = 0
failed_count = 0
processed_count = 0


def print_progress_bar(current, total, bar_length=50, prefix="", suffix=""):
    """
    打印进度条
    """
    progress = current / total
    block = int(bar_length * progress)
    bar = "█" * block + "─" * (bar_length - block)
    percent = progress * 100
    with print_lock:
        sys.stdout.write(f"\r{prefix} |{bar}| {percent:.1f}% {suffix}")
        sys.stdout.flush()


def print_step(message, step_type="info"):
    """
    打印带时间戳的步骤信息
    """
    timestamp = time.strftime("%H:%M:%S")
    if step_type == "info":
        prefix = "ℹ️"
    elif step_type == "success":
        prefix = "✅"
    elif step_type == "warning":
        prefix = "⚠️"
    elif step_type == "error":
        prefix = "❌"
    elif step_type == "api":
        prefix = "🌐"
    elif step_type == "file":
        prefix = "📄"
    elif step_type == "save":
        prefix = "💾"
    else:
        prefix = " "

    with print_lock:
        print(f"\n[{timestamp}] {prefix} {message}")


def call_api(prompt_text, law_file_name, api_key):
    """
    调用API生成三元组，带进度显示
    """
    full_prompt = f"""识别该法律条文中的所有可用于法律只是图谱的全部三元组，将本条款、该条款等代词替换为当前条款的名称，补充主语，比如什么会员，什么理事会，什么组织等，尽可能完整的表述比如某经济组织某组织理事会。因为很多不同法律条文的prompt都以这句话为始。

{prompt_text}

输出格式：直接给我Subject|Relation|Object，不要输出任何其他内容。"""

    conn = http.client.HTTPSConnection("dpapi.cn")
    payload = json.dumps({
        "model": "deepseek-v3",
        "messages": [
            {
                "role": "user",
                "content": full_prompt
            }
        ]
    })
    headers = {
        'Authorization': api_key,
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }

    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print_step(f"API调用重试 {attempt + 1}/{max_retries}... (文件: {law_file_name})", "api")

            print_step(f"正在发送API请求 (文件: {law_file_name})...", "api")

            # 记录开始时间
            start_time = time.time()

            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            data = res.read()

            # 计算耗时
            elapsed_time = time.time() - start_time

            response = json.loads(data.decode("utf-8"))

            if 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0]['message']['content']
                # 统计生成的三元组数量（粗略估计）
                triplet_count = len([line for line in content.split('\n') if '|' in line])
                print_step(f"API调用成功！耗时: {elapsed_time:.1f}秒, 生成约 {triplet_count} 个三元组", "success")
                return content
            else:
                print_step(f"API响应格式异常: {response}", "error")
                return None

        except Exception as e:
            print_step(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)}", "error")
            if attempt < max_retries - 1:
                print_step(f"等待 {retry_delay} 秒后重试...", "api")
                time.sleep(retry_delay)
            else:
                print_step(f"文件 {law_file_name} API调用最终失败", "error")
                return None


def parse_triplets(response_text):
    """
    解析API返回的三元组文本
    """
    triplets = []
    if not response_text:
        return triplets

    lines = response_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and '|' in line:
            parts = line.split('|')
            if len(parts) >= 3:
                subject = parts[0].strip()
                relation = parts[1].strip()
                obj = '|'.join(parts[2:]).strip()  # 处理object中可能包含|的情况
                triplets.append({
                    "subject": subject,
                    "relation": relation,
                    "object": obj
                })

    return triplets


def process_law_file(input_file_path, output_dir, api_key, total_files):
    """
    处理单个法律文件，带详细进度显示
    """
    global successful_count, failed_count, processed_count

    file_name = os.path.basename(input_file_path)
    print_step(f"开始处理文件: {file_name}", "file")

    # 读取法律文本
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            law_text = f.read()
        print_step(f"成功读取文件，大小: {len(law_text)} 字符", "success")
    except UnicodeDecodeError:
        try:
            with open(input_file_path, 'r', encoding='gbk') as f:
                law_text = f.read()
            print_step(f"使用GBK编码成功读取文件，大小: {len(law_text)} 字符", "success")
        except Exception as e:
            print_step(f"无法读取文件 {file_name}: {str(e)}", "error")
            with progress_lock:
                failed_count += 1
                processed_count += 1
            return False

    # 调用API
    print_step("开始API调用...", "api")
    response = call_api(law_text, file_name, api_key)
    if not response:
        with progress_lock:
            failed_count += 1
            processed_count += 1
        return False

    # 解析三元组
    print_step("正在解析API返回的三元组...", "info")
    triplets = parse_triplets(response)

    if triplets:
        # 准备输出数据
        output_data = {
            "source_file": file_name,
            "processed_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "triplets": triplets,
            "count": len(triplets)
        }

        # 保存为JSON文件
        output_file_name = file_name.replace('.txt', '.json')
        output_file_path = os.path.join(output_dir, output_file_name)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print_step(f"已保存 {len(triplets)} 个三元组到: {output_file_name}", "save")

        # 可选：显示前几个三元组作为预览
        preview_count = min(3, len(triplets))
        print_step(f"三元组预览 (前{preview_count}个):", "info")
        for i in range(preview_count):
            t = triplets[i]
            print(f"  {i + 1}. {t['subject']} | {t['relation']} | {t['object'][:50]}...")

        # 只有成功生成JSON才删除原文件
        try:
            os.remove(input_file_path)
            print_step(f"已删除原文件: {file_name}", "success")
        except Exception as e:
            print_step(f"删除原文件失败: {str(e)}", "error")

        with progress_lock:
            successful_count += 1
            processed_count += 1

        return True
    else:
        print_step(f"警告: 文件 {file_name} 未生成有效的三元组", "warning")
        with progress_lock:
            failed_count += 1
            processed_count += 1
        return False


def worker(queue, output_dir, total_files, api_key_index):
    """
    工作线程函数
    """
    while True:
        task = queue.get()
        if task is None:
            break

        input_file_path = task
        api_key = API_KEYS[api_key_index]
        process_law_file(input_file_path, output_dir, api_key, total_files)

        queue.task_done()


def main():
    global successful_count, failed_count, processed_count

    print_step("🚀 法律文本三元组提取工具启动", "info")

    # 设置目录路径
    input_dir = "./lawtxtversion"
    output_dir = "./3entity"

    # 检查输入目录
    if not os.path.exists(input_dir):
        print_step(f"错误: 输入目录 {input_dir} 不存在", "error")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print_step(f"输出目录: {output_dir}", "info")

    # 获取所有txt文件
    txt_files = list(Path(input_dir).glob("*.txt"))
    total_files = len(txt_files)
    print_step(f"找到 {total_files} 个txt文件需要处理", "info")

    if total_files == 0:
        print_step("没有找到txt文件，程序退出", "warning")
        return

    print_step("\n开始批量处理...", "info")
    print("=" * 70)

    # 创建队列和线程池
    queue = Queue()
    num_threads = 4  # 4个线程
    threads = []

    # 创建并启动工作线程
    for i in range(num_threads):
        api_key_index = i % len(API_KEYS)  # 轮流使用两个API密钥
        t = threading.Thread(target=worker, args=(queue, output_dir, total_files, api_key_index))
        t.daemon = True
        t.start()
        threads.append(t)

    # 将所有文件添加到队列
    for txt_file in txt_files:
        queue.put(str(txt_file))

    # 等待所有任务完成
    while processed_count < total_files:
        print_progress_bar(processed_count, total_files, prefix="总体进度", suffix=f"{processed_count}/{total_files}")
        time.sleep(0.5)

    # 等待所有线程完成
    for _ in range(num_threads):
        queue.put(None)
    for t in threads:
        t.join()

    # 完成，更新进度条到100%
    print_progress_bar(total_files, total_files, prefix="总体进度", suffix=f"{total_files}/{total_files}")
    print("\n")

    # 输出最终统计信息
    print_step("🏁 批量处理完成！", "success")
    print("=" * 70)
    print(f"✅ 成功: {successful_count} 个文件")
    print(f"❌ 失败: {failed_count} 个文件")
    print(f"📁 三元组JSON文件已保存到: {output_dir}")

    # 如果输出目录有文件，显示汇总信息
    json_files = list(Path(output_dir).glob("*.json"))
    if json_files:
        total_triplets = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    total_triplets += data.get('count', 0)
            except:
                pass
        print(f"📊 总计生成三元组数量: {total_triplets} 个")


if __name__ == "__main__":
    main()
