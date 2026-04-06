import os
from docx import Document

# 你的 Word 文件根目录（可以包含子文件夹）
root_folder = "./laws"
# 输出文件
output_file = "knowledge.txt"

# 遍历所有子文件夹和文件
docx_files = []
for dirpath, dirnames, filenames in os.walk(root_folder):
    for f in filenames:
        if f.endswith(".docx"):
            docx_files.append(os.path.join(dirpath, f))

with open(output_file, "w", encoding="utf-8") as out_file:
    for file_path in docx_files:
        try:
            doc = Document(file_path)
            full_text = [para.text for para in doc.paragraphs if para.text.strip()]
            content = "\n".join(full_text)

            # 写入 txt 文件
            out_file.write(f"标题: {os.path.basename(file_path)}\n")
            out_file.write(content)
            out_file.write("\n" + "=" * 80 + "\n")

            print(f"已处理: {file_path}")
        except Exception as e:
            print(f"处理 {file_path} 出错: {e}")

print(f"全部 Word 文件已合并到 {output_file}")