from docx import Document
from pathlib import Path

# 根目录
ROOT = Path(".")

# docx文件夹
DOCX_DIR = Path("./laws")

# txt输出文件夹
TXT_DIR = Path("./lawtxtversion")

# 如果不存在就创建
TXT_DIR.mkdir(exist_ok=True)


def convert_docx_to_txt(docx_path, txt_path):

    doc = Document(docx_path)

    with open(txt_path, "w", encoding="utf-8") as f:

        for para in doc.paragraphs:

            text = para.text.strip()

            if text:
                f.write(text + "\n")


def main():

    docx_files = list(DOCX_DIR.glob("*.docx"))

    print(f"发现 {len(docx_files)} 个docx文件")

    for file in docx_files:

        txt_file = TXT_DIR / (file.stem + ".txt")

        convert_docx_to_txt(file, txt_file)

        print("转换完成:", file.name)


if __name__ == "__main__":
    main()