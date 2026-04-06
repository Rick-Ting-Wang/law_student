import re
import json
from pathlib import Path


class RuleEngine:
    def __init__(self):
        self.rules = []

    def add_rule(self, func):
        self.rules.append(func)

    def apply(self, text):
        for r in self.rules:
            text = r(text)
        return text


class ChinaLawParser:

    def __init__(self):

        # ===== 规则引擎 =====
        self.engine = RuleEngine()
        self.engine.add_rule(self.clean_noise)
        self.engine.add_rule(self.normalize_numbers)
        self.engine.add_rule(self.fix_article_breaks)
        self.engine.add_rule(self.fix_chapter_breaks)

        # ===== regex =====
        self.title_pattern = re.compile(r"标题[:：]\s*(.+)")

        self.chapter_pattern = re.compile(
            r"(?m)^\s*第[一二三四五六七八九十百千万零0-9]+章[^\n]*"
        )

        self.section_pattern = re.compile(
            r"(?m)^\s*第[一二三四五六七八九十百千万零0-9]+节[^\n]*"
        )

        self.article_pattern = re.compile(
            r"(?m)^第[一二三四五六七八九十百千万零0-9]+条(?=[　\s\u4e00-\u9fff])"
        )

        self.item_pattern = re.compile(
            r"(?m)^\s*[一二三四五六七八九十]+、"
        )

        # ✅ 判断是否是"断头内容"的特征词
        # 这些词出现在条文开头，说明这一段是上一条的延续
        self.continuation_pattern = re.compile(
            r"^(规定的|所称|前款|本条|前条|以上|以下|之规定|"
            r"款规定|条规定|项规定|的规定|情形之一|有下列|"
            r"前述|上述|是指|包括|但是|否则|除外|不得|应当|"
            r"，|。|；|的，|者，)"
        )

    # =====================
    # 规则1：清理噪声
    # =====================
    def clean_noise(self, text):
        patterns = [
            r"第\d+页",
            r"共\d+页",
            r"打印日期.*",
            r"发布日期.*",
            r"实施日期.*"
        ]
        for p in patterns:
            text = re.sub(p, "", text)
        return text

    # =====================
    # 规则2：数字标准化
    # =====================
    def normalize_numbers(self, text):
        mapping = {
            "１": "1", "２": "2", "３": "3",
            "４": "4", "５": "5", "６": "6",
            "７": "7", "８": "8", "９": "9", "０": "0",
        }
        for k, v in mapping.items():
            text = text.replace(k, v)
        return text

    # =====================
    # 规则3：修复条文换行
    # =====================
    def fix_article_breaks(self, text):
        return re.sub(
            r"(?<!\n)(第[一二三四五六七八九十百千万零0-9]+条(?=[　\s\u4e00-\u9fff]))",
            r"\n\1",
            text
        )

    # =====================
    # 规则4：修复章换行
    # =====================
    def fix_chapter_breaks(self, text):
        return re.sub(
            r"(?<!\n)(第[一二三四五六七八九十百千万零0-9]+章)",
            r"\n\1",
            text
        )

    # =====================
    # 文档拆分
    # =====================
    def split_documents(self, text):
        docs = re.split(r"={10,}", text)
        return [d.strip() for d in docs if len(d.strip()) > 50]

    # =====================
    # 提取标题（清理文件名）
    # =====================
    def extract_title(self, doc):
        m = self.title_pattern.search(doc)
        raw = m.group(1).strip() if m else ""

        if not raw:
            lines = doc.split("\n")
            for line in lines[:5]:
                line = line.strip()
                if 6 < len(line) < 80 and not re.match(r"\d{4}年|\(|（|标题[:：]", line):
                    raw = line
                    break

        return self._clean_filename(raw) if raw else "未知法律"

    def _clean_filename(self, name: str) -> str:
        name = re.sub(r"\.(docx|doc|txt|pdf)$", "", name, flags=re.IGNORECASE)
        name = re.sub(r"[_\-]\d{6,8}$", "", name)
        return name.strip()

    def extract_date_from_title_line(self, doc: str) -> str:
        m = self.title_pattern.search(doc)
        raw = m.group(1).strip() if m else ""
        date_m = re.search(r"(\d{4})(\d{2})(\d{2})", raw)
        if date_m:
            return f"{date_m.group(1)}-{date_m.group(2)}-{date_m.group(3)}"
        return ""

    # =====================
    # 章节解析
    # =====================
    def parse_chapters(self, doc):
        matches = list(self.chapter_pattern.finditer(doc))
        chapters = []

        if not matches:
            chapters.append({"title": "", "text": doc})
            return chapters

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(doc)
            chapters.append({
                "title": m.group().strip(),
                "text": doc[start:end]
            })

        return chapters

    # =====================
    # 条文解析
    # ✅ 核心修复：解析完之后合并断头条文
    # =====================
    def parse_articles(self, text):
        matches = list(self.article_pattern.finditer(text))
        raw_articles = []

        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            raw_articles.append({
                "article": m.group().strip(),
                "text": text[start:end].strip()
            })

        # ✅ 合并断头条文
        return self._merge_broken_articles(raw_articles)

    def _merge_broken_articles(self, articles: list) -> list:
        """
        检测并合并被错误切断的条文。

        判断逻辑：
        如果某条的 text 以"continuation_pattern"开头，
        说明它是上一条被截断后的延续，拼回去。

        例如：
            第九条  → "金融机构违反本规定"         ← 不完整
            第七条  → "规定的，由中国人民银行..."   ← 开头是"规定的" → 断头！

        合并后：
            第九条  → "金融机构违反本规定第七条规定的，由中国人民银行..."
        """
        if not articles:
            return articles

        merged = [articles[0].copy()]

        for curr in articles[1:]:
            text = curr["text"]
            is_broken = (
                # 特征1：内容以续接词开头
                self.continuation_pattern.match(text)
                # 特征2：内容极短（不可能是完整条文）
                or len(text) < 15
                # 特征3：上一条内容末尾没有句号/分号（句子没结束）
                or (merged and not re.search(r"[。；]$", merged[-1]["text"].rstrip()))
                and len(text) < 50
            )

            if is_broken and merged:
                # 拼回上一条，中间加上被切掉的条号
                merged[-1]["text"] = (
                    merged[-1]["text"]
                    + curr["article"]      # 把误切的"第七条"补回正文
                    + text
                )
            else:
                merged.append(curr.copy())

        return merged

    # =====================
    # fallback 解析
    # =====================
    def parse_items(self, text):
        parts = re.split(self.item_pattern, text)
        items = []

        for i, p in enumerate(parts):
            if len(p.strip()) < 20:
                continue
            items.append({
                "article": f"条款{i + 1}",
                "text": p.strip()
            })

        return items

    # =====================
    # chunk切分
    # =====================
    def split_long(self, text, size=500):
        if len(text) <= size:
            return [text]
        return [text[i:i + size] for i in range(0, len(text), size)]

    # =====================
    # 生成chunk
    # =====================
    def build_chunk(self, law, chapter, article, text):
        return f"法律：{law}\n章节：{chapter}\n条文：{article}\n\n{text}".strip()

    # =====================
    # 解析单法律
    # =====================
    def parse_document(self, doc):
        title = self.extract_title(doc)
        publish_date = self.extract_date_from_title_line(doc)
        chapters = self.parse_chapters(doc)
        results = []

        for ch in chapters:
            chapter = ch["title"]
            articles = self.parse_articles(ch["text"])

            if not articles:
                articles = self.parse_items(ch["text"])

            for a in articles:
                pieces = self.split_long(a["text"])

                for p in pieces:
                    results.append({
                        "law": title,
                        "publish_date": publish_date,
                        "chapter": chapter,
                        "article": a["article"],
                        "text": p,
                        "chunk": self.build_chunk(title, chapter, a["article"], p)
                    })

        return results

    # =====================
    # 主解析
    # =====================
    def parse(self, text):
        text = self.engine.apply(text)
        docs = self.split_documents(text)
        results = []

        for d in docs:
            parsed = self.parse_document(d)
            results.extend(parsed)

        return results


if __name__ == "__main__":

    text = Path("knowledge.txt").read_text(encoding="utf-8")

    parser = ChinaLawParser()
    data = parser.parse(text)

    with open("law_structured.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("解析完成")
    print("chunk数量:", len(data))
