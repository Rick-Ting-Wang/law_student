import os
import json
import time
import sys
from pathlib import Path
from threading import Lock
from neo4j import GraphDatabase, exceptions as neo4j_exceptions

# ─── Neo4j 连接配置 ────────────────────────────────────────────────────────────
NEO4J_URI      = "bolt://localhost:7687"   # 修改为你的 Neo4j 地址
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "neo4jneo4j"           # 修改为你的密码
NEO4J_DATABASE = "laws"                   # 企业版可指定库名，社区版保持 neo4j

# ─── 批量写入配置 ──────────────────────────────────────────────────────────────
BATCH_SIZE = 500   # 每批提交的三元组数量，可根据内存调整

# ─── 全局锁 ───────────────────────────────────────────────────────────────────
print_lock = Lock()


# ══════════════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════════════

def log(message: str, level: str = "info"):
    icons = {
        "info":    "[INFO]",
        "success": "[OK]  ",
        "warning": "[WARN]",
        "error":   "[ERR] ",
        "step":    "[-->] ",
    }
    ts = time.strftime("%H:%M:%S")
    icon = icons.get(level, "[ ]  ")
    with print_lock:
        print(f"[{ts}] {icon} {message}")


def print_progress(current: int, total: int, prefix: str = "Progress", bar_len: int = 50):
    pct = current / total if total else 0
    filled = int(bar_len * pct)
    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stdout.write(f"\r{prefix} |{bar}| {pct*100:.1f}%  {current}/{total}  ")
    sys.stdout.flush()


def sanitize(text: str) -> str:
    """移除 Cypher 注入风险字符，保持中文完整。"""
    return text.strip().replace("\\", "\\\\").replace('"', '\\"')


# ══════════════════════════════════════════════════════════════════════════════
#  Neo4j 操作封装
# ══════════════════════════════════════════════════════════════════════════════

class KnowledgeGraphBuilder:
    def __init__(self, uri: str, user: str, password: str, database: str):
        log(f"连接 Neo4j: {uri}", "step")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self._verify_connection()

    def _verify_connection(self):
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS ping")
                result.single()
            log("Neo4j 连接成功", "success")
        except Exception as e:
            log(f"Neo4j 连接失败: {e}", "error")
            raise

    def close(self):
        self.driver.close()

    # ── 索引 & 约束 ────────────────────────────────────────────────────────────
    def create_indexes(self):
        """
        为 Entity 节点的 name 属性创建全文索引和唯一约束，
        加速 MERGE 操作。
        """
        log("创建索引与约束...", "step")
        queries = [
            # 唯一约束（同时隐式创建 B-Tree 索引）
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE",

            # 全文索引（支持中文模糊搜索）
            "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.name]",
        ]
        with self.driver.session(database=self.database) as session:
            for q in queries:
                try:
                    session.run(q)
                except neo4j_exceptions.ClientError as e:
                    # 约束/索引已存在时忽略
                    if "already exists" not in str(e):
                        log(f"索引创建警告: {e}", "warning")
        log("索引创建完毕", "success")

    # ── 批量写入 ────────────────────────────────────────────────────────────────
    def import_batch(self, session, batch: list[dict], source_file: str):
        """
        使用 UNWIND 批量 MERGE 节点和关系，一次事务写入 BATCH_SIZE 条。
        关系类型动态来自 relation 字段。
        """
        # Neo4j 关系类型不能含特殊字符，需规范化
        # 用参数传递 relation 字符串，借助 apoc.merge.relationship（若有 APOC）
        # 无 APOC 时，同一 batch 内按 relation 分组分别写入
        by_relation: dict[str, list] = {}
        for row in batch:
            rel = row["relation"]
            by_relation.setdefault(rel, []).append(row)

        total_merged = 0
        for relation, rows in by_relation.items():
            # 关系类型只保留字母/数字/中文，空格转下划线
            safe_rel = _normalize_relation(relation)
            cypher = f"""
            UNWIND $rows AS row
            MERGE (s:Entity {{name: row.subject}})
              ON CREATE SET s.created_at = timestamp()
            MERGE (o:Entity {{name: row.object}})
              ON CREATE SET o.created_at = timestamp()
            MERGE (s)-[r:`{safe_rel}`]->(o)
              ON CREATE SET
                r.source     = row.source,
                r.created_at = timestamp(),
                r.raw_rel    = row.relation
              ON MATCH SET
                r.source     = row.source
            """
            params = [
                {
                    "subject":  r["subject"],
                    "object":   r["object"],
                    "source":   source_file,
                    "relation": r["relation"],
                }
                for r in rows
            ]
            session.run(cypher, rows=params)
            total_merged += len(rows)
        return total_merged

    def import_json_file(self, json_path: str) -> tuple[int, int]:
        """
        导入单个 JSON 文件，返回 (成功三元组数, 失败三元组数)。
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log(f"读取 JSON 失败 {json_path}: {e}", "error")
            return 0, 0

        triplets: list[dict] = data.get("triplets", [])
        source_file: str     = data.get("source_file", os.path.basename(json_path))

        if not triplets:
            log(f"文件无三元组，跳过: {source_file}", "warning")
            return 0, 0

        # 过滤无效条目
        valid   = [t for t in triplets if t.get("subject") and t.get("relation") and t.get("object")]
        invalid = len(triplets) - len(valid)

        imported = 0
        with self.driver.session(database=self.database) as session:
            # 分批写入
            for start in range(0, len(valid), BATCH_SIZE):
                batch = valid[start: start + BATCH_SIZE]
                try:
                    with session.begin_transaction() as tx:
                        count = self.import_batch(tx, batch, source_file)
                        tx.commit()
                        imported += count
                except Exception as e:
                    log(f"批次写入失败 ({source_file} 第{start//BATCH_SIZE+1}批): {e}", "error")

        return imported, invalid

    # ── 统计查询 ────────────────────────────────────────────────────────────────
    def get_statistics(self) -> dict:
        with self.driver.session(database=self.database) as session:
            node_count = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            rel_count  = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            rel_types  = session.run(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN count(relationshipType) AS c"
            ).single()["c"]
        return {
            "nodes":         node_count,
            "relationships": rel_count,
            "rel_types":     rel_types,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_relation(rel: str) -> str:
    """
    将关系字符串规范化为合法的 Neo4j 关系类型：
    - 保留中文、英文字母、数字、下划线
    - 空格 -> 下划线
    - 其余特殊字符剥除
    """
    import re
    rel = rel.strip().replace(" ", "_")
    # 保留汉字、字母、数字、下划线
    rel = re.sub(r"[^\u4e00-\u9fff\w]", "", rel)
    return rel or "RELATED_TO"


# ══════════════════════════════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    input_dir = "./3entity"   # 上一步生成的 JSON 目录

    log("法律知识图谱构建工具启动", "info")
    log(f"读取目录: {input_dir}", "info")

    # 检查目录
    if not os.path.exists(input_dir):
        log(f"目录不存在: {input_dir}", "error")
        return

    json_files = sorted(Path(input_dir).glob("*.json"))
    total = len(json_files)
    if total == 0:
        log("未找到 JSON 文件，退出", "warning")
        return
    log(f"共发现 {total} 个 JSON 文件", "info")

    # 连接并初始化图谱
    builder = KnowledgeGraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE)
    builder.create_indexes()

    # 开始导入
    print("=" * 70)
    log("开始批量导入...", "step")

    total_imported = 0
    total_invalid  = 0
    file_success   = 0
    file_failed    = 0
    start_time     = time.time()

    for idx, json_file in enumerate(json_files, start=1):
        print_progress(idx - 1, total, prefix="导入进度")
        log(f"处理: {json_file.name}", "step")

        imported, invalid = builder.import_json_file(str(json_file))

        if imported > 0:
            log(f"导入 {imported} 条三元组 (跳过无效: {invalid})", "success")
            total_imported += imported
            total_invalid  += invalid
            file_success   += 1
        else:
            log(f"文件导入失败或无内容: {json_file.name}", "warning")
            file_failed += 1

    print_progress(total, total, prefix="导入进度")
    print("\n")

    # 统计
    elapsed = time.time() - start_time
    stats   = builder.get_statistics()
    builder.close()

    print("=" * 70)
    log("知识图谱构建完成！", "success")
    print(f"  耗时            : {elapsed:.1f} 秒")
    print(f"  处理文件        : {total} 个  (成功 {file_success} / 失败 {file_failed})")
    print(f"  导入三元组      : {total_imported} 条  (无效跳过 {total_invalid} 条)")
    print("-" * 70)
    print(f"  [Neo4j 图谱统计]")
    print(f"  实体节点 (Entity): {stats['nodes']:,}")
    print(f"  关系总数         : {stats['relationships']:,}")
    print(f"  关系类型数       : {stats['rel_types']}")
    print("=" * 70)
    log("可在 Neo4j Browser 中使用以下 Cypher 探索图谱:", "info")
    print("""
  // 查看所有节点数量
  MATCH (n:Entity) RETURN count(n)

  // 查看某实体的所有关系
  MATCH (n:Entity {name: '你的实体名'})-[r]->(m) RETURN n,r,m LIMIT 50

  // 查看出度最高的实体（关键节点）
  MATCH (n:Entity)-[r]->()
  RETURN n.name, count(r) AS degree
  ORDER BY degree DESC LIMIT 20

  // 全文检索（需全文索引）
  CALL db.index.fulltext.queryNodes('entity_fulltext', '合同')
  YIELD node, score RETURN node.name, score LIMIT 20

  // 两实体间最短路径
  MATCH p=shortestPath(
    (a:Entity {name:'甲方'})-[*..10]-(b:Entity {name:'违约金'})
  ) RETURN p
    """)


if __name__ == "__main__":
    main()
