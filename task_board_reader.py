#!/usr/bin/env python3
"""Notion Task Board から「📥 新規」セクションのタスクを読み出して表示する。"""

import os
import sys
from datetime import datetime, timezone, timedelta

try:
    from notion_client import Client
    from notion_client.errors import APIResponseError
except ImportError:
    print("エラー：notion-client がインストールされていません。")
    print("  pip install notion-client を実行してください。")
    sys.exit(1)

PAGE_ID = "35b5ae2b8d6a8198852ddedbf8621eb5"
TARGET_HEADING = "📥 新規"
JST = timezone(timedelta(hours=9))


def get_api_key() -> str:
    key = os.environ.get("NOTION_API_KEY", "")
    if not key:
        print("エラー：環境変数 NOTION_API_KEY が設定されていません。")
        print("  export NOTION_API_KEY='your_secret_key' を実行してください。")
        sys.exit(1)
    return key


def fetch_page_blocks(client: Client, block_id: str) -> list:
    """ページ内のすべてのブロックを再帰なしで取得する（ページネーション対応）。"""
    blocks = []
    cursor = None
    while True:
        kwargs = {"block_id": block_id}
        if cursor:
            kwargs["start_cursor"] = cursor
        try:
            response = client.blocks.children.list(**kwargs)
        except APIResponseError as e:
            if e.status == 401:
                print("エラー：API キーが無効です。NOTION_API_KEY を確認してください。")
            elif e.status == 403:
                print("エラー：このページへのアクセス権がありません。")
                print("  Notion でインテグレーションをページに招待してください。")
            elif e.status == 404:
                print("エラー：指定されたページが見つかりません。PAGE_ID を確認してください。")
            else:
                print(f"エラー：Notion API エラー（ステータス {e.status}）: {e.body}")
            sys.exit(1)
        blocks.extend(response["results"])
        if not response.get("has_more"):
            break
        cursor = response["next_cursor"]
    return blocks


def extract_rich_text(rich_text_list: list) -> str:
    return "".join(rt["plain_text"] for rt in rich_text_list)


def get_block_text(block: dict) -> str:
    btype = block["type"]
    content = block.get(btype, {})
    rich = content.get("rich_text", [])
    return extract_rich_text(rich)


def is_target_heading(block: dict) -> bool:
    btype = block["type"]
    if btype not in ("heading_1", "heading_2", "heading_3"):
        return False
    text = get_block_text(block)
    return TARGET_HEADING in text


def is_any_heading(block: dict) -> bool:
    return block["type"] in ("heading_1", "heading_2", "heading_3")


def fetch_children_if_any(client: Client, block: dict) -> list:
    if block.get("has_children"):
        return fetch_page_blocks(client, block["id"])
    return []


def collect_tasks_from_blocks(client: Client, blocks: list) -> list[dict]:
    """
    ブロック列から TARGET_HEADING の直後にあるタスクを収集する。
    heading_1/2/3 の has_children が True の場合は子ブロックも探索する。
    子ブロック内に TARGET_HEADING がある場合も対応。
    """
    tasks = []

    def walk(block_list: list):
        in_section = False
        for block in block_list:
            btype = block["type"]

            if is_target_heading(block):
                in_section = True
                # heading 自体が子を持つ場合（Notion のトグル heading 等）
                if block.get("has_children"):
                    children = fetch_children_if_any(client, block)
                    tasks.extend(collect_tasks_from_children(client, children))
                continue

            if in_section:
                if is_any_heading(block):
                    in_section = False
                    # 再帰で子を確認
                    if block.get("has_children"):
                        children = fetch_children_if_any(client, block)
                        walk(children)
                    continue

                task = extract_task(client, block)
                if task:
                    tasks.append(task)

            # TARGET_HEADING がまだ見つかっていない場合も子を探索
            if not in_section and block.get("has_children"):
                children = fetch_children_if_any(client, block)
                walk(children)

    walk(blocks)
    return tasks


def collect_tasks_from_children(client: Client, blocks: list) -> list[dict]:
    """子ブロック群からタスクを直接収集する（見出しで区切らない）。"""
    tasks = []
    for block in blocks:
        task = extract_task(client, block)
        if task:
            tasks.append(task)
    return tasks


def extract_task(client: Client, block: dict) -> dict | None:
    """ブロックからタスク情報を抽出する。"""
    btype = block["type"]

    # to_do ブロック
    if btype == "to_do":
        content = block["to_do"]
        title = extract_rich_text(content.get("rich_text", []))
        checked = content.get("checked", False)
        detail_lines = []
        if block.get("has_children"):
            children = fetch_page_blocks(client, block["id"])
            for child in children:
                line = get_block_text(child)
                if line:
                    detail_lines.append(line)
        return {
            "title": title or "(無題)",
            "checked": checked,
            "details": detail_lines,
            "type": "todo",
        }

    # bulleted / numbered list
    if btype in ("bulleted_list_item", "numbered_list_item"):
        title = get_block_text(block)
        detail_lines = []
        if block.get("has_children"):
            children = fetch_page_blocks(client, block["id"])
            for child in children:
                line = get_block_text(child)
                if line:
                    detail_lines.append(line)
        return {
            "title": title or "(無題)",
            "checked": False,
            "details": detail_lines,
            "type": btype,
        }

    # paragraph（空でなければタスクとして扱う）
    if btype == "paragraph":
        title = get_block_text(block)
        if not title.strip():
            return None
        detail_lines = []
        if block.get("has_children"):
            children = fetch_page_blocks(client, block["id"])
            for child in children:
                line = get_block_text(child)
                if line:
                    detail_lines.append(line)
        return {
            "title": title,
            "checked": False,
            "details": detail_lines,
            "type": "paragraph",
        }

    # child_page
    if btype == "child_page":
        title = block["child_page"].get("title", "(無題)")
        return {
            "title": title,
            "checked": False,
            "details": [],
            "type": "child_page",
        }

    return None


def print_tasks(tasks: list[dict]):
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S JST")
    print("=== Task Board：新規タスク ===")
    print(f"[{now}]")
    print()

    if not tasks:
        print("現在新規タスクはありません")
        return

    for i, task in enumerate(tasks, 1):
        status = "[完了] " if task.get("checked") else ""
        print(f"{i}. {status}{task['title']}")
        if task["details"]:
            for line in task["details"]:
                print(f"   内容：{line}")
        print()


def main():
    api_key = get_api_key()

    try:
        client = Client(auth=api_key)
    except Exception as e:
        print(f"エラー：Notion クライアントの初期化に失敗しました: {e}")
        sys.exit(1)

    blocks = fetch_page_blocks(client, PAGE_ID)
    tasks = collect_tasks_from_blocks(client, blocks)
    print_tasks(tasks)


if __name__ == "__main__":
    main()
