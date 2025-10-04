import os
import json
from typing import Dict, List, Any
from jload import jsave

all_train = []

def fix_code_block_ending(text):
    if text.endswith('```') and not text.endswith('\n```'):
        text = text[:-3] + '\n```'
    return text

def extract_last_lean4_block(text: str) -> tuple[str, str]:
    marker = "```lean4"
    start_positions = []
    pos = 0

    while True:
        pos = text.find(marker, pos)
        if pos == -1:
            break
        start_positions.append(pos)
        pos += len(marker)

    if not start_positions:
        return (text, "")

    last_start = start_positions[-1]

    end_marker = "```"
    search_from = last_start + len(marker)
    end_pos = text.find(end_marker, search_from)

    if end_pos == -1:
        lean4_block = text[last_start:]
    else:
        lean4_block = text[last_start:end_pos + len(end_marker)]

    before_content = text[:last_start].strip()

    before_lines = before_content.splitlines()
    if before_lines and before_lines[-1].strip().startswith('#'):
        before_lines = before_lines[:-1]
    before_content = '\n'.join(before_lines)

    return before_content, fix_code_block_ending(lean4_block.strip())

def process_data(
    data1: List[Dict[str, Any]],
    data2: Dict[str, Any],
    metadata: Dict[str, str]
) -> None:
    print("─" * 50)
    print(f"🚀 開始處理一組新的數據...")
    print(f"   - 來源檔案 1 (data1): {metadata['code_compilation_path']}")
    print(f"   - 來源檔案 2 (data2): {metadata['full_records_path']}")
    
    
    correct_names = set()
    for d1 in data1:
        if d1["compilation_result"]["pass"] and d1["compilation_result"]["complete"]:
            correct_names.add(d1["name"])
    for d2 in data2:
        if d2["problem_id"] in correct_names:
            train_entry = d2["messages_history_for_this_attempt"]
            train_entry.append({
                "role": "assistant",
                "content": d2["model_output"]
            })
            for t in train_entry:
                if t["role"] is "assistant":
                    a, b = extract_last_lean4_block(t["content"])
                    t["content"] = f"<think>\n{a}\n</think>\n{b}"
            all_train.append({"messages": train_entry})

def find_and_load_correction_files(base_dir: str) -> None:
    """
    遍歷指定的基本目錄，尋找所有 _corr* 文件對並進行處理。

    Args:
        base_dir (str): 要搜索的根目錄路徑。
    """
    if not os.path.isdir(base_dir):
        print(f"錯誤：找不到目錄 '{base_dir}'")
        return

    print(f"正在掃描目錄: {base_dir}\n")
    
    # os.walk 會遞歸地遍歷所有子目錄
    for root, _, files in os.walk(base_dir):
        # 找出所有 code_compilation 的修正檔案
        code_files = [f for f in files if f.startswith('code_compilation_repl') and f.endswith('.json')]
        
        if not code_files:
            continue

        for code_file in code_files:
            # 從 'code_compilation_repl_corr1.json' 中提取後綴 '_corr1.json'
            suffix = code_file.replace('code_compilation_repl', '')
            
            # 構建對應的 full_records 檔案名
            records_file = f'full_records{suffix}'
            
            # 檢查配對的檔案是否存在
            if records_file in files:
                code_file_path = os.path.join(root, code_file)
                records_file_path = os.path.join(root, records_file)
                
                try:
                    # 讀取 data1
                    with open(code_file_path, 'r', encoding='utf-8') as f1:
                        data1 = json.load(f1)
                    
                    # 讀取 data2
                    with open(records_file_path, 'r', encoding='utf-8') as f2:
                        data2 = json.load(f2)

                    # 準備元數據
                    metadata = {
                        "base_directory": root,
                        "code_compilation_path": code_file_path,
                        "full_records_path": records_file_path,
                        "correction_suffix": suffix
                    }
                    
                    # 將兩個數據傳遞給處理函數
                    process_data(data1, data2, metadata)
                
                except json.JSONDecodeError as e:
                    print(f"JSON 解碼錯誤，檔案: {code_file_path} 或 {records_file_path} -> {e}")
                except Exception as e:
                    print(f"讀取或處理檔案時發生未知錯誤: {e}")


if __name__ == "__main__":
    base_directory = "/scratch/gpfs/yl7690/projects/DeepSeek-Prover-V1.5/results_unified/OMR_statements_iter1_filtered"
    
    find_and_load_correction_files(base_directory)
    
    jsave(all_train, "omni32_train.jsonl")