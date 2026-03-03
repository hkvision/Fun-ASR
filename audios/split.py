import os
import glob
import random
import math

# 设置随机种子，保证每次运行划分结果一致
random.seed(42)

def get_files_data(base_path, source_category, target_hotwords):
    """
    遍历指定目录，收集符合热词要求的数据。
    """
    collected_data = []
    
    # 获取绝对路径
    search_root = os.path.join(os.getcwd(), base_path)
    
    if not os.path.exists(search_root):
        print(f"[Error] 路径不存在: {search_root}")
        return collected_data, []

    # 记录找到的热词文件夹
    found_hotwords = []
    
    # 获取目录下实际存在的文件夹列表
    existing_folders = os.listdir(search_root)

    for folder_name in existing_folders:
        # 1. 文件夹名称过滤 (精确匹配)
        if folder_name not in target_hotwords:
            continue
            
        hotword_path = os.path.join(search_root, folder_name)
        if not os.path.isdir(hotword_path):
            continue

        found_hotwords.append(folder_name)

        # ==========================================
        # 修复点：使用 set() 去重，防止 Windows 下 *.wav 和 *.WAV 重复扫描同一文件
        # ==========================================
        raw_files = glob.glob(os.path.join(hotword_path, "*.wav")) + \
                    glob.glob(os.path.join(hotword_path, "*.WAV"))
        wav_files = list(set(raw_files)) # 去重操作
        
        for wav_path in wav_files:
            file_name = os.path.basename(wav_path)
            base_name = os.path.splitext(file_name)[0]
            
            final_txt_path = None
            
            # ==========================================
            # 针对 TTS 的特殊匹配逻辑
            # ==========================================
            if source_category == "tts":
                # 规则：01_v1_....wav -> 01.txt
                try:
                    prefix = file_name.split('_')[0] # 获取 "01"
                    if prefix.isdigit():
                        txt_candidate = os.path.join(hotword_path, f"{prefix}.txt")
                        if os.path.exists(txt_candidate):
                            final_txt_path = txt_candidate
                except Exception:
                    pass
            
            # ==========================================
            # 普通数据的匹配逻辑
            # ==========================================
            else:
                # 优先级: _m.txt > .txt
                txt_m = os.path.join(hotword_path, base_name + "_m.txt")
                txt_n = os.path.join(hotword_path, base_name + ".txt")
                
                if os.path.exists(txt_m):
                    final_txt_path = txt_m
                elif os.path.exists(txt_n):
                    final_txt_path = txt_n
            
            # ==========================================
            # 如果找到了对应的 txt，读取并添加
            # ==========================================
            if final_txt_path:
                try:
                    with open(final_txt_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip().replace('\n', ' ').replace('\t', ' ')
                    
                    # 生成唯一ID: 热词_来源_文件名
                    unique_id = f"{folder_name}_{source_category}_{base_name}" 

                    collected_data.append({
                        'id': unique_id,
                        'wav_path': wav_path,
                        'text': content,
                        'hotword': folder_name
                    })
                except Exception as e:
                    print(f"[Warning] 读取文本失败 {final_txt_path}: {e}")

    return collected_data, found_hotwords

def smart_split(data_list, ratios, priority_mode='fraction'):
    """
    智能划分数据集
    priority_mode: 
      - 'real_meeting': 优先填补 Train 和 Test
      - 'fraction': 最大余额法 (谁的小数部分大给谁)
    """
    total = len(data_list)
    if total == 0:
        return [], [], []

    random.shuffle(data_list)

    # 1. 计算理论数量
    float_counts = [total * r for r in ratios] 
    
    # 2. 初始整数分配
    int_counts = [int(x) for x in float_counts]
    
    # 3. 计算剩余名额
    remainder = total - sum(int_counts)
    
    # 4. 分配剩余名额
    if remainder > 0:
        if priority_mode == 'real_meeting':
            # 规则：优先给 Train (0) 和 Test (2)，最后 Val (1)
            priority_indices = [0, 2, 1] 
            for i in range(remainder):
                idx = priority_indices[i % 3]
                int_counts[idx] += 1
        else:
            # 规则：最大余额法
            fractions = [(i, c - int(c)) for i, c in enumerate(float_counts)]
            fractions.sort(key=lambda x: x[1], reverse=True)
            
            for i in range(remainder):
                idx = fractions[i][0]
                int_counts[idx] += 1

    # 5. 切分
    n_train = int_counts[0]
    n_val = int_counts[1]
    n_test = int_counts[2]
    
    train_set = data_list[:n_train]
    val_set = data_list[n_train : n_train + n_val]
    test_set = data_list[n_train + n_val :]
    
    return train_set, val_set, test_set

def main():
    # 目标热词列表
    with open("hotwords.txt", "r", encoding="utf-8") as f:
        TARGET_HOTWORDS = [line.strip() for line in f if line.strip()]

    # 定义源: (文件夹路径, 标识名, (Train/Val/Test比例), 划分模式)
    sources = [
        ("real_meeting", "real", (0.4, 0.2, 0.4), "real_meeting"),
        # (os.path.join("Intel_hotword_99", "Junlong"), "junlong", (0.6, 0.2, 0.2), "fraction"),
        # (os.path.join("Intel_hotword_99", "Sunny"), "sunny", (0.6, 0.2, 0.2), "fraction"),
        (os.path.join("Intel_hotword_99", "tts"), "tts", (0.6, 0.2, 0.2), "fraction")
    ]
    
    # sources = [
    #     (os.path.join("Intel_hotword_99", "Junlong"), "junlong", (0.0, 0.0, 1.0), "fraction"),
    #     (os.path.join("Intel_hotword_99", "Sunny"), "sunny", (0.0, 0.0, 1.0), "fraction"),
    #     (os.path.join("Intel_hotword_99", "Yanzhang"), "yanzhang", (0.0, 0.0, 1.0), "fraction")
    # ]

    all_train = []
    all_val = []
    all_test = []

    print(">>> 开始处理数据...")

    for path, source_name, ratios, mode in sources:
        print(f"\n正在扫描: {path} ...")
        
        raw_data, found_hws = get_files_data(path, source_name, TARGET_HOTWORDS)
        
        if not found_hws:
            print(f"  [提示] 未在此路径下找到目标热词文件夹 (正常跳过)")
            continue
        
        # 按热词归类
        data_by_hotword = {hw: [] for hw in TARGET_HOTWORDS}
        for item in raw_data:
            data_by_hotword[item['hotword']].append(item)
        
        # 对每个热词进行划分
        for hw in TARGET_HOTWORDS:
            items = data_by_hotword[hw]
            if not items:
                continue
                
            tr, va, te = smart_split(items, ratios, priority_mode=mode)
            
            all_train.extend(tr)
            all_val.extend(va)
            all_test.extend(te)
            
            print(f"  - [{source_name}] {hw:<15}: 总数 {len(items):<3} -> Train {len(tr)}, Val {len(va)}, Test {len(te)}")

    # 写入文件
    print("\n>>> 正在写入输出文件...")
    
    outputs = [
        ("train", all_train),
        ("val", all_val),
        ("test", all_test)
        # ("different_humans", all_test)
    ]
    
    for prefix, dataset in outputs:
        if len(dataset) > 0:
            wav_scp_name = f"{prefix}_wav.scp"
            text_txt_name = f"{prefix}_text.txt"
            
            dataset.sort(key=lambda x: x['id'])
            
            with open(wav_scp_name, 'w', encoding='utf-8') as f_wav, \
                open(text_txt_name, 'w', encoding='utf-8') as f_txt:
                for item in dataset:
                    f_wav.write(f"{item['id']}\t{item['wav_path']}\n")
                    f_txt.write(f"{item['id']}\t{item['text']}\n")
                    
            print(f"已生成: {wav_scp_name:<15} (共 {len(dataset)} 条)")
            print(f"已生成: {text_txt_name:<15}")

    print("\n所有操作完成！")

if __name__ == "__main__":
    main()