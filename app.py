#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import jieba
import jieba.posseg as pseg
from gensim.models import FastText
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import numpy as np
import re
import os
import streamlit.components.v1 as components

from lime.lime_text import LimeTextExplainer

# ==========================================
# 0. 页面配置与字体安全检查
# ==========================================
st.set_page_config(page_title="文风分析实验室", layout="wide")

@st.cache_resource
def get_font_prop():
    font_path = "simhei.ttf"
    if not os.path.exists(font_path): return None
    try:
        if os.path.getsize(font_path) / (1024 * 1024) < 1: return None
        return fm.FontProperties(fname=font_path)
    except: return None

my_font_prop = get_font_prop()

# ==========================================
# 1. 核心工具函数
# ==========================================

def read_content_safe(file_obj, limit=None):
    """安全读取文件内容"""
    try:
        file_obj.seek(0)
        content_bytes = file_obj.read()
        text = content_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = content_bytes.decode('gbk')
        except UnicodeDecodeError:
            text = content_bytes.decode('utf-8', errors='ignore')
    if limit: return text[:limit]
    return text

def basic_clean(text):
    """基础清洗"""
    if not isinstance(text, str): return ""
    text = re.sub(r'第.+?章.*', '', text)
    text = re.sub(r'Chapter.*', '', text)
    text = re.sub(r'\[\d+\]|[\u2460-\u24FF]', '', text)
    text = re.sub(r'["“”]', '', text)
    punctuation_map = {',': '，', '!': '！', '?': '？', '(': '（', ')': '）', ':': '：', ';': '；'}
    for eng_punc, chi_punc in punctuation_map.items():
        text = text.replace(eng_punc, chi_punc)
    return text

def split_sentences_custom(text, min_len=30):
    """
    自定义分句函数：
    1. 凑够 min_len (30字) 才断句（针对逗号）。
    2. 遇到强结束符（句号/感叹号/问号/换行）必须立刻断句，不管长度够不够。
    """
    # 切分：保留标点
    raw_sents = re.split(r'([,，。！？\n]+)', text)
    merged_sents = []
    buffer = ""
    
    strong_terminators = {'。', '！', '？', '\n', '!', '?'}
    
    for i in range(0, len(raw_sents) - 1, 2):
        content = raw_sents[i]
        punct = raw_sents[i+1]
        
        segment = content + punct
        buffer += segment
        
        is_strong_end = any(c in punct for c in strong_terminators)
        if len(buffer) >= min_len or is_strong_end:
            merged_sents.append(buffer)
            buffer = ""
    if raw_sents[-1]: buffer += raw_sents[-1]
    if buffer: merged_sents.append(buffer)
    
    return [s for s in merged_sents if s.strip()]

def smart_chunking(text, min_length=300):
    """智能切分长文本用于训练"""
    lines = text.split("\n")
    final_chunks = []
    current_chunk = ""
    for line in lines:
        line = line.strip()
        if len(line) < 2: continue
        current_chunk += line + " "
        if len(current_chunk) >= min_length:
            final_chunks.append(current_chunk)
            current_chunk = ""
    if len(current_chunk) > 50:
        final_chunks.append(current_chunk)
    return final_chunks

def get_style_tokens(text, blocklist):
    """文风分词：基于停用词表过滤"""
    text = basic_clean(text)
    words = jieba.lcut(text)
    return [w for w in words if w not in blocklist and not w.isspace()]

def generate_blocklist_from_files(uploaded_files):
    """自动生成停用词表：实词"""
    sample_text = ""
    for uploaded_file in uploaded_files:
        content = read_content_safe(uploaded_file)
        sample_text += basic_clean(content)[:200000]
    words = pseg.cut(sample_text)
    candidates = []
    target_flags = {'nr', 'ns', 'nz', 'nt', 'per', 'loc'}
    for w, f in words:
        if len(w) > 1 and f in target_flags:
            candidates.append(w)
    from collections import Counter
    blocklist = set([w for w, c in Counter(candidates).most_common(500)])
    return blocklist

def get_color_html(text, weight):
    val = abs(weight)
    if val < 0.001: return text 
    
    intensity = min(val * 10, 0.7) 
    intensity = max(intensity, 0.15)

    if weight > 0:
        # 正向：亮红色
        rgba = f"rgba(255, 60, 60, {intensity})" 
    else:
        # 负向：亮蓝色
        rgba = f"rgba(0, 160, 255, {intensity})"
        
    return f"<span style='background-color: {rgba}; padding: 2px 4px; border-radius: 4px;'>{text}</span>"

# ==========================================
# 2. 网站界面 UI
# ==========================================

st.title("🕵️‍♂️ 文风分析实验室")
st.markdown("""
上传某位作家的原著，再输入你的同人文本，算法将根据虚词、句式等（而非剧情内容）计算文风相似度，并输出最具原著味的句子。
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Step 1: 上传原著文本")
    st.info("请上传原著 TXT 文件（可多选）")
    uploaded_originals = st.file_uploader("上传原著 (支持 .txt)", type="txt", accept_multiple_files=True)

    st.header("Step 2: 输入测试文本")
    fanfic_text = st.text_area("在此粘贴你的同人文本：", height=200, placeholder="建议粘贴 500 字以上的文本...")

    start_btn = st.button("🚀 开始文风分析", type="primary")

# ==========================================
# 3. 主逻辑控制器
# ==========================================

if start_btn:
    if not uploaded_originals:
        st.error("❌ 请先上传原著文件！")
    elif not fanfic_text.strip():
        st.error("❌ 请输入测试文本！")
    else:
        with col2:
            # === 阶段一：基础模型构建与计算 ===
            status = st.status("正在进行全流程分析...", expanded=True)
            
            # 1. 预处理
            status.write("📖 正在扫描原著并构建实词停用词表...")
            blocklist = generate_blocklist_from_files(uploaded_originals)
            status.write(f"✅ 已停用 {len(blocklist)} 个高频专有名词（如：{list(blocklist)[:5]}...）")
            
            # 2. 数据切分
            status.write("✂️ 正在进行文本切片与清洗...")
            original_docs = []
            for u_file in uploaded_originals:
                content = read_content_safe(u_file)
                chunks = smart_chunking(content)
                for chunk in chunks:
                    tokens = get_style_tokens(chunk, blocklist)
                    if len(tokens) > 50: original_docs.append(tokens)
            
            # 3. 处理同人文本
            preview_text = fanfic_text # [:3000] 
            test_tokens = get_style_tokens(preview_text, blocklist)
            
            if len(test_tokens) < 10:
                status.update(label="分析失败：有效词汇不足", state="error")
                st.stop()

            # 4. 训练模型
            status.write("🧠 正在训练 FastText 文风向量空间...")
            all_docs = original_docs + [test_tokens]
            model = FastText(sentences=all_docs, vector_size=100, window=5, min_count=1, epochs=20, seed=42)
            
            # 5. 计算基准与得分
            def get_vec(tokens):
                vecs = [model.wv[w] for w in tokens if w in model.wv]
                return np.mean(vecs, axis=0) if vecs else np.zeros(100)

            orig_vecs = np.array([get_vec(d) for d in original_docs])
            gold_standard = np.mean(orig_vecs, axis=0) 
            test_vecs = get_vec(test_tokens)
            
            similarity = cosine_similarity([test_vecs], [gold_standard])[0][0]
            final_score = similarity * 100
            
            status.write("✅ 基础分析完成，准备进行句子归因...")

            # === 阶段二：结果展示 (基础部分) ===
            st.divider()
            st.subheader("📊 相似度分析报告")
            
            res_c1, res_c2 = st.columns([1, 1])
            
            with res_c1:
                st.metric(label="整体文风相似度", value=f"{final_score:.2f}%")
                
                if final_score > 90:
                    st.success("""
                    **判定：疑似作者小号（Tier S）** 
                    😭 **救命！这是哪位神仙太太下凡？** 这简直就是原著！若不是作者的小号，建议严查是否偷了存稿硬盘。  
                    *评价：绝赞好粮，垂直入坑，请受我一拜！*
                    """)
                elif final_score > 75:
                    st.info("""
                    **判定：美味（Tier A）** 
                    😋 **好一口美味的粮！** 虽然在细节处能看出太太自己的行文习惯，但整体还原度极高。  
                    *评价：是不可多得的优质粮，这就加入书架！*
                    """)
                elif final_score > 60:
                    st.warning("""
                    **判定：自带滤镜的AU感（Tier B）** 
                    🤔 **这是什么奇怪的pa吗？** 虽然还在同人的范畴里，但是私设比较多呢。  
                    *评价：熟悉的陌生人，仿佛在OOC边缘试探（）*
                    """)
                else:
                    st.error("""
                    **判定：OOC预警 / 纯属原创（Tier C）** 
                    😨 **确定这是同人？** 这独特的文风已经完全脱离了原著的引力圈，如果不看角色名，机器还以为误入了隔壁片场。  
                    *评价：这是极致的OOC，还是披着同人皮的原创大作？这很难评，祝您开心就好。*
                    """)
            
            with res_c2:
                st.write("### 向量空间投影")
                if len(orig_vecs) > 0:
                    try:
                        pca = PCA(n_components=2)
                        X_all = np.vstack([orig_vecs, [test_vecs]])
                        X_pca = pca.fit_transform(X_all)
                        n_orig = len(orig_vecs)

                        fig, ax = plt.subplots(figsize=(6, 4))
                        fig.patch.set_alpha(0.0) 
                        ax.patch.set_alpha(0.0)   

                        ax.scatter(X_pca[:n_orig, 0], X_pca[:n_orig, 1], c='lightgray', s=10, alpha=0.5, label='原著切片')
                        center = pca.transform([gold_standard])
                        ax.scatter(center[:,0], center[:,1], c='red', marker='*', s=200, label='原著基准')
                        ax.scatter(X_pca[n_orig:, 0], X_pca[n_orig:, 1], c='blue', s=80, marker='X', label='你的文本')

                        if my_font_prop:
                            ax.legend(prop=my_font_prop, frameon=False) 
                            ax.set_title("文风落点分布", fontproperties=my_font_prop)
                        else:
                            ax.legend(frameon=False)
                            ax.set_title("Style Distribution (Font Missing)")

                        ax.axis('off')
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"绘图出错: {e}")
                        
                        
            # === 阶段三：句子级 LIME 进阶分析 ===
            st.divider()
            st.subheader("🔍 深度归因：哪些句子最像原著？")
            st.info("正在逐句分析文风贡献度（红色=加分项，蓝色=减分项）...")
            
            # 1. 切分句子
            sentences_list = split_sentences_custom(preview_text)
            wrapped_text = " ".join([str(i) for i in range(len(sentences_list))])
            
            def sentence_predict_proba(str_indices_list):
                results = []
                for str_indices in str_indices_list:
                    indices = [int(i) for i in str_indices.split()]
                    reconstructed_text = "".join([sentences_list[i] for i in indices])
                    
                    t_tokens = get_style_tokens(reconstructed_text, blocklist)
                    if not t_tokens:
                        results.append([1.0, 0.0])
                        continue
                    
                    vec = get_vec(t_tokens)
                    sim = cosine_similarity([vec], [gold_standard])[0][0]
                    
                    sim_scaled = sim ** 3
                    results.append([1 - sim_scaled, sim_scaled])
                return np.array(results)

            try:
                explainer = LimeTextExplainer(class_names=['差异', '原著风'])
                
                exp = explainer.explain_instance(
                    wrapped_text, 
                    sentence_predict_proba, 
                    num_features=len(sentences_list), 
                    num_samples=150 
                )
                
                weights = exp.as_list()
                weight_map = {int(k): v for k, v in weights}
                
                num_sentences = len(sentences_list)
                top_k_count = max(int(num_sentences * 0.1), 1)
                
                sorted_by_val = sorted(weight_map.items(), key=lambda x: x[1], reverse=True)
                top_pos_indices = set(k for k, v in sorted_by_val[:top_k_count] if v > 0)
                
                sorted_by_val_asc = sorted(weight_map.items(), key=lambda x: x[1])
                top_neg_indices = set(k for k, v in sorted_by_val_asc[:top_k_count] if v < 0)
                
                highlight_indices = top_pos_indices.union(top_neg_indices)

                st.write(f"### 📜 全文文风热力图 ")
                st.caption("红色 = 极具原著神韵的短句；蓝色 = 明显偏离原著风格的短句；无底色 = 文风特征不明显")
                
                html_parts = []
                for idx, sentence in enumerate(sentences_list):
                    weight = weight_map.get(idx, 0)
                    if idx in highlight_indices:
                        html_parts.append(get_color_html(sentence, weight))
                    else:
                        html_parts.append(f"<span>{sentence}</span>")
                
                full_html = f"<div style='line-height: 1.8; font-family: serif; padding: 15px; border: 1px solid #ddd; border-radius: 5px; height: 400px; overflow-y: auto;'>{''.join(html_parts)}</div>"
                st.markdown(full_html, unsafe_allow_html=True)
                
                st.write("### 🏆 最具“原著味”的短句 TOP 5")
                top_sentences_data = []
                for idx, score in sorted_by_val[:5]:
                    if score > 0:
                        top_sentences_data.append({
                            "排名": len(top_sentences_data) + 1,
                            "句子内容": sentences_list[idx],
                            "贡献度得分": f"{score:.4f}"
                        })
                
                if top_sentences_data:
                    st.table(pd.DataFrame(top_sentences_data).set_index("排名"))
                else:
                    st.write("未检测到显著的正向特征句子。")

                status.update(label="全流程分析圆满完成！", state="complete", expanded=False)

            except Exception as e:
                st.error(f"LIME 分析过程出错: {e}")
                status.update(label="分析中断", state="error")

