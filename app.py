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
import numpy as np
import re
import os

# ==========================================
# 0. 页面配置与字体处理
# ==========================================
st.set_page_config(page_title="文风指纹分析实验室", layout="wide")

# 解决绘图中文乱码（自动下载字体）
@st.cache_resource
def get_font():
    font_path = "simhei.ttf"
    if not os.path.exists(font_path):
        os.system('wget -O simhei.ttf "https://www.wfonts.com/download/data/2014/06/01/simhei/simhei.ttf"')
    return fm.FontProperties(fname=font_path)

my_font = get_font()
plt.rcParams['font.family'] = my_font.get_name()

# ==========================================
# 1. 核心算法函数 (复用之前的逻辑)
# ==========================================

def basic_clean(text):
    """基础清洗：去章节头、统一标点"""
    if not isinstance(text, str): return ""
    text = re.sub(r'第.+?章.*', '', text)
    text = re.sub(r'Chapter.*', '', text)
    text = re.sub(r'\[\d+\]|[\u2460-\u24FF]', '', text)
    punctuation_map = {',': '，', '!': '！', '?': '？', '(': '（', ')': '）', ':': '：', ';': '；'}
    for eng_punc, chi_punc in punctuation_map.items():
        text = text.replace(eng_punc, chi_punc)
    return text

def smart_chunking(text, min_length=300):
    """智能切分：将文本切分为长段落"""
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
    """文风分词：基于黑名单过滤"""
    text = basic_clean(text)
    words = jieba.lcut(text)
    # 核心：保留不在黑名单里的词（保留虚词、标点、普通动词）
    return [w for w in words if w not in blocklist and not w.isspace()]

def generate_blocklist_from_files(uploaded_files):
    """从上传的文件对象中自动生成黑名单"""
    sample_text = ""
    # 读取所有原著的前 50000 字
    for uploaded_file in uploaded_files:
        # 指针归零，防止二次读取为空
        uploaded_file.seek(0)
        content = uploaded_file.read().decode('utf-8', errors='ignore')
        sample_text += basic_clean(content)[:50000]
    
    words = pseg.cut(sample_text)
    candidates = []
    target_flags = {'nr', 'ns', 'nz', 'nt', 'per', 'loc'}
    
    for w, f in words:
        if len(w) > 1 and f in target_flags:
            candidates.append(w)
            
    # 截取 Top 500 高频实体
    from collections import Counter
    blocklist = set([w for w, c in Counter(candidates).most_common(500)])
    return blocklist

# ==========================================
# 2. 网站界面 UI
# ==========================================

st.title("🕵️‍♂️ 文风指纹分析实验室")
st.markdown("""
这是一个基于 **FastText** 和 **Stylometry (文体学)** 的分析工具。
上传某位作家的原著（如《盗墓笔记》），再输入你的同人文本，算法将自动剥离“内容”，仅根据“文风”计算相似度。
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Step 1: 建立基准")
    st.info("请上传原著 TXT 文件（可多选）。系统将自动学习其文风并建立黑名单。")
    uploaded_originals = st.file_uploader("上传原著 (支持 .txt)", type="txt", accept_multiple_files=True)

    st.header("Step 2: 输入测试文本")
    fanfic_text = st.text_area("在此粘贴你的同人/测试文本：", height=200, placeholder="把要测试的小说片段粘贴在这里...")

    start_btn = st.button("🚀 开始文风分析", type="primary")

# ==========================================
# 3. 主逻辑控制器
# ==========================================

if start_btn:
    if not uploaded_originals:
        st.error("请先上传原著文件！")
    elif not fanfic_text.strip():
        st.error("请输入测试文本！")
    else:
        with col2:
            with st.status("正在进行深度分析...", expanded=True) as status:
                
                # --- 阶段 A: 预处理原著 ---
                st.write("📖 正在读取原著并生成实体黑名单...")
                blocklist = generate_blocklist_from_files(uploaded_originals)
                st.write(f"✅ 已自动屏蔽 {len(blocklist)} 个高频专有名词（如：{list(blocklist)[:5]}...）")

                st.write("✂️ 正在进行智能分段与去噪...")
                original_docs = []
                for u_file in uploaded_originals:
                    u_file.seek(0)
                    content = u_file.read().decode('utf-8', errors='ignore')
                    chunks = smart_chunking(content)
                    for chunk in chunks:
                        tokens = get_style_tokens(chunk, blocklist)
                        if len(tokens) > 50:
                            original_docs.append(tokens)
                
                # --- 阶段 B: 处理测试文本 ---
                test_docs = []
                test_chunks = smart_chunking(fanfic_text, min_length=200) # 测试文本也可以切片
                for chunk in test_chunks:
                    tokens = get_style_tokens(chunk, blocklist)
                    if len(tokens) > 50:
                        test_docs.append(tokens)
                
                if not test_docs:
                    st.error("测试文本太短或有效词汇太少，无法分析。")
                    st.stop()

                # --- 阶段 C: 训练模型 ---
                st.write("🧠 正在训练 FastText 文风模型 (这可能需要几秒钟)...")
                # 混合训练
                all_docs = original_docs + test_docs
                model = FastText(sentences=all_docs, vector_size=100, window=5, min_count=1, epochs=20, seed=42)
                
                # --- 阶段 D: 计算相似度 ---
                def get_vec(tokens):
                    vecs = [model.wv[w] for w in tokens if w in model.wv]
                    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

                orig_vecs = np.array([get_vec(d) for d in original_docs])
                test_vecs = np.array([get_vec(d) for d in test_docs])
                
                gold_standard = np.mean(orig_vecs, axis=0) # 原著质心
                test_centroid = np.mean(test_vecs, axis=0) # 测试文本质心（如果有多段）
                
                similarity = cosine_similarity([test_centroid], [gold_standard])[0][0]
                final_score = similarity * 100
                
                status.update(label="分析完成！", state="complete", expanded=False)

            # ==========================================
            # 4. 结果展示
            # ==========================================
            st.divider()
            st.subheader("分析结果")
            
            # 仪表盘样式显示分数
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(label="文风相似度", value=f"{final_score:.2f}%")
                if final_score > 90:
                    st.success("判定：极度相似（可能是真爱粉或高度模仿）")
                elif final_score > 75:
                    st.info("判定：风格接近（抓住了语感，但略有差异）")
                else:
                    st.warning("判定：差异显著（可能是由于OOC或个人风格强烈）")

            # 可视化绘图
            with metric_col2:
                st.write("### 向量空间投影")
                if len(orig_vecs) > 0:
                    # PCA 降维
                    pca = PCA(n_components=2)
                    X_all = np.vstack([orig_vecs, test_vecs])
                    X_pca = pca.fit_transform(X_all)
                    
                    n_orig = len(orig_vecs)
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    # 原著点（背景）
                    ax.scatter(X_pca[:n_orig, 0], X_pca[:n_orig, 1], c='lightgray', s=10, alpha=0.5, label='原著切片')
                    # 原著中心
                    center = pca.transform([gold_standard])
                    ax.scatter(center[:,0], center[:,1], c='red', marker='*', s=200, label='原著基准')
                    # 测试文本点
                    ax.scatter(X_pca[n_orig:, 0], X_pca[n_orig:, 1], c='blue', s=80, marker='X', label='你的文本')
                    
                    ax.legend(prop=my_font)
                    ax.set_title("文风落点分布", fontproperties=my_font)
                    ax.axis('off') # 去掉坐标轴更美观
                    st.pyplot(fig)

