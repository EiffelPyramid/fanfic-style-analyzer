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

# 引入 LIME 库
from lime.lime_text import LimeTextExplainer

# ==========================================
# 0. 页面配置与字体安全检查
# ==========================================
st.set_page_config(page_title="文风指纹分析实验室 (Pro Plus)", layout="wide")

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
    punctuation_map = {',': '，', '!': '！', '?': '？', '(': '（', ')': '）', ':': '：', ';': '；'}
    for eng_punc, chi_punc in punctuation_map.items():
        text = text.replace(eng_punc, chi_punc)
    return text

def split_sentences_custom(text):
    """
    自定义分句函数：保留标点符号
    """
    # 使用正则按句号、感叹号、问号、换行符切分，并保留分隔符
    # 模式解释：([^。！？\n]+[。！？\n]?) 匹配非分隔符开头，以分隔符或结尾结束的串
    sents = re.split(r'([。！？\n]+)', text)
    # re.split 会把分隔符单独切出来，我们需要把它们拼回去
    # 例子：['你好', '！', '再见', '。', '']
    new_sents = []
    for i in range(0, len(sents) - 1, 2):
        new_sents.append(sents[i] + sents[i+1])
    if sents[-1]: new_sents.append(sents[-1])
    # 过滤空句
    return [s for s in new_sents if s.strip()]

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
    """文风分词：基于黑名单过滤"""
    text = basic_clean(text)
    words = jieba.lcut(text)
    # 过滤逻辑：保留非黑名单词且非纯空白
    return [w for w in words if w not in blocklist and not w.isspace()]

def generate_blocklist_from_files(uploaded_files):
    """自动生成内容词黑名单"""
    sample_text = ""
    for uploaded_file in uploaded_files:
        content = read_content_safe(uploaded_file)
        sample_text += basic_clean(content)[:50000]
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
    """根据权重生成带背景色的 HTML span"""
    # 权重通常在 -0.1 到 0.1 之间
    # 正数(红) = 像原著，负数(蓝) = 不像
    # 归一化颜色强度
    intensity = min(abs(weight) * 5, 1.0) # 放大系数，让颜色更明显
    
    if weight > 0:
        # 红色 (255, 0, 0)，透明度变化
        rgba = f"rgba(255, 0, 0, {intensity * 0.5})" 
    else:
        # 蓝色 (0, 0, 255)
        rgba = f"rgba(0, 0, 255, {intensity * 0.5})"
        
    return f"<span style='background-color: {rgba}; padding: 2px; border-radius: 3px;'>{text}</span>"

# ==========================================
# 2. 网站界面 UI
# ==========================================

st.title("🕵️‍♂️ 文风指纹分析实验室 (Sentence LIME Edition)")
st.markdown("""
本系统已升级 **句子级可解释性分析 (Sentence-Level Explainability)**：
AI 将自动识别文中 **最具有原著神韵的句子**（高亮为红色），以及 **最偏离原著风格的句子**（高亮为蓝色）。
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Step 1: 建立基准")
    uploaded_originals = st.file_uploader("上传原著 (支持 .txt)", type="txt", accept_multiple_files=True)
    
    st.header("Step 2: 输入测试文本")
    fanfic_text = st.text_area("在此粘贴同人/测试文本：", height=250, placeholder="建议粘贴 500 字以上的段落...")

    # 一键启动按钮
    start_btn = st.button("🚀 开始全流程分析", type="primary")

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
            status.write("📖 正在扫描原著并构建去噪黑名单...")
            blocklist = generate_blocklist_from_files(uploaded_originals)
            
            # 2. 数据切分
            status.write("✂️ 正在进行文本切片与清洗...")
            original_docs = []
            for u_file in uploaded_originals:
                content = read_content_safe(u_file)
                chunks = smart_chunking(content)
                for chunk in chunks:
                    tokens = get_style_tokens(chunk, blocklist)
                    if len(tokens) > 50: original_docs.append(tokens)
            
            # 3. 处理同人文本 (用于 FastText 训练的 Token)
            # 我们用稍微长一点的文本训练模型，保证语境
            preview_text = fanfic_text[:3000] 
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
            gold_standard = np.mean(orig_vecs, axis=0) # 原著质心
            test_vec = get_vec(test_tokens)
            
            similarity = cosine_similarity([test_vec], [gold_standard])[0][0]
            final_score = similarity * 100
            
            status.write("✅ 基础分析完成，准备进行句子级归因...")

            # === 阶段二：结果展示 (基础部分) ===
            st.divider()
            st.subheader("📊 基础分析报告")
            
            res_c1, res_c2 = st.columns([1, 1])
            
            with res_c1:
                st.metric(label="整体文风相似度", value=f"{final_score:.2f}%")
                
                # 评语逻辑
                if final_score > 90:
                    st.success("**判定：极度相似（Tier S）**\n\n该文本在虚词韵律与句式结构上与原著高度一致，机器判定其具有极高的还原度。")
                elif final_score > 75:
                    st.info("**判定：风格接近（Tier A）**\n\n文本抓住了原著的语感特征，读起来很有原作的味道，但在细节上略有个人色彩。")
                elif final_score > 60:
                    st.warning("**判定：略有差异（Tier B）**\n\n虽然属于同人范畴，但作者保留了强烈的个人叙述风格，文风与原著有明显区别。")
                else:
                    st.error("**判定：差异显著（Tier C）**\n\n机器难以识别出这是基于原著的仿写，可能是一篇完全架空的现代文或OOC作品。")

            with res_c2:
                # 向量图
                if len(orig_vecs) > 0:
                    try:
                        pca = PCA(n_components=2)
                        X_all = np.vstack([orig_vecs, [test_vec]])
                        X_pca = pca.fit_transform(X_all)
                        
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.scatter(X_pca[:-1, 0], X_pca[:-1, 1], c='lightgray', s=15, alpha=0.6, label='原著切片')
                        center = pca.transform([gold_standard])
                        ax.scatter(center[:,0], center[:,1], c='red', marker='*', s=200, label='原著基准')
                        ax.scatter(X_pca[-1, 0], X_pca[-1, 1], c='blue', s=100, marker='X', edgecolors='white', label='你的文本')
                        
                        if my_font_prop:
                            ax.legend(prop=my_font_prop)
                            ax.set_title("文风向量空间分布", fontproperties=my_font_prop)
                        else:
                            ax.axis('off')
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"绘图错误: {e}")

            # === 阶段三：句子级 LIME 进阶分析 ===
            st.divider()
            st.subheader("🔍 深度归因：哪些句子最像原著？")
            st.info("AI 正在逐句分析文风贡献度（红色=加分项，蓝色=减分项）...")
            
            # --- 核心黑科技：句子级 LIME ---
            # 1. 将文本拆分成句子列表
            sentences_list = split_sentences_custom(preview_text)
            
            # 2. 构造“代理文本”：用索引号 "0 1 2 3" 代替实际句子传给 LIME
            # 这样 LIME 就会以为 "0" 是一个词，其实它是第0句
            wrapped_text = " ".join([str(i) for i in range(len(sentences_list))])
            
            # 3. 定义预测函数：LIME 会传进来 ["0 2 3", "1 4"] 这样的索引组合
            # 我们需要把它们还原成句子，再算相似度
            def sentence_predict_proba(str_indices_list):
                results = []
                for str_indices in str_indices_list:
                    # 还原句子
                    indices = [int(i) for i in str_indices.split()]
                    # 拼接成文本
                    reconstructed_text = "".join([sentences_list[i] for i in indices])
                    
                    # 算分
                    t_tokens = get_style_tokens(reconstructed_text, blocklist)
                    if not t_tokens:
                        results.append([1.0, 0.0]) # 空文本不像
                        continue
                    
                    vec = get_vec(t_tokens)
                    sim = cosine_similarity([vec], [gold_standard])[0][0]
                    
                    # 放大差异以便可视化
                    sim_scaled = sim ** 3
                    results.append([1 - sim_scaled, sim_scaled])
                return np.array(results)

            try:
                # 4. 初始化解释器
                explainer = LimeTextExplainer(class_names=['差异', '原著风'])
                
                # 5. 生成解释 (num_features=所有句子)
                # num_samples 可以调低一点提高速度，比如 100-200
                exp = explainer.explain_instance(
                    wrapped_text, 
                    sentence_predict_proba, 
                    num_features=len(sentences_list), 
                    num_samples=150 
                )
                
                # 6. 获取权重：格式为 [('3', 0.12), ('0', -0.05)...]
                weights = exp.as_list()
                # 转换成字典: {句子的Index: 权重}
                weight_map = {int(k): v for k, v in weights}
                
                # === 结果展示 A: 全文热力图 ===
                st.write("### 📜 全文文风热力图")
                st.caption("红色越深代表该句越接近原著文风；蓝色代表该句与原著差异较大。")
                
                html_parts = []
                for idx, sentence in enumerate(sentences_list):
                    weight = weight_map.get(idx, 0)
                    html_parts.append(get_color_html(sentence, weight))
                
                # 拼接并显示
                full_html = f"<div style='line-height: 1.8; font-family: serif; padding: 15px; border: 1px solid #ddd; border-radius: 5px;'>{''.join(html_parts)}</div>"
                st.markdown(full_html, unsafe_allow_html=True)
                
                # === 结果展示 B: 最具贡献度的句子排行 ===
                st.write("### 🏆 最具“原著味”的句子 TOP 5")
                # 排序：权重从大到小
                sorted_indices = sorted(weight_map.keys(), key=lambda k: weight_map[k], reverse=True)
                
                top_sentences_data = []
                for idx in sorted_indices[:5]:
                    if weight_map[idx] > 0: # 只显示正向贡献
                        top_sentences_data.append({
                            "排名": len(top_sentences_data) + 1,
                            "句子内容": sentences_list[idx],
                            "贡献度得分": f"{weight_map[idx]:.4f}"
                        })
                
                if top_sentences_data:
                    st.table(pd.DataFrame(top_sentences_data).set_index("排名"))
                else:
                    st.write("未检测到显著的正向特征句子。")

                status.update(label="全流程分析圆满完成！", state="complete", expanded=False)

            except Exception as e:
                st.error(f"LIME 分析过程出错: {e}")
                status.update(label="分析中断", state="error")

