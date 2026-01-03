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
    """文风分词：基于停用词表过滤"""
    text = basic_clean(text)
    words = jieba.lcut(text)
    # 过滤逻辑：保留非黑名单词且非纯空白
    return [w for w in words if w not in blocklist and not w.isspace()]

def generate_blocklist_from_files(uploaded_files):
    """自动生成停用词表：实词"""
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
    """
    优化版：增强颜色可视性，适配深色/浅色模式
    """
    # 1. 动态放大权重
    # LIME 的句子权重通常较小，我们将其放大 10 倍，并限制最大透明度为 0.7
    # 限制为 0.7 是为了保证文字（无论是黑字还是白字）依然清晰可读
    val = abs(weight)
    if val < 0.001: return text # 权重太小不染色
    
    intensity = min(val * 10, 0.7) 
    
    # 2. 设定“保底”透明度
    # 只要有权重，至少给 0.15 的透明度，防止颜色太浅看不见
    intensity = max(intensity, 0.15)

    if weight > 0:
        # 正向（像原著）：使用亮红色 (255, 60, 60)
        # 原来的 (255, 0, 0) 在黑底上容易显得暗沉，加一点绿蓝分量会更亮
        rgba = f"rgba(255, 60, 60, {intensity})" 
    else:
        # 负向（不像原著）：使用亮蓝色 (0, 160, 255)
        # 纯蓝 (0, 0, 255) 在暗夜模式下几乎隐形，必须提高绿色分量变成“天蓝”
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
    st.header("Step 1: 建立基准")
    st.info("请上传原著 TXT 文件（可多选）")
    uploaded_originals = st.file_uploader("上传原著 (支持 .txt)", type="txt", accept_multiple_files=True)

    st.header("Step 2: 输入测试文本")
    fanfic_text = st.text_area("在此粘贴你的同人文本：", height=200, placeholder="建议粘贴 500 字以上的段落...")

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
            
            status.write("✅ 基础分析完成，准备进行句子归因...")

            # === 阶段二：结果展示 (基础部分) ===
            st.divider()
            st.subheader("📊 相似度分析报告")
            
            res_c1, res_c2 = st.columns([1, 1])
            
            with res_c1:
                st.metric(label="整体文风相似度", value=f"{final_score:.2f}%")
                
                # 评语逻辑 (同人圈特供版)
                if final_score > 90:
                    st.success("""
                    **判定：疑似作者小号（Tier S）** 😭 **救命！这是哪位神仙太太下凡？** 这简直就是原著！若不是作者的小号，建议严查是否偷了存稿硬盘。  
                    *评价：绝赞好粮，垂直入坑，请受我一拜！*
                    """)
                elif final_score > 75:
                    st.info("""
                    **判定：美味（Tier A）** 😋 **好一口美味的粮！** 虽然在细节处能看出太太自己的行文习惯，但整体还原度极高。  
                    *评价：是不可多得的优质粮，这就加入书架！*
                    """)
                elif final_score > 60:
                    st.warning("""
                    **判定：自带滤镜的AU感（Tier B）** 🤔 **这是什么奇怪的pa吗？** 虽然还在同人的范畴里，但是私设比较多呢。  
                    *评价：熟悉的陌生人，仿佛在OOC边缘试探（）*
                    """)
                else:
                    st.error("""
                    **判定：OOC预警 / 纯属原创（Tier C）** 😨 **确定这是同人？** 这独特的文风已经完全脱离了原著的引力圈，如果不看角色名，机器还以为误入了隔壁片场。  
                    *评价：这是极致的OOC，还是披着同人皮的原创大作？这很难评，祝您开心就好。*
                    """)
            
            with metric_col2:
                st.write("### 向量空间投影")
                if len(orig_vecs) > 0:
                    try:
                        pca = PCA(n_components=2)
                        X_all = np.vstack([orig_vecs, [test_vec]])
                        X_pca = pca.fit_transform(X_all)
                        n_orig = len(orig_vecs)

                        fig, ax = plt.subplots(figsize=(6, 4))

                        # 【关键修改1】设置背景透明
                        fig.patch.set_alpha(0.0)  # 将图片底色设为透明
                        ax.patch.set_alpha(0.0)   # 将绘图区底色设为透明

                        # 绘图部分
                        ax.scatter(X_pca[:n_orig, 0], X_pca[:n_orig, 1], c='lightgray', s=10, alpha=0.5, label='原著切片')
                        center = pca.transform([gold_standard])
                        ax.scatter(center[:,0], center[:,1], c='red', marker='*', s=200, label='原著基准')
                        ax.scatter(X_pca[n_orig:, 0], X_pca[n_orig:, 1], c='blue', s=80, marker='X', label='你的文本')

                        # 【安全绘图】只有当字体对象有效时，才应用字体
                        if my_font_prop:
                            # 【关键修改2】frameon=False 去除图例的边框
                            ax.legend(prop=my_font_prop, frameon=False) 
                            ax.set_title("文风落点分布", fontproperties=my_font_prop)
                        else:
                            ax.legend(frameon=False)
                            ax.set_title("Style Distribution (Font Missing)")

                        # 关闭坐标轴（这一步本身就去除了大部分边框）
                        ax.axis('off')

                        # 渲染图片
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"绘图出错: {e}")
                        
                        
            # === 阶段三：句子级 LIME 进阶分析 ===
            st.divider()
            st.subheader("🔍 深度归因：哪些句子最像原著？")
            st.info("正在逐句分析文风贡献度（红色=加分项，蓝色=减分项）...")
            
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

