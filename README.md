# HCIE Data Mining V3.0 · 可视化学习指南

基于华为 **HCIE-Big Data-Data Mining V3.0** 培训教材与实验手册构建的可视化学习网站。采用苹果科技风格设计，帮助学习者以清晰、优雅、交互式的方式掌握数据挖掘知识体系。

---

## 项目简介

本项目将教材内容按照 **MECE 原则** 进行结构化梳理，并通过网页形式呈现：

- **知识框架总览**：三大模块、四十余个知识点的层次化展示
- **CRISP-DM 流程**：六阶段数据挖掘标准流程可视化
- **数据预处理与特征工程**：清洗 → 转换 → 描述 → 特征工程流程
- **回归与分类建模**：分类、回归、集成算法的系统化展示
- **学习路径建议**：四阶段递进式学习计划

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 页面结构 | HTML5 |
| 样式设计 | CSS3（自定义设计系统） |
| 交互逻辑 | 原生 JavaScript（ES6+） |
| 数据存储 | JSON（`data/outline.json`） |
| 字体 | Inter + Noto Sans SC |

---

## 设计风格

- 苹果科技风：大量留白、圆角卡片、柔和阴影、玻璃态导航
- 渐变与高对比度排版
- 响应式布局，适配桌面与移动端
- 平滑滚动与入场动画

---

## 项目结构

```
HCIE_Data_Mining_VisLab/
├── index.html          # 单页应用主页面
├── css/
│   └── style.css       # 苹果风格样式系统
├── js/
│   └── app.js          # 导航、交互、动画逻辑
├── data/
│   └── outline.json    # 知识结构数据（待填充）
├── docs/
│   └── HCIE_Big_Data_Data_Mining_V3.0_教材结构化梳理.md  # 原始教材梳理
└── README.md           # 项目说明
```

---

## 本地运行

本项目为纯静态网站，无需构建工具。

### 方式一：直接打开
双击 `index.html`，使用浏览器即可查看。

### 方式二：本地服务器
```bash
# Python 3
python -m http.server 8000

# 或 Node.js
npx serve .
```

然后在浏览器访问 `http://localhost:8000`。

---

## 后续计划

- [ ] 将 Markdown 知识点结构化写入 `data/outline.json`
- [ ] 引入 ECharts / D3.js 绘制可交互知识树
- [ ] 完善 CRISP-DM 流程动态演示
- [ ] 填充算法卡片详情页（原理、公式、代码示例）
- [ ] 接入 MathJax / KaTeX 渲染数学公式
- [ ] 部署到 GitHub Pages

---

## 许可证

本项目为个人学习整理，教材内容版权归华为所有。

---

*基于华为 HCIE-Big Data-Data Mining V3.0 培训教材结构化梳理构建。*
