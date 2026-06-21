(function () {
    'use strict';

    const navToggle = document.getElementById('navToggle');
    const navMenu = document.getElementById('navMenu');
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.section');
    const stageItems = document.querySelectorAll('.stage-item');
    const stageTitle = document.querySelector('.stage-title');
    const stageText = document.querySelector('.stage-text');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');
    const knowledgeGrid = document.getElementById('knowledgeGrid');

    // 全局数据缓存
    let outlineData = null;

    const crispDmData = [
        { title: '商业理解', text: '明确业务目标，将业务问题转化为数据挖掘目标，制定项目计划与成功标准。' },
        { title: '数据理解', text: '收集数据、进行描述性统计、探索数据特征并评估数据质量。' },
        { title: '数据准备', text: '清洗、转换、编码、缩放与特征工程，构建最终建模数据集。' },
        { title: '建模', text: '选择合适算法，训练模型，调整参数，获得候选模型。' },
        { title: '评估', text: '验证模型是否满足业务目标，对比模型效果并分析误差。' },
        { title: '模型实施', text: '将模型部署到生产环境，监控效果并持续迭代优化。' }
    ];

    // 移动导航切换
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', () => {
            navMenu.classList.toggle('open');
        });

        navLinks.forEach(link => {
            link.addEventListener('click', () => {
                navMenu.classList.remove('open');
            });
        });
    }

    // 滚动时高亮当前导航
    function updateActiveNav() {
        const scrollPos = window.scrollY + 120;

        sections.forEach(section => {
            const top = section.offsetTop;
            const height = section.offsetHeight;
            const id = section.getAttribute('id');

            if (scrollPos >= top && scrollPos < top + height) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${id}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }

    window.addEventListener('scroll', updateActiveNav, { passive: true });

    // CRISP-DM 阶段切换
    function setStage(index) {
        stageItems.forEach((item, i) => {
            item.classList.toggle('active', i === index);
        });

        if (stageTitle && stageText) {
            stageTitle.textContent = crispDmData[index].title;
            stageText.textContent = crispDmData[index].text;
        }

        const progress = index === 0 ? 0 : ((index) / (stageItems.length - 1)) * 100;
        document.querySelector('.stage-track').style.setProperty('--progress', `${progress}%`);
    }

    stageItems.forEach((item, index) => {
        item.addEventListener('click', () => setStage(index));
    });

    // 建模算法选项卡
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;

            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            tabPanels.forEach(panel => {
                panel.classList.toggle('active', panel.dataset.panel === tab);
            });
        });
    });

    // 加载 outline.json
    async function loadOutline() {
        try {
            const res = await fetch('data/outline.json');
            outlineData = await res.json();
            renderAll();
        } catch (err) {
            console.error('Failed to load outline.json:', err);
            renderFallback();
        }
    }

    // 渲染知识总览卡片
    function renderKnowledgeCards() {
        if (!knowledgeGrid || !outlineData) return;
        const modules = outlineData.modules || [];
        knowledgeGrid.innerHTML = modules.map((m, i) => `
            <article class="knowledge-card" style="animation-delay: ${i * 0.1}s">
                <p class="eyebrow">${m.chapter}</p>
                <h3>${m.title}</h3>
                <p>${m.summary}</p>
                <div class="card-topics">
                    ${(m.sections || []).map(s => `<span class="topic-tag">${s.title}</span>`).join('')}
                </div>
            </article>
        `).join('');
    }

    // 渲染公式面板
    function renderFormulas() {
        const container = document.getElementById('formulaGrid');
        if (!container || !outlineData) return;
        const formulas = outlineData.formulas || [];
        const categories = [...new Set(formulas.map(f => f.category))];

        container.innerHTML = categories.map(cat => `
            <div class="formula-group">
                <h4 class="formula-category">${cat}</h4>
                <div class="formula-list">
                    ${formulas.filter(f => f.category === cat).map(f => `
                        <div class="formula-item">
                            <span class="formula-name">${f.name}</span>
                            <span class="formula-expr">${escapeHtml(f.expression)}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `).join('');
    }

    // 渲染实验列表
    function renderExperiments() {
        const container = document.getElementById('experimentsList');
        if (!container || !outlineData) return;
        const experiments = outlineData.experiments || [];

        container.innerHTML = experiments.map((exp, i) => `
            <article class="experiment-card" style="animation-delay: ${i * 0.05}s">
                <div class="experiment-header">
                    <h4>${exp.title}</h4>
                    <span class="dataset-tag">${exp.dataset}</span>
                </div>
                <div class="experiment-tools">
                    ${(exp.tools || []).map(t => `<span class="tool-tag">${t}</span>`).join('')}
                </div>
                <ol class="experiment-steps">
                    ${(exp.steps || []).slice(0, 4).map(step => `<li>${step}</li>`).join('')}
                </ol>
                <p class="experiment-takeaway">${exp.keyTakeaway || ''}</p>
            </article>
        `).join('');
    }

    // 渲染案例研究
    function renderCaseStudies() {
        const container = document.getElementById('caseStudiesGrid');
        if (!container || !outlineData) return;
        const cases = outlineData.caseStudies || [];

        container.innerHTML = cases.map((c, i) => `
            <article class="knowledge-card case-card" style="animation-delay: ${i * 0.1}s">
                <div class="case-meta">
                    <span class="case-type">${c.type}</span>
                    <span class="case-dataset">${c.dataset}</span>
                </div>
                <h3>${c.title}</h3>
                <div class="card-topics">
                    ${(c.algorithms || []).map(a => `<span class="topic-tag">${a}</span>`).join('')}
                </div>
                <ul class="case-steps">
                    ${(c.keySteps || []).slice(0, 4).map(s => `<li>${s}</li>`).join('')}
                </ul>
            </article>
        `).join('');
    }

    // 失败回退
    function renderFallback() {
        if (knowledgeGrid) {
            knowledgeGrid.innerHTML = `
                <article class="knowledge-card"><p class="eyebrow">提示</p><h3>数据加载失败</h3><p>请通过本地服务器打开本站，或检查 data/outline.json 是否存在。</p></article>
            `;
        }
    }

    function escapeHtml(str) {
        return String(str).replace(/[&<>"']/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
    }

    // 渲染所有动态内容
    function renderAll() {
        renderKnowledgeCards();
        renderFormulas();
        renderExperiments();
        renderCaseStudies();
        updateActiveNav();
    }

    // 滚动入场动画
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.12
    };

    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('revealed');
                revealObserver.unobserve(entry.target);
            }
        });
    }, observerOptions);

    function initReveal() {
        const revealElements = document.querySelectorAll('.section-header, .knowledge-card, .step-card, .algo-card, .timeline-item, .stage-detail, .formula-group, .experiment-card, .case-card');
        revealElements.forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(24px)';
            el.style.transition = 'opacity 0.7s ease, transform 0.7s ease';
            revealObserver.observe(el);
        });
    }

    document.addEventListener('DOMContentLoaded', () => {
        setStage(0);
        loadOutline();
        initReveal();

        // 定义 revealed 样式
        const style = document.createElement('style');
        style.textContent = `
            .revealed { opacity: 1 !important; transform: translateY(0) !important; }
        `;
        document.head.appendChild(style);
    });
})();
