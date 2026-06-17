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

    const crispDmData = [
        {
            title: '商业理解',
            text: '明确业务目标，将业务问题转化为数据挖掘目标，制定项目计划与成功标准。'
        },
        {
            title: '数据理解',
            text: '收集数据、进行描述性统计、探索数据特征并评估数据质量。'
        },
        {
            title: '数据准备',
            text: '清洗、转换、编码、缩放与特征工程，构建最终建模数据集。'
        },
        {
            title: '建模',
            text: '选择合适算法，训练模型，调整参数，获得候选模型。'
        },
        {
            title: '评估',
            text: '验证模型是否满足业务目标，对比模型效果并分析误差。'
        },
        {
            title: '模型实施',
            text: '将模型部署到生产环境，监控效果并持续迭代优化。'
        }
    ];

    const knowledgeModules = [
        {
            title: '数据挖掘基础',
            subtitle: '第一章',
            desc: '概念、方法体系、开发工具链与 CRISP-DM 标准流程。'
        },
        {
            title: '数据预处理与特征工程',
            subtitle: '第二章',
            desc: '数据清洗、转换、描述可视化与特征选择。'
        },
        {
            title: '回归与分类建模',
            subtitle: '第三章',
            desc: '基础概念、分类/回归算法与集成学习方法。'
        }
    ];

    // 渲染知识模块卡片
    function renderKnowledgeCards() {
        if (!knowledgeGrid) return;
        knowledgeGrid.innerHTML = knowledgeModules.map((m, i) => `
            <article class="knowledge-card" style="animation-delay: ${i * 0.1}s">
                <p class="eyebrow">${m.subtitle}</p>
                <h3>${m.title}</h3>
                <p>${m.desc}</p>
            </article>
        `).join('');
    }

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
    updateActiveNav();

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

    // 简单的滚动入场动画
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
        const revealElements = document.querySelectorAll('.section-header, .knowledge-card, .step-card, .algo-card, .timeline-item, .stage-detail');
        revealElements.forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(24px)';
            el.style.transition = 'opacity 0.7s ease, transform 0.7s ease';
            revealObserver.observe(el);
        });
    }

    document.addEventListener('DOMContentLoaded', () => {
        renderKnowledgeCards();
        setStage(0);
        initReveal();

        // 定义 revealed 样式
        const style = document.createElement('style');
        style.textContent = `
            .revealed {
                opacity: 1 !important;
                transform: translateY(0) !important;
            }
        `;
        document.head.appendChild(style);
    });
})();
