(function () {
  const GLOBAL_KEY = "__seBanditModelSelectionDemoInstances";

  function template() {
    return `
      <section class="se-bandit-demo__hero">
        <div class="se-bandit-demo__eyebrow">UCB1 Bandit LLM Selection</div>
        <h2 class="se-bandit-demo__hero-title">Watch the posterior move as cost pressure changes.</h2>
        <p class="se-bandit-demo__hero-copy">
          The demo stays inside the docs layout and mirrors the current
          <code>AsymmetricUCB</code> selection shape: reward, exploration, and
          optional cheapness blending.
        </p>
      </section>

      <div class="se-bandit-demo__layout">
        <section class="se-bandit-demo__panel">
          <div class="se-bandit-demo__arena-head">
            <div>
              <h3 data-role="arena-title">Model Arena · Generation 0</h3>
              <p class="se-bandit-demo__microcopy">Cost increases to the right. Reward signal increases upward.</p>
            </div>
            <div class="se-bandit-demo__actions" aria-label="Simulation controls">
              <button type="button" class="se-bandit-demo__button" data-action="play-pause">Pause</button>
              <button type="button" class="se-bandit-demo__button se-bandit-demo__button--secondary" data-action="reset">Reset</button>
            </div>
          </div>
          <div class="se-bandit-demo__control se-bandit-demo__control--inline">
            <div class="se-bandit-demo__control-top">
              <strong>Cost Awareness</strong>
              <span data-role="cost-aware-label">0.50</span>
            </div>
            <input type="range" min="0" max="1" step="0.01" value="0.5" data-input="cost-aware" aria-label="Cost awareness coefficient">
            <div class="se-bandit-demo__microcopy">
              Lower values lean toward stronger or higher-upside models. Higher values push the policy toward cheaper ones.
            </div>
          </div>
          <svg class="se-bandit-demo__svg" data-role="scatter" viewBox="0 0 860 320" role="img" aria-label="Cost versus reward scatter plot"></svg>
          <section class="se-bandit-demo__posterior-panel">
            <div class="se-bandit-demo__posterior-chart" data-role="posterior-chart" role="img" aria-label="Posterior probability bar plot"></div>
          </section>
          <section class="se-bandit-demo__timeline-panel">
            <svg class="se-bandit-demo__svg se-bandit-demo__svg--timeline" data-role="sample-chart" viewBox="0 0 860 320" role="img" aria-label="Cumulative samples across generations"></svg>
          </section>
        </section>
      </div>
    `;
  }

  function bindDom(root) {
    const get = (selector) => root.querySelector(selector);
    return {
      root,
      arenaTitle: get("[data-role='arena-title']"),
      costAware: get("[data-input='cost-aware']"),
      costAwareLabel: get("[data-role='cost-aware-label']"),
      scatter: get("[data-role='scatter']"),
      posteriorChart: get("[data-role='posterior-chart']"),
      sampleChart: get("[data-role='sample-chart']"),
      playPause: get("[data-action='play-pause']"),
      reset: get("[data-action='reset']")
    };
  }

  function render(instance) {
    const rows = window.ShinkaBanditModelSelectionDemo.renderStatic(instance.dom, instance.state);
    window.ShinkaBanditModelSelectionDemo.updateSummary(instance.dom, instance.state, rows);
  }

  function wireControls(instance) {
    instance.dom.costAware.addEventListener("input", () => {
      instance.state.config.costAwareCoef = Number(instance.dom.costAware.value);
      render(instance);
    });

    instance.dom.playPause.addEventListener("click", () => {
      instance.state.running = !instance.state.running;
      render(instance);
    });

    instance.dom.reset.addEventListener("click", () => {
      const nextConfig = {
        ...instance.state.config,
        costAwareCoef: Number(instance.dom.costAware.value)
      };
      instance.state = window.ShinkaBanditModelSelectionDemo.createState(
        nextConfig,
        instance.state.scenarioKey,
        instance.state.running
      );
      instance.accumulator = 0;
      render(instance);
    });
  }

  function tick(instance) {
    let lastFrame = null;

    function frame(timestamp) {
      if (instance.destroyed || !instance.root.isConnected) {
        return;
      }
      if (lastFrame === null) {
        lastFrame = timestamp;
      }
      const dt = Math.min(0.2, (timestamp - lastFrame) / 1000);
      lastFrame = timestamp;
      if (instance.state.running) {
        instance.accumulator += dt * instance.state.config.speed;
        while (
          instance.accumulator >= 1 &&
          instance.state.round < instance.state.config.maxRounds
        ) {
          window.ShinkaBanditModelSelectionDemo.stepRound(instance.state);
          instance.accumulator -= 1;
        }
        if (instance.state.round >= instance.state.config.maxRounds) {
          instance.state.running = false;
          instance.accumulator = 0;
        }
        render(instance);
      }
      window.requestAnimationFrame(frame);
    }

    window.requestAnimationFrame(frame);
  }

  function mount(root) {
    if (!window.ShinkaBanditModelSelectionDemo || root.dataset.mounted === "true") {
      return;
    }
    root.dataset.mounted = "true";
    root.classList.add("se-bandit-demo");
    root.innerHTML = template();

    const instance = {
      root,
      dom: bindDom(root),
      destroyed: false,
      accumulator: 0,
      state: window.ShinkaBanditModelSelectionDemo.createState(
        window.ShinkaBanditModelSelectionDemo.DEFAULT_CONFIG,
        "balanced",
        true
      )
    };

    window[GLOBAL_KEY].push(instance);
    wireControls(instance);
    render(instance);
    tick(instance);
  }

  function init() {
    window[GLOBAL_KEY] = window[GLOBAL_KEY] || [];
    window[GLOBAL_KEY].forEach((instance) => {
      instance.destroyed = true;
    });
    window[GLOBAL_KEY] = [];
    document.querySelectorAll("#bandit-selection-widget").forEach(mount);
  }

  if (typeof document$ !== "undefined" && document$ && document$.subscribe) {
    document$.subscribe(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
