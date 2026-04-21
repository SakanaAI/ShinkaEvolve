(function () {
  const GLOBAL_KEY = "__seThroughputDemoInstances";

  function template() {
    return `
      <div class="se-throughput-demo__header">
        <div class="se-throughput-demo__top-row">
          <div class="se-throughput-demo__eyebrow">Interactive Throughput Model</div>
          <div class="se-throughput-demo__header-actions">
            <div class="se-throughput-demo__mode-actions">
              <button class="se-throughput-demo__button" data-eval-mode="slow">Slow Eval</button>
              <button class="se-throughput-demo__button is-active" data-eval-mode="fast">Fast Eval</button>
            </div>
            <button class="se-throughput-demo__button is-active" data-action="toggle">Pause</button>
            <button class="se-throughput-demo__button" data-action="reset">Reset</button>
          </div>
        </div>
        <div class="se-throughput-demo__title-row">
          <div>
            <h3 class="se-throughput-demo__title">Tune worker pools and watch the pipeline rebalance.</h3>
            <p class="se-throughput-demo__summary">Sampling hands candidates off to evaluation, and database workers finalize completed generations.</p>
          </div>
        </div>

        <div class="se-throughput-demo__controls">
          ${control("sampling", "Sampling workers")}
          ${control("evaluation", "Evaluation workers")}
          ${control("database", "Database workers")}
        </div>
      </div>

      <div class="se-throughput-demo__body">
        <section class="se-throughput-demo__panel">
          <div class="se-throughput-demo__panel-head">
            <div class="se-throughput-demo__label">Stage Timeline</div>
          </div>
          <svg class="se-throughput-demo__stage-svg" data-role="diagram" viewBox="0 0 1080 620" role="img" aria-label="Interactive throughput diagram"></svg>
        </section>
      </div>
    `;
  }

  function control(field, label) {
    return `
      <label class="se-throughput-demo__control">
        <div class="se-throughput-demo__control-top">
          <span class="se-throughput-demo__label">${label}</span>
          <span class="se-throughput-demo__control-value" data-role="${field}-value"></span>
        </div>
        <input type="range" min="1" max="8" step="1" value="1" data-input="${field}">
      </label>
    `;
  }

  function bindDom(root) {
    const get = (selector) => root.querySelector(selector);
    return {
      root,
      diagram: get("[data-role='diagram']"),
      inputs: {
        sampling: get("[data-input='sampling']"),
        evaluation: get("[data-input='evaluation']"),
        database: get("[data-input='database']")
      },
      valueLabels: {
        sampling: get("[data-role='sampling-value']"),
        evaluation: get("[data-role='evaluation-value']"),
        database: get("[data-role='database-value']")
      },
      actionButtons: {
        toggle: get("[data-action='toggle']"),
        reset: get("[data-action='reset']")
      },
      evalModeButtons: Array.from(root.querySelectorAll("[data-eval-mode]"))
    };
  }

  function syncInputs(instance, config) {
    ["sampling", "evaluation", "database"].forEach((field) => {
      instance.dom.inputs[field].value = String(config[field]);
      instance.dom.valueLabels[field].textContent = String(config[field]);
    });
  }

  function refresh(instance) {
    const runtime = window.ShinkaAsyncThroughputDemo;
    const { dom, state } = instance;
    syncInputs(instance, state.config);
    dom.evalModeButtons.forEach((button) => {
      button.classList.toggle("is-active", button.dataset.evalMode === state.evalMode);
    });
    runtime.renderDiagram(dom.diagram, state);
  }

  function resetSimulation(instance) {
    const runtime = window.ShinkaAsyncThroughputDemo;
    const next = runtime.createState(instance.state.config, instance.state.evalMode);
    next.playing = instance.state.playing;
    instance.state = next;
    runtime.startSampling(next);
    refresh(instance);
  }

  function wireControls(instance) {
    Object.entries(instance.dom.inputs).forEach(([field, input]) => {
      input.addEventListener("input", () => {
        instance.state.config[field] = Number(input.value);
        resetSimulation(instance);
      });
    });

    instance.dom.actionButtons.toggle.addEventListener("click", () => {
      instance.state.playing = !instance.state.playing;
      instance.dom.actionButtons.toggle.textContent = instance.state.playing
        ? "Pause"
        : "Play";
      instance.dom.actionButtons.toggle.classList.toggle(
        "is-active",
        instance.state.playing
      );
    });

    instance.dom.actionButtons.reset.addEventListener("click", () => {
      resetSimulation(instance);
    });

    instance.dom.evalModeButtons.forEach((button) => {
      button.addEventListener("click", () => {
        instance.state.evalMode = button.dataset.evalMode;
        resetSimulation(instance);
      });
    });
  }

  function tick(instance) {
    const runtime = window.ShinkaAsyncThroughputDemo;
    let lastFrame = null;

    function frame(timestamp) {
      if (instance.destroyed || !instance.root.isConnected) {
        return;
      }
      if (lastFrame === null) {
        lastFrame = timestamp;
      }
      const dt = Math.min(0.12, (timestamp - lastFrame) / 1000);
      lastFrame = timestamp;
      if (instance.state.playing) {
        runtime.advance(instance.state, dt);
      }
      refresh(instance);
      window.requestAnimationFrame(frame);
    }

    runtime.startSampling(instance.state);
    window.requestAnimationFrame(frame);
  }

  function mount(root) {
    if (!window.ShinkaAsyncThroughputDemo || root.dataset.mounted === "true") {
      return;
    }
    root.dataset.mounted = "true";
    root.classList.add("se-throughput-demo");
    root.innerHTML = template();

    const runtime = window.ShinkaAsyncThroughputDemo;
    const instance = {
      root,
      destroyed: false
    };
    window[GLOBAL_KEY].push(instance);

    instance.dom = bindDom(root);
    instance.state = runtime.createState(
      runtime.DEFAULT_CONFIG,
      runtime.DEFAULT_EVAL_MODE
    );

    wireControls(instance);
    syncInputs(instance, instance.state.config);
    refresh(instance);
    tick(instance);
  }

  function init() {
    window[GLOBAL_KEY] = window[GLOBAL_KEY] || [];
    window[GLOBAL_KEY].forEach((instance) => {
      instance.destroyed = true;
    });
    window[GLOBAL_KEY] = [];
    document.querySelectorAll("#async-throughput-demo").forEach(mount);
  }

  if (typeof document$ !== "undefined" && document$ && document$.subscribe) {
    document$.subscribe(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
