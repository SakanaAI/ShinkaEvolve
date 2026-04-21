(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const DEFAULT_CONFIG = { sampling: 5, evaluation: 4, database: 3 };
  const DEFAULT_EVAL_MODE = "fast";
  const EVAL_BASE_SECONDS = { fast: 1.6, slow: 6.8 };
  const STAGES = [{ key: "sampling", label: "Proposal", color: "#2563eb" }, { key: "evaluation", label: "Evaluation", color: "#f59e0b" }, { key: "database", label: "Database", color: "#10b981" }];
  const FACTORS = { sampling: [0.84, 1.12, 0.93, 1.08, 1.18, 0.9, 1.04, 0.96], evaluation: [1.06, 0.92, 1.11, 0.97, 1.02, 0.88, 1.08, 0.95], database: [0.91, 1.04, 1.12, 0.96, 1.08, 0.93, 1.02, 0.98] };

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function formatPercent(value) {
    return `${Math.round(value)}%`;
  }

  function factorFor(stageKey, id) {
    const factors = FACTORS[stageKey];
    return factors[(id + stageKey.length) % factors.length];
  }

  function normalizeConfig(config) {
    return {
      sampling: clamp(Number(config.sampling), 1, 8),
      evaluation: clamp(Number(config.evaluation), 1, 8),
      database: clamp(Number(config.database), 1, 8)
    };
  }

  function createState(config, evalMode) {
    return {
      config: normalizeConfig(config),
      evalMode: evalMode === "slow" ? "slow" : DEFAULT_EVAL_MODE,
      totalGenerations: 36,
      time: 0,
      playing: true,
      nextJobId: 0,
      generatedCount: 0,
      completedCount: 0,
      active: { sampling: [], evaluation: [], database: [] },
      queues: { proposalReady: [], databaseReady: [] },
      occupiedTime: { sampling: 0, evaluation: 0, database: 0 },
      lastEvent: "Bootstrapping worker pools.",
      finishedAt: null
    };
  }

  function proposalCapacity(state) {
    return state.config.sampling;
  }

  function availableLane(state, stageKey) {
    const used = new Set(state.active[stageKey].map((job) => job.lane));
    const cap = state.config[stageKey];
    for (let lane = 0; lane < cap; lane += 1) {
      if (!used.has(lane)) {
        return lane;
      }
    }
    return -1;
  }

  function createJob(state, id) {
    const base = {
      sampling: 4.9,
      evaluation: EVAL_BASE_SECONDS[state.evalMode],
      database: 1.35
    };
    return {
      id,
      durations: {
        sampling: base.sampling * factorFor("sampling", id),
        evaluation: base.evaluation * factorFor("evaluation", id),
        database: base.database * factorFor("database", id)
      }
    };
  }

  function startSampling(state) {
    let changed = false;
    while (
      state.active.sampling.length < state.config.sampling &&
      state.generatedCount < state.totalGenerations &&
      state.active.sampling.length + state.queues.proposalReady.length <
        proposalCapacity(state)
    ) {
      const lane = availableLane(state, "sampling");
      if (lane < 0) {
        break;
      }
      const job = createJob(state, state.nextJobId);
      state.active.sampling.push({
        ...job,
        lane,
        startedAt: state.time,
        endsAt: state.time + job.durations.sampling
      });
      state.nextJobId += 1;
      state.generatedCount += 1;
      changed = true;
    }
    if (changed) {
      state.lastEvent =
        state.evalMode === "slow"
          ? "Sampling is outpacing evaluation."
          : "Sampling workers are feeding evaluation, which is clearing work quickly.";
    }
    return changed;
  }

  function startQueued(state, queueKey, stageKey) {
    let changed = false;
    const queue = state.queues[queueKey];
    while (
      queue.length > 0 &&
      state.active[stageKey].length < state.config[stageKey]
    ) {
      const lane = availableLane(state, stageKey);
      if (lane < 0) {
        break;
      }
      const queued = queue.shift();
      state.active[stageKey].push({
        ...queued,
        lane,
        startedAt: state.time,
        endsAt: state.time + queued.durations[stageKey]
      });
      changed = true;
    }
    return changed;
  }

  function removeFinished(state, stageKey) {
    const remaining = [];
    let changed = false;
    state.active[stageKey].forEach((job) => {
      if (job.endsAt > state.time) {
        remaining.push(job);
        return;
      }
      changed = true;
      if (stageKey === "sampling") {
        state.queues.proposalReady.push({ id: job.id, durations: job.durations });
        state.lastEvent = `G${job.id} is ready for evaluation.`;
        return;
      }
      if (stageKey === "evaluation") {
        state.queues.databaseReady.push({ id: job.id, durations: job.durations });
        state.lastEvent =
          `G${job.id} finished evaluation and is waiting for database finalization.`;
        return;
      }
      state.completedCount += 1;
      state.lastEvent = `G${job.id} fully finalized. Throughput increased.`;
    });
    state.active[stageKey] = remaining;
    return changed;
  }

  function advance(state, dt) {
    STAGES.forEach((stage) => {
      state.occupiedTime[stage.key] += state.active[stage.key].length * dt;
    });
    state.time += dt;

    let changed = false;
    let guard = 0;
    do {
      changed = false;
      changed = removeFinished(state, "sampling") || changed;
      changed = removeFinished(state, "evaluation") || changed;
      changed = removeFinished(state, "database") || changed;
      changed = startQueued(state, "proposalReady", "evaluation") || changed;
      changed = startQueued(state, "databaseReady", "database") || changed;
      changed = startSampling(state) || changed;
      guard += 1;
    } while (changed && guard < 12);

    const done =
      state.completedCount >= state.totalGenerations &&
      state.active.sampling.length === 0 &&
      state.active.evaluation.length === 0 &&
      state.active.database.length === 0 &&
      state.queues.proposalReady.length === 0 &&
      state.queues.databaseReady.length === 0;

    if (done && state.finishedAt === null) {
      state.finishedAt = state.time;
      state.lastEvent =
        "Run complete. Auto-resetting to replay the throughput pattern.";
    }
    if (state.finishedAt !== null && state.time - state.finishedAt > 2.2) {
      const config = state.config;
      const evalMode = state.evalMode;
      Object.assign(state, createState(config, evalMode));
      startSampling(state);
    }
  }

  function avgUtilization(state, stageKey) {
    const cap = state.config[stageKey];
    if (state.time <= 0 || cap === 0) {
      return 0;
    }
    return Math.min(100, (state.occupiedTime[stageKey] / (state.time * cap)) * 100);
  }

  function throughput(state) {
    if (state.time <= 0) {
      return 0;
    }
    return (state.completedCount / state.time) * 60;
  }

  function describeBottleneck(state) {
    if (
      state.queues.databaseReady.length > 0 &&
      state.active.database.length === state.config.database
    ) {
      return {
        title: "Database workers are now the limiting pool.",
        copy:
          "Evaluated generations are waiting to be finalized, so more DB capacity would increase end-to-end throughput."
      };
    }
    if (
      state.queues.proposalReady.length === 0 &&
      state.active.evaluation.length < state.config.evaluation &&
      state.time > 2
    ) {
      return {
        title: "Sampling is starving evaluation.",
        copy:
          "Evaluation lanes are ready to work, but the eval handoff queue is empty. This is the fast-eval case: sampling is the bottleneck."
      };
    }
    if (
      state.queues.proposalReady.length > 0 &&
      state.active.evaluation.length === state.config.evaluation
    ) {
      return {
        title: "Evaluation is saturated.",
        copy:
          "The eval handoff queue is filling because evaluation is slower than sampling. This is the slow-eval case."
      };
    }
    return {
      title: "The pipeline is flowing with bounded overlap.",
      copy:
        "Sampling, evaluation, and database finalization are all contributing, with no single pool fully dominating yet."
    };
  }

  function rect(x, y, width, height, fill, stroke, strokeWidth, dasharray) {
    const node = document.createElementNS(SVG_NS, "rect");
    node.setAttribute("x", x);
    node.setAttribute("y", y);
    node.setAttribute("width", width);
    node.setAttribute("height", height);
    node.setAttribute("fill", fill);
    if (stroke) {
      node.setAttribute("stroke", stroke);
      node.setAttribute("stroke-width", strokeWidth);
    }
    if (dasharray) {
      node.setAttribute("stroke-dasharray", dasharray);
    }
    return node;
  }

  function textNode(x, y, value, size, weight, fill, anchor) {
    const node = document.createElementNS(SVG_NS, "text");
    node.setAttribute("x", x);
    node.setAttribute("y", y);
    node.setAttribute("font-size", size);
    node.setAttribute("font-weight", weight);
    node.setAttribute("fill", fill);
    if (anchor) {
      node.setAttribute("text-anchor", anchor);
    }
    node.textContent = value;
    return node;
  }

  function line(x1, y1, x2, y2) {
    const node = document.createElementNS(SVG_NS, "line");
    node.setAttribute("x1", x1);
    node.setAttribute("y1", y1);
    node.setAttribute("x2", x2);
    node.setAttribute("y2", y2);
    node.setAttribute("stroke", "#111111");
    node.setAttribute("stroke-width", "3");
    return node;
  }

  function jobCard(x, y, width, height, color, label, fontSize) {
    const group = document.createElementNS(SVG_NS, "g");
    group.appendChild(rect(x, y, width, height, color, "#111111", 2));
    group.appendChild(
      textNode(
        x + width / 2,
        y + height / 2 + 4,
        label,
        fontSize || 15,
        900,
        "#081018",
        "middle"
      )
    );
    return group;
  }

  function drawStage(svg, box, stage, capacity, maxRows, top, rowHeight) {
    svg.appendChild(rect(box.x, 88, box.width, 50, stage.color, "#111111", 3));
    svg.appendChild(textNode(box.x + 14, 119, stage.label, 18, 900, "#0b0b0b"));
    svg.appendChild(
      textNode(
        box.x + box.width - 14,
        119,
        `${capacity} workers`,
        14,
        800,
        "#0b0b0b",
        "end"
      )
    );
    for (let lane = 0; lane < maxRows; lane += 1) {
      const active = lane < capacity;
      svg.appendChild(
        rect(
          box.x,
          top + lane * rowHeight,
          box.width,
          38,
          active ? "rgba(255,255,255,0.78)" : "rgba(227,216,203,0.38)",
          "#111111",
          1.5,
          active ? null : "6 6"
        )
      );
      svg.appendChild(
        textNode(
          box.x + 12,
          top + lane * rowHeight + 25,
          active ? `W${lane + 1}` : "unused",
          14,
          700,
          active ? "#111111" : "#978b7e"
        )
      );
    }
  }

  function drawQueue(svg, box, label, jobs, color, columns, rows) {
    const panelTop = 184;
    const panelHeight = 186;
    const headerBottom = 240;
    const sidePadding = columns === 1 ? 12 : 10;
    const bottomPadding = 12;
    const colGap = columns === 1 ? 0 : 6;
    const rowGap = 6;
    const slotWidth = Math.floor(
      (box.width - sidePadding * 2 - colGap * Math.max(columns - 1, 0)) / columns
    );
    const slotHeight = Math.floor(
      (panelTop + panelHeight - headerBottom - bottomPadding - rowGap * (rows - 1)) / rows
    );
    const cardWidth = Math.max(20, slotWidth - 6);
    const cardHeight = Math.max(18, slotHeight - 6);
    const cardFontSize = cardWidth < 24 ? 9 : 10;

    svg.appendChild(rect(box.x, panelTop, box.width, panelHeight, "#f7efe2", "#111111", 3));
    svg.appendChild(
      textNode(box.x + box.width / 2, 210, label, 15, 900, "#111111", "middle")
    );
    svg.appendChild(
      textNode(
        box.x + box.width / 2,
        230,
        `${jobs.length} queued`,
        13,
        700,
        "#6b6258",
        "middle"
      )
    );
    const startX = box.x + sidePadding;
    const startY = headerBottom;

    for (let index = 0; index < columns * rows; index += 1) {
      const col = Math.floor(index / rows);
      const row = index % rows;
      const slotX = startX + col * (slotWidth + colGap);
      const slotY = startY + row * (slotHeight + rowGap);
      svg.appendChild(
        rect(
          slotX,
          slotY,
          slotWidth,
          slotHeight,
          "rgba(227,216,203,0.4)",
          "#111111",
          1.3,
          "4 4"
        )
      );
    }

    jobs.slice(0, columns * rows).forEach((job, index) => {
      const col = Math.floor(index / rows);
      const row = index % rows;
      const slotX = startX + col * (slotWidth + colGap);
      const slotY = startY + row * (slotHeight + rowGap);
      const x = slotX + (slotWidth - cardWidth) / 2;
      const y = slotY + (slotHeight - cardHeight) / 2;
      svg.appendChild(
        jobCard(x, y, cardWidth, cardHeight, color, `G${job.id}`, cardFontSize)
      );
    });
  }

  function renderDiagram(svg, state) {
    while (svg.firstChild) {
      svg.removeChild(svg.firstChild);
    }

    const maxRows = 8;
    const rowHeight = 48;
    const laneHeight = 38;
    const top = 146;
    const cardWidth = 72;
    const cardHeight = 28;
    const boxes = {
      sampling: { x: 30, width: 270 },
      proposalQueue: { x: 320, width: 90 },
      evaluation: { x: 430, width: 270 },
      databaseQueue: { x: 720, width: 74 },
      database: { x: 812, width: 238 }
    };

    svg.appendChild(rect(0, 0, 1080, 620, "#fffdf8", "none", 0));
    svg.appendChild(rect(18, 18, 1044, 584, "none", "#111111", 3));
    svg.appendChild(
      textNode(
        42,
        46,
        "Generation cards move left to right through the worker pools.",
        16,
        900,
        "#cc0000"
      )
    );
    svg.appendChild(line(300, 116, 418, 116));
    svg.appendChild(line(700, 116, 800, 116));
    svg.appendChild(textNode(332, 100, "eval handoff", 13, 800, "#6b6258"));
    svg.appendChild(textNode(718, 100, "db handoff", 13, 800, "#6b6258"));

    STAGES.forEach((stage) => {
      drawStage(svg, boxes[stage.key], stage, state.config[stage.key], maxRows, top, rowHeight);
    });
    drawQueue(svg, boxes.proposalQueue, "Ready", state.queues.proposalReady, STAGES[0].color, 2, 4);
    drawQueue(svg, boxes.databaseQueue, "DB", state.queues.databaseReady, STAGES[2].color, 1, 4);

    STAGES.forEach((stage) => {
      state.active[stage.key].forEach((job) => {
        const box = boxes[stage.key];
        const duration = job.durations[stage.key];
        const progress = clamp((state.time - job.startedAt) / duration, 0, 1);
        const x = box.x + 56 + progress * (box.width - cardWidth - 74);
        const y = top + job.lane * rowHeight + (laneHeight - cardHeight) / 2;
        svg.appendChild(jobCard(x, y, cardWidth, cardHeight, stage.color, `G${job.id}`));
      });
    });
  }

  window.ShinkaAsyncThroughputDemo = {
    DEFAULT_CONFIG,
    DEFAULT_EVAL_MODE,
    STAGES,
    createState,
    startSampling,
    advance,
    avgUtilization,
    throughput,
    proposalCapacity,
    describeBottleneck,
    renderDiagram,
    formatPercent,
    normalizeConfig
  };
})();
