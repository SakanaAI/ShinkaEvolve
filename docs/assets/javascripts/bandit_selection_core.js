(function () {
  const MODELS = [
    { name: "gpt-5-mini", color: "#e07a5f" },
    { name: "gemini-3-flash-preview", color: "#3d8c95" },
    { name: "gemini-3.1-pro-preview", color: "#5d4fa1" },
    { name: "gpt-5.4", color: "#d4a017" }
  ];

  const SCENARIOS = {
    balanced: {
      label: "Balanced",
      baseline: 0.54,
      arms: [
        { rewardMean: 0.58, rewardStd: 0.03, costMean: 0.01, costStd: 0.002 },
        { rewardMean: 0.64, rewardStd: 0.05, costMean: 0.022, costStd: 0.005 },
        { rewardMean: 0.71, rewardStd: 0.06, costMean: 0.046, costStd: 0.007 },
        { rewardMean: 0.74, rewardStd: 0.08, costMean: 0.072, costStd: 0.011 }
      ]
    },
    quality: {
      label: "Quality First",
      baseline: 0.58,
      arms: [
        { rewardMean: 0.60, rewardStd: 0.03, costMean: 0.012, costStd: 0.002 },
        { rewardMean: 0.67, rewardStd: 0.04, costMean: 0.024, costStd: 0.004 },
        { rewardMean: 0.77, rewardStd: 0.06, costMean: 0.054, costStd: 0.008 },
        { rewardMean: 0.83, rewardStd: 0.09, costMean: 0.095, costStd: 0.014 }
      ]
    },
    budget: {
      label: "Budget Tight",
      baseline: 0.51,
      arms: [
        { rewardMean: 0.57, rewardStd: 0.03, costMean: 0.007, costStd: 0.0015 },
        { rewardMean: 0.60, rewardStd: 0.04, costMean: 0.013, costStd: 0.0025 },
        { rewardMean: 0.67, rewardStd: 0.05, costMean: 0.035, costStd: 0.006 },
        { rewardMean: 0.73, rewardStd: 0.07, costMean: 0.082, costStd: 0.013 }
      ]
    }
  };

  const DEFAULT_CONFIG = {
    epsilon: 0.2,
    explorationCoef: 1.0,
    costExplorationCoef: 0.1,
    costPower: 1.0,
    costRefPercentile: 50,
    costAwareCoef: 0.5,
    speed: 2.2,
    maxRounds: 100
  };

  function percentile(values, pct) {
    const sorted = [...values].sort((a, b) => a - b);
    if (!sorted.length) {
      return 0;
    }
    const pos = (sorted.length - 1) * (pct / 100);
    const base = Math.floor(pos);
    const rest = pos - base;
    return sorted[base + 1] === undefined
      ? sorted[base]
      : sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function formatMoney(value) {
    return `$${value.toFixed(3)}`;
  }

  function formatScore(value) {
    return value.toFixed(2);
  }

  function formatTick(value) {
    return value >= 0.1 ? value.toFixed(2) : value.toFixed(3);
  }

  function formatPosterior(value) {
    return `${(value * 100).toFixed(0)}%`;
  }

  function compactModelName(name) {
    return name.replace("gemini-", "gm-");
  }

  function compactPosteriorName(name) {
    return compactModelName(name).replace(/-preview$/u, "");
  }

  function logExpm1(z) {
    return z > 50 ? z : Math.log(Math.expm1(Math.max(z, 1e-9)));
  }

  function logAdd(a, b) {
    if (!Number.isFinite(a)) {
      return b;
    }
    if (!Number.isFinite(b)) {
      return a;
    }
    const top = Math.max(a, b);
    return top + Math.log(Math.exp(a - top) + Math.exp(b - top));
  }

  function seeded(seed) {
    return function rng() {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    };
  }

  function gaussian(rng, mean, std) {
    const u1 = Math.max(rng(), 1e-9);
    const u2 = rng();
    const mag = Math.sqrt(-2 * Math.log(u1));
    return mean + std * mag * Math.cos(2 * Math.PI * u2);
  }

  function createArmState(model) {
    return {
      name: model.name,
      color: model.color,
      nSubmitted: 0,
      nCompleted: 0,
      nCosts: 0,
      totalCosts: 0,
      s: -Infinity,
      divs: 0
    };
  }

  function currentSampleCounts(arms) {
    return arms.map((arm) => arm.nSubmitted);
  }

  function createState(config, scenarioKey, running) {
    const arms = MODELS.map(createArmState);
    return {
      config: { ...config },
      scenarioKey,
      scenario: SCENARIOS[scenarioKey],
      rng: seeded(20260421),
      round: 0,
      running,
      history: [],
      totalUplift: 0,
      totalCost: 0,
      obsMax: -Infinity,
      maxCostObserved: -Infinity,
      minCostObserved: Infinity,
      lastDecision: null,
      arms,
      sampleTimeline: [
        {
          round: 0,
          counts: currentSampleCounts(arms)
        }
      ]
    };
  }

  function normalizedMean(state, arm) {
    if (!arm.divs || !Number.isFinite(state.obsMax)) {
      return 0;
    }
    return Math.exp((arm.s - Math.log(Math.max(arm.divs, 1e-9))) - state.obsMax);
  }

  function costRatio(state, arm) {
    if (!Number.isFinite(state.maxCostObserved) || !Number.isFinite(state.minCostObserved)) {
      return null;
    }
    const allMeans = state.arms
      .filter((candidate) => candidate.nCosts > 0)
      .map((candidate) => candidate.totalCosts / candidate.nCosts);
    if (!allMeans.length) {
      return null;
    }
    const totalPulls = state.arms.reduce(
      (sum, candidate) => sum + Math.max(candidate.nSubmitted, candidate.nCompleted),
      0
    );
    const num = 2 * Math.log(Math.max(totalPulls, 2));
    const meanCost = arm.nCosts > 0
      ? arm.totalCosts / arm.nCosts
      : percentile(allMeans, state.config.costRefPercentile);
    const costRange = Math.max(state.maxCostObserved - state.minCostObserved, 0);
    const nCost = Math.max(arm.nCosts, 1);
    const bonus = state.config.costExplorationCoef * costRange * Math.sqrt(num / nCost);
    const denom = Math.max(meanCost - bonus, 1e-7);
    const ref = Math.max(percentile(allMeans, state.config.costRefPercentile), 1e-7);
    return ref / denom;
  }

  function computePosterior(state) {
    const unseen = state.arms.filter(
      (arm) => Math.max(arm.nSubmitted, arm.nCompleted) <= 0
    );
    if (unseen.length > 0) {
      const probability = 1 / unseen.length;
      return state.arms.map((arm) => ({
        arm,
        posterior: unseen.includes(arm) ? probability : 0,
        exploit: 0,
        explore: 0,
        rewardScore: 0,
        costScore: 0,
        finalScore: unseen.includes(arm) ? 1 : 0
      }));
    }

    const totalPulls = state.arms.reduce(
      (sum, arm) => sum + Math.max(arm.nSubmitted, arm.nCompleted),
      0
    );
    const num = 2 * Math.log(Math.max(totalPulls, 2));
    const rows = state.arms.map((arm) => {
      const n = Math.max(arm.nSubmitted, arm.nCompleted, 1);
      const exploit = normalizedMean(state, arm);
      const explore = state.config.explorationCoef * Math.sqrt(num / n);
      return {
        arm,
        exploit,
        explore,
        rewardScore: exploit + explore,
        costRatio: costRatio(state, arm)
      };
    });

    const ratios = rows
      .map((row) => row.costRatio)
      .filter((value) => value !== null);
    const ratioMax = ratios.length ? Math.max(...ratios, 1e-9) : 1;

    rows.forEach((row) => {
      row.costScore = row.costRatio === null
        ? 0
        : Math.pow(row.costRatio / ratioMax, state.config.costPower);
      row.finalScore =
        (1 - state.config.costAwareCoef) * row.rewardScore +
        state.config.costAwareCoef * row.costScore;
    });

    const best = Math.max(...rows.map((row) => row.finalScore));
    const winners = rows.filter((row) => row.finalScore === best);
    const rem = rows.length - winners.length;

    rows.forEach((row) => {
      if (winners.length === rows.length) {
        row.posterior = 1 / rows.length;
      } else if (winners.includes(row)) {
        row.posterior = (1 - state.config.epsilon) / winners.length;
      } else {
        row.posterior = rem > 0 ? state.config.epsilon / rem : 0;
      }
    });

    return rows;
  }

  function sampleIndex(state, rows) {
    const threshold = state.rng();
    let accum = 0;
    for (let i = 0; i < rows.length; i += 1) {
      accum += rows[i].posterior;
      if (threshold <= accum || i === rows.length - 1) {
        return i;
      }
    }
    return 0;
  }

  function stepRound(state) {
    const rows = computePosterior(state);
    const choiceIndex = sampleIndex(state, rows);
    const chosen = rows[choiceIndex].arm;
    const profile = state.scenario.arms[choiceIndex];

    chosen.nSubmitted += 1;
    const rawReward = clamp(
      gaussian(state.rng, profile.rewardMean, profile.rewardStd),
      0.35,
      0.95
    );
    const uplift = Math.max(rawReward - state.scenario.baseline, 0);
    const contrib = uplift > 0 ? logExpm1(uplift) : -Infinity;
    chosen.s = logAdd(chosen.s, contrib);
    chosen.divs += 1;
    chosen.nCompleted += 1;
    if (Number.isFinite(contrib)) {
      state.obsMax = Math.max(state.obsMax, contrib);
    }

    const cost = Math.max(
      gaussian(state.rng, profile.costMean, profile.costStd),
      0.001
    );
    chosen.totalCosts += cost;
    chosen.nCosts += 1;
    state.maxCostObserved = Math.max(state.maxCostObserved, cost);
    state.minCostObserved = Math.min(state.minCostObserved, cost);

    state.round += 1;
    state.totalUplift += uplift;
    state.totalCost += cost;
    state.lastDecision = {
      name: chosen.name,
      color: chosen.color,
      uplift,
      cost
    };
    state.history.unshift({
      round: state.round,
      name: chosen.name,
      color: chosen.color,
      uplift,
      cost,
      posterior: rows[choiceIndex].posterior
    });
    state.history = state.history.slice(0, 30);
    state.sampleTimeline.push({
      round: state.round,
      counts: currentSampleCounts(state.arms)
    });
  }

  function renderScatter(dom, state, rows) {
    const width = 860;
    const height = 320;
    const pad = { top: 42, right: 34, bottom: 58, left: 126 };
    const means = rows.map((row, index) =>
      row.arm.nCosts > 0
        ? row.arm.totalCosts / row.arm.nCosts
        : state.scenario.arms[index].costMean
    );
    const rewardValues = rows.map((row) => row.rewardScore);
    const maxCost = Math.max(...means) * 1.15;
    const rewardMin = Math.max(0, Math.min(...rewardValues));
    const rewardMax = Math.max(...rewardValues, rewardMin + 0.1);
    const rewardSpan = Math.max(rewardMax - rewardMin, 0.12);
    const yDomainMin = Math.max(0, rewardMin - rewardSpan * 0.12);
    const yDomainMax = rewardMax + rewardSpan * 0.2;
    const plotWidth = width - pad.left - pad.right;
    const plotHeight = height - pad.top - pad.bottom;
    const y = (value) =>
      height - pad.bottom - ((value - yDomainMin) / (yDomainMax - yDomainMin)) * plotHeight;
    const x = (value) => pad.left + (value / maxCost) * plotWidth;
    const leader = [...rows].sort((a, b) => b.posterior - a.posterior)[0].arm.name;
    const yTicks = 4;
    const xTicks = 4;
    const yGrid = Array.from({ length: yTicks + 1 }, (_, index) => {
      const value = yDomainMin + ((yDomainMax - yDomainMin) * index) / yTicks;
      const yPos = y(value);
      return `
        <line x1="${pad.left}" y1="${yPos}" x2="${width - pad.right}" y2="${yPos}" stroke="rgba(24,33,43,0.08)" />
        <text x="${pad.left - 12}" y="${yPos + 4}" text-anchor="end" font-size="12" fill="#5b6672">${formatScore(value)}</text>
      `;
    }).join("");
    const xGrid = Array.from({ length: xTicks + 1 }, (_, index) => {
      const value = (maxCost * index) / xTicks;
      const xPos = x(value);
      return `
        <line x1="${xPos}" y1="${pad.top}" x2="${xPos}" y2="${height - pad.bottom}" stroke="rgba(24,33,43,0.08)" />
        <text x="${xPos}" y="${height - pad.bottom + 22}" text-anchor="middle" font-size="12" fill="#5b6672">${formatTick(value)}</text>
      `;
    }).join("");
    const axis = `
      ${yGrid}
      ${xGrid}
      <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="rgba(24,33,43,0.28)" />
      <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${pad.left}" y2="${pad.top}" stroke="rgba(24,33,43,0.28)" />
      <text x="${pad.left + plotWidth / 2}" y="${height - 12}" text-anchor="middle" font-size="22" fill="#5b6672">Mean cost per generation</text>
      <text x="30" y="${pad.top + plotHeight / 2}" text-anchor="middle" font-size="22" fill="#5b6672" transform="rotate(-90 30 ${pad.top + plotHeight / 2})">Reward score</text>
    `;
    const points = rows
      .map((row, index) => {
        const meanCost = means[index];
        const cx = x(meanCost);
        const cy = y(row.rewardScore);
        const isLeader = row.arm.name === leader;
        const labelY = Math.max(cy - 18, pad.top + 12);
        return `
          <circle cx="${cx}" cy="${cy}" r="${isLeader ? 14 : 10}" fill="${row.arm.color}" fill-opacity="${isLeader ? 0.92 : 0.76}" stroke="${isLeader ? "#18212b" : "#fff"}" stroke-width="${isLeader ? 3 : 2}" />
          <text x="${cx}" y="${labelY}" text-anchor="middle" font-size="11" fill="#18212b">${compactModelName(row.arm.name)}</text>
        `;
      })
      .join("");
    dom.scatter.innerHTML = axis + points;
  }

  function renderPosteriorBars(dom, rows) {
    const bars = rows
      .map((row) => {
        const compactName = compactPosteriorName(row.arm.name);
        return `
          <div class="se-bandit-demo__posterior-bar" aria-label="${row.arm.name}: ${formatPosterior(row.posterior)}">
            <div class="se-bandit-demo__posterior-value">${formatPosterior(row.posterior)}</div>
            <div class="se-bandit-demo__posterior-track">
              <div
                class="se-bandit-demo__posterior-fill"
                style="height: ${Math.max(row.posterior * 100, 2)}%; background: ${row.arm.color};"
              ></div>
            </div>
            <div class="se-bandit-demo__posterior-name">${compactName}</div>
          </div>
        `;
      })
      .join("");

    dom.posteriorChart.innerHTML = `
      <div class="se-bandit-demo__posterior-y-axis" aria-hidden="true">
        <div class="se-bandit-demo__posterior-axis-label">Posterior Probabilities</div>
        <div class="se-bandit-demo__posterior-axis-ticks">
          <span>100%</span>
          <span>50%</span>
          <span>0%</span>
        </div>
      </div>
      <div class="se-bandit-demo__posterior-bars">${bars}</div>
    `;
  }

  function renderSampleTimeline(dom, state) {
    const width = 860;
    const height = 320;
    const pad = { top: 28, right: 34, bottom: 56, left: 88 };
    const plotWidth = width - pad.left - pad.right;
    const plotHeight = height - pad.top - pad.bottom;
    const timeline = state.sampleTimeline;
    const maxRound = Math.max(state.round, 1);
    const maxCount = Math.max(
      1,
      ...timeline.flatMap((entry) => entry.counts)
    );
    const x = (value) => pad.left + (value / maxRound) * plotWidth;
    const y = (value) => height - pad.bottom - (value / maxCount) * plotHeight;
    const xTicks = Math.min(5, maxRound);
    const yTicks = Math.min(4, maxCount);

    const yGrid = Array.from({ length: yTicks + 1 }, (_, index) => {
      const value = (maxCount * index) / yTicks;
      const yPos = y(value);
      return `
        <line x1="${pad.left}" y1="${yPos}" x2="${width - pad.right}" y2="${yPos}" stroke="rgba(24,33,43,0.08)" />
        <text x="${pad.left - 12}" y="${yPos + 4}" text-anchor="end" font-size="12" fill="#5b6672">${Math.round(value)}</text>
      `;
    }).join("");

    const xGrid = Array.from({ length: xTicks + 1 }, (_, index) => {
      const value = (maxRound * index) / xTicks;
      const xPos = x(value);
      return `
        <line x1="${xPos}" y1="${pad.top}" x2="${xPos}" y2="${height - pad.bottom}" stroke="rgba(24,33,43,0.08)" />
        <text x="${xPos}" y="${height - pad.bottom + 22}" text-anchor="middle" font-size="12" fill="#5b6672">${Math.round(value)}</text>
      `;
    }).join("");

    const series = MODELS.map((model, index) => {
      const linePoints = timeline.map((entry) => ({
        x: x(entry.round),
        y: y(entry.counts[index])
      }));
      const pathData = linePoints
        .map((point, pointIndex) => `${pointIndex === 0 ? "M" : "L"} ${point.x} ${point.y}`)
        .join(" ");
      const markers = linePoints
        .filter((_, pointIndex) => pointIndex === 0 || pointIndex === linePoints.length - 1 || pointIndex % 8 === 0)
        .map((point) => `
          <circle
            cx="${point.x}"
            cy="${point.y}"
            r="2.1"
            fill="${model.color}"
            fill-opacity="0.38"
          />
        `)
        .join("");
      const last = timeline[timeline.length - 1];
      const lastX = x(last.round);
      const lastY = y(last.counts[index]);
      return `
        <path
          d="${pathData}"
          fill="none"
          stroke="${model.color}"
          stroke-opacity="0.18"
          stroke-width="6"
          stroke-linecap="round"
          stroke-linejoin="round"
        />
        <path
          class="se-bandit-demo__timeline-line"
          d="${pathData}"
          fill="none"
          stroke="${model.color}"
          stroke-width="3.25"
          stroke-linecap="round"
          stroke-linejoin="round"
        />
        ${markers}
        <circle
          class="se-bandit-demo__timeline-dot"
          cx="${lastX}"
          cy="${lastY}"
          r="5.5"
          fill="${model.color}"
          stroke="rgba(255,255,255,0.9)"
          stroke-width="2"
        />
      `;
    }).join("");

    const legendX = pad.left + 16;
    const legendY = pad.top + 14;
    const legendRowHeight = 20;
    const legendWidth = 150;
    const legendHeight = 16 + MODELS.length * legendRowHeight;
    const legendRows = MODELS.map((model, index) => {
      const rowY = legendY + 18 + index * legendRowHeight;
      return `
        <line x1="${legendX + 10}" y1="${rowY - 5}" x2="${legendX + 28}" y2="${rowY - 5}" stroke="${model.color}" stroke-width="3.25" stroke-linecap="round" />
        <circle cx="${legendX + 19}" cy="${rowY - 5}" r="3.2" fill="${model.color}" />
        <text x="${legendX + 36}" y="${rowY - 1}" font-size="11" fill="#18212b">${compactPosteriorName(model.name)}</text>
      `;
    }).join("");

    dom.sampleChart.innerHTML = `
      ${yGrid}
      ${xGrid}
      <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="rgba(24,33,43,0.28)" />
      <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${pad.left}" y2="${pad.top}" stroke="rgba(24,33,43,0.28)" />
      <rect x="${legendX}" y="${legendY}" width="${legendWidth}" height="${legendHeight}" rx="10" fill="rgba(255,255,255,0.88)" stroke="rgba(24,33,43,0.12)" />
      ${legendRows}
      <text x="${pad.left + plotWidth / 2}" y="${height - 12}" text-anchor="middle" font-size="22" fill="#5b6672">Generation</text>
      <text x="30" y="${pad.top + plotHeight / 2}" text-anchor="middle" font-size="22" fill="#5b6672" transform="rotate(-90 30 ${pad.top + plotHeight / 2})">Cumulative Samples</text>
      ${series}
    `;
  }

  function renderStatic(dom, state) {
    const rows = computePosterior(state);
    renderScatter(dom, state, rows);
    renderPosteriorBars(dom, rows);
    renderSampleTimeline(dom, state);
    return rows;
  }

  function updateSummary(dom, state, rows) {
    dom.costAware.value = String(state.config.costAwareCoef);
    dom.costAwareLabel.textContent = state.config.costAwareCoef.toFixed(2);
    dom.arenaTitle.textContent = `Model Arena · Generation ${state.round}`;
    dom.playPause.textContent = state.running ? "Pause" : "Play";
    dom.playPause.setAttribute(
      "aria-label",
      state.running ? "Pause simulation" : "Play simulation"
    );
  }

  window.ShinkaBanditModelSelectionDemo = {
    DEFAULT_CONFIG,
    SCENARIOS,
    createState,
    stepRound,
    renderStatic,
    updateSummary
  };
})();
