// Renders the showcase from data/cases.json (written by `ncc showcase`).
// All data-derived text goes through textContent, never innerHTML.

const SITE = {
  repoUrl: "https://github.com/parishamaheshj18/context-based-color-correction",
  author: "Parisha",
  featured: "lab-shelf-warm", // the before/after shown at the top
};

// Bar order within each scene; slots follow that order (validated as adjacent pairs).
const SERIES = [
  { key: "ours", label: "With agent review", color: "var(--series-1)" },
  { key: "first_pass", label: "Before agent review", color: "var(--series-2)" },
  { key: "gray_world", label: "Gray world", color: "var(--series-3)" },
  { key: "before", label: "Camera", color: "var(--series-4)" },
];
const LABELS = { foliage: "Foliage", skin: "Skin", concrete: "Concrete", paper: "White paper", snow: "Snow", cloud: "Cloud", neutral_surface: "Neutral surface", known_color: "Your match" };
const VERDICTS = { consistent: "consistent", doubtful: "doubtful", wrong: "wrong" };
const SVG_NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (name === "text") node.textContent = value;
    else if (name === "class") node.className = value;
    else if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
    else node.setAttribute(name, value === true ? "" : value);
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function svg(tag, attrs = {}, children = []) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (name === "text") node.textContent = value;
    else node.setAttribute(name, value);
  }
  for (const child of [].concat(children)) if (child) node.append(child);
  return node;
}

const fmt = {
  deg: (v) => `${v.toFixed(v < 10 ? 2 : 1)}°`,
  kelvin: (v) => `${Math.round(v / 10) * 10} K`,
};
const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;

// ---------- tooltip ----------
const tooltip = document.getElementById("tooltip");
function showTooltip(event, title, rows) {
  tooltip.replaceChildren(
    el("div", { text: title, style: "font-weight:600;margin-bottom:0.2rem" }),
    ...rows.map(([color, value, label]) =>
      el("div", { class: "row" }, [
        el("i", { class: "key", style: `background:${color}` }),
        el("strong", { text: value }),
        el("span", { text: label }),
      ]),
    ),
  );
  tooltip.hidden = false;
  const box = tooltip.getBoundingClientRect();
  const x = Math.min(event.clientX + 14, window.innerWidth - box.width - 8);
  const y = Math.max(8, event.clientY - box.height - 12);
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}
function hideTooltip() { tooltip.hidden = true; }

// ---------- before/after slider ----------
function compareSlider(afterSrc, beforeSrc, afterLabel, beforeLabel, alt) {
  const beforeImg = el("img", { src: beforeSrc, alt: `${alt} (${beforeLabel})`, loading: "lazy" });
  const beforeTag = el("span", { class: "tag left", text: beforeLabel });
  const beforeLayer = el("div", { class: "before-layer" }, [beforeImg]);
  const divider = el("div", { class: "divider", "aria-hidden": "true" });
  const range = el("input", { type: "range", min: 0, max: 100, value: 50, "aria-label": `Reveal ${beforeLabel} vs ${afterLabel}` });
  const root = el("div", { class: "compare" }, [
    el("img", { src: afterSrc, alt: `${alt} (${afterLabel})`, loading: "lazy" }),
    beforeLayer, range, divider, beforeTag, el("span", { class: "tag right", text: afterLabel }),
  ]);
  const update = () => {
    beforeLayer.style.clipPath = `inset(0 ${100 - range.value}% 0 0)`;
    divider.style.left = `${range.value}%`;
  };
  range.addEventListener("input", update);
  update();
  return {
    root,
    setBefore(src, label) {
      beforeImg.src = src;
      beforeImg.alt = `${alt} (${label})`;
      beforeTag.textContent = label;
      range.setAttribute("aria-label", `Reveal ${label} vs ${afterLabel}`);
    },
  };
}

function cameraLabel(c) { return c.input_mode === "raster" ? "Original" : "Camera"; }
// True when a revised round, not the first correction, was delivered.
function revisedDelivered(c) { return (c.review?.rounds.find((r) => r.delivered)?.round ?? 1) > 1; }

// Buttons that swap what the slider compares the corrected photo against.
function compareChoices(c, slider) {
  const options = [
    { src: c.images.before, label: cameraLabel(c) },
    { src: c.images.gray_world, label: "Gray world" },
  ];
  if (c.images.first_pass && revisedDelivered(c)) {
    options.push({ src: c.images.first_pass, label: "Before agent review" });
  }
  const buttons = options.map((option, i) =>
    el("button", {
      type: "button", text: option.label, "aria-pressed": i === 0 ? "true" : "false",
      onclick: () => {
        slider.setBefore(option.src, option.label);
        buttons.forEach((b) => b.setAttribute("aria-pressed", b === buttons[i] ? "true" : "false"));
      },
    }),
  );
  return el("div", { class: "compare-with", role: "group", "aria-label": "Compare the corrected photo with" }, [
    el("span", { text: "Compare with" }), ...buttons,
  ]);
}

// ---------- hero ----------
function renderStats(cases) {
  const charted = cases.filter((c) => c.ground_truth);
  const cameras = new Set(cases.map((c) => c.camera).filter(Boolean));
  const reviews = cases.reduce((n, c) => n + (c.review?.rounds.length ?? 0), 0);
  const stats = [
    [String(cases.length), `photos, ${cameras.size} cameras`],
    ["2", reviews ? `Claude agents, ${reviews} critic reviews` : "Claude agents"],
    ["0", "AI-generated pixels"],
  ];
  if (charted.length) {
    const camera = mean(charted.map((c) => c.ground_truth.before.angular_error_deg));
    const ours = mean(charted.map((c) => c.ground_truth.ours.angular_error_deg));
    stats.splice(2, 0, [`${fmt.deg(camera)} → ${fmt.deg(ours)}`, "error vs. hidden chart: camera → ours"]);
  }
  document.getElementById("stats").append(
    ...stats.map(([value, label]) => el("div", { class: "stat" }, [el("dt", { text: value }), el("dd", { text: label })])),
  );
}

function renderFeatured(c) {
  if (!c) return;
  const slider = compareSlider(c.images.ours, c.images.before, "Corrected", cameraLabel(c), c.title);
  document.getElementById("featured").append(
    slider.root,
    el("p", {
      class: "note center",
      text: c.ground_truth ? "Drag to compare. The chart was hidden from the agents." : "Drag to compare.",
    }),
  );
}

function renderSheetExample(cases) {
  const c = cases.find((x) => x.id === SITE.featured && x.review) ?? cases.find((x) => x.review);
  if (!c) return;
  const round = c.review.rounds[0];
  const figure = document.getElementById("sheet-example");
  figure.querySelector("img").src = round.sheet;
  figure.querySelector("figcaption").textContent = "What the critic sees: before | after, and each surface next to its true color.";
  figure.hidden = false;
}

// ---------- charts ----------
function roundedTopBar(x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height);
  return `M${x},${y + height}V${y + r}Q${x},${y} ${x + r},${y}H${x + width - r}Q${x + width},${y} ${x + width},${y + r}V${y + height}Z`;
}

function niceMax(value) {
  const steps = [1, 2, 2.5, 5, 10];
  const magnitude = 10 ** Math.floor(Math.log10(value || 1));
  return steps.map((s) => s * magnitude).find((s) => s >= value) ?? 10 * magnitude;
}

function accuracyChart(cases, series) {
  const width = 900, height = 320, margin = { top: 28, right: 12, bottom: 54, left: 44 };
  const innerW = width - margin.left - margin.right, innerH = height - margin.top - margin.bottom;
  const values = cases.flatMap((c) => series.map((s) => c.ground_truth[s.key].angular_error_deg));
  const yMax = niceMax(Math.max(...values) * 1.15);
  const y = (v) => margin.top + innerH * (1 - v / yMax);
  const groupW = innerW / cases.length;
  const barW = Math.min(46, (groupW * 0.62) / series.length);
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Light-color error per scene and method" });
  for (let t = 0; t <= 4; t++) {
    const v = (yMax / 4) * t;
    root.append(svg("line", { class: "gridline", x1: margin.left, x2: width - margin.right, y1: y(v), y2: y(v) }));
    root.append(svg("text", { x: margin.left - 8, y: y(v) + 4, "text-anchor": "end", text: `${+v.toFixed(2)}°` }));
  }
  cases.forEach((c, i) => {
    const groupX = margin.left + groupW * i + (groupW - barW * series.length - 2 * (series.length - 1)) / 2;
    const rows = series.map((s) => [s.color, fmt.deg(c.ground_truth[s.key].angular_error_deg), s.label]);
    series.forEach((s, j) => {
      const value = c.ground_truth[s.key].angular_error_deg;
      const x = groupX + j * (barW + 2);
      const bar = svg("path", { d: roundedTopBar(x, y(value), barW, y(0) - y(value), 4), fill: s.color, tabindex: 0 });
      bar.setAttribute("aria-label", `${c.title}: ${s.label} ${fmt.deg(value)}`);
      const hit = svg("rect", { x: x - 1, y: margin.top, width: barW + 2, height: innerH, fill: "transparent" });
      for (const target of [bar, hit]) {
        target.addEventListener("pointermove", (e) => showTooltip(e, c.title, rows));
        target.addEventListener("pointerleave", hideTooltip);
      }
      bar.addEventListener("focus", () => { const r = bar.getBoundingClientRect(); showTooltip({ clientX: r.right, clientY: r.top }, c.title, rows); });
      bar.addEventListener("blur", hideTooltip);
      root.append(hit, bar, svg("text", { class: "value-label", x: x + barW / 2, y: y(value) - 5, "text-anchor": "middle", text: value.toFixed(2) }));
    });
    const label = svg("text", { x: margin.left + groupW * (i + 0.5), y: height - margin.bottom + 20, "text-anchor": "middle" });
    wrapLabel(label, c.title, 34);
    root.append(label);
  });
  root.append(svg("line", { class: "axis", x1: margin.left, x2: width - margin.right, y1: y(0), y2: y(0) }));
  return root;
}

function wrapLabel(textNode, text, maxChars) {
  const words = text.split(" ");
  const lines = [""];
  for (const word of words) {
    if ((lines.at(-1) + " " + word).trim().length > maxChars && lines.length < 2) lines.push(word);
    else lines[lines.length - 1] = (lines.at(-1) + " " + word).trim();
  }
  lines.forEach((line, i) => textNode.append(svg("tspan", { x: textNode.getAttribute("x"), dy: i ? 15 : 0, text: line })));
}

function accuracyTable(cases, series) {
  const head = el("tr", {}, [el("th", { text: "Scene" }), ...series.map((s) => el("th", { text: `${s.label}: angle / ΔE00` }))]);
  const rows = cases.map((c) =>
    el("tr", {}, [
      el("td", { text: c.title }),
      ...series.map((s) => el("td", { text: `${fmt.deg(c.ground_truth[s.key].angular_error_deg)} / ${c.ground_truth[s.key].delta_e00.toFixed(1)}` })),
    ]),
  );
  return el("table", { class: "data-table" }, [el("thead", {}, [head]), el("tbody", {}, rows)]);
}

function accuracyNote(charted) {
  const parts = charted.map((c) => {
    const gt = c.ground_truth;
    if (!gt.first_pass || !revisedDelivered(c)) return `${c.title}: first correction kept.`;
    return `${c.title}: agent review ${fmt.deg(gt.first_pass.angular_error_deg)} → ${fmt.deg(gt.ours.angular_error_deg)}.`;
  });
  return `${parts.join(" ")} Spot checks on ${charted.length} photos, not a benchmark.`;
}

function renderAccuracy(cases) {
  const container = document.getElementById("accuracy-chart");
  const charted = cases.filter((c) => c.ground_truth);
  if (!charted.length) { document.getElementById("accuracy").hidden = true; return; }
  const series = SERIES.filter((s) => charted.every((c) => c.ground_truth[s.key]));
  const table = accuracyTable(charted, series);
  table.hidden = true;
  const toggle = el("button", { class: "toggle-table", text: "Show table" });
  toggle.addEventListener("click", () => {
    table.hidden = !table.hidden;
    toggle.textContent = table.hidden ? "Show table" : "Hide table";
  });
  container.append(
    el("div", { class: "chart-header" }, [
      el("div", { class: "legend" }, series.map((s) => el("span", {}, [el("i", { style: `background:${s.color}` }), s.label]))),
      toggle,
    ]),
    el("div", { class: "chart" }, [accuracyChart(charted, series)]),
    table,
  );
  document.getElementById("accuracy-note").textContent = accuracyNote(charted);
}

// ---------- case ----------
function lightText(physics) {
  if (!physics.cct_k) return null;
  const kind = physics.family === "blackbody" ? "tungsten-like" : "daylight";
  return `Light ≈ ${fmt.kelvin(physics.cct_k)}, ${kind}`;
}

function roundsTimeline(review) {
  return el("ol", { class: "rounds" }, review.rounds.map((r) => {
    const critique = r.critique;
    const cast = critique?.remaining_cast === "none" ? "no cast left" : `${critique?.remaining_cast} cast`;
    const text = critique ? `severity ${critique.cast_severity.toFixed(2)}, ${cast}` : "critic unavailable";
    return el("li", { class: r.delivered ? "delivered" : "" }, [
      el("strong", { text: `Round ${r.round}` }), el("span", { text }),
      r.delivered ? el("span", { class: "badge", text: "delivered" }) : null,
    ]);
  }));
}

function quote(text) {
  const node = el("blockquote", { class: "rationale clamped", text, title: "Click to expand" });
  node.addEventListener("click", () => node.classList.toggle("clamped"));
  return node;
}

function referenceList(round) {
  const verdicts = new Map((round.critique?.verdicts ?? []).map((v) => [v.reference_id, v]));
  return el("ul", { class: "picks" }, round.references.map((ref) => {
    const verdict = verdicts.get(ref.id);
    const difference = ref.delta_e === null ? "" : ` · ΔE ${ref.delta_e}`;
    return el("li", { class: ref.used ? "yes" : "no" }, [
      el("span", { class: "mark", text: ref.used ? "✓" : "✗", "aria-label": ref.used ? "used" : "not used" }),
      el("span", {}, [
        el("strong", { text: `${ref.id} ${LABELS[ref.label] ?? ref.label}` }),
        ` — ${ref.name}${difference}`,
        verdict ? el("span", { class: `verdict ${verdict.verdict}`, text: `critic: ${VERDICTS[verdict.verdict] ?? verdict.verdict}` }) : null,
      ]),
    ]);
  }));
}

function reviewDetails(c) {
  const review = c.review;
  const delivered = review.rounds.find((r) => r.delivered) ?? review.rounds.at(-1);
  const gt = c.ground_truth;
  return el("details", { class: "case-details" }, [
    el("summary", { text: "How the agents handled this photo" }),
    el("div", { class: "review" }, [
      roundsTimeline(review),
      review.stop_reason ? el("p", { class: "note", text: `Stopped because ${review.stop_reason}.` }) : null,
      el("div", { class: "details-grid" }, [
        el("a", { href: delivered.sheet, target: "_blank", rel: "noopener", class: "sheet-link" }, [
          el("img", { src: delivered.sheet, alt: `${c.title}: review sheet from round ${delivered.round}`, loading: "lazy" }),
          el("span", { class: "note", text: `Review sheet, round ${delivered.round} (open full size)` }),
        ]),
        el("div", {}, [
          delivered.critique ? el("h4", { text: "Agent 2 · Critic" }) : null,
          delivered.critique ? quote(delivered.critique.observations) : null,
          el("h4", { text: "Agent 1 · Scene analyst" }),
          delivered.rationale ? quote(delivered.rationale) : null,
          el("h4", { text: "Surfaces used" }),
          referenceList(delivered),
          gt ? el("p", { class: "note", text: `Error vs. hidden chart: ${SERIES.filter((s) => gt[s.key]).map((s) => `${s.label} ${fmt.deg(gt[s.key].angular_error_deg)}`).join(" · ")}` }) : null,
        ]),
      ]),
    ]),
  ]);
}

// Older data without an agent review: show the single scene analysis.
function analysisDetails(c) {
  const ai = c.ai;
  const picks = el("ul", { class: "picks" }, ai.candidates.map((m) =>
    el("li", { class: m.usable ? "yes" : "no" }, [
      el("span", { class: "mark", text: m.usable ? "✓" : "✗", "aria-label": m.usable ? "trusted" : "rejected" }),
      el("span", {}, [el("strong", { text: LABELS[m.label] ?? m.label }), m.description ? ` — ${m.description}` : ""]),
    ]),
  ));
  return el("details", { class: "case-details" }, [
    el("summary", { text: "What Claude saw" }),
    el("div", { class: "details-grid" }, [
      el("img", { src: c.images.overlay, alt: `${c.title}: regions Claude chose`, loading: "lazy" }),
      el("div", {}, [ai.rationale ? el("blockquote", { class: "rationale", text: ai.rationale }) : null, picks]),
    ]),
  ]);
}

function caseCard(c) {
  const slider = compareSlider(c.images.ours, c.images.before, "Corrected", cameraLabel(c), c.title);
  const rounds = c.review?.rounds.length;
  const meta = [c.camera, lightText(c.physics), rounds ? `${rounds} review round${rounds > 1 ? "s" : ""}` : null].filter(Boolean).join(" · ");
  return el("article", { class: "case", id: c.id }, [
    el("div", { class: "case-head" }, [el("h3", { text: c.title }), meta ? el("p", { text: meta }) : null]),
    c.caption ? el("p", { class: "note caption", text: c.caption }) : null,
    slider.root,
    compareChoices(c, slider),
    c.review ? reviewDetails(c) : analysisDetails(c),
  ]);
}

// ---------- page ----------
function showToolScreenshot() {
  const shot = document.getElementById("match-tool-shot");
  const reveal = () => { document.getElementById("match-tool").hidden = false; };
  if (shot.complete && shot.naturalWidth) reveal();
  else shot.addEventListener("load", reveal, { once: true });
}

async function main() {
  for (const link of document.querySelectorAll(".repo-link")) {
    if (SITE.repoUrl) link.href = SITE.repoUrl; else link.remove();
  }
  showToolScreenshot();
  const data = await (await fetch("data/cases.json")).json();
  const cases = data.cases;
  renderStats(cases);
  renderFeatured(cases.find((c) => c.id === SITE.featured) ?? cases[0]);
  renderSheetExample(cases);
  renderAccuracy(cases);
  document.getElementById("cases").append(...cases.map(caseCard));
  const footer = document.getElementById("footer-meta");
  footer.append(`Built by ${SITE.author} · agents: ${data.pipeline.vlm ?? "Claude"} · `);
  if (SITE.repoUrl) footer.append(el("a", { href: SITE.repoUrl, text: "source on GitHub" }));
}

main().catch((error) => {
  document.getElementById("cases").append(el("p", { class: "note", text: `Could not load results: ${error.message}` }));
});
