// Renders the showcase from data/cases.json (written by `ncc showcase`).
// All data-derived text goes through textContent, never innerHTML.

const SITE = {
  repoUrl: "", // e.g. "https://github.com/<user>/natural-color-charts"
  author: "Parisha",
};

const SERIES = [
  { key: "ours", label: "This project", color: "var(--series-ours)" },
  { key: "gray_world", label: "Gray world", color: "var(--series-gray-world)" },
  { key: "before", label: "Camera", color: "var(--series-before)" },
];
const LABELS = { foliage: "Foliage", skin: "Skin", concrete: "Concrete", paper: "White paper", snow: "Snow", cloud: "Cloud", neutral_surface: "Neutral surface" };
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
  pct: (v) => `${Math.round(v * 100)}%`,
};

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

function accuracyChart(cases) {
  const width = 900, height = 320, margin = { top: 28, right: 12, bottom: 54, left: 44 };
  const innerW = width - margin.left - margin.right, innerH = height - margin.top - margin.bottom;
  const values = cases.flatMap((c) => SERIES.map((s) => c.ground_truth[s.key]?.angular_error_deg ?? 0));
  const yMax = niceMax(Math.max(...values) * 1.15);
  const y = (v) => margin.top + innerH * (1 - v / yMax);
  const groupW = innerW / cases.length;
  const barW = Math.min(46, (groupW * 0.62) / SERIES.length);
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Angular error per scene and method" });
  for (let t = 0; t <= 4; t++) {
    const v = (yMax / 4) * t;
    root.append(svg("line", { class: "gridline", x1: margin.left, x2: width - margin.right, y1: y(v), y2: y(v) }));
    root.append(svg("text", { x: margin.left - 8, y: y(v) + 4, "text-anchor": "end", text: `${+v.toFixed(2)}°` }));
  }
  cases.forEach((c, i) => {
    const groupX = margin.left + groupW * i + (groupW - barW * SERIES.length - 2 * (SERIES.length - 1)) / 2;
    const rows = SERIES.map((s) => [s.color, fmt.deg(c.ground_truth[s.key].angular_error_deg), s.label]);
    SERIES.forEach((s, j) => {
      const value = c.ground_truth[s.key].angular_error_deg;
      const x = groupX + j * (barW + 2);
      const bar = svg("path", { d: roundedTopBar(x, y(value), barW, y(0) - y(value), 4), fill: s.color, tabindex: 0 });
      bar.setAttribute("aria-label", `${c.title}: ${s.label} ${fmt.deg(value)}`);
      const hit = svg("rect", { x: x - 1, y: margin.top, width: barW + 2, height: innerH, fill: "transparent" });
      for (const target of [bar, hit]) {
        target.addEventListener("pointermove", (e) => showTooltip(e, c.title, rows));
        target.addEventListener("pointerleave", hideTooltip);
      }
      bar.addEventListener("focus", (e) => { const r = bar.getBoundingClientRect(); showTooltip({ clientX: r.right, clientY: r.top }, c.title, rows); });
      bar.addEventListener("blur", hideTooltip);
      root.append(hit, bar, svg("text", { class: "value-label", x: x + barW / 2, y: y(value) - 5, "text-anchor": "middle", text: value.toFixed(1) }));
    });
    const label = svg("text", { x: margin.left + groupW * (i + 0.5), y: height - margin.bottom + 20, "text-anchor": "middle" });
    wrapLabel(label, c.title, 26);
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

function accuracyTable(cases) {
  const head = el("tr", {}, [el("th", { text: "Scene" }), ...SERIES.map((s) => el("th", { text: `${s.label}: angle / ΔE00` }))]);
  const rows = cases.map((c) =>
    el("tr", {}, [
      el("td", { text: c.title }),
      ...SERIES.map((s) => el("td", { text: `${fmt.deg(c.ground_truth[s.key].angular_error_deg)} / ${c.ground_truth[s.key].delta_e00.toFixed(1)}` })),
    ]),
  );
  return el("table", { class: "data-table" }, [el("thead", {}, [head]), el("tbody", {}, rows)]);
}

// ---------- case ----------
function lightText(physics) {
  if (!physics.cct_k) return null;
  const kind = physics.family === "blackbody" ? "tungsten-like" : "daylight";
  return `Light ≈ ${fmt.kelvin(physics.cct_k)}, ${kind}`;
}

function details(c) {
  const ai = c.ai;
  const picks = el("ul", { class: "picks" }, ai.candidates.map((m) =>
    el("li", { class: m.usable ? "yes" : "no" }, [
      el("span", { class: "mark", text: m.usable ? "✓" : "✗", "aria-label": m.usable ? "trusted" : "rejected" }),
      el("span", {}, [el("strong", { text: LABELS[m.label] ?? m.label }), m.description ? ` — ${m.description}` : ""]),
    ]),
  ));
  const gt = c.ground_truth;
  return el("details", { class: "case-details" }, [
    el("summary", { text: "What Claude saw" }),
    el("div", { class: "details-grid" }, [
      el("img", { src: c.images.overlay, alt: `${c.title}: regions Claude chose`, loading: "lazy" }),
      el("div", {}, [
        ai.rationale ? el("blockquote", { class: "rationale", text: ai.rationale }) : null,
        picks,
        gt ? el("p", { class: "note", text: `Error vs. hidden chart: ${SERIES.map((s) => `${s.label} ${fmt.deg(gt[s.key].angular_error_deg)}`).join(" · ")}` }) : null,
      ]),
    ]),
  ]);
}

function caseCard(c) {
  const beforeLabel = c.input_mode === "raster" ? "Original" : "Camera";
  const slider = compareSlider(c.images.ours, c.images.before, "Corrected", beforeLabel, c.title);
  const meta = [c.camera, lightText(c.physics)].filter(Boolean).join(" · ");
  return el("article", { class: "case", id: c.id }, [
    el("div", { class: "case-head" }, [el("h3", { text: c.title }), meta ? el("p", { text: meta }) : null]),
    slider.root,
    details(c),
  ]);
}

// ---------- page ----------
function renderAccuracy(cases) {
  const container = document.getElementById("accuracy-chart");
  const charted = cases.filter((c) => c.ground_truth);
  if (!charted.length) { document.getElementById("accuracy").hidden = true; return; }
  const table = accuracyTable(charted);
  table.hidden = true;
  const toggle = el("button", { class: "toggle-table", text: "Table" });
  toggle.addEventListener("click", () => { table.hidden = !table.hidden; });
  container.append(
    el("div", { class: "chart-header" }, [
      el("div", { class: "legend" }, SERIES.map((s) => el("span", {}, [el("i", { style: `background:${s.color}` }), s.label]))),
      toggle,
    ]),
    el("div", { class: "chart" }, [accuracyChart(charted)]),
    table,
  );
}

function showToolScreenshot() {
  const shot = document.getElementById("match-tool-shot");
  const reveal = () => { document.getElementById("match-tool").hidden = false; };
  if (shot.complete && shot.naturalWidth) reveal();
  else shot.addEventListener("load", reveal, { once: true });
}

async function main() {
  const repo = document.getElementById("repo-link");
  if (SITE.repoUrl) repo.href = SITE.repoUrl; else repo.remove();
  showToolScreenshot();
  const data = await (await fetch("data/cases.json")).json();
  renderAccuracy(data.cases);
  document.getElementById("cases").append(...data.cases.map(caseCard));
  document.getElementById("footer-meta").textContent = `Built by ${SITE.author} · Python, colour-science, Claude, Segment Anything`;
}

main().catch((error) => {
  document.getElementById("cases").append(el("p", { class: "note", text: `Could not load results: ${error.message}` }));
});
