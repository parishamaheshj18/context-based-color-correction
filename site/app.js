// Renders the showcase from data/cases.json (written by `ncc showcase`).
// All data-derived text goes through textContent, never innerHTML.

const SITE = {
  repoUrl: "", // e.g. "https://github.com/<user>/natural-color-charts"
  author: "Parisha",
};

const SERIES = [
  { key: "ours", label: "This project", color: "var(--series-ours)" },
  { key: "gray_world", label: "Gray world (classic AWB)", color: "var(--series-gray-world)" },
  { key: "before", label: "Camera as-shot", color: "var(--series-before)" },
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

function spectrumChart(spectrum, caption) {
  const width = 560, height = 220, margin = { top: 12, right: 12, bottom: 34, left: 40 };
  const innerW = width - margin.left - margin.right, innerH = height - margin.top - margin.bottom;
  const wl = spectrum.wavelengths_nm;
  const x = (v) => margin.left + ((v - wl[0]) / (wl.at(-1) - wl[0])) * innerW;
  const y = (v) => margin.top + innerH * (1 - v / 1.05);
  const path = (values) => values.map((v, i) => `${i ? "L" : "M"}${x(wl[i]).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": caption });
  for (const v of [0, 0.5, 1]) {
    root.append(svg("line", { class: "gridline", x1: margin.left, x2: width - margin.right, y1: y(v), y2: y(v) }));
    root.append(svg("text", { x: margin.left - 6, y: y(v) + 4, "text-anchor": "end", text: v.toFixed(1) }));
  }
  for (const v of [400, 500, 600, 700]) root.append(svg("text", { x: x(v), y: height - 12, "text-anchor": "middle", text: `${v} nm` }));
  root.append(svg("path", { d: path(spectrum.d65_relative_power), fill: "none", stroke: "var(--text-muted)", "stroke-width": 2, "stroke-dasharray": "5 4" }));
  root.append(svg("path", { d: path(spectrum.relative_power), fill: "none", stroke: "var(--series-ours)", "stroke-width": 2, "stroke-linejoin": "round" }));
  const cross = svg("line", { y1: margin.top, y2: margin.top + innerH, stroke: "var(--text-muted)", "stroke-width": 1, visibility: "hidden" });
  const hit = svg("rect", { x: margin.left, y: margin.top, width: innerW, height: innerH, fill: "transparent" });
  root.append(cross, hit);
  hit.addEventListener("pointermove", (e) => {
    const box = root.getBoundingClientRect();
    const svgX = ((e.clientX - box.left) / box.width) * width;
    const i = Math.max(0, Math.min(wl.length - 1, Math.round(((svgX - margin.left) / innerW) * (wl.length - 1))));
    cross.setAttribute("x1", x(wl[i])); cross.setAttribute("x2", x(wl[i])); cross.setAttribute("visibility", "visible");
    showTooltip(e, `${wl[i]} nm`, [
      ["var(--series-ours)", spectrum.relative_power[i].toFixed(2), "Estimated light"],
      ["var(--text-muted)", spectrum.d65_relative_power[i].toFixed(2), "D65 daylight (target)"],
    ]);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTooltip(); });
  return el("figure", { class: "chart", style: "margin:0 0 0.75rem" }, [
    el("div", { class: "legend" }, [
      el("span", {}, [el("i", { class: "line", style: "background:var(--series-ours)" }), "Estimated light"]),
      el("span", {}, [el("i", { class: "dashed" }), "D65 daylight (target white)"]),
    ]),
    root,
    el("figcaption", { class: "note", text: caption }),
  ]);
}

// ---------- case card ----------
function tabSet(definitions) {
  const tabs = el("div", { class: "tabs", role: "tablist" });
  const panels = [];
  definitions.forEach((def, i) => {
    const id = `${def.id}-panel`;
    const button = el("button", { role: "tab", "aria-selected": i === 0 ? "true" : "false", "aria-controls": id, id: `${def.id}-tab` }, [
      def.kind ? el("span", { class: `dot ${def.kind}`, "aria-hidden": "true" }) : null,
      def.label,
    ]);
    const panel = el("div", { class: "panel", role: "tabpanel", id, "aria-labelledby": `${def.id}-tab`, hidden: i !== 0 }, def.content());
    button.addEventListener("click", () => {
      tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", "false"));
      panels.forEach((p) => (p.hidden = true));
      button.setAttribute("aria-selected", "true");
      panel.hidden = false;
    });
    tabs.append(button);
    panels.push(panel);
  });
  return [tabs, ...panels];
}

function aiPanel(c) {
  const ai = c.ai;
  const used = ai.source === "vlm";
  const dl = el("dl", {}, [
    el("dt", { text: "Setting" }), el("dd", { text: [ai.environment, ai.time_of_day, ai.weather].filter((v) => v && v !== "unknown").join(" · ") || "not determined" }),
    el("dt", { text: "Light sources" }), el("dd", { text: ai.light_sources.length ? ai.light_sources.join(", ") : "none named" }),
    el("dt", { text: "Light guess" }), el("dd", { text: ai.hypotheses.filter((h) => h.confidence > 0).map((h) => `${h.family} ${h.cct_range_k[0]}–${h.cct_range_k[1]} K (${fmt.pct(h.confidence)})`).join("; ") || "no prior" }),
    el("dt", { text: "Your intent" }), el("dd", { text: `${ai.intent.adaptation.replace("_", " ")}${ai.intent.adaptation === "preserve_ambiance" ? `, ${fmt.pct(ai.intent.strength)} correction` : ""}` }),
  ]);
  const materials = el("ul", { class: "materials" }, ai.candidates.map((m) =>
    el("li", {}, [
      el("span", { class: `verdict ${m.usable ? "yes" : "no"}`, text: m.usable ? "✓ trust" : "✗ reject" }),
      el("span", {}, [el("strong", { text: LABELS[m.label] ?? m.label }), m.description ? ` — ${m.description}` : ""]),
      m.concerns.length ? el("small", { text: `Concerns: ${m.concerns.join("; ")}` }) : null,
    ]),
  ));
  return [
    used ? null : el("p", { class: "note", text: "No vision model was used for this case; all materials were assumed possible." }),
    ai.rationale ? el("blockquote", { class: "rationale" }, [ai.rationale, el("cite", { text: `${ai.model ?? "model"}, answering in a strict JSON schema` })]) : null,
    dl,
    el("h4", { text: "Reference materials the model proposed", style: "margin:0 0 0.5rem;font-size:0.95rem" }),
    materials,
  ];
}

function regionsPanel(c) {
  const rows = c.ai.regions.map((r) =>
    el("tr", {}, [
      el("td", { text: LABELS[r.label] ?? r.label }),
      el("td", { text: r.detector_score?.toFixed(2) ?? "–" }),
      el("td", { text: r.pixels.toLocaleString() }),
      el("td", { text: r.kept ? r.weight.toFixed(2) : "–" }),
      el("td", {}, [r.used ? el("span", { class: "chip good", text: "used" }) : el("span", { class: "chip bad", text: r.reason || "not used" })]),
    ]),
  );
  return [
    el("figure", { class: "overlay-figure" }, [
      el("img", { src: c.images.overlay, alt: `${c.title}: reference regions chosen by the AI`, loading: "lazy" }),
      el("figcaption", { text: `Masks from ${c.ai.detector} + ${c.ai.segmenter}, tinted by material; stronger tint = more trusted. Red = rejected.${c.ground_truth ? " The black box is the hidden color chart." : ""}` }),
    ]),
    rows.length
      ? el("table", { class: "data-table" }, [
          el("thead", {}, [el("tr", {}, ["Material", "Detector score", "Pixels", "Weight", "Status"].map((t) => el("th", { text: t })))]),
          el("tbody", {}, rows),
        ])
      : el("p", { class: "note", text: "No usable reference regions were found, so the estimate fell back to image statistics." }),
  ];
}

function physicsPanel(c) {
  const p = c.physics;
  const metrics = el("div", { class: "metric-row" }, [
    p.cct_k ? el("div", { class: "metric" }, [el("strong", { text: `${fmt.kelvin(p.cct_k)}${p.cct_uncertainty_k ? ` ± ${Math.round(p.cct_uncertainty_k)}` : ""}` }), el("span", { text: `${p.family} light` })]) : null,
    el("div", { class: "metric" }, [el("strong", { text: p.estimator.replaceAll("_", " ") }), el("span", { text: p.is_fallback ? "estimator (fallback)" : "estimator" })]),
    el("div", { class: "metric" }, [el("strong", { text: fmt.pct(p.confidence) }), el("span", { text: "confidence" })]),
    el("div", { class: "metric" }, [el("strong", { text: p.correction.replaceAll("_", " ") }), el("span", { text: "color matrix" })]),
  ]);
  return [
    metrics,
    p.spectrum ? spectrumChart(p.spectrum, `Light spectrum fitted to the reference materials (${p.family}).`) : null,
    el("p", { class: "note", text: p.camera_curves ? `Camera curves: ${p.camera_curves} (${p.camera_curves_reason}).` : "No spectral curves for this camera, so the camera's own color matrix was used." }),
    ...p.notes.filter((n) => !n.startsWith("skipped")).map((n) => el("p", { class: "note", text: n })),
  ];
}

function truthPanel(c) {
  const gt = c.ground_truth;
  return [
    el("div", { class: "metric-row" }, SERIES.map((s) =>
      el("div", { class: "metric" }, [el("strong", { text: fmt.deg(gt[s.key].angular_error_deg) }), el("span", { text: s.label })]),
    )),
    el("p", { class: "note", text: "Angular error between each method's estimate of the light and the light measured on the chart's gray patches. The chart was hidden from every stage of the pipeline." }),
    el("p", { class: "note", text: `ΔE00 on the 24 patches: ${SERIES.map((s) => `${s.label} ${gt[s.key].delta_e00.toFixed(1)}`).join(", ")}. Patch color accuracy is limited by the camera's factory color matrix, which all three share.` }),
  ];
}

function caseCard(c) {
  const baselines = [["before", c.input_mode === "raster" ? "Original" : "Camera as-shot"], ["gray_world", "Gray world"]];
  const slider = compareSlider(c.images.ours, c.images.before, "This project", baselines[0][1], c.title);
  const switcher = el("div", { class: "baseline-switch", role: "group", "aria-label": "Compare against" }, ["Compare against:"]);
  baselines.forEach(([key, label], i) => {
    const button = el("button", { "aria-pressed": i === 0 ? "true" : "false", text: label });
    button.addEventListener("click", () => {
      switcher.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false"));
      button.setAttribute("aria-pressed", "true");
      slider.setBefore(c.images[key], label);
    });
    switcher.append(button);
  });
  const tags = el("div", { class: "case-tags" }, [
    c.camera ? el("span", { class: "chip", text: c.camera }) : null,
    el("span", { class: "chip", text: c.input_mode === "raw" ? "RAW" : "JPEG/PNG (approximate)" }),
    c.ground_truth ? el("span", { class: "chip good", text: `chart-verified · ${fmt.deg(c.ground_truth.ours.angular_error_deg)} error` }) : null,
    c.ai.source === "vlm" ? el("span", { class: "chip ai", text: "AI context" }) : null,
  ]);
  const tabs = [
    { id: `${c.id}-ai`, label: "AI reasoning", kind: "ai", content: () => aiPanel(c) },
    { id: `${c.id}-regions`, label: "Regions", kind: "ai", content: () => regionsPanel(c) },
    { id: `${c.id}-physics`, label: "Physics", kind: "physics", content: () => physicsPanel(c) },
  ];
  if (c.ground_truth) tabs.push({ id: `${c.id}-truth`, label: "Ground truth", content: () => truthPanel(c) });
  return el("article", { class: "case", id: c.id }, [
    el("div", { class: "case-head" }, [el("div", {}, [el("h3", { text: c.title }), c.caption ? el("p", { text: c.caption }) : null]), tags]),
    el("div", {}, [slider.root, switcher]),
    el("div", {}, tabSet(tabs)),
  ]);
}

// ---------- page ----------
function mean(values) { return values.reduce((a, b) => a + b, 0) / values.length; }

function renderHero(cases) {
  const charted = cases.filter((c) => c.ground_truth);
  const stats = document.getElementById("hero-stats");
  if (charted.length) {
    const avg = (key) => mean(charted.map((c) => c.ground_truth[key].angular_error_deg));
    stats.append(
      el("li", {}, [el("strong", { text: fmt.deg(avg("ours")) }), el("span", { text: `mean error, this project (${charted.length} chart scenes)` })]),
      el("li", {}, [el("strong", { text: fmt.deg(avg("gray_world")) }), el("span", { text: "gray world, the classic auto white balance" })]),
      el("li", {}, [el("strong", { text: String(cases.length) }), el("span", { text: "real photos, RAW and JPEG" })]),
    );
  }
  const featured = charted.length
    ? charted.reduce((best, c) => (c.ground_truth.before.angular_error_deg - c.ground_truth.ours.angular_error_deg > best.ground_truth.before.angular_error_deg - best.ground_truth.ours.angular_error_deg ? c : best))
    : cases[0];
  if (!featured) return;
  const slider = compareSlider(featured.images.ours, featured.images.before, "This project", "Camera as-shot", featured.title);
  document.getElementById("hero-figure").append(slider.root, el("figcaption", { text: `${featured.title}. Drag to compare; details in the gallery below.` }));
}

function renderAccuracy(cases) {
  const container = document.getElementById("accuracy-chart");
  const charted = cases.filter((c) => c.ground_truth);
  if (!charted.length) { document.getElementById("accuracy").hidden = true; return; }
  const table = accuracyTable(charted);
  table.hidden = true;
  const toggle = el("button", { class: "toggle-table", text: "Show as table" });
  toggle.addEventListener("click", () => { table.hidden = !table.hidden; toggle.textContent = table.hidden ? "Show as table" : "Hide table"; });
  container.append(
    el("div", { class: "chart-header" }, [
      el("div", { class: "legend" }, SERIES.map((s) => el("span", {}, [el("i", { style: `background:${s.color}` }), s.label]))),
      toggle,
    ]),
    el("div", { class: "chart" }, [accuracyChart(charted)]),
    table,
  );
  document.getElementById("accuracy-footnote").textContent =
    "Angular error between the estimated and the measured light color, in degrees. “Camera as-shot” is the white balance the photographer locked in (5200 K).";
}

async function main() {
  const repo = document.getElementById("repo-link");
  if (SITE.repoUrl) repo.href = SITE.repoUrl; else repo.remove();
  const response = await fetch("data/cases.json");
  const data = await response.json();
  const cases = data.cases;
  if (data.pipeline?.vlm) {
    document.querySelectorAll(".stage.ai .kind")[0].textContent = `AI · ${data.pipeline.vlm}`;
  }
  renderHero(cases);
  renderAccuracy(cases);
  document.getElementById("cases").append(...cases.map(caseCard));
  document.getElementById("footer-meta").textContent = `Built by ${SITE.author}. Every number on this page is generated by the code in the repository.`;
}

main().catch((error) => {
  document.getElementById("cases").append(el("p", { class: "note", text: `Could not load results: ${error.message}` }));
});
