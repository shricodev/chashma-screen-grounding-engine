(opts) => {
  const MIN_SIZE = opts.minSize ?? 6;
  const MIN_VISIBLE = opts.minVisibleFraction ?? 0.6;
  const MAX_ELEMENTS = opts.maxElements ?? 250;
  const MAX_TEXT = 120;

  const SELECTOR = [
    "a[href]", "button", "input:not([type=hidden])", "select", "textarea", "summary",
    "[role=button]", "[role=link]", "[role=tab]", "[role=menuitem]", "[role=checkbox]",
    "[role=radio]", "[role=switch]", "[role=option]", "[role=combobox]", "[role=searchbox]",
    "[role=textbox]", "[contenteditable=''], [contenteditable=true]", "[onclick]",
    "[tabindex]:not([tabindex='-1'])",
  ].join(",");

  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim().slice(0, MAX_TEXT);

  const implicitRole = (el) => {
    const tag = el.tagName.toLowerCase();
    if (tag === "a") return "link";
    if (tag === "button" || tag === "summary") return "button";
    if (tag === "select") return "combobox";
    if (tag === "textarea") return "textbox";
    if (tag === "input") {
      const t = (el.getAttribute("type") || "text").toLowerCase();
      if (["button", "submit", "reset", "image"].includes(t)) return "button";
      if (t === "checkbox") return "checkbox";
      if (t === "radio") return "radio";
      if (t === "search") return "searchbox";
      if (t === "range") return "slider";
      return "textbox";
    }
    if (el.isContentEditable) return "textbox";
    return "generic";
  };

  const accessibleName = (el) => {
    const labelledBy = el.getAttribute("aria-labelledby");
    if (labelledBy) {
      const text = labelledBy.split(/\s+/)
        .map((id) => document.getElementById(id)?.innerText || "")
        .join(" ");
      if (clean(text)) return clean(text);
    }
    const aria = el.getAttribute("aria-label");
    if (clean(aria)) return clean(aria);
    if (el.labels && el.labels.length) {
      const text = Array.from(el.labels).map((l) => l.innerText).join(" ");
      if (clean(text)) return clean(text);
    }
    const tag = el.tagName.toLowerCase();
    if (tag === "input") {
      const t = (el.getAttribute("type") || "").toLowerCase();
      if (["button", "submit", "reset"].includes(t) && clean(el.value)) return clean(el.value);
      if (clean(el.getAttribute("placeholder"))) return clean(el.getAttribute("placeholder"));
    }
    const text = clean(el.innerText);
    if (text) return text;
    const img = el.querySelector("img[alt]");
    if (img && clean(img.getAttribute("alt"))) return clean(img.getAttribute("alt"));
    const svgTitle = el.querySelector("svg title");
    if (svgTitle && clean(svgTitle.textContent)) return clean(svgTitle.textContent);
    return clean(el.getAttribute("title"));
  };

  const isRendered = (el) => {
    if (el.closest("[aria-hidden=true], [inert]")) return false;
    for (let node = el; node && node.nodeType === 1; node = node.parentElement) {
      const s = getComputedStyle(node);
      if (s.display === "none" || s.visibility === "hidden" || s.visibility === "collapse") {
        return false;
      }
      if (parseFloat(s.opacity) < 0.1) return false;
    }
    return true;
  };

  const isOnTop = (el, r) => {
    const probes = [
      [0.5, 0.5], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
    ];
    let hits = 0;
    let centerHit = false;
    probes.forEach(([fx, fy], i) => {
      const x = Math.min(vw - 1, Math.max(0, r.left + r.width * fx));
      const y = Math.min(vh - 1, Math.max(0, r.top + r.height * fy));
      const top = document.elementFromPoint(x, y);
      const hit = !!top && (top === el || el.contains(top));
      if (hit) hits++;
      if (hit && i === 0) centerHit = true;
    });
    return centerHit && hits >= 3;
  };

  const hasVisual = (el) =>
    !!el.querySelector("svg, img, i, [class*=icon]") ||
    getComputedStyle(el).backgroundImage !== "none";

  const out = [];
  for (const el of document.querySelectorAll(SELECTOR)) {
    if (out.length >= MAX_ELEMENTS) break;
    if (el.disabled) continue;
    const r = el.getBoundingClientRect();
    if (r.width < MIN_SIZE || r.height < MIN_SIZE) continue;

    const vx1 = Math.max(0, r.left), vy1 = Math.max(0, r.top);
    const vx2 = Math.min(vw, r.right), vy2 = Math.min(vh, r.bottom);
    if (vx2 <= vx1 || vy2 <= vy1) continue;
    if (((vx2 - vx1) * (vy2 - vy1)) / (r.width * r.height) < MIN_VISIBLE) continue;
    if (r.width * r.height > 0.5 * vw * vh) continue;

    if (!isRendered(el) || !isOnTop(el, r)) continue;

    const text = clean(el.innerText);
    const role = el.getAttribute("role") || implicitRole(el);
    out.push({
      tag: el.tagName.toLowerCase(),
      role,
      name: accessibleName(el),
      text,
      type: el.getAttribute("type"),
      href: el.tagName.toLowerCase() === "a" ? el.href.slice(0, 300) : null,
      placeholder: el.getAttribute("placeholder"),
      is_icon: !text && hasVisual(el),
      rect: [vx1, vy1, vx2 - vx1, vy2 - vy1],
    });
  }

  return {
    elements: out,
    viewport: [vw, vh],
    title: document.title,
    lang: document.documentElement.lang || null,
    scroll_y: window.scrollY,
    page_height: document.documentElement.scrollHeight,
  };
}
