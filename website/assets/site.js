/* Aerolume website behaviour. Everything here is progressive enhancement: with scripts off the
   page shows the system theme, a still hero background colour, every block in place and plain
   code blocks. The live flow behind the home page's hero is assets/hero/hero.js. */
(function () {
  "use strict";
  var root = document.documentElement;
  var dark = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

  function store(key, value) {
    try { window.localStorage.setItem(key, value); } catch (e) { /* storage blocked: session only */ }
  }
  function effectiveTheme() {
    var t = root.getAttribute("data-theme");
    if (t === "light" || t === "dark") return t;
    return dark && dark.matches ? "dark" : "light";
  }

  /* Theme switch: flips between light and dark and remembers the choice. The button shows an
     icon, so its accessible name says what a click does. */
  var toggle = document.querySelector("[data-theme-toggle]");
  function label() {
    if (!toggle) return;
    var next = effectiveTheme() === "dark" ? "light" : "dark";
    toggle.setAttribute("aria-label", "Switch to " + next + " theme");
    toggle.setAttribute("title", "Switch to " + next + " theme");
  }
  if (toggle) {
    toggle.hidden = false;
    toggle.addEventListener("click", function () {
      var next = effectiveTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      store("aerolume-theme", next);
      label();
    });
    if (dark && dark.addEventListener) dark.addEventListener("change", label);
    label();
  }

  /* Scroll reveals (home page). Blocks marked data-reveal settle into place as they come into
     view: a section head and its text as one block, a picture (data-reveal="wipe") uncovered in
     the direction its flow runs, the ledger as one table whose band gauges then draw row by row,
     the cavity plot whose line draws through the published values, and the tally's pass bar
     (site.css). No number is ever animated: a count-in shows wrong values on the way. Transform and opacity only, each sequence done within about 600 ms, so nothing
     reflows and reading never waits.
     The hiding itself is opt-in: CSS only hides a block once this script has put "motion" on
     the root element, which it does only with IntersectionObserver available and reduced motion off, so
     without scripts, with reduced motion, or in print every block is simply there.
     An observer only reports blocks whose visibility changes, so a jump (an anchor link, the
     End key) would leave the blocks it skipped hidden above the viewport; a cheap scroll check
     reveals those too, without animation. */
  var reduce = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
  // The ledger's gauges draw one row after another (25 ms apart, at most 150 ms in all).
  var gauged = document.querySelectorAll(".ledger tbody tr");
  for (var g = 0, gi = 0; g < gauged.length; g++) {
    if (gauged[g].querySelector(".gauge")) gauged[g].style.setProperty("--gi", String(Math.min(gi++, 6)));
  }
  var blocks = Array.prototype.slice.call(document.querySelectorAll("[data-reveal]"));

  function reveal(el, delay) {
    if (el.classList.contains("is-in")) return;
    el.style.setProperty("--reveal-delay", delay + "ms");
    el.classList.add("is-in");
  }

  if (blocks.length && "IntersectionObserver" in window && !(reduce && reduce.matches)) {
    root.classList.add("motion");
    var pending = blocks.length;
    var io = new IntersectionObserver(function (entries) {
      // Blocks that arrive together are staggered top to bottom, 60 ms apart (at most 180 ms).
      var hits = entries.filter(function (e) { return e.isIntersecting; })
        .sort(function (a, b) { return a.boundingClientRect.top - b.boundingClientRect.top; });
      for (var h = 0; h < hits.length; h++) {
        reveal(hits[h].target, Math.min(h * 60, 180));
        io.unobserve(hits[h].target);
        pending--;
      }
      if (pending <= 0) io.disconnect();
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0 });
    blocks.forEach(function (b) { io.observe(b); });

    var queued = false;
    var skipped = function () {
      queued = false;
      for (var s = 0; s < blocks.length; s++) {
        var b = blocks[s];
        if (!b.classList.contains("is-in") && b.getBoundingClientRect().bottom < 0) {
          b.classList.add("no-anim");
          reveal(b, -1);
          io.unobserve(b);
          pending--;
        }
      }
      if (pending <= 0) window.removeEventListener("scroll", onScroll);
    };
    var onScroll = function () { if (!queued) { queued = true; window.requestAnimationFrame(skipped); } };
    window.addEventListener("scroll", onScroll, { passive: true });
    // Printing shows everything, whatever was scrolled past.
    window.addEventListener("beforeprint", function () { blocks.forEach(function (b) { reveal(b, -1); }); });
    // A keyboard user can tab into a block that is scrolled into view but still inside the
    // observer's bottom margin: show it at once, so a focused link is never invisible.
    document.addEventListener("focusin", function (ev) {
      var b = ev.target && ev.target.closest ? ev.target.closest("[data-reveal]") : null;
      if (b && !b.classList.contains("is-in")) { b.classList.add("no-anim"); reveal(b, -1); }
    });
    if (reduce && reduce.addEventListener) {
      reduce.addEventListener("change", function () { if (reduce.matches) blocks.forEach(function (b) { reveal(b, -1); }); });
    }
  }

  /* Copy buttons on install commands. */
  var copies = document.querySelectorAll("[data-copy]");
  for (var c = 0; c < copies.length; c++) {
    copies[c].hidden = false;
    copies[c].addEventListener("click", function (ev) {
      var btn = ev.currentTarget;
      var pre = document.getElementById(btn.getAttribute("data-copy"));
      if (!pre) return;
      var text = pre.textContent.replace(/\n$/, "");
      var done = function () {
        btn.textContent = "Copied";
        setTimeout(function () { btn.textContent = "Copy"; }, 1600);
      };
      var fallback = function () {
        var range = document.createRange();
        range.selectNodeContents(pre);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        btn.textContent = "Selected";
        setTimeout(function () { btn.textContent = "Copy"; }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, fallback);
      } else {
        fallback();
      }
    });
  }
})();
