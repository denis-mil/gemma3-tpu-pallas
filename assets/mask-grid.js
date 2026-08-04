/* mask-grid.js — reusable tile-level attention mask visualiser.
 *
 * Classic script (no ES modules) so it loads over file://.
 *
 *   MaskGrid.mount('#grid', {
 *     seqLen: 2048, window: 512, block: 256,
 *     controls: ['seqLen', 'window', 'block']   // omit for a static figure
 *   });
 *
 * It draws the (query block × key block) grid and classifies every tile:
 *
 *   dead    — the mask forbids every (q,k) pair in it. Never load it.
 *   partial — some pairs allowed, some not. Load it, then mask elementwise.
 *   full    — every pair allowed. Load it, no mask needed.
 *
 * Conventions match reference.attention_mask() in src/gemma3_pallas/reference.py:
 *   causal   k <= q
 *   windowed k > q - W        (so a query sees W keys, itself included)
 *
 * A tile spans q in [qlo, qhi], k in [klo, khi]. The diagonal d = q - k then
 * ranges over the contiguous interval [qlo - khi, qhi - klo], so:
 *
 *   live iff  qhi >= klo  and  qlo - khi <= W - 1
 *   full iff  khi <= qlo  and  qhi - klo <= W - 1
 */
(function (global) {
  'use strict';

  var SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768];
  var BLOCKS = [128, 256, 512, 1024];

  function css(el, name, fallback) {
    var v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }

  function classify(i, j, bq, bk, W) {
    var qlo = i * bq, qhi = qlo + bq - 1;
    var klo = j * bk, khi = klo + bk - 1;
    if (qhi < klo) { return 0; }                 // entirely above the diagonal
    if (qlo - khi > W - 1) { return 0; }         // entirely behind the window
    if (khi <= qlo && qhi - klo <= W - 1) { return 2; }
    return 1;
  }

  function tally(N, W, bq, bk) {
    var rows = Math.ceil(N / bq), cols = Math.ceil(N / bk);
    var dead = 0, partial = 0, full = 0;
    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < cols; j++) {
        var c = classify(i, j, bq, bk, W);
        if (c === 0) { dead++; } else if (c === 1) { partial++; } else { full++; }
      }
    }
    return { rows: rows, cols: cols, dead: dead, partial: partial, full: full,
             total: rows * cols, live: partial + full };
  }

  function fmt(n) { return n.toLocaleString('en-US'); }

  function mount(target, spec) {
    var root = typeof target === 'string' ? document.querySelector(target) : target;
    if (!root) { return; }

    var state = {
      seqLen: spec.seqLen || 2048,
      window: spec.window === null ? Infinity : (spec.window || 512),
      block: spec.block || 256
    };
    var controls = spec.controls || [];

    root.classList.add('widget');
    root.innerHTML = '';

    if (spec.title) {
      var t = document.createElement('p');
      t.className = 'widget-title';
      t.textContent = spec.title;
      root.appendChild(t);
    }

    // ---- controls -------------------------------------------------------
    if (controls.length) {
      var bar = document.createElement('div');
      bar.className = 'controls';

      var defs = {
        seqLen: { label: 'seq_len', opts: SEQ_LENS.map(function (n) { return [n, fmt(n)]; }) },
        window: { label: 'layer', opts: [[512, 'local · W=512'], [Infinity, 'global · no window']] },
        block:  { label: 'block size', opts: BLOCKS.map(function (n) { return [n, String(n)]; }) }
      };

      controls.forEach(function (key) {
        var def = defs[key];
        if (!def) { return; }
        var wrap = document.createElement('div');
        wrap.className = 'control';
        var lab = document.createElement('label');
        lab.textContent = def.label;
        lab.htmlFor = key + '-' + Math.random().toString(36).slice(2, 7);
        var sel = document.createElement('select');
        sel.id = lab.htmlFor;
        def.opts.forEach(function (pair) {
          var o = document.createElement('option');
          o.value = String(pair[0]);
          o.textContent = pair[1];
          if (Number(pair[0]) === state[key]) { o.selected = true; }
          sel.appendChild(o);
        });
        sel.addEventListener('change', function () {
          state[key] = Number(sel.value);
          draw();
        });
        wrap.appendChild(lab);
        wrap.appendChild(sel);
        bar.appendChild(wrap);
      });
      root.appendChild(bar);
    }

    // ---- canvas ---------------------------------------------------------
    var canvas = document.createElement('canvas');
    canvas.style.width = '100%';
    canvas.style.maxWidth = '30rem';
    canvas.style.display = 'block';
    canvas.style.margin = '0 auto';
    root.appendChild(canvas);

    var legend = document.createElement('div');
    legend.className = 'legend';
    legend.innerHTML =
      '<span><i class="swatch" style="background:var(--accent)"></i>full — no mask needed</span>' +
      '<span><i class="swatch" style="background:var(--accent-soft)"></i>partial — mask elementwise</span>' +
      '<span><i class="swatch" style="background:var(--dead)"></i>dead — never load</span>';
    root.appendChild(legend);

    var readout = document.createElement('div');
    readout.className = 'readout';
    root.appendChild(readout);

    function draw() {
      var N = state.seqLen, W = state.window, B = state.block;
      var s = tally(N, W, B, B);

      var dpr = global.devicePixelRatio || 1;
      var cssW = canvas.clientWidth || 480;
      var cell = cssW / s.cols;
      var cssH = cell * s.rows;
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
      canvas.style.height = cssH + 'px';

      var ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);

      var colours = [
        css(root, '--dead', '#c9c6bd'),
        css(root, '--accent-soft', '#f0e4df'),
        css(root, '--accent', '#7a2e1d')
      ];
      var grid = css(root, '--rule', '#d8d5cc');
      var stroke = cell > 7;

      for (var i = 0; i < s.rows; i++) {
        for (var j = 0; j < s.cols; j++) {
          var c = classify(i, j, B, B, W);
          ctx.fillStyle = colours[c];
          ctx.fillRect(j * cell, i * cell, cell, cell);
          if (stroke) {
            ctx.strokeStyle = grid;
            ctx.lineWidth = 0.5;
            ctx.strokeRect(j * cell + 0.25, i * cell + 0.25, cell - 0.5, cell - 0.5);
          }
        }
      }

      var pct = (100 * s.live / s.total);
      var saving = s.total / s.live;
      readout.innerHTML =
        'grid <b>' + s.rows + ' × ' + s.cols + '</b> = ' + fmt(s.total) + ' tiles ' +
        '(queries down, keys across)<br>' +
        'live <b>' + fmt(s.live) + '</b> = ' + fmt(s.full) + ' full + ' + fmt(s.partial) + ' partial · ' +
        'dead ' + fmt(s.dead) + '<br>' +
        'a kernel that skips dead tiles loads <b>' + pct.toFixed(2) + '%</b> of the KV ' +
        'the naive baseline does — <b>' + saving.toFixed(1) + '×</b> less<br>' +
        '<span style="color:var(--ink-muted)">live tiles per query row: ' +
        (s.live / s.rows).toFixed(2) +
        (W === Infinity ? ' (grows with seq_len)' : ' → W/B + 1 = ' + (W / B + 1) + ' once the row clears the start') +
        '</span>';
    }

    draw();
    global.addEventListener('resize', draw);
    if (global.matchMedia) {
      var mq = global.matchMedia('(prefers-color-scheme: dark)');
      if (mq.addEventListener) { mq.addEventListener('change', draw); }
    }
  }

  global.MaskGrid = { mount: mount, tally: tally, classify: classify };
})(window);
