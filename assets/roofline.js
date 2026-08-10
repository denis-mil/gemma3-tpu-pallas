/* roofline.js — where a kernel sits between the two roofs of a TPU v5e.
 *
 * Classic script (no ES modules) so it loads over file://.
 *
 *   Roofline.mount('#roof', {
 *     title: 'Which roof binds',
 *     seqLen: 32768, window: 512, block: 512,
 *     controls: ['seqLen', 'window', 'block', 'bandwidth']
 *   });
 *
 * Requires mask-grid.js, pipeline-grid.js and skip-table.js first. As with
 * skip-table.js that is deliberate: this widget must not invent its own idea of
 * how much traffic a kernel moves.
 *
 *   - the live/partial/full predicate comes from MaskGrid.classify (lesson 01)
 *   - the shrunk tables come from SkipTable.shrink / .blockMask (lesson 05)
 *   - EVERY DMA count comes from PipelineGrid.trace (lesson 03's elision rule)
 *
 * so the bytes on the x-axis here are the same bytes lesson 05 counted, and a
 * roofline drawn from them cannot quietly disagree with the lesson that
 * produced them.
 *
 * ---- the model ------------------------------------------------------------
 *
 * One local attention layer, bf16 operands, splash_attention's grid order
 * (head, q_block, kv_block) with kv innermost. Q, K, V and O blocks are all
 * (block x head_dim), so one DMA moves the same number of bytes whichever
 * operand it belongs to.
 *
 *   FLOPs = 4 * pairs * head_dim,  pairs counted at BLOCK granularity, because
 *           the kernel computes whole blocks — reference/block-skipping-arithmetic.html
 *   bytes = (total copies issued) * block * head_dim * 2
 *
 * Model constants are Gemma 3 1B's, from src/gemma3_pallas/shapes.py.
 * Hardware constants are stated with their source in BANDWIDTHS below, because
 * the sources disagree by 4.9% and averaging them would hide that.
 *
 * ---- what the picture can and cannot say ----------------------------------
 *
 * The y-axis is a RATE, not a runtime. Two kernels that do wildly different
 * amounts of work land on the same spot if their arithmetic intensity matches.
 * That is why block skipping is invisible here and visible in the table below:
 * it changes the work, not the rate.
 */
(function (global) {
  'use strict';

  /* ---- constants ---------------------------------------------------- */

  var HEAD_DIM = 256;      // shapes.py Gemma3Config.head_dim
  var Q_HEADS = 4;         // shapes.py num_heads
  var KV_HEADS = 1;        // shapes.py num_kv_heads — MQA
  var BYTES = 2;           // bf16

  /* One roof, always. v5e has no fp32 matmul unit, so an fp32-precision matmul is
     emulated with 3 or 6 bf16 passes -- but that pass count belongs in `flops`, not
     as a divisor here (ADR-0004). BYTES = 2 above means every kernel on this page is
     bf16, one pass, so its `flops` values are already hardware counts and `computeS`
     and `ridge` below are already in the current convention. */
  var PEAK_BF16 = 197e12;  // cloud docs AND jax tpu_info agree on this one

  /* Three sources, three numbers. The reader can pick and watch the ridge move. */
  var BANDWIDTHS = [
    { label: 'jax tpu_info.py — 820e9 B/s', value: 820e9 },
    { label: 'shapes.py — 819e9 B/s', value: 819e9 },
    { label: "cloud docs '800 GiBps', literal", value: 800 * 1024 * 1024 * 1024 }
  ];

  var SEQ_LENS = [2048, 4096, 8192, 16384, 32768];
  var BLOCKS = [128, 256, 512, 1024];
  var WINDOWS = [64, 128, 256, 512, 1024, 2048];

  /* ---- costing ------------------------------------------------------- */

  function denseTables(nq) {
    var dataNext = [], blockMask = [];
    for (var i = 0; i < nq; i++) {
      var dn = [], bm = [];
      for (var j = 0; j < nq; j++) { dn.push(j); bm.push(1); }
      dataNext.push(dn); blockMask.push(bm);
    }
    return { width: nq, dataNext: dataNext, blockMask: blockMask };
  }

  /* Copies for a (head, q, kv) grid, derived from a ONE-HEAD trace.
   *
   * The head axis is outermost, so each head replays an identical step
   * sequence. An operand whose index_map mentions the head cannot elide across
   * the seam; one that ignores it (K and V, which MQA shares) elides exactly
   * when the head's last block index equals its first. Both facts are read off
   * PipelineGrid.trace's own `indices`, not re-derived from the rule. */
  function copiesAcrossHeads(row, heads) {
    var idx = row.indices;
    var seam = idx.length > 1 &&
               String(idx[idx.length - 1]) === String(idx[0]) &&
               row.kind !== 'out';
    return heads * row.copies - (seam ? heads - 1 : 0);
  }

  function prefill(seqLen, window_, block, skip) {
    var nq = Math.ceil(seqLen / block);
    var t = skip ? global.SkipTable.shrink(global.SkipTable.blockMask(nq, nq, block, window_))
                 : denseTables(nq);

    var tr = global.PipelineGrid.trace(
      [['i', nq], ['j', t.width]],
      [
        { name: 'Q', kind: 'in', map: function (ix) { return [ix.i]; } },
        { name: 'K', kind: 'in', map: function (ix) { return [t.dataNext[ix.i][ix.j]]; } },
        { name: 'V', kind: 'in', map: function (ix) { return [t.dataNext[ix.i][ix.j]]; } },
        { name: 'O', kind: 'out', map: function (ix) { return [ix.i]; } }
      ]
    );

    var copies = 0, per = {};
    tr.rows.forEach(function (r) {
      var c = r.name === 'Q' || r.name === 'O' ? Q_HEADS * r.copies
                                               : copiesAcrossHeads(r, Q_HEADS);
      per[r.name] = c;
      copies += c;
    });

    var computed = 0;
    for (var i = 0; i < nq; i++) {
      for (var j = 0; j < t.width; j++) { if (t.blockMask[i][j] > 0) { computed += 1; } }
    }
    computed *= Q_HEADS;

    var blockBytes = block * HEAD_DIM * BYTES;
    return {
      name: skip ? 'prefill · block-skipped' : 'prefill · dense grid',
      short: skip ? 'prefill' : 'dense',
      flops: 4 * computed * block * block * HEAD_DIM,
      bytes: copies * blockBytes,
      steps: Q_HEADS * nq * t.width,
      computed: computed,
      grid: '(' + Q_HEADS + ', ' + nq + ', ' + t.width + ')',
      dmas: per
    };
  }

  /* One decode step: one query row against the cached window. */
  function decode(seqLen, window_) {
    var span = Math.min(seqLen, window_);
    return {
      name: 'decode · one step',
      short: 'decode',
      flops: 4 * span * Q_HEADS * HEAD_DIM,
      bytes: 2 * KV_HEADS * span * HEAD_DIM * BYTES + 2 * Q_HEADS * HEAD_DIM * BYTES,
      steps: null,
      computed: null,
      grid: '—',
      dmas: null
    };
  }

  function evaluate(k, bw) {
    var ai = k.flops / k.bytes;
    var computeS = k.flops / PEAK_BF16;
    var memoryS = k.bytes / bw;
    k.ai = ai;
    k.attainable = Math.min(ai * bw, PEAK_BF16);
    k.binds = computeS >= memoryS ? 'compute' : 'memory';
    k.floorS = Math.max(computeS, memoryS);
    k.margin = k.binds === 'compute' ? ai / (PEAK_BF16 / bw) : (PEAK_BF16 / bw) / ai;
    return k;
  }

  /* ---- formatting ---------------------------------------------------- */

  function si(x, unit) {
    var steps = [[1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'k']];
    for (var i = 0; i < steps.length; i++) {
      if (Math.abs(x) >= steps[i][0]) {
        return (x / steps[i][0]).toFixed(x / steps[i][0] < 10 ? 2 : 1) + ' ' + steps[i][1] + unit;
      }
    }
    return x.toFixed(0) + ' ' + unit;
  }

  function bytesFmt(b) {
    var MiB = 1024 * 1024;
    if (b >= 1024 * MiB) { return (b / (1024 * MiB)).toFixed(2) + ' GiB'; }
    if (b >= MiB) { return (b / MiB).toFixed(b / MiB < 10 ? 2 : 0) + ' MiB'; }
    return (b / 1024).toFixed(0) + ' KiB';
  }

  function timeFmt(s) {
    if (s >= 1e-3) { return (s * 1e3).toFixed(2) + ' ms'; }
    if (s >= 1e-6) { return (s * 1e6).toFixed(1) + ' µs'; }
    return (s * 1e9).toFixed(0) + ' ns';
  }

  /* ---- the chart ----------------------------------------------------- */

  var W = 640, H = 392;
  var PAD = { l: 62, r: 18, t: 22, b: 46 };
  var X0 = 0, X1 = 4;     // log10 arithmetic intensity, 1 .. 10^4 FLOPs/byte
  var Y0 = 11, Y1 = 15;   // log10 attainable FLOP/s, 10^11 .. 10^15

  function sx(v) { return PAD.l + (v - X0) / (X1 - X0) * (W - PAD.l - PAD.r); }
  function sy(v) { return H - PAD.b - (v - Y0) / (Y1 - Y0) * (H - PAD.t - PAD.b); }
  function lg(v) { return Math.log(v) / Math.LN10; }

  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function chart(kernels, bw) {
    var ridge = PEAK_BF16 / bw;
    var lo = PEAK_BF16 / (800 * 1024 * 1024 * 1024);   // the three sources' spread
    var hi = PEAK_BF16 / 819e9;
    var yPeak = sy(lg(PEAK_BF16));
    var s = [];

    s.push('<svg class="rf-chart" viewBox="0 0 ' + W + ' ' + H + '" role="img" ' +
           'aria-label="Roofline: attainable rate against arithmetic intensity">');

    /* recessive grid, solid hairlines */
    var d;
    for (d = X0; d <= X1; d++) {
      s.push('<line class="rf-grid" x1="' + sx(d).toFixed(1) + '" y1="' + sy(Y0) +
             '" x2="' + sx(d).toFixed(1) + '" y2="' + sy(Y1) + '"/>');
    }
    for (d = Y0; d <= Y1; d++) {
      s.push('<line class="rf-grid" x1="' + sx(X0) + '" y1="' + sy(d).toFixed(1) +
             '" x2="' + sx(X1) + '" y2="' + sy(d).toFixed(1) + '"/>');
    }

    /* the band the three published bandwidths put the ridge in */
    s.push('<rect class="rf-band" x="' + sx(lg(lo)).toFixed(1) + '" y="' + sy(Y1) +
           '" width="' + (sx(lg(hi)) - sx(lg(lo))).toFixed(1) +
           '" height="' + (sy(Y0) - sy(Y1)).toFixed(1) + '"/>');

    /* both roofs full length, hairline — then the binding envelope over the top */
    s.push('<line class="rf-roof-off" x1="' + sx(X0) + '" y1="' + yPeak.toFixed(1) +
           '" x2="' + sx(X1) + '" y2="' + yPeak.toFixed(1) + '"/>');
    s.push('<line class="rf-roof-off" x1="' + sx(X0) + '" y1="' + sy(lg(bw) + X0).toFixed(1) +
           '" x2="' + sx(X1) + '" y2="' + sy(lg(bw) + X1).toFixed(1) + '"/>');

    var xr = sx(lg(ridge));
    s.push('<path class="rf-envelope" d="M' + sx(X0) + ' ' + sy(lg(bw) + X0).toFixed(1) +
           ' L' + xr.toFixed(1) + ' ' + yPeak.toFixed(1) +
           ' L' + sx(X1) + ' ' + yPeak.toFixed(1) + '"/>');

    s.push('<text class="rf-anno" x="' + (xr + 6).toFixed(1) + '" y="' + (sy(Y1) + 13) +
           '">ridge ' + ridge.toFixed(0) + ' FLOPs/byte</text>');
    s.push('<text class="rf-anno rf-end" x="' + (sx(X1) - 4) + '" y="' + (yPeak - 7).toFixed(1) +
           '">compute roof · 197 TFLOP/s</text>');
    s.push('<text class="rf-anno" x="' + (sx(X0) + 5) + '" y="' + (sy(lg(bw) + X0) - 7).toFixed(1) +
           '">memory roof · slope = bandwidth</text>');

    /* marks: droplines first so they sit under the dots */
    kernels.forEach(function (k) {
      var x = sx(lg(k.ai)), y = sy(lg(k.attainable));
      s.push('<line class="rf-drop" x1="' + x.toFixed(1) + '" y1="' + y.toFixed(1) +
             '" x2="' + x.toFixed(1) + '" y2="' + sy(Y0) + '"/>');
    });
    kernels.forEach(function (k, n) {
      var x = sx(lg(k.ai)), y = sy(lg(k.attainable));
      s.push('<g class="rf-mark" data-k="' + n + '" tabindex="0" role="button" ' +
             'aria-label="' + esc(k.name + ', ' + k.ai.toFixed(0) + ' FLOPs per byte, ' +
                                 k.binds + '-bound') + '">');
      s.push('<title>' + esc(k.name + ' — ' + k.ai.toFixed(1) + ' FLOPs/byte, ' +
                             k.binds + '-bound') + '</title>');
      s.push('<circle class="rf-hit" cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="14"/>');
      s.push('<circle class="rf-dot" cx="' + x.toFixed(1) + '" cy="' + y.toFixed(1) + '" r="5"/>');
      s.push('<text class="rf-label' + (x > W * 0.62 ? ' rf-end' : '') + '" x="' +
             (x + (x > W * 0.62 ? -11 : 11)).toFixed(1) + '" y="' + (y + 4).toFixed(1) + '">' +
             esc(k.short) + '</text>');
      s.push('</g>');
    });

    /* axes */
    s.push('<line class="rf-axis" x1="' + sx(X0) + '" y1="' + sy(Y0) + '" x2="' + sx(X1) +
           '" y2="' + sy(Y0) + '"/>');
    s.push('<line class="rf-axis" x1="' + sx(X0) + '" y1="' + sy(Y0) + '" x2="' + sx(X0) +
           '" y2="' + sy(Y1) + '"/>');

    var XT = ['1', '10', '100', '1k', '10k'];
    for (d = X0; d <= X1; d++) {
      s.push('<text class="rf-tick rf-mid" x="' + sx(d).toFixed(1) + '" y="' + (sy(Y0) + 15) +
             '">' + XT[d - X0] + '</text>');
    }
    var YT = ['0.1', '1', '10', '100', '1000'];
    for (d = Y0; d <= Y1; d++) {
      s.push('<text class="rf-tick rf-end" x="' + (sx(X0) - 8) + '" y="' + (sy(d) + 3.5).toFixed(1) +
             '">' + YT[d - Y0] + '</text>');
    }
    s.push('<text class="rf-axis-label rf-mid" x="' + ((sx(X0) + sx(X1)) / 2).toFixed(1) +
           '" y="' + (H - 8) + '">arithmetic intensity — FLOPs per byte of HBM traffic</text>');
    s.push('<text class="rf-axis-label rf-mid" transform="translate(14,' +
           ((sy(Y0) + sy(Y1)) / 2).toFixed(1) + ') rotate(-90)" x="0" y="0">' +
           'attainable TFLOP/s</text>');

    s.push('</svg>');
    return s.join('');
  }

  /* ---- controls ------------------------------------------------------ */

  function select(labelText, values, current, fmt, onChange) {
    var wrap = document.createElement('div');
    wrap.className = 'control';
    var id = 'rf-' + Math.random().toString(36).slice(2, 7);
    var lab = document.createElement('label');
    lab.textContent = labelText;
    lab.htmlFor = id;
    var sel = document.createElement('select');
    sel.id = id;
    values.forEach(function (v, i) {
      var o = document.createElement('option');
      o.value = String(i);
      o.textContent = fmt ? fmt(v) : String(v);
      if (v === current) { o.selected = true; }
      sel.appendChild(o);
    });
    sel.addEventListener('change', function () { onChange(values[Number(sel.value)]); });
    wrap.appendChild(lab);
    wrap.appendChild(sel);
    return wrap;
  }

  /* ---- mount --------------------------------------------------------- */

  function mount(target, spec) {
    var root = typeof target === 'string' ? document.querySelector(target) : target;
    if (!root) { return; }
    if (!global.SkipTable || !global.PipelineGrid || !global.MaskGrid) {
      root.textContent = 'roofline.js needs mask-grid.js, pipeline-grid.js and skip-table.js.';
      return;
    }

    var seqLen = spec.seqLen || 32768;
    var window_ = spec.window || 512;
    var block = spec.block || 512;
    var bw = BANDWIDTHS[0];
    var controls = spec.controls || [];

    root.classList.add('widget');
    root.innerHTML = '';

    if (spec.title) {
      var t = document.createElement('p');
      t.className = 'widget-title';
      t.textContent = spec.title;
      root.appendChild(t);
    }

    var bar = document.createElement('div');
    bar.className = 'controls';
    if (controls.indexOf('seqLen') !== -1) {
      bar.appendChild(select('sequence length N', SEQ_LENS, seqLen, null,
        function (v) { seqLen = v; draw(); }));
    }
    if (controls.indexOf('window') !== -1) {
      bar.appendChild(select('window W', WINDOWS, window_, null,
        function (v) { window_ = v; draw(); }));
    }
    if (controls.indexOf('block') !== -1) {
      bar.appendChild(select('block B (bq = bkv)', BLOCKS, block, null,
        function (v) { block = v; draw(); }));
    }
    if (controls.indexOf('bandwidth') !== -1) {
      bar.appendChild(select('which published bandwidth', BANDWIDTHS, bw,
        function (v) { return v.label; }, function (v) { bw = v; draw(); }));
    }
    if (bar.children.length) { root.appendChild(bar); }

    var plot = document.createElement('div');
    root.appendChild(plot);

    var legend = document.createElement('div');
    legend.className = 'legend';
    legend.innerHTML =
      '<span><i class="swatch" style="background:var(--accent)"></i>the roof that binds</span>' +
      '<span><i class="swatch" style="background:var(--dead)"></i>the roof you are not on</span>' +
      '<span><i class="swatch" style="background:var(--accent-soft)"></i>ridge, across all three published bandwidths</span>';
    root.appendChild(legend);

    var holder = document.createElement('div');
    holder.className = 'scroll-x';
    root.appendChild(holder);

    var readout = document.createElement('div');
    readout.className = 'readout';
    root.appendChild(readout);

    function describe(k) {
      return '<b>' + esc(k.name) + '</b> &nbsp; ' + k.ai.toFixed(1) + ' FLOPs/byte &rarr; <b>' +
        k.binds + '-bound</b> by ' + k.margin.toFixed(2) + '&times;<br>' +
        '<span style="color:var(--ink-muted)">' + si(k.flops, 'FLOP') + ' over ' +
        bytesFmt(k.bytes) + ' &nbsp;·&nbsp; floor ' + timeFmt(k.floorS) +
        (k.grid !== '—' ? ' &nbsp;·&nbsp; grid ' + k.grid : '') + '</span>';
    }

    function draw() {
      var dense = evaluate(prefill(seqLen, window_, block, false), bw.value);
      var skip = evaluate(prefill(seqLen, window_, block, true), bw.value);
      var dec = evaluate(decode(seqLen, window_), bw.value);
      var rows = [dense, skip, dec];
      var marks = [skip, dec];

      plot.innerHTML = chart(marks, bw.value);

      var html = '<table class="rf-tab"><thead><tr><th>kernel</th>' +
        '<th class="num">FLOPs</th><th class="num">HBM bytes</th>' +
        '<th class="num">FLOPs/byte</th><th>binds</th><th class="num">time floor</th>' +
        '</tr></thead><tbody>';
      rows.forEach(function (k, n) {
        html += '<tr' + (n === 1 ? ' class="highlight"' : '') + '><td>' + esc(k.name) + '</td>' +
          '<td class="num">' + si(k.flops, 'F') + '</td>' +
          '<td class="num">' + bytesFmt(k.bytes) + '</td>' +
          '<td class="num">' + k.ai.toFixed(1) + '</td>' +
          '<td>' + k.binds + '</td>' +
          '<td class="num">' + timeFmt(k.floorS) + '</td></tr>';
      });
      html += '</tbody></table>';
      holder.innerHTML = html;

      var base = describe(skip) + '<br>' +
        '<span style="color:var(--ink-muted)">block skipping cuts FLOPs ' +
        (dense.flops / skip.flops).toFixed(2) + '&times; and bytes ' +
        (dense.bytes / skip.bytes).toFixed(2) + '&times;, so intensity moves ' +
        (skip.ai / dense.ai).toFixed(3) + '&times; — the two prefill rows are the same ' +
        'point on the chart and ' + (dense.floorS / skip.floorS).toFixed(0) +
        '&times; apart in time</span>';
      readout.innerHTML = base;

      var nodes = plot.querySelectorAll ? plot.querySelectorAll('.rf-mark') : [];
      Array.prototype.forEach.call(nodes, function (g) {
        var k = marks[Number(g.getAttribute('data-k'))];
        function show() { readout.innerHTML = describe(k); }
        function hide() { readout.innerHTML = base; }
        g.addEventListener('mouseenter', show);
        g.addEventListener('focus', show);
        g.addEventListener('mouseleave', hide);
        g.addEventListener('blur', hide);
      });
    }

    draw();
  }

  global.Roofline = {
    mount: mount, prefill: prefill, decode: decode, evaluate: evaluate,
    PEAK_BF16: PEAK_BF16, BANDWIDTHS: BANDWIDTHS
  };
})(window);
