/* softmax-stream.js — reusable single-pass (online) softmax stepper.
 *
 * Classic script (no ES modules) so it loads over file://.
 *
 *   SoftmaxStream.mount('#sim', {
 *     title: 'One pass, one tile at a time',
 *     preset: 'gentle',              // key into PRESETS, or pass scores/tile
 *     scores: [1, 3, 5, 2], tile: 2,
 *     mode: 'running-max',           // 'running-max' | 'none' | 'first-tile'
 *     precision: 'float32',          // 'float32' | 'float64'
 *     controls: ['preset', 'tile', 'mode', 'precision']
 *   });
 *
 * It walks the recurrence of Milakov & Gimelshein (arXiv:1805.02867, Alg. 3),
 * which is the same one FlashAttention runs per key-block (arXiv:2205.14135 §3.1):
 *
 *   m_j = max(m_{j-1}, rowmax(S_j))
 *   a_j = exp(m_{j-1} - m_j)                        <- the rescale, always <= 1
 *   l_j = a_j * l_{j-1} + sum(exp(S_j - m_j))
 *
 * with m_0 = -inf, l_0 = 0. The invariant that makes it checkable at every step:
 *
 *   l_j == sum over ALL scores seen so far of exp(s - m_j)
 *
 * i.e. the accumulator is always a correct normaliser for the data seen so far,
 * re-based to the current max. That is why one pass suffices.
 *
 * Modes exist to show what the max is actually buying:
 *   running-max  the real algorithm
 *   none         subtract nothing (m := 0). Still algebraically exact; overflows.
 *   first-tile   freeze m at tile 0's max. Also exact; underflows/overflows badly.
 * The last two are the point: the RECURRENCE is exact for any m sequence. Only
 * the running max keeps every exponent <= 0, hence every term representable.
 *
 * float32 is simulated with Math.fround after every operation. JS has no true
 * f32 arithmetic, so intermediates are f64-rounded-to-f32 rather than genuinely
 * f32 — close enough to show overflow honestly, not close enough to trust the
 * last ulp. Verified against numpy float32 for the overflow threshold only.
 */
(function (global) {
  'use strict';

  var PRESETS = {
    gentle: {
      label: 'gentle · max arrives late',
      scores: [1, 3, 5, 2],
      note: 'The worked example. Small enough to check by hand.'
    },
    spike: {
      label: 'spike · one huge score',
      scores: [1, 3, 2, 4, 90, 3, 2, 1],
      note: 'A score of 90. exp(90) overflows float32 — ln(3.40e38) = 88.72.'
    },
    rising: {
      label: 'rising · max moves every tile',
      scores: [1, 2, 4, 6, 9, 12, 16, 20],
      note: 'The max increases at every tile, so every step pays a rescale.'
    },
    settled: {
      label: 'settled · max arrives first',
      scores: [20, 16, 12, 9, 6, 4, 2, 1],
      note: 'Same numbers reversed. m never moves after tile 0, so every alpha is 1.'
    }
  };

  var MODES = {
    'running-max': 'running max (the algorithm)',
    'none': 'subtract nothing (m = 0)',
    'first-tile': 'freeze m at tile 0'
  };

  function ident(x) { return x; }

  function fmt(x, digits) {
    if (!isFinite(x)) { return x > 0 ? '∞' : (x < 0 ? '−∞' : 'NaN'); }
    if (Number.isNaN(x)) { return 'NaN'; }
    if (x !== 0 && (Math.abs(x) >= 1e6 || Math.abs(x) < 1e-4)) {
      return x.toExponential(2);
    }
    return x.toFixed(digits === undefined ? 6 : digits);
  }

  /* Walk the recurrence. Returns one record per tile plus the final state. */
  function run(scores, tile, mode, f32) {
    var r = f32 ? Math.fround : ident;
    var steps = [];
    var m = -Infinity;
    var l = 0;
    var seen = [];

    for (var s = 0; s < scores.length; s += tile) {
      var t = scores.slice(s, s + tile);
      var tmax = Math.max.apply(null, t);
      var mOld = m;
      var lOld = l;
      var mNew, alpha;

      if (mode === 'none') {
        mNew = 0;
        alpha = 1;
      } else if (mode === 'first-tile') {
        mNew = (mOld === -Infinity) ? tmax : mOld;
        alpha = 1;
      } else {
        mNew = Math.max(mOld, tmax);
        alpha = (mOld === -Infinity) ? 0 : r(Math.exp(r(mOld - mNew)));
      }

      var sum = 0;
      for (var i = 0; i < t.length; i++) {
        sum = r(sum + r(Math.exp(r(t[i] - mNew))));
      }
      l = r(r(alpha * lOld) + sum);
      seen = seen.concat(t);

      /* The invariant, recomputed from scratch over everything seen so far. */
      var direct = 0;
      for (var k = 0; k < seen.length; k++) {
        direct = r(direct + r(Math.exp(r(seen[k] - mNew))));
      }
      var holds = (direct === l) ||
                  (isFinite(direct) && isFinite(l) &&
                   Math.abs(direct - l) <= 1e-6 * Math.max(1, Math.abs(direct)));

      steps.push({
        index: steps.length, tile: t, tmax: tmax, mOld: mOld, mNew: mNew,
        alpha: alpha, sum: sum, l: l, direct: direct, holds: holds,
        exact: direct === l
      });
      m = mNew;
    }

    /* Final probabilities, and the float64 safe-softmax truth to score against. */
    var probs = scores.map(function (x) { return r(Math.exp(r(x - m))) / l; });
    var gmax = Math.max.apply(null, scores);
    var num = scores.map(function (x) { return Math.exp(x - gmax); });
    var den = num.reduce(function (a, b) { return a + b; }, 0);
    var truth = num.map(function (x) { return x / den; });

    var err = 0;
    var broken = false;
    for (var j = 0; j < probs.length; j++) {
      if (!isFinite(probs[j])) { broken = true; }
      else { err = Math.max(err, Math.abs(probs[j] - truth[j])); }
    }

    return { steps: steps, m: m, l: l, probs: probs, truth: truth,
             err: err, broken: broken || !isFinite(l) };
  }

  function mount(target, spec) {
    var root = typeof target === 'string' ? document.querySelector(target) : target;
    if (!root) { return; }

    var presetKey = spec.preset || 'gentle';
    var state = {
      preset: presetKey,
      scores: spec.scores || PRESETS[presetKey].scores,
      tile: spec.tile || 2,
      mode: spec.mode || 'running-max',
      precision: spec.precision || 'float32'
    };
    var shown = 0;

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
    root.appendChild(bar);

    var strip = document.createElement('div');
    strip.className = 'stream-strip';
    root.appendChild(strip);

    var buttons = document.createElement('div');
    buttons.className = 'stream-buttons';
    root.appendChild(buttons);

    var tableWrap = document.createElement('div');
    tableWrap.className = 'scroll-x';
    root.appendChild(tableWrap);

    var readout = document.createElement('div');
    readout.className = 'readout';
    root.appendChild(readout);

    /* ---- controls ------------------------------------------------------- */
    var defs = {
      preset: {
        label: 'scores',
        opts: Object.keys(PRESETS).map(function (k) { return [k, PRESETS[k].label]; })
      },
      tile: { label: 'tile size', opts: [[1, '1'], [2, '2'], [4, '4']] },
      mode: {
        label: 'subtract',
        opts: Object.keys(MODES).map(function (k) { return [k, MODES[k]]; })
      },
      precision: { label: 'precision', opts: [['float32', 'float32'], ['float64', 'float64']] }
    };

    (spec.controls || []).forEach(function (key) {
      var def = defs[key];
      if (!def) { return; }
      var wrap = document.createElement('div');
      wrap.className = 'control';
      var lab = document.createElement('label');
      lab.textContent = def.label;
      lab.htmlFor = 'ss-' + key + '-' + Math.random().toString(36).slice(2, 7);
      var sel = document.createElement('select');
      sel.id = lab.htmlFor;
      def.opts.forEach(function (pair) {
        var o = document.createElement('option');
        o.value = String(pair[0]);
        o.textContent = pair[1];
        if (String(pair[0]) === String(state[key])) { o.selected = true; }
        sel.appendChild(o);
      });
      sel.addEventListener('change', function () {
        var v = sel.value;
        state[key] = (key === 'tile') ? Number(v) : v;
        if (key === 'preset') { state.scores = PRESETS[v].scores; }
        shown = 0;
        draw();
      });
      wrap.appendChild(lab);
      wrap.appendChild(sel);
      bar.appendChild(wrap);
    });

    /* ---- buttons -------------------------------------------------------- */
    function mkBtn(text, fn) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = 'stream-btn';
      b.textContent = text;
      b.addEventListener('click', fn);
      buttons.appendChild(b);
      return b;
    }
    var stepBtn = mkBtn('Step ▸', function () { shown++; draw(); });
    mkBtn('All ▸▸', function () { shown = 1e9; draw(); });
    mkBtn('Reset', function () { shown = 0; draw(); });

    /* ---- draw ----------------------------------------------------------- */
    function draw() {
      var f32 = state.precision === 'float32';
      var res = run(state.scores, state.tile, state.mode, f32);
      var total = res.steps.length;
      if (shown > total) { shown = total; }
      stepBtn.disabled = shown >= total;

      /* score strip, grouped into tiles */
      strip.innerHTML = '';
      state.scores.forEach(function (x, i) {
        var tileIndex = Math.floor(i / state.tile);
        var cell = document.createElement('span');
        cell.className = 'stream-cell';
        if (tileIndex < shown) { cell.classList.add('done'); }
        if (tileIndex === shown - 1) { cell.classList.add('current'); }
        if (i % state.tile === 0 && i > 0) { cell.classList.add('tile-start'); }
        cell.textContent = x;
        strip.appendChild(cell);
      });

      /* state table */
      var rows = res.steps.slice(0, shown);
      if (!rows.length) {
        tableWrap.innerHTML =
          '<p class="stream-empty">Nothing consumed yet. ' +
          'm = −∞, ℓ = 0. Press <b>Step</b>.</p>';
      } else {
        var html = '<table><tr>' +
          '<th class="num">tile</th><th class="num">rowmax</th>' +
          '<th class="num">m</th><th class="num">α</th>' +
          '<th class="num">ℓ</th><th>invariant</th></tr>';
        rows.forEach(function (s) {
          html += '<tr' + (s.index === shown - 1 ? ' class="highlight"' : '') + '>' +
            '<td class="num">' + s.index + '</td>' +
            '<td class="num">' + fmt(s.tmax, 0) + '</td>' +
            '<td class="num">' + fmt(s.mNew, 0) + '</td>' +
            '<td class="num">' + (s.index === 0 ? '—' : fmt(s.alpha)) + '</td>' +
            '<td class="num">' + fmt(s.l) + '</td>' +
            '<td>' + (s.holds
              ? (s.exact ? 'holds · to the bit' : 'holds · to ~1 ulp')
              : '<b>broken</b>') + '</td>' +
            '</tr>';
        });
        tableWrap.innerHTML = html + '</table>';
      }

      /* readout */
      var note = PRESETS[state.preset] ? PRESETS[state.preset].note : '';
      var out = '<span style="color:var(--ink-muted)">' + note + '</span><br>';

      if (shown < total) {
        out += (total - shown) + ' of ' + total + ' tiles still unread — ' +
               'ℓ is already a valid normaliser for the ' + (shown * state.tile) +
               ' scores seen so far.';
      } else if (res.broken) {
        out += 'Final ℓ = <b>' + fmt(res.l) + '</b>. The softmax is <b>destroyed</b> — ' +
               'the exponent left the range of ' + state.precision + '. ' +
               'Note the recurrence itself never made an error.';
      } else {
        out += 'Final m = <b>' + fmt(res.m, 0) + '</b>, ℓ = <b>' + fmt(res.l) + '</b><br>' +
               'largest disagreement with a two-pass safe softmax: <b>' +
               (res.err === 0 ? '0 — bit-exact' : res.err.toExponential(1)) + '</b>';
      }
      readout.innerHTML = out;
    }

    draw();
  }

  global.SoftmaxStream = { mount: mount, run: run, PRESETS: PRESETS };
})(window);
