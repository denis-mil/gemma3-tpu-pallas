/* block-shape-check.js — apply the TPU block-shape rule by hand.
 *
 * Classic script (no ES modules) so it loads over file://.
 *
 *   BlockShapeCheck.mount('#chk', {
 *     array: [512, 256],
 *     block: [512, 256],
 *     dtype: 'bf16',
 *     presets: [{ label: 'Q tile', array: [...], block: [...], dtype: 'bf16' }]
 *   });
 *
 * This exists because nothing on a CPU will do it for you. Pallas interpret
 * mode accepts an illegal block shape and runs it — verified in both interpret
 * modes, session 6, recorded in docs/research/0001 §5.4. Block-shape legality
 * is a hardware-only check, so on a develop-on-CPU / validate-on-TPU workflow
 * it has to be applied at authoring time. That is what this widget is.
 *
 * The rule, from Grids and BlockSpecs:
 *
 *   "On TPU, only blocks with rank at least 1 are supported. Furthermore, the
 *    last two dimensions of your block shape must be equal to the respective
 *    dimension of the overall array, or be divisible by 8 and 128 respectively.
 *    For blocks of rank 1, the block dimension must be equal to the array
 *    dimension, or be a multiple of 1024, or be a power of 2 and at least
 *    128 * (32 / bitwidth(dtype))."
 *
 * AMBIGUITY, flagged rather than silently resolved. That sentence has two
 * readings and the docs do not disambiguate them:
 *
 *   per-axis  (blk[-2] == arr[-2] || blk[-2] % 8 == 0)
 *          && (blk[-1] == arr[-1] || blk[-1] % 128 == 0)
 *   paired    (blk[-2:] == arr[-2:])
 *          || (blk[-2] % 8 == 0 && blk[-1] % 128 == 0)
 *
 * They agree on almost everything. They disagree exactly when one trailing axis
 * is satisfied by equality and the other by divisibility — e.g. a (16, 7) array
 * with an (8, 7) block. This widget evaluates BOTH and says so when they split.
 * Do not report a shape as legal on the strength of the reading you preferred.
 */
(function (global) {
  'use strict';

  var SUBLANES = 8;    // tpu_info.py: num_sublanes, all TensorCores
  var LANES = 128;     // tpu_info.py: num_lanes,    all TensorCores

  var BITWIDTH = { f32: 32, bf16: 16, f16: 16, int8: 8 };

  function isPow2(n) { return n > 0 && (n & (n - 1)) === 0; }

  /* Rank-1 clause: equal to the array dim, or a multiple of 1024, or a power
   * of two and at least 128 * (32 / bitwidth). */
  function checkRank1(arr, blk, dtype) {
    var floor = LANES * (32 / BITWIDTH[dtype]);
    if (blk[0] === arr[0]) {
      return { ok: true, why: 'block equals the array dimension (' + arr[0] + ')' };
    }
    if (blk[0] % 1024 === 0) {
      return { ok: true, why: blk[0] + ' is a multiple of 1024' };
    }
    if (isPow2(blk[0]) && blk[0] >= floor) {
      return { ok: true, why: blk[0] + ' is a power of 2 and ≥ 128×(32/' +
                             BITWIDTH[dtype] + ') = ' + floor };
    }
    return {
      ok: false,
      why: blk[0] + ' is not the array dim (' + arr[0] + '), not a multiple of 1024, and not ' +
           'a power of 2 ≥ ' + floor + ' for ' + dtype
    };
  }

  /* One trailing axis under the per-axis reading. */
  function axisVerdict(arrDim, blkDim, divisor, name) {
    if (blkDim === arrDim) {
      return { ok: true, by: 'equal', why: name + ' block ' + blkDim +
               ' equals the array dimension' };
    }
    if (blkDim % divisor === 0) {
      return { ok: true, by: 'divides', why: name + ' block ' + blkDim +
               ' is divisible by ' + divisor };
    }
    return { ok: false, by: 'none', why: name + ' block ' + blkDim +
             ' is neither the array dimension (' + arrDim + ') nor divisible by ' + divisor };
  }

  function check(arr, blk, dtype) {
    if (!arr.length || !blk.length) {
      return { verdict: 'error', lines: ['Give both an array shape and a block shape.'] };
    }
    if (arr.length !== blk.length) {
      return { verdict: 'error', lines: [
        'Ranks differ: block has rank ' + blk.length + ', array has rank ' + arr.length +
        '. A BlockSpec block_shape must have one entry per array axis.'] };
    }
    if (blk.some(function (n) { return !(n > 0); })) {
      return { verdict: 'error', lines: ['Every block dimension must be a positive integer.'] };
    }
    if (blk.some(function (n, i) { return n > arr[i]; })) {
      return { verdict: 'error', lines: [
        'A block is larger than the array on some axis. Legal in principle (the tail is ' +
        'padded) but almost always a typo — check the shape you meant.'] };
    }

    if (arr.length === 1) {
      var r1 = checkRank1(arr, blk, dtype);
      return {
        verdict: r1.ok ? 'legal' : 'illegal',
        lines: ['Rank-1 block, so the rank-1 clause applies.', r1.why],
        divides: arr[0] % blk[0] === 0
      };
    }

    var n = arr.length;
    var minor = axisVerdict(arr[n - 1], blk[n - 1], LANES, 'minor (last)');
    var second = axisVerdict(arr[n - 2], blk[n - 2], SUBLANES, 'second-minor');

    var perAxis = minor.ok && second.ok;
    var paired = (blk[n - 1] === arr[n - 1] && blk[n - 2] === arr[n - 2]) ||
                 (blk[n - 1] % LANES === 0 && blk[n - 2] % SUBLANES === 0);

    var lines = [second.why, minor.why];
    if (n > 2) {
      lines.unshift('Leading axes (' + blk.slice(0, n - 2).join(', ') +
                    ') are unconstrained — the rule only governs the last two.');
    }

    var verdict;
    if (perAxis === paired) {
      verdict = perAxis ? 'legal' : 'illegal';
    } else {
      verdict = 'ambiguous';
      lines.push('The two readings of the rule disagree here: per-axis says ' +
                 (perAxis ? 'legal' : 'illegal') + ', paired says ' +
                 (paired ? 'legal' : 'illegal') + '. The docs do not disambiguate. ' +
                 'Treat as illegal until a chip says otherwise.');
    }

    var divides = arr.every(function (a, i) { return a % blk[i] === 0; });
    return { verdict: verdict, lines: lines, divides: divides };
  }

  /* Grid steps and wasted lanes if the block does not divide the array. */
  function tiling(arr, blk) {
    var steps = 1, padded = 1, real = 1;
    var per = arr.map(function (a, i) {
      var c = Math.ceil(a / blk[i]);
      steps *= c;
      padded *= c * blk[i];
      real *= a;
      return c;
    });
    return { counts: per, steps: steps, padded: padded, real: real,
             waste: padded === 0 ? 0 : 1 - real / padded };
  }

  function parseShape(s) {
    return s.split(/[,\s x×()]+/)
      .filter(function (t) { return t.length; })
      .map(Number)
      .filter(function (n) { return Number.isFinite(n); });
  }

  function mount(target, spec) {
    var root = typeof target === 'string' ? document.querySelector(target) : target;
    if (!root) { return; }

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

    function field(labelText, value) {
      var wrap = document.createElement('div');
      wrap.className = 'control';
      var lab = document.createElement('label');
      lab.textContent = labelText;
      var inp = document.createElement('input');
      inp.type = 'text';
      inp.size = 12;
      inp.value = value;
      inp.addEventListener('input', draw);
      wrap.appendChild(lab);
      wrap.appendChild(inp);
      bar.appendChild(wrap);
      return inp;
    }

    var arrIn = field('array shape', (spec.array || [512, 256]).join(', '));
    var blkIn = field('block_shape', (spec.block || [512, 256]).join(', '));

    var dWrap = document.createElement('div');
    dWrap.className = 'control';
    var dLab = document.createElement('label');
    dLab.textContent = 'dtype';
    var dSel = document.createElement('select');
    ['bf16', 'f32', 'int8'].forEach(function (d) {
      var o = document.createElement('option');
      o.value = d; o.textContent = d;
      if (d === (spec.dtype || 'bf16')) { o.selected = true; }
      dSel.appendChild(o);
    });
    dSel.addEventListener('change', draw);
    dWrap.appendChild(dLab);
    dWrap.appendChild(dSel);
    bar.appendChild(dWrap);
    root.appendChild(bar);

    if (spec.presets && spec.presets.length) {
      var pres = document.createElement('div');
      pres.className = 'preset-row';
      spec.presets.forEach(function (p) {
        var b = document.createElement('button');
        b.type = 'button';
        b.className = 'btn';
        b.textContent = p.label;
        b.addEventListener('click', function () {
          arrIn.value = p.array.join(', ');
          blkIn.value = p.block.join(', ');
          if (p.dtype) { dSel.value = p.dtype; }
          draw();
        });
        pres.appendChild(b);
      });
      root.appendChild(pres);
    }

    var out = document.createElement('div');
    out.className = 'readout';
    root.appendChild(out);

    function draw() {
      var arr = parseShape(arrIn.value);
      var blk = parseShape(blkIn.value);
      var res = check(arr, blk, dSel.value);

      var badge = { legal: 'LEGAL', illegal: 'ILLEGAL', ambiguous: 'AMBIGUOUS',
                    error: '—' }[res.verdict];
      var html = '<span class="verdict verdict-' + res.verdict + '">' + badge + '</span> ';
      if (res.verdict !== 'error') {
        html += '<b>' + blk.join('×') + '</b> block on a <b>' +
                arr.join('×') + '</b> array<br>';
      }
      html += '<span style="color:var(--ink-muted)">' + res.lines.join('<br>') + '</span>';

      if (res.verdict !== 'error' && arr.length === blk.length) {
        var t = tiling(arr, blk);
        html += '<br><br>grid would be (' + t.counts.join(', ') + ') = <b>' + t.steps +
                '</b> steps';
        if (!res.divides) {
          html += '<br><span style="color:var(--ink-muted)">Does not divide evenly. The ' +
                  'tail blocks are padded on input and discarded on output, and the grid ' +
                  'still runs every step — <b>' +
                  (t.waste * 100).toFixed(1) + '%</b> of the elements computed are ' +
                  'padding. Padding is not masking: the values are unspecified garbage, ' +
                  'so your kernel still needs its mask.</span>';
        } else {
          html += '<br><span style="color:var(--ink-muted)">Divides evenly — no ' +
                  'padded tail blocks.</span>';
        }
      }
      out.innerHTML = html;
    }

    draw();
  }

  global.BlockShapeCheck = { mount: mount, check: check, tiling: tiling,
                             parseShape: parseShape };
})(window);
