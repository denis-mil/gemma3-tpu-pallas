/* vmem-budget.js — add up what a kernel asks VMEM to hold at once.
 *
 * Classic script (no ES modules) so it loads over file://.
 *
 *   VmemBudget.mount('#budget', {
 *     params: [{ name: 'bq', label: 'query block', values: [128, 256, 512] },
 *              { name: 'bkv', label: 'kv block',   values: [128, 256, 512] }],
 *     dtypes: { out: ['bf16', 'f32'] },          // optional dtype pickers
 *     operands: [
 *       { name: 'Q', elements: function (p) { return p.bq * 256; },
 *         dtype: 'bf16', buffers: 2, note: 'pipelined input' }
 *     ],
 *     ceilings: [{ label: 'tpu_info.py — v5e', bytes: 128 * 1024 * 1024 }]
 *   });
 *
 * The rule it applies, from TPU Pipelining § Multiple Buffering and § TPU
 * Memory Spaces (see docs/research/0001 §4.2):
 *
 *     bytes(operand) = prod(block_shape) × itemsize × buffer_count
 *
 * with buffer_count = 2 by default for every pipelined input and output,
 * dropping to 1 for a whole-array trivial window; and scratch buffers counted
 * ONCE, because scratch "is persistent across kernel iterations".
 *
 * The widget separates operand terms (linear in the block size) from the score
 * tile (quadratic in it), because that split — not any single total — is the
 * thing that transfers to the next kernel.
 */
(function (global) {
  'use strict';

  var KiB = 1024, MiB = 1024 * 1024;
  var ITEMSIZE = { bf16: 2, f16: 2, f32: 4, int8: 1 };

  function fmt(bytes) {
    if (bytes >= MiB) { return (bytes / MiB).toFixed(bytes >= 10 * MiB ? 1 : 2) + ' MiB'; }
    return Math.round(bytes / KiB) + ' KiB';
  }

  function resolve(v, p) { return typeof v === 'function' ? v(p) : v; }

  /* Returns { rows, linear, quadratic, total }. Exported for headless checks. */
  function compute(spec, p) {
    var linear = 0, quadratic = 0;
    var rows = (spec.operands || []).map(function (op) {
      var elems = resolve(op.elements, p);
      var dtype = resolve(op.dtype, p);
      var buffers = resolve(op.buffers, p);
      var bytes = elems * ITEMSIZE[dtype] * buffers;
      if (op.quadratic) { quadratic += bytes; } else { linear += bytes; }
      return { name: op.name, elements: elems, dtype: dtype, buffers: buffers,
               bytes: bytes, note: resolve(op.note, p) || '',
               quadratic: !!op.quadratic, shape: resolve(op.shape, p) || '' };
    });
    return { rows: rows, linear: linear, quadratic: quadratic,
             total: linear + quadratic };
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

    var state = {};
    var bar = document.createElement('div');
    bar.className = 'controls';

    (spec.params || []).forEach(function (prm) {
      state[prm.name] = prm.value != null ? prm.value : prm.values[prm.values.length - 1];
      var wrap = document.createElement('div');
      wrap.className = 'control';
      var lab = document.createElement('label');
      lab.textContent = prm.label || prm.name;
      var sel = document.createElement('select');
      prm.values.forEach(function (v) {
        var o = document.createElement('option');
        o.value = String(v);
        o.textContent = prm.name + ' = ' + v;
        if (v === state[prm.name]) { o.selected = true; }
        sel.appendChild(o);
      });
      sel.addEventListener('change', function () {
        state[prm.name] = Number(sel.value);
        draw();
      });
      wrap.appendChild(lab);
      wrap.appendChild(sel);
      bar.appendChild(wrap);
    });

    Object.keys(spec.dtypes || {}).forEach(function (key) {
      var opts = spec.dtypes[key];
      state[key] = opts[0];
      var wrap = document.createElement('div');
      wrap.className = 'control';
      var lab = document.createElement('label');
      lab.textContent = key + ' dtype';
      var sel = document.createElement('select');
      opts.forEach(function (d) {
        var o = document.createElement('option');
        o.value = d; o.textContent = d;
        sel.appendChild(o);
      });
      sel.addEventListener('change', function () { state[key] = sel.value; draw(); });
      wrap.appendChild(lab);
      wrap.appendChild(sel);
      bar.appendChild(wrap);
    });

    var ceilings = spec.ceilings || [];
    var ceilIdx = 0;
    if (ceilings.length > 1) {
      var cWrap = document.createElement('div');
      cWrap.className = 'control';
      var cLab = document.createElement('label');
      cLab.textContent = 'VMEM ceiling — sources disagree';
      var cSel = document.createElement('select');
      ceilings.forEach(function (c, i) {
        var o = document.createElement('option');
        o.value = String(i);
        o.textContent = c.label;
        cSel.appendChild(o);
      });
      cSel.addEventListener('change', function () { ceilIdx = Number(cSel.value); draw(); });
      cWrap.appendChild(cLab);
      cWrap.appendChild(cSel);
      bar.appendChild(cWrap);
    }

    root.appendChild(bar);

    var holder = document.createElement('div');
    holder.className = 'scroll-x';
    root.appendChild(holder);

    var meter = document.createElement('div');
    root.appendChild(meter);

    var readout = document.createElement('div');
    readout.className = 'readout';
    root.appendChild(readout);

    function draw() {
      var r = compute(spec, state);

      var html = '<table class="pg budget"><tbody>';
      html += '<tr class="pg-step"><th>buffer</th><th>block</th><th>dtype</th>' +
              '<th>bufs</th><th>VMEM</th><th></th></tr>';
      r.rows.forEach(function (row) {
        html += '<tr class="pg-op' + (row.quadratic ? ' budget-quad' : '') + '">' +
                '<th>' + row.name + '</th>' +
                '<td>' + (row.shape || row.elements + ' el') + '</td>' +
                '<td>' + row.dtype + '</td>' +
                '<td>×' + row.buffers + '</td>' +
                '<td class="budget-bytes">' + fmt(row.bytes) + '</td>' +
                '<td class="budget-note">' + row.note + '</td></tr>';
      });
      html += '<tr class="pg-op budget-total"><th>total</th><td colspan="3"></td>' +
              '<td class="budget-bytes">' + fmt(r.total) + '</td><td></td></tr>';
      html += '</tbody></table>';
      holder.innerHTML = html;

      var cap = ceilings.length ? ceilings[ceilIdx].bytes : 0;
      if (cap) {
        var pct = Math.min(100, (r.total / cap) * 100);
        var linPct = Math.min(100, (r.linear / cap) * 100);
        meter.innerHTML =
          '<div class="budget-bar" title="' + fmt(r.total) + ' of ' + fmt(cap) + '">' +
          '<i class="budget-fill-lin" style="width:' + linPct.toFixed(2) + '%"></i>' +
          '<i class="budget-fill-quad" style="width:' + (pct - linPct).toFixed(2) + '%"></i>' +
          '</div>' +
          '<div class="legend">' +
          '<span><i class="swatch" style="background:var(--accent)"></i>operands + scratch — ' +
          'linear in the block size</span>' +
          '<span><i class="swatch" style="background:var(--dead)"></i>score tile — ' +
          'quadratic in it</span>' +
          '<span>' + (pct < 1 ? '<1' : pct.toFixed(1)) + '% of ' +
          ceilings[ceilIdx].label + '</span></div>';
      }

      var lines = [];
      lines.push('operands + scratch <b>' + fmt(r.linear) + '</b> · score tile <b>' +
                 fmt(r.quadratic) + '</b> · total <b>' + fmt(r.total) + '</b>');
      if (cap) {
        lines.push('against ' + ceilings[ceilIdx].label + ' (' + fmt(cap) + '): ' +
                   (r.total > cap
                     ? '<b>over budget</b> — Mosaic would refuse this at compile time'
                     : 'fits, with ' + fmt(cap - r.total) + ' to spare'));
      }
      if (spec.footnote) { lines.push('<span style="color:var(--ink-muted)">' +
                                      spec.footnote + '</span>'); }
      readout.innerHTML = lines.join('<br>');
    }

    draw();
  }

  global.VmemBudget = { mount: mount, compute: compute, fmt: fmt };
})(window);
