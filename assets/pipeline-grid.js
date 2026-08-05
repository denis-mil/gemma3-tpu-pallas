/* pipeline-grid.js — reusable Pallas grid / BlockSpec traffic visualiser.
 *
 * Classic script (no ES modules) so it loads over file://.
 *
 *   PipelineGrid.mount('#dma', {
 *     axes: [['i_q', 3], ['j_kv', 4]],          // outermost first
 *     operands: [
 *       { name: 'Q', kind: 'in',  map: function (ix) { return [ix.i_q, 0]; } },
 *       { name: 'K', kind: 'in',  map: function (ix) { return [ix.j_kv, 0]; } },
 *       { name: 'O', kind: 'out', map: function (ix) { return [ix.i_q, 0]; } }
 *     ],
 *     controls: ['order']                        // omit for a static figure
 *   });
 *
 * It walks the grid in lexicographic order with the LAST axis varying fastest,
 * and for every operand at every step decides whether a copy is issued.
 *
 * The rule, as JAX implements it in both the interpreter and emit_pipeline
 * (see docs/research/0001-pallas-grid-blockspec-index-map.md §3.4):
 *
 *   inputs  look BACKWARDS  — copy in  on the first step, or when the block
 *                             index differs from the previous step
 *   outputs look FORWARDS   — copy out on the last step, or when the block
 *                             index will differ on the next step
 *
 * That asymmetry is not decoration: it is what makes a revisited output block
 * behave as an accumulator, and it is why a reduction has to sit on the last
 * grid axis.
 *
 * `map` is written in terms of NAMED invocation indices, never positional
 * ones, so permuting the axis order leaves every index_map untouched. That is
 * the whole point of the 'order' control — the BlockSpecs do not change, only
 * the traversal does, and the traffic moves by 4x anyway.
 */
(function (global) {
  'use strict';

  function css(el, name, fallback) {
    var v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }

  function same(a, b) {
    if (!a || !b || a.length !== b.length) { return false; }
    for (var i = 0; i < a.length; i++) { if (a[i] !== b[i]) { return false; } }
    return true;
  }

  /* Lexicographic, so perms[0] is always the IDENTITY permutation and a widget
   * with no 'order' control renders the axes exactly as the caller wrote them. */
  function permutations(n) {
    var rest = [];
    for (var i = 0; i < n; i++) { rest.push(i); }
    var out = [];
    (function rec(cur, left) {
      if (!left.length) { out.push(cur); return; }
      for (var k = 0; k < left.length; k++) {
        rec(cur.concat([left[k]]), left.slice(0, k).concat(left.slice(k + 1)));
      }
    })([], rest);
    return out;
  }

  /* Walk the grid under one axis ordering and classify every cell.
   * Returns { steps: [{ix, coords}], rows: [{name, kind, cells, copies}] }.
   * cell is 'copy' (a DMA was issued) or 'held' (the buffer was reused). */
  function trace(axes, operands) {
    var sizes = axes.map(function (a) { return a[1]; });
    var total = sizes.reduce(function (a, b) { return a * b; }, 1);

    var steps = [];
    for (var s = 0; s < total; s++) {
      var rem = s, coords = new Array(sizes.length), ix = {};
      for (var d = sizes.length - 1; d >= 0; d--) {
        coords[d] = rem % sizes[d];
        rem = Math.floor(rem / sizes[d]);
      }
      for (var k = 0; k < axes.length; k++) { ix[axes[k][0]] = coords[k]; }
      steps.push({ coords: coords, ix: ix });
    }

    var rows = operands.map(function (op) {
      var idx = steps.map(function (st) { return op.map(st.ix); });
      var cells = idx.map(function (cur, i) {
        if (op.kind === 'out') {
          return (i === total - 1 || !same(cur, idx[i + 1])) ? 'copy' : 'held';
        }
        return (i === 0 || !same(cur, idx[i - 1])) ? 'copy' : 'held';
      });
      var copies = cells.filter(function (c) { return c === 'copy'; }).length;
      return { name: op.name, kind: op.kind || 'in', note: op.note || '',
               cells: cells, copies: copies, indices: idx };
    });

    return { steps: steps, rows: rows, total: total };
  }

  function mount(target, spec) {
    var root = typeof target === 'string' ? document.querySelector(target) : target;
    if (!root) { return; }

    var axes = spec.axes;
    var operands = spec.operands;
    var controls = spec.controls || [];
    var perms = permutations(axes.length);
    var order = perms[0];

    root.classList.add('widget');
    root.innerHTML = '';

    if (spec.title) {
      var t = document.createElement('p');
      t.className = 'widget-title';
      t.textContent = spec.title;
      root.appendChild(t);
    }

    if (controls.indexOf('order') !== -1 && perms.length > 1) {
      var bar = document.createElement('div');
      bar.className = 'controls';
      var wrap = document.createElement('div');
      wrap.className = 'control';
      var id = 'pg-order-' + Math.random().toString(36).slice(2, 7);
      var lab = document.createElement('label');
      lab.textContent = 'grid axis order (outermost first)';
      lab.htmlFor = id;
      var sel = document.createElement('select');
      sel.id = id;
      perms.forEach(function (p, i) {
        var o = document.createElement('option');
        o.value = String(i);
        o.textContent = 'grid = (' + p.map(function (d) {
          return axes[d][1];
        }).join(', ') + ')  ' + p.map(function (d) { return axes[d][0]; }).join(', ') +
          '   — ' + axes[p[p.length - 1]][0] + ' fastest';
        sel.appendChild(o);
      });
      sel.addEventListener('change', function () {
        order = perms[Number(sel.value)];
        draw();
      });
      wrap.appendChild(lab);
      wrap.appendChild(sel);
      bar.appendChild(wrap);
      root.appendChild(bar);
    }

    var holder = document.createElement('div');
    holder.className = 'scroll-x';
    root.appendChild(holder);

    var legend = document.createElement('div');
    legend.className = 'legend';
    legend.innerHTML =
      '<span><i class="swatch" style="background:var(--accent)"></i>' +
      '<b>D</b> — copy issued, HBM ↔ VMEM</span>' +
      '<span><i class="swatch" style="background:var(--dead)"></i>' +
      '· — same block as the neighbouring step, copy skipped</span>';
    root.appendChild(legend);

    var readout = document.createElement('div');
    readout.className = 'readout';
    root.appendChild(readout);

    function draw() {
      var ordered = order.map(function (d) { return axes[d]; });
      var tr = trace(ordered, operands);

      var html = '<table class="pg"><tbody>';

      html += '<tr class="pg-step"><th>step</th>';
      for (var s = 0; s < tr.total; s++) { html += '<td>' + (s % 10) + '</td>'; }
      html += '<th class="pg-count"></th></tr>';

      ordered.forEach(function (ax, d) {
        html += '<tr class="pg-axis' + (d === ordered.length - 1 ? ' pg-fast' : '') +
                '"><th>' + ax[0] + '</th>';
        tr.steps.forEach(function (st) { html += '<td>' + st.coords[d] + '</td>'; });
        html += '<th class="pg-count">' + (d === ordered.length - 1 ? 'fastest' : '') +
                '</th></tr>';
      });

      tr.rows.forEach(function (r) {
        html += '<tr class="pg-op"><th>' + r.name +
                (r.kind === 'out' ? ' <span class="pg-kind">out</span>' : '') + '</th>';
        r.cells.forEach(function (c) {
          html += '<td class="pg-' + c + '">' + (c === 'copy' ? 'D' : '·') + '</td>';
        });
        html += '<th class="pg-count">' + r.copies + '</th></tr>';
      });

      html += '</tbody></table>';
      holder.innerHTML = html;

      var lines = tr.rows.map(function (r) {
        var per = tr.total / r.copies;
        return '<b>' + r.name + '</b> ' + r.copies + ' ' +
               (r.kind === 'out' ? 'write-backs' : 'DMAs') +
               (per >= 2 ? ' — one per ' + (Number.isInteger(per) ? per : per.toFixed(1)) +
                           ' steps, resident in between'
                         : ' — refetched every step');
      });
      var fastest = ordered[ordered.length - 1][0];
      readout.innerHTML =
        tr.total + ' grid steps, ' + fastest + ' varying fastest<br>' +
        lines.join('<br>') +
        '<br><span style="color:var(--ink-muted)">an operand is resident exactly when its ' +
        'index_map ignores <b>' + fastest + '</b></span>';
    }

    draw();
  }

  global.PipelineGrid = { mount: mount, trace: trace };
})(window);
