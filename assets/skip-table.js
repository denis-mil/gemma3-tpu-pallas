/* skip-table.js — the prefetched SMEM tables that drive a shrunk grid.
 *
 * Classic script (no ES modules) so it loads over file://.
 *
 *   SkipTable.mount('#skip', {
 *     seqLen: 2048, window: 512, block: 512,
 *     controls: ['seqLen', 'window', 'block']   // omit for a static figure
 *   });
 *
 * Requires mask-grid.js and pipeline-grid.js to be loaded first. That is
 * deliberate rather than incidental:
 *
 *   - the live/partial/full predicate comes from MaskGrid.classify, so this
 *     widget and lesson 01's picture cannot drift apart;
 *   - the copy-elision rule comes from PipelineGrid.trace, so the DMA counts
 *     here obey exactly the rule lesson 03 taught, and there is one
 *     implementation of it in the workspace, not two.
 *
 * What it builds, in splash_attention's own vocabulary
 * (jax/experimental/pallas/ops/tpu/splash_attention/splash_attention_mask_info.py):
 *
 *   block_mask[i][j]  0 dead · 1 partial · 2 full   — MaskInfo's docstring
 *                     ("0 ... all zeros, 1 ... both zeros and ones, 2 ...
 *                     entirely ones"), which is lesson 01's trichotomy.
 *   data_next[i][j]   the kv block index this step should actually fetch.
 *                     Unshrunk: the next live column at or after j, else the
 *                     last live column at or before j. Shrunk: row i's live
 *                     columns, left-packed and padded to the widest row.
 *
 * Both constructions were checked against process_mask(..., shrink_grid=)
 * for N=2048, W=512, B=512 and reproduce its tables exactly.
 *
 * Three variants are traced, and the difference between the last two is the
 * whole point of the widget:
 *
 *   no table        index_map returns j. Every block fetched, every block
 *                   computed.
 *   table, full     index_map returns data_next[i][j] on the rectangular
 *                   grid. Dead blocks resolve to a neighbour's index, so the
 *                   fetch is elided — but the step still runs.
 *   table, shrunk   same, on a grid only as wide as the busiest row. The
 *                   steps go too.
 */
(function (global) {
  'use strict';

  var SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768];
  var BLOCKS = [128, 256, 512, 1024];
  var WINDOWS = [128, 256, 512, 1024, 2048];
  var MAX_PRINTED_ROWS = 10;

  /* ---- tables ------------------------------------------------------ */

  function blockMask(nq, nkv, b, w) {
    var m = [];
    for (var i = 0; i < nq; i++) {
      var row = [];
      for (var j = 0; j < nkv; j++) { row.push(global.MaskGrid.classify(i, j, b, b, w)); }
      m.push(row);
    }
    return m;
  }

  function dataNextFull(mask) {
    return mask.map(function (row) {
      var live = [];
      row.forEach(function (v, j) { if (v > 0) { live.push(j); } });
      return row.map(function (_, j) {
        for (var a = 0; a < live.length; a++) { if (live[a] >= j) { return live[a]; } }
        return live.length ? live[live.length - 1] : 0;
      });
    });
  }

  function shrink(mask) {
    var lives = mask.map(function (row) {
      var live = [];
      row.forEach(function (v, j) { if (v > 0) { live.push(j); } });
      return live;
    });
    var width = Math.max(1, Math.max.apply(null, lives.map(function (l) { return l.length; })));
    var dataNext = [], blockMaskS = [];
    lives.forEach(function (live, i) {
      var dn = [], bm = [];
      for (var s = 0; s < width; s++) {
        if (s < live.length) { dn.push(live[s]); bm.push(mask[i][live[s]]); }
        else { dn.push(0); bm.push(0); }        // pad: index 0, do not compute
      }
      dataNext.push(dn); blockMaskS.push(bm);
    });
    return { width: width, dataNext: dataNext, blockMask: blockMaskS };
  }

  /* ---- tracing ----------------------------------------------------- */

  /* Delegates the elision rule to pipeline-grid.js. */
  function count(nq, width, dataNext, blockMask) {
    var tr = global.PipelineGrid.trace(
      [['i_q', nq], ['j', width]],
      [
        { name: 'Q', kind: 'in', map: function (ix) { return [ix.i_q, 0]; } },
        { name: 'K', kind: 'in', map: function (ix) { return [dataNext[ix.i_q][ix.j], 0]; } },
        { name: 'O', kind: 'out', map: function (ix) { return [ix.i_q, 0]; } }
      ]
    );
    var byName = {};
    tr.rows.forEach(function (r) { byName[r.name] = r.copies; });
    var compute = 0;
    for (var i = 0; i < nq; i++) {
      for (var j = 0; j < width; j++) { if (blockMask[i][j] > 0) { compute += 1; } }
    }
    return { steps: tr.total, q: byName.Q, k: byName.K, o: byName.O, compute: compute };
  }

  /* ---- rendering --------------------------------------------------- */

  function intTable(caption, rows, classOf) {
    var html = '<div class="st-tab"><p class="st-cap">' + caption + '</p><table class="pg"><tbody>';
    rows.forEach(function (row, i) {
      html += '<tr class="pg-op"><th>i=' + i + '</th>';
      row.forEach(function (v, j) {
        html += '<td class="' + (classOf ? classOf(v, i, j) : '') + '">' + v + '</td>';
      });
      html += '</tr>';
    });
    return html + '</tbody></table></div>';
  }

  function select(labelText, values, current, fmt, onChange) {
    var wrap = document.createElement('div');
    wrap.className = 'control';
    var id = 'st-' + Math.random().toString(36).slice(2, 7);
    var lab = document.createElement('label');
    lab.textContent = labelText;
    lab.htmlFor = id;
    var sel = document.createElement('select');
    sel.id = id;
    values.forEach(function (v) {
      var o = document.createElement('option');
      o.value = String(v);
      o.textContent = fmt ? fmt(v) : String(v);
      if (v === current) { o.selected = true; }
      sel.appendChild(o);
    });
    sel.addEventListener('change', function () { onChange(Number(sel.value)); });
    wrap.appendChild(lab);
    wrap.appendChild(sel);
    return wrap;
  }

  function mount(target, spec) {
    var root = typeof target === 'string' ? document.querySelector(target) : target;
    if (!root) { return; }
    if (!global.MaskGrid || !global.PipelineGrid) {
      root.textContent = 'skip-table.js needs mask-grid.js and pipeline-grid.js.';
      return;
    }

    var seqLen = spec.seqLen || 2048;
    var window_ = spec.window || 512;
    var block = spec.block || 512;
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
      bar.appendChild(select('sequence length', SEQ_LENS, seqLen, null,
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
    if (bar.children.length) { root.appendChild(bar); }

    var tables = document.createElement('div');
    tables.className = 'st-tables';
    root.appendChild(tables);

    var holder = document.createElement('div');
    holder.className = 'scroll-x';
    root.appendChild(holder);

    var legend = document.createElement('div');
    legend.className = 'legend';
    legend.innerHTML =
      '<span><i class="swatch" style="background:var(--accent)"></i>2 — full, no elementwise mask</span>' +
      '<span><i class="swatch" style="background:var(--accent-soft)"></i>1 — partial, mask it</span>' +
      '<span><i class="swatch" style="background:var(--dead)"></i>0 — dead, do not compute</span>';
    root.appendChild(legend);

    var readout = document.createElement('div');
    readout.className = 'readout';
    root.appendChild(readout);

    function draw() {
      var nq = Math.ceil(seqLen / block), nkv = nq;
      var mask = blockMask(nq, nkv, block, window_);
      var dnFull = dataNextFull(mask);
      var sh = shrink(mask);

      var dense = { dataNext: [], blockMask: [] };
      for (var i = 0; i < nq; i++) {
        var dnRow = [], bmRow = [];
        for (var j = 0; j < nkv; j++) { dnRow.push(j); bmRow.push(1); }
        dense.dataNext.push(dnRow); dense.blockMask.push(bmRow);
      }

      var variants = [
        { label: 'no table — index_map returns j', width: nkv,
          r: count(nq, nkv, dense.dataNext, dense.blockMask) },
        { label: 'table, full grid', width: nkv,
          r: count(nq, nkv, dnFull, mask) },
        { label: 'table, shrunk grid', width: sh.width,
          r: count(nq, sh.width, sh.dataNext, sh.blockMask) }
      ];

      if (nq <= MAX_PRINTED_ROWS) {
        tables.innerHTML =
          intTable('block_mask — shrunk, ' + nq + ' × ' + sh.width, sh.blockMask,
            function (v) { return v === 0 ? 'st-dead' : (v === 2 ? 'st-full' : 'st-partial'); }) +
          intTable('data_next — shrunk, ' + nq + ' × ' + sh.width, sh.dataNext,
            function (v, i, j) { return sh.blockMask[i][j] === 0 ? 'st-dead' : ''; });
      } else {
        tables.innerHTML = '<p class="st-cap">' + nq + ' × ' + sh.width +
          ' tables — too tall to print here; the counts below are over the whole thing.</p>';
      }

      var html = '<table class="st-cmp"><thead><tr>' +
        '<th>index_map</th><th class="num">grid</th><th class="num">steps</th>' +
        '<th class="num">computed</th><th class="num">K DMAs</th><th class="num">Q DMAs</th>' +
        '</tr></thead><tbody>';
      variants.forEach(function (v, n) {
        html += '<tr' + (n === 2 ? ' class="highlight"' : '') + '><td>' + v.label + '</td>' +
          '<td class="num">(' + nq + ', ' + v.width + ')</td>' +
          '<td class="num">' + v.r.steps + '</td>' +
          '<td class="num">' + v.r.compute + '</td>' +
          '<td class="num">' + v.r.k + '</td>' +
          '<td class="num">' + v.r.q + '</td></tr>';
      });
      html += '</tbody></table>';
      holder.innerHTML = html;

      var noTable = variants[0].r, full = variants[1].r, shr = variants[2].r;
      var distinct = nkv;
      readout.innerHTML =
        'the <b>table</b> cuts K traffic ' + noTable.k + ' &rarr; <b>' + full.k +
        '</b> DMAs and leaves the step count at ' + full.steps + '<br>' +
        'the <b>shrink</b> cuts steps ' + full.steps + ' &rarr; <b>' + shr.steps +
        '</b> and leaves the traffic at ' + shr.k + ' DMAs<br>' +
        '<span style="color:var(--ink-muted)">' +
        (shr.k === distinct
          ? 'K DMAs = ' + distinct + ' = one per kv block — every key block is loaded ' +
            'exactly once for the whole layer'
          : 'K DMAs = ' + shr.k + ' against ' + distinct + ' distinct kv blocks — ' +
            (shr.k > distinct ? 'some blocks are loaded more than once' : '')) +
        '</span>';
    }

    draw();
  }

  global.SkipTable = { mount: mount, blockMask: blockMask, shrink: shrink,
                       dataNextFull: dataNextFull, count: count };
})(window);
