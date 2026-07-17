/* ═══════════════════════════════════════════════════════════════════════
   Graph view for Define Columns — a ComfyUI-style node canvas that is a
   second, toggleable presentation of the exact same .tag-card elements
   the list view uses. No parallel data model: node "state" is just the
   existing hidden inputs/chips/condition fields read and driven through
   their existing update functions (rebuildTagChips, updateTagColsHidden,
   etc. — defined earlier in this page's inline <script>).

   Loaded after the inline list-view script, as a classic (non-module)
   script, so it shares the same global scope and can freely reuse those
   functions. It attaches its own independent event listeners rather than
   editing the existing ones, so list view behaviour is untouched.
   ═══════════════════════════════════════════════════════════════════════ */
(function () {
    var tagViewport = document.getElementById('tag-viewport');
    var tagContainer = document.getElementById('tag-container');
    var listBtn      = document.getElementById('view-list-btn');
    var graphBtn     = document.getElementById('view-graph-btn');
    var toolbar      = document.getElementById('graph-toolbar');
    var pageWrap     = document.getElementById('page-wrap');
    var zoomPct      = document.getElementById('zoom-pct');
    if (!tagViewport || !tagContainer || !listBtn || !graphBtn) return;

    var graphActive = false;
    var didInitialFit = false;
    var canvas = { scale: 1, tx: 40, ty: 40 };

    var MIN_SCALE = 0.3, MAX_SCALE = 1.75;
    var NODE_START_X = 360, NODE_SPACING_X = 460, NODE_START_Y = 40;

    /* ═══ Node position (persisted via --nx/--ny + hidden node_x/node_y) ═══ */
    function getPos(card) {
        var x = parseFloat(card.dataset.nodeX);
        var y = parseFloat(card.dataset.nodeY);
        if (isNaN(x) || isNaN(y)) return null;
        return { x: x, y: y };
    }

    function setPos(card, x, y) {
        x = Math.round(x); y = Math.round(y);
        card.dataset.nodeX = String(x);
        card.dataset.nodeY = String(y);
        card.style.setProperty('--nx', x + 'px');
        card.style.setProperty('--ny', y + 'px');
        var hx = card.querySelector('.node-x-hidden');
        var hy = card.querySelector('.node-y-hidden');
        if (hx) hx.value = String(x);
        if (hy) hy.value = String(y);
    }

    function autoLayoutIndex(idx) {
        return { x: NODE_START_X + idx * NODE_SPACING_X, y: NODE_START_Y };
    }

    function initNodePositions() {
        Array.from(tagContainer.querySelectorAll('.tag-card')).forEach(function (card, idx) {
            var pos = getPos(card) || autoLayoutIndex(idx);
            setPos(card, pos.x, pos.y);
        });
    }
    window.__gvInitNodePositions = initNodePositions;

    /* ═══ Canvas transform (pan/zoom) ══════════════════════════════════ */
    function applyCanvasTransform() {
        tagContainer.style.transform = 'translate(' + canvas.tx + 'px,' + canvas.ty + 'px) scale(' + canvas.scale + ')';
        if (zoomPct) zoomPct.textContent = Math.round(canvas.scale * 100) + '%';
    }

    function zoomAround(cx, cy, factor) {
        var newScale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, canvas.scale * factor));
        canvas.tx = cx - (cx - canvas.tx) * (newScale / canvas.scale);
        canvas.ty = cy - (cy - canvas.ty) * (newScale / canvas.scale);
        canvas.scale = newScale;
        applyCanvasTransform();
    }

    tagViewport.addEventListener('wheel', function (e) {
        if (!graphActive) return;
        e.preventDefault();
        var rect = tagViewport.getBoundingClientRect();
        var factor = e.deltaY < 0 ? 1.1 : (1 / 1.1);
        zoomAround(e.clientX - rect.left, e.clientY - rect.top, factor);
    }, { passive: false });

    var zoomInBtn = document.getElementById('zoom-in-btn');
    var zoomOutBtn = document.getElementById('zoom-out-btn');
    var fitBtn = document.getElementById('fit-view-btn');
    if (zoomInBtn) zoomInBtn.addEventListener('click', function () {
        var rect = tagViewport.getBoundingClientRect();
        zoomAround(rect.width / 2, rect.height / 2, 1.15);
    });
    if (zoomOutBtn) zoomOutBtn.addEventListener('click', function () {
        var rect = tagViewport.getBoundingClientRect();
        zoomAround(rect.width / 2, rect.height / 2, 1 / 1.15);
    });
    if (fitBtn) fitBtn.addEventListener('click', fitView);

    function fitView() {
        var nodes = Array.from(tagContainer.children).filter(function (el) {
            return el.classList && (el.classList.contains('tag-card') || el.id === 'graph-source-node');
        });
        if (!nodes.length) { canvas.scale = 1; canvas.tx = 40; canvas.ty = 40; applyCanvasTransform(); return; }
        var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        nodes.forEach(function (n) {
            minX = Math.min(minX, n.offsetLeft);
            minY = Math.min(minY, n.offsetTop);
            maxX = Math.max(maxX, n.offsetLeft + n.offsetWidth);
            maxY = Math.max(maxY, n.offsetTop + n.offsetHeight);
        });
        var pad = 60;
        minX -= pad; minY -= pad; maxX += pad; maxY += pad;
        var rect = tagViewport.getBoundingClientRect();
        var contentW = Math.max(1, maxX - minX), contentH = Math.max(1, maxY - minY);
        var scale = Math.min(rect.width / contentW, rect.height / contentH, MAX_SCALE);
        scale = Math.max(scale, MIN_SCALE);
        canvas.scale = scale;
        canvas.tx = -minX * scale;
        canvas.ty = -minY * scale;
        applyCanvasTransform();
    }
    window.__gvFitView = fitView;

    /* ═══ Panning (drag empty canvas) ══════════════════════════════════ */
    var isPanning = false, panStartX = 0, panStartY = 0, panStartTx = 0, panStartTy = 0;
    tagViewport.addEventListener('mousedown', function (e) {
        if (!graphActive) return;
        if (e.button !== 0) return;
        if (e.target.closest('.tag-card, #graph-source-node')) return;
        isPanning = true;
        panStartX = e.clientX; panStartY = e.clientY;
        panStartTx = canvas.tx; panStartTy = canvas.ty;
        tagViewport.classList.add('panning');
        document.body.classList.add('select-none');
        e.preventDefault();
    });
    document.addEventListener('mousemove', function (e) {
        if (!isPanning) return;
        canvas.tx = panStartTx + (e.clientX - panStartX);
        canvas.ty = panStartTy + (e.clientY - panStartY);
        applyCanvasTransform();
    });
    document.addEventListener('mouseup', function () {
        if (isPanning) {
            isPanning = false;
            tagViewport.classList.remove('panning');
            document.body.classList.remove('select-none');
        }
    });

    /* ═══ Node dragging (via the "Step N" badge handle) ════════════════ */
    tagContainer.addEventListener('mousedown', function (e) {
        if (!graphActive) return;
        var badge = e.target.closest('.card-badge');
        if (!badge) return;
        var card = badge.closest('.tag-card');
        if (!card) return;
        e.preventDefault();
        e.stopPropagation();
        var startX = e.clientX, startY = e.clientY;
        var pos = getPos(card) || { x: 0, y: 0 };
        var startNX = pos.x, startNY = pos.y;
        card.classList.add('dragging-node');
        document.body.classList.add('select-none');

        function onMove(ev) {
            var dx = (ev.clientX - startX) / canvas.scale;
            var dy = (ev.clientY - startY) / canvas.scale;
            setPos(card, startNX + dx, startNY + dy);
            if (window.__gvRedrawWires) window.__gvRedrawWires();
        }
        function onUp() {
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
            card.classList.remove('dragging-node');
            document.body.classList.remove('select-none');
        }
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });

    /* ═══ Redraw hook — recompute wires on anything that could change
       chip selections, conditions, or output-column names. A no-op until
       redrawWires() exists (added alongside the source node). ═══ */
    ['input', 'change'].forEach(function (evt) {
        tagContainer.addEventListener(evt, function () {
            if (graphActive && window.__gvRedrawWires) window.__gvRedrawWires();
        });
    });
    tagContainer.addEventListener('click', function () {
        if (graphActive && window.__gvRedrawWires) window.__gvRedrawWires();
    });

    /* New card added via "+ Add Tag" — give it a position once it exists.
       Runs after the existing add-tag-btn handler (attached earlier in
       document order), so the card is already appended. */
    var addTagBtn = document.getElementById('add-tag-btn');
    if (addTagBtn) addTagBtn.addEventListener('click', function () {
        var cards = tagContainer.querySelectorAll('.tag-card');
        var newCard = cards[cards.length - 1];
        if (!newCard) return;
        var pos = autoLayoutIndex(cards.length - 1);
        setPos(newCard, pos.x, pos.y);
        if (graphActive && window.__gvRedrawWires) window.__gvRedrawWires();
    });

    /* ═══ Source node ("CSV Columns") — built once, pins re-highlighted
       whenever the global Step-1 chip selection changes. Fixed position,
       not draggable, not persisted (purely a visual anchor). ═══ */
    var sourceNode = document.getElementById('graph-source-node');

    function buildSourceNodeOnce() {
        if (!sourceNode || sourceNode.dataset.built) return;
        sourceNode.dataset.built = '1';
        sourceNode.style.setProperty('--nx', '40px');
        sourceNode.style.setProperty('--ny', '40px');
        var html = '';
        html += '<div class="px-3 py-2 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 rounded-t-lg">';
        html += '  <span class="text-xs font-semibold text-gray-700 dark:text-gray-200">CSV Columns</span>';
        html += '</div>';
        html += '<div class="src-pin-list py-1">';
        (window.ALL_COLUMNS || []).forEach(function (col) {
            html += '<div class="flex items-center gap-2 px-3 py-1 text-xs" style="white-space:nowrap;">'
                + '<span class="src-pin-dot" data-col="' + escAttr(col) + '"></span>'
                + '<span class="src-pin-label truncate text-gray-400 dark:text-gray-500" style="max-width:170px;">' + escHtml(col) + '</span>'
                + '</div>';
        });
        html += '</div>';
        sourceNode.innerHTML = html;
    }

    function updateSourceHighlights() {
        if (!sourceNode) return;
        var selected = {};
        document.querySelectorAll('.chip-cb:checked').forEach(function (cb) { selected[cb.value] = true; });
        sourceNode.querySelectorAll('.src-pin-dot').forEach(function (dot) {
            var active = !!selected[dot.dataset.col];
            dot.classList.toggle('active', active);
            var label = dot.nextElementSibling;
            if (!label) return;
            label.classList.toggle('text-gray-400', !active);
            label.classList.toggle('dark:text-gray-500', !active);
            label.classList.toggle('text-gray-800', active);
            label.classList.toggle('dark:text-gray-100', active);
            label.classList.toggle('font-medium', active);
        });
    }

    var chipContainer = document.getElementById('chip-container');
    if (chipContainer) chipContainer.addEventListener('change', function () {
        updateSourceHighlights();
        if (graphActive && window.__gvRedrawWires) window.__gvRedrawWires();
    });

    /* ═══ Wires — recomputed from existing state on every redraw, never
       stored. A column feeding a tag node's context/condition resolves to
       either a source-node pin (a raw CSV column) or an earlier tag's
       output pin (its OutputColumn), mirroring real run-time precedence
       (a generated column shadows a same-named CSV column). ═══ */
    function contextColumnsForCard(card, idx) {
        var hidden = card.querySelector('.tag-cols-hidden');
        var val = hidden ? hidden.value.trim() : '';
        var avail = getAvailableColsForCard(idx);
        if (!val) return avail; // empty = "use all available"
        var wanted = val.split(',').map(function (s) { return s.trim(); }).filter(Boolean);
        return wanted.filter(function (c) { return avail.indexOf(c) !== -1; });
    }

    function conditionColumnForCard(card) {
        var toggle = card.querySelector('.cond-toggle');
        if (!toggle || !toggle.checked) return null;
        var field = card.querySelector('input[name="condition_field"]');
        var val = field ? field.value.trim() : '';
        return val || null;
    }

    function findUpstreamPin(colName, cardIdx, cards) {
        for (var j = cardIdx - 1; j >= 0; j--) {
            var nameInput = cards[j].querySelector('input[name="output_column"]');
            if (nameInput && nameInput.value.trim() === colName) {
                return cards[j].querySelector('.out-pin');
            }
        }
        if (sourceNode) {
            var dots = sourceNode.querySelectorAll('.src-pin-dot');
            for (var i = 0; i < dots.length; i++) {
                if (dots[i].dataset.col === colName) return dots[i];
            }
        }
        return null;
    }

    function redrawWires() {
        var svg = document.getElementById('graph-wires-svg');
        if (!svg) return;
        while (svg.firstChild) svg.removeChild(svg.firstChild);
        if (!graphActive) return;

        var containerRect = tagContainer.getBoundingClientRect();
        function anchor(el) {
            var r = el.getBoundingClientRect();
            return {
                x: (r.left + r.width / 2 - containerRect.left) / canvas.scale,
                y: (r.top + r.height / 2 - containerRect.top) / canvas.scale
            };
        }
        function drawWire(fromEl, toEl, extraClass) {
            if (!fromEl || !toEl) return;
            var a = anchor(fromEl), b = anchor(toEl);
            var bend = Math.max(60, Math.abs(b.x - a.x) * 0.5);
            var d = 'M ' + a.x + ' ' + a.y
                + ' C ' + (a.x + bend) + ' ' + a.y + ', '
                + (b.x - bend) + ' ' + b.y + ', '
                + b.x + ' ' + b.y;
            var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', d);
            path.setAttribute('class', 'gv-wire' + (extraClass ? ' ' + extraClass : ''));
            svg.appendChild(path);
        }

        var cards = Array.from(tagContainer.querySelectorAll('.tag-card'));
        cards.forEach(function (card, idx) {
            var ctxSocket = card.querySelector('.ctx-socket');
            var condSocket = card.querySelector('.cond-socket');

            contextColumnsForCard(card, idx).forEach(function (col) {
                drawWire(findUpstreamPin(col, idx, cards), ctxSocket, 'gv-wire-ctx');
            });

            var condCol = conditionColumnForCard(card);
            if (condCol) drawWire(findUpstreamPin(condCol, idx, cards), condSocket, 'gv-wire-cond');
        });
    }
    window.__gvRedrawWires = redrawWires;

    buildSourceNodeOnce();
    updateSourceHighlights();

    /* ═══ Mode toggle ═══════════════════════════════════════════════════ */
    function setMode(mode) {
        graphActive = (mode === 'graph');
        tagContainer.classList.toggle('graph-active', graphActive);
        tagViewport.classList.toggle('graph-active', graphActive);
        if (toolbar) toolbar.classList.toggle('active', graphActive);
        listBtn.classList.toggle('active', !graphActive);
        graphBtn.classList.toggle('active', graphActive);
        if (pageWrap) {
            pageWrap.classList.toggle('max-w-4xl', !graphActive);
            pageWrap.classList.toggle('max-w-6xl', graphActive);
        }
        if (graphActive) {
            initNodePositions();
            applyCanvasTransform();
            if (window.__gvRedrawWires) window.__gvRedrawWires();
            if (!didInitialFit) { didInitialFit = true; fitView(); }
        }
        try { localStorage.setItem('odt_graph_view_mode', mode); } catch (e) {}
    }

    listBtn.addEventListener('click', function () { setMode('list'); });
    graphBtn.addEventListener('click', function () { setMode('graph'); });

    var savedMode = null;
    try { savedMode = localStorage.getItem('odt_graph_view_mode'); } catch (e) {}
    setMode(savedMode === 'graph' ? 'graph' : 'list');
})();
