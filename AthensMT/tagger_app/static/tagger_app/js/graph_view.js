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
    var zoomPct      = document.getElementById('zoom-pct');
    var tagsCard     = document.getElementById('tags-card');
    var pageHeading  = document.getElementById('page-heading');
    var globalContextCard = document.getElementById('global-context-card');
    var imageSettingsCard = document.getElementById('image-settings-card');
    var tagsHeaderRow = document.getElementById('tags-header-row');
    var listSubmitBtn = document.getElementById('list-submit-btn');
    var backToListBtn = document.getElementById('gv-back-to-list-btn');
    if (!tagViewport || !tagContainer || !listBtn || !graphBtn) return;

    var graphActive = false;
    var didInitialFit = false;
    var canvas = { scale: 1, tx: 40, ty: 40 };

    var MIN_SCALE = 0.3, MAX_SCALE = 1.75;
    var NODE_START_X = 360, NODE_SPACING_X = 460, NODE_START_Y = 40;

    /* ═══ Per-column color coding — every distinct column (CSV or a tag's
       own output) gets a consistent color, applied to its source/output pin
       and every wire that carries it, so you can trace a column across the
       graph at a glance. Assigned by position (CSV columns in order, then
       each tag's output continuing the sequence) so it's stable for a given
       pipeline shape without needing to persist anything. ═══ */
    var WIRE_PALETTE = [
        '#f43f5e', '#f97316', '#eab308', '#84cc16', '#22c55e', '#14b8a6',
        '#06b6d4', '#3b82f6', '#8b5cf6', '#d946ef', '#ec4899', '#78716c'
    ];
    function colorIndexForColumn(colName) {
        var all = window.ALL_COLUMNS || [];
        var srcIdx = all.indexOf(colName);
        if (srcIdx !== -1) return srcIdx;
        var cards = Array.from(tagContainer.querySelectorAll('.tag-card'));
        for (var j = 0; j < cards.length; j++) {
            var inp = cards[j].querySelector('input[name="output_column"]');
            if (inp && inp.value.trim() === colName) return all.length + j;
        }
        return 0;
    }
    function colorForColumn(colName) {
        return WIRE_PALETTE[colorIndexForColumn(colName) % WIRE_PALETTE.length];
    }

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

    /* A wheel over a textarea (prompt template, negative prompt, ...) or a
       node body that's internally scrolling (tall image-mode nodes) should
       scroll that element, not zoom the canvas — but only while it still
       has room to scroll in that direction; once it hits its own top/bottom
       the wheel "escapes" to zoom, matching how nested-scroll areas usually
       feel. */
    function scrollableAncestor(el) {
        while (el && el !== tagViewport) {
            if (el.scrollHeight > el.clientHeight + 1) {
                var overflowY = getComputedStyle(el).overflowY;
                if (overflowY === 'auto' || overflowY === 'scroll') return el;
            }
            el = el.parentElement;
        }
        return null;
    }

    tagViewport.addEventListener('wheel', function (e) {
        if (!graphActive) return;
        var scrollEl = scrollableAncestor(e.target);
        if (scrollEl) {
            var atTop = scrollEl.scrollTop <= 0;
            var atBottom = scrollEl.scrollTop + scrollEl.clientHeight >= scrollEl.scrollHeight - 1;
            if ((e.deltaY < 0 && !atTop) || (e.deltaY > 0 && !atBottom)) return;
        }
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

    /* ═══ Node dragging — anywhere on the card's own background starts a
       drag (not just the small "Step N" badge, which alone is too small
       and mostly crowded out by the output-column input anyway). Every
       actual control (inputs, buttons, selects, labels/chips, pins) is
       excluded so normal interaction is untouched. ═══ */
    var NODE_DRAG_EXCLUDE = 'input, button, textarea, select, a, label, .node-pin';
    tagContainer.addEventListener('mousedown', function (e) {
        if (!graphActive) return;
        var card = e.target.closest('.tag-card');
        if (!card) return;
        if (e.target.closest(NODE_DRAG_EXCLUDE)) return;
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
    tagContainer.addEventListener('click', function (e) {
        // Wire clicks (selection) don't change chip/condition state — handled
        // separately below, and a redraw here would destroy the very wire
        // element that click just tried to select.
        if (e.target.closest('#graph-wires-svg')) return;
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
        ensureImgCollapseToggle(newCard);
        if (graphActive && window.__gvRedrawWires) window.__gvRedrawWires();
    });

    /* Image Settings starts collapsed in graph view (image-mode projects
       only — enhanceCardForImage, defined in the inline script, is what
       actually builds .img-section, and it already ran for every card
       that exists by the time this script runs). List view is untouched:
       the toggle button and collapsed state are both graph-mode-only via
       CSS scoped under #tag-container.graph-active. */
    function ensureImgCollapseToggle(card) {
        var sec = card.querySelector('.img-section');
        if (!sec || sec.dataset.gvCollapseReady) return;
        sec.dataset.gvCollapseReady = '1';
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'gv-img-collapse-btn';
        sec.insertBefore(btn, sec.firstChild);
        sec.classList.add('gv-collapsed');
        btn.textContent = 'Image Settings ▸ (click to expand)';
        btn.addEventListener('click', function () {
            var collapsed = sec.classList.toggle('gv-collapsed');
            btn.textContent = 'Image Settings ' + (collapsed ? '▸ (click to expand)' : '▾ (click to collapse)');
        });
    }
    function ensureAllImgCollapseToggles() {
        Array.from(tagContainer.querySelectorAll('.tag-card')).forEach(ensureImgCollapseToggle);
    }

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
        (window.ALL_COLUMNS || []).forEach(function (col, i) {
            var color = WIRE_PALETTE[i % WIRE_PALETTE.length];
            html += '<div class="flex items-center gap-2 px-3 py-1 text-xs" style="white-space:nowrap;">'
                + '<span class="src-pin-dot" data-col="' + escAttr(col) + '" style="background:' + color + ';"></span>'
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
        if (val === (window.NO_CONTEXT_COLUMNS || '__NONE__')) return []; // explicitly zero
        var avail = getAvailableColsForCard(idx);
        if (!val) return avail; // unset = "use all available"
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

    var SVGNS = 'http://www.w3.org/2000/svg';

    function anchor(el) {
        var r = el.getBoundingClientRect();
        var rect = tagContainer.getBoundingClientRect();
        return {
            x: (r.left + r.width / 2 - rect.left) / canvas.scale,
            y: (r.top + r.height / 2 - rect.top) / canvas.scale
        };
    }

    function bezierD(a, b) {
        var bend = Math.max(60, Math.abs(b.x - a.x) * 0.5);
        return 'M ' + a.x + ' ' + a.y
            + ' C ' + (a.x + bend) + ' ' + a.y + ', '
            + (b.x - bend) + ' ' + b.y + ', '
            + b.x + ' ' + b.y;
    }

    function cssEsc(s) {
        return (window.CSS && CSS.escape) ? CSS.escape(s) : String(s).replace(/["\\]/g, '\\$&');
    }

    var selectedWireHit = null;

    function showDeleteBadge(hit) {
        var badge = document.getElementById('gv-wire-delete-badge');
        if (!badge || !hit._fromEl || !hit._toEl) return;
        var a = anchor(hit._fromEl), b = anchor(hit._toEl);
        badge.style.left = ((a.x + b.x) / 2) + 'px';
        badge.style.top = ((a.y + b.y) / 2) + 'px';
        badge.style.display = 'flex';
        badge.dataset.kind = hit.dataset.kind;
        badge.dataset.targetIdx = hit.dataset.targetIdx;
        badge.dataset.col = hit.dataset.col;
    }
    function hideDeleteBadge() {
        var badge = document.getElementById('gv-wire-delete-badge');
        if (badge) badge.style.display = 'none';
    }
    function selectWire(hit) {
        clearWireSelection();
        selectedWireHit = hit;
        hit.classList.add('selected');
        if (hit._visiblePath) hit._visiblePath.classList.add('selected');
        showDeleteBadge(hit);
    }
    function clearWireSelection() {
        if (selectedWireHit) {
            selectedWireHit.classList.remove('selected');
            if (selectedWireHit._visiblePath) selectedWireHit._visiblePath.classList.remove('selected');
        }
        selectedWireHit = null;
        hideDeleteBadge();
    }

    function disconnectContext(card, col) {
        var cb = card.querySelector('.tag-chip-cb[data-col="' + cssEsc(col) + '"]');
        if (cb && cb.checked) {
            cb.checked = false;
            applyTagChipStyle(cb);
            updateTagColsHidden(card);
        }
    }
    function disconnectCondition(card, col) {
        var field = card.querySelector('input[name="condition_field"]');
        if (field && field.value.trim() === col) field.value = '';
    }
    function deleteWire(kind, targetIdx, col) {
        var cards = Array.from(tagContainer.querySelectorAll('.tag-card'));
        var card = cards[targetIdx];
        if (!card) return;
        if (kind === 'ctx') disconnectContext(card, col);
        else disconnectCondition(card, col);
    }

    function redrawWires() {
        var svg = document.getElementById('graph-wires-svg');
        if (!svg) return;
        clearWireSelection();
        while (svg.firstChild) svg.removeChild(svg.firstChild);
        if (!graphActive) return;

        function drawWire(fromEl, toEl, extraClass, meta) {
            if (!fromEl || !toEl) return;
            var d = bezierD(anchor(fromEl), anchor(toEl));
            var color = colorForColumn(meta.col);

            var hit = document.createElementNS(SVGNS, 'path');
            hit.setAttribute('d', d);
            hit.setAttribute('class', 'gv-wire-hit');
            hit.dataset.kind = meta.kind;
            hit.dataset.targetIdx = String(meta.targetIdx);
            hit.dataset.col = meta.col;
            hit._fromEl = fromEl;
            hit._toEl = toEl;
            svg.appendChild(hit);

            var path = document.createElementNS(SVGNS, 'path');
            path.setAttribute('d', d);
            path.setAttribute('class', 'gv-wire' + (extraClass ? ' ' + extraClass : ''));
            path.style.stroke = color;
            svg.appendChild(path);
            hit._visiblePath = path;
        }

        var cards = Array.from(tagContainer.querySelectorAll('.tag-card'));
        cards.forEach(function (card, idx) {
            var ctxSocket = card.querySelector('.ctx-socket');
            var condSocket = card.querySelector('.cond-socket');
            var outPin = card.querySelector('.out-pin');
            var outNameInput = card.querySelector('input[name="output_column"]');
            var outName = outNameInput ? outNameInput.value.trim() : '';
            if (outPin) outPin.style.background = outName ? colorForColumn(outName) : '#9ca3af';

            contextColumnsForCard(card, idx).forEach(function (col) {
                drawWire(findUpstreamPin(col, idx, cards), ctxSocket, 'gv-wire-ctx', { kind: 'ctx', targetIdx: idx, col: col });
            });

            var condCol = conditionColumnForCard(card);
            if (condCol) drawWire(findUpstreamPin(condCol, idx, cards), condSocket, 'gv-wire-cond', { kind: 'cond', targetIdx: idx, col: condCol });
        });
    }
    window.__gvRedrawWires = redrawWires;

    /* Wire selection (click) + deletion (Delete/Backspace or the × badge). */
    tagContainer.addEventListener('click', function (e) {
        var hit = e.target.closest('.gv-wire-hit');
        if (!hit) return;
        e.stopPropagation();
        selectWire(hit);
    });
    tagViewport.addEventListener('click', function () {
        if (graphActive) clearWireSelection();
    });
    document.addEventListener('keydown', function (e) {
        if (!graphActive || !selectedWireHit) return;
        if (e.key !== 'Delete' && e.key !== 'Backspace') return;
        var activeTag = (document.activeElement && document.activeElement.tagName) || '';
        if (activeTag === 'INPUT' || activeTag === 'TEXTAREA') return;
        e.preventDefault();
        deleteWire(selectedWireHit.dataset.kind, parseInt(selectedWireHit.dataset.targetIdx, 10), selectedWireHit.dataset.col);
        clearWireSelection();
        redrawWires();
    });
    var deleteBadge = document.getElementById('gv-wire-delete-badge');
    if (deleteBadge) deleteBadge.addEventListener('click', function (e) {
        e.stopPropagation();
        deleteWire(this.dataset.kind, parseInt(this.dataset.targetIdx, 10), this.dataset.col);
        clearWireSelection();
        redrawWires();
    });

    /* ═══ Drag-to-connect — drag from a source column pin or a tag's output
       pin onto a downstream tag's context/condition socket. Both ends are
       resolved through the same chip/condition-field controls the node
       body already exposes, so a drag is just a faster way to trigger
       something you could also do by hand inside the node. ═══ */
    function pinColumnName(pinEl) {
        if (pinEl.classList.contains('src-pin-dot')) return pinEl.dataset.col;
        if (pinEl.classList.contains('out-pin')) {
            var card = pinEl.closest('.tag-card');
            var inp = card ? card.querySelector('input[name="output_column"]') : null;
            return inp ? inp.value.trim() : '';
        }
        return '';
    }

    function connectContext(card, colName) {
        var cb = card.querySelector('.tag-chip-cb[data-col="' + cssEsc(colName) + '"]');
        if (cb && !cb.checked) {
            cb.checked = true;
            applyTagChipStyle(cb);
            updateTagColsHidden(card);
        }
    }
    function connectCondition(card, colName) {
        var toggle = card.querySelector('.cond-toggle');
        var field = card.querySelector('input[name="condition_field"]');
        if (!field) return;
        field.value = colName;
        if (toggle && !toggle.checked) {
            toggle.checked = true;
            toggle.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }

    var wireDrag = null;

    function containerLocalFromClient(clientX, clientY) {
        var rect = tagContainer.getBoundingClientRect();
        return { x: (clientX - rect.left) / canvas.scale, y: (clientY - rect.top) / canvas.scale };
    }

    function isValidDropTarget(socket, drag) {
        var card = socket.closest('.tag-card');
        if (!card) return false;
        var cards = Array.from(tagContainer.querySelectorAll('.tag-card'));
        var targetIdx = cards.indexOf(card);
        if (drag.cardIdx !== -1 && targetIdx <= drag.cardIdx) return false;
        return true;
    }

    function socketUnderPoint(clientX, clientY) {
        var target = document.elementFromPoint(clientX, clientY);
        return target ? target.closest('.ctx-socket, .cond-socket') : null;
    }

    function onWireDragMove(e) {
        if (!wireDrag) return;
        var a = anchor(wireDrag.fromEl);
        var b = containerLocalFromClient(e.clientX, e.clientY);
        wireDrag.tempPath.setAttribute('d', bezierD(a, b));

        document.querySelectorAll('.node-pin.gv-drop-target').forEach(function (el) { el.classList.remove('gv-drop-target'); });
        var socket = socketUnderPoint(e.clientX, e.clientY);
        if (socket && isValidDropTarget(socket, wireDrag)) socket.classList.add('gv-drop-target');
    }

    function onWireDragUp(e) {
        document.removeEventListener('mousemove', onWireDragMove);
        document.removeEventListener('mouseup', onWireDragUp);
        document.body.classList.remove('select-none');
        document.querySelectorAll('.node-pin.gv-drop-target').forEach(function (el) { el.classList.remove('gv-drop-target'); });

        var drag = wireDrag;
        wireDrag = null;
        if (drag && drag.tempPath && drag.tempPath.parentNode) drag.tempPath.parentNode.removeChild(drag.tempPath);
        if (!drag) return;

        var socket = socketUnderPoint(e.clientX, e.clientY);
        if (!socket || !isValidDropTarget(socket, drag)) return;

        var colName = pinColumnName(drag.fromEl);
        if (!colName) return;
        var targetCard = socket.closest('.tag-card');
        if (socket.classList.contains('ctx-socket')) connectContext(targetCard, colName);
        else connectCondition(targetCard, colName);
        redrawWires();
    }

    function startWireDrag(pinEl, clientX, clientY) {
        var cardIdx = -1;
        var card = pinEl.closest('.tag-card');
        if (card) cardIdx = Array.from(tagContainer.querySelectorAll('.tag-card')).indexOf(card);

        var svg = document.getElementById('graph-wires-svg');
        var tempPath = document.createElementNS(SVGNS, 'path');
        tempPath.setAttribute('class', 'gv-wire gv-wire-temp');
        svg.appendChild(tempPath);

        wireDrag = { fromEl: pinEl, cardIdx: cardIdx, tempPath: tempPath };
        tempPath.setAttribute('d', bezierD(anchor(pinEl), containerLocalFromClient(clientX, clientY)));
        document.body.classList.add('select-none');

        document.addEventListener('mousemove', onWireDragMove);
        document.addEventListener('mouseup', onWireDragUp);
    }

    tagContainer.addEventListener('mousedown', function (e) {
        if (!graphActive) return;
        var pin = e.target.closest('.src-pin-dot, .out-pin');
        if (!pin) return;
        e.preventDefault();
        e.stopPropagation();
        startWireDrag(pin, e.clientX, e.clientY);
    });

    buildSourceNodeOnce();
    updateSourceHighlights();
    ensureAllImgCollapseToggles();

    /* ═══ Mode toggle ═══════════════════════════════════════════════════ */
    function setMode(mode) {
        graphActive = (mode === 'graph');
        tagContainer.classList.toggle('graph-active', graphActive);
        tagViewport.classList.toggle('graph-active', graphActive);
        if (toolbar) toolbar.classList.toggle('active', graphActive);
        listBtn.classList.toggle('active', !graphActive);
        graphBtn.classList.toggle('active', graphActive);
        // Graph mode takes over the whole content area: the heading and the
        // Global Context card (redundant with the source node's pins) hide,
        // and the tags card becomes a fixed full-screen overlay (CSS above).
        if (tagsCard) tagsCard.classList.toggle('gv-fullscreen', graphActive);
        if (pageHeading) pageHeading.style.display = graphActive ? 'none' : '';
        if (globalContextCard) globalContextCard.style.display = graphActive ? 'none' : '';
        if (imageSettingsCard) imageSettingsCard.style.display = graphActive ? 'none' : '';
        // No container header in graph mode at all — title/description and the
        // original toggle+add-tag row hide too; the floating back-to-list
        // button and bottom-right cluster (toolbar) are the only chrome left.
        if (tagsHeaderRow) tagsHeaderRow.style.display = graphActive ? 'none' : '';
        if (listSubmitBtn) listSubmitBtn.style.display = graphActive ? 'none' : '';
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
    if (backToListBtn) backToListBtn.addEventListener('click', function () { setMode('list'); });

    // The floating "+ Add Tag" in the bottom-right cluster delegates to the
    // real (now-hidden) button rather than duplicating its card-creation
    // logic, which lives in the inline list-view script.
    var gvAddTagBtn = document.getElementById('gv-add-tag-btn');
    if (gvAddTagBtn && addTagBtn) gvAddTagBtn.addEventListener('click', function () { addTagBtn.click(); });

    var savedMode = null;
    try { savedMode = localStorage.getItem('odt_graph_view_mode'); } catch (e) {}
    setMode(savedMode === 'graph' ? 'graph' : 'list');
})();
