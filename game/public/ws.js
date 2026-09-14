// Bridges the QWOP game running in this page to the Python RL agent, via server.js acting
// as a relay between this browser tab and the agent's WebSocket connection.
//
// Protocol (JSON messages):
//   server -> browser: {type: "step", action: [q, w, o, p]} | {type: "reset"}
//   browser -> server: {type: "state", posx, posy, ..., velx, diffx, reward, terminated, truncated, terminal}
//
// State/terminal come straight from `window.state` / `window.terminal`, which QWOP.min.js
// already sets on every physics tick (in addition to writing them into the #state/#terminal
// DOM elements for the older Selenium-based environment).

// Headless/backgrounded browser tabs are typically reported as hidden even when they're
// the only tab open. The game engine listens for visibilitychange and auto-pauses physics
// when the page looks hidden, which would silently freeze training. Neutralize that before
// QWOP.min.js (injected later, on window.onload) gets a chance to register its listener.
Object.defineProperty(document, 'hidden', { get: function () { return false }, configurable: true })
Object.defineProperty(document, 'visibilityState', { get: function () { return 'visible' }, configurable: true })
var _origAddEventListener = document.addEventListener.bind(document)
document.addEventListener = function (type, listener, options) {
    if (/visibilitychange/i.test(type)) return
    return _origAddEventListener(type, listener, options)
}

// Physics tick counter, incremented once per simulated 0.04s world.step() (see the patched
// call site in QWOP.min.js: `window.__qwopStepWorld(this.m_world)` in place of a raw
// `this.m_world.step(0.04, 5, 5)`). This is the source of truth for how much *simulated*
// time has actually elapsed - callers should compute velocity/reward from
// (posx delta) / (ticks delta * 0.04) instead of measuring wall-clock time.
window.physicsTick = 0
window.__qwopStepWorld = function (world) {
    world.step(0.04, 5, 5)
    window.physicsTick += 1
}

// Lock-step physics: the game schedules its loop via a plain window.requestAnimationFrame
// call, looked up fresh each time rather than cached, so replacing it here (before
// QWOP.min.js's loop ever starts) is enough to take over scheduling entirely. Instead of
// auto-firing on a timer, the frame callback is captured and only actually runs when
// explicitly pumped from Python via execute_script - normally once per env.step()/reset().
// This makes "how much simulated time elapses per RL step" a deliberate choice instead of
// depending on wall-clock/system timing (Selenium latency, GC pauses, CPU contention from
// other parallel workers).
window.__qwopFrameCallback = null
window.requestAnimationFrame = function (callback) {
    window.__qwopFrameCallback = callback
}

// The frame callback's own timestamp argument isn't what actually gates physics stepping -
// the engine's internal frame accumulator (core.timestamp()) reads window.performance.now()
// directly, bypassing whatever we pass into the callback. So a fake, callback-argument-only
// clock doesn't work: calling the callback back-to-back with real wall-clock time between
// calls made the engine's own accumulator batch a variable, unpredictable number of physics
// steps per call (sometimes 0, sometimes several), which is exactly the wall-clock coupling
// we're trying to eliminate. Overriding performance.now() itself - so the engine's own
// accumulator is driven by a fake clock we advance by a fixed amount per callback - fixes
// this: empirically, incrementing by FRAME_INTERVAL_MS per call yields (very close to)
// exactly one physics tick per call, regardless of how fast we actually call it.
window.__qwopFakeNow = performance.now()
Object.defineProperty(window.performance, 'now', {
    value: function () { return window.__qwopFakeNow },
    writable: true,
    configurable: true,
})
var FRAME_INTERVAL_MS = 33.34 // ~1000/30; empirically matches the engine's internal frame_time

// Runs the currently-queued frame callback exactly once, with no guarantee it results in a
// physics tick (e.g. before the game's click-to-start has happened). Used only to pump the
// loop during initial connection, to get window.state populated - state capture happens
// whenever the player object exists, independent of whether physics is stepping yet.
window.__qwopPumpFrame = function () {
    var cb = window.__qwopFrameCallback
    window.__qwopFrameCallback = null
    window.__qwopFakeNow += FRAME_INTERVAL_MS
    if (cb) cb(window.__qwopFakeNow)
}

// Runs the frame callback n times, each advancing the fake clock by exactly one frame
// interval - which (see above) reliably yields one physics tick per call.
window.__qwopAdvance = function (n) {
    for (var i = 0; i < n; i++) {
        var cb = window.__qwopFrameCallback
        window.__qwopFrameCallback = null
        window.__qwopFakeNow += FRAME_INTERVAL_MS
        if (cb) cb(window.__qwopFakeNow)
    }
}

var ws
var keyStates = {}
var lastPosx = 0
var lastTime = null

function qwopLoaded() {
    ws = new WebSocket("ws://" + window.location.host)

    ws.onopen = function () {
        console.log("Opened web socket.")
        ws.send(JSON.stringify({ type: "hello", role: "browser" }))
    }

    ws.onmessage = function (event) {
        var message
        try {
            message = JSON.parse(event.data)
        } catch (e) {
            console.error("Invalid message from server:", event.data)
            return
        }

        if (message.type === "reset") {
            handleReset()
        } else if (message.type === "step") {
            handleStep(message.action)
        }
    }
}

function handleStep(action) {
    pressKeys(actionToKeys(action))
    waitFrames(2).then(sendState)
}

function handleReset() {
    pressKeys([])

    var canvas = document.getElementsByTagName('canvas')[0]
    dispatchMouseEvent(canvas, 'mouseover')
    dispatchMouseEvent(canvas, 'mousedown')
    dispatchMouseEvent(canvas, 'click')
    dispatchMouseEvent(canvas, 'mouseup')

    simulateKeydown(getCode("r"))
    simulateKeyup(getCode("r"))

    lastPosx = 0
    lastTime = null

    waitFrames(2).then(sendState)
}

function actionToKeys(action) {
    var checkKeys = ["q", "w", "o", "p"]
    var keys = []
    for (var i = 0; i < checkKeys.length; i++) {
        if (action[i]) keys.push(checkKeys[i])
    }
    return keys
}

// Exposed for the Selenium-driven environment (qwop_env.py) to apply an action via
// execute_script - dispatching synthetic key events directly is one script call, vs.
// Selenium's ActionChains which is a separate WebDriver round trip. Folding this into the
// same execute_script call as window.__qwopAdvance (see _advance_and_get_state in
// qwop_env.py) collapses step() to a single round trip instead of two.
window.__qwopApplyAction = function (action) {
    pressKeys(actionToKeys(action))
}

function sendState() {
    var state = window.state || {}
    var terminal = !!window.terminal

    var posx = state.posx || 0
    var now = performance.now()
    var diffx = lastTime === null ? 0 : posx - lastPosx
    var dt = lastTime === null ? 0 : Math.max((now - lastTime) / 1000, 0.001)
    var velx = dt > 0 ? diffx / dt : 0

    lastPosx = posx
    lastTime = now

    var reward = terminal ? -5.0 : velx

    var payload = Object.assign({}, state, {
        type: "state",
        terminal: terminal,
        velx: velx,
        diffx: diffx,
        reward: reward,
        terminated: terminal,
        truncated: false,
    })

    ws.send(JSON.stringify(payload))
}

function waitFrames(n) {
    // requestAnimationFrame no longer auto-fires (see the lock-step override above) - drive
    // it explicitly instead of waiting for something that will never happen on its own.
    window.__qwopAdvance(n)
    return Promise.resolve()
}

function pressKeys(keys) {
    var checkKeys = ["q", "w", "o", "p"]
    for (var i = 0; i < checkKeys.length; i++) {
        var key = checkKeys[i]
        var code = getCode(key)
        var shouldBeDown = keys.indexOf(key) > -1
        if (shouldBeDown && !keyStates[code]) {
            simulateKeydown(code)
            keyStates[code] = true
        } else if (!shouldBeDown && keyStates[code]) {
            simulateKeyup(code)
            keyStates[code] = false
        }
    }
}

function getCode(keyChar) {
    if (keyChar === 'q') return 81
    if (keyChar === 'w') return 87
    if (keyChar === 'o') return 79
    if (keyChar === 'p') return 80
    if (keyChar === 'r') return 82
    return 32
}

function dispatchMouseEvent(target, type) {
    // The game only unpauses physics on a real click on the canvas. The legacy
    // document.createEvent("MouseEvents")/initEvent construction dispatches an event with
    // no coordinates/button, which the game doesn't recognize as a valid click - use a
    // modern MouseEvent aimed at the canvas center instead.
    var rect = target.getBoundingClientRect()
    var x = rect.left + rect.width / 2
    var y = rect.top + rect.height / 2
    var e = new MouseEvent(type, {
        bubbles: true,
        cancelable: true,
        view: window,
        clientX: x,
        clientY: y,
        button: 0,
        buttons: type === 'mousedown' ? 1 : 0,
    })
    target.dispatchEvent(e)
}

function simulateKeydown(keycode, isCtrl, isAlt, isShift) {
    var e = new KeyboardEvent("keydown", { bubbles: true, cancelable: true, char: String.fromCharCode(keycode), key: String.fromCharCode(keycode), shiftKey: isShift, ctrlKey: isCtrl, altKey: isAlt })
    Object.defineProperty(e, 'keyCode', { get: function () { return this.keyCodeVal } })
    e.keyCodeVal = keycode
    document.dispatchEvent(e)
}

function simulateKeyup(keycode, isCtrl, isAlt, isShift) {
    var e = new KeyboardEvent("keyup", { bubbles: true, cancelable: true, char: String.fromCharCode(keycode), key: String.fromCharCode(keycode), shiftKey: isShift, ctrlKey: isCtrl, altKey: isAlt })
    Object.defineProperty(e, 'keyCode', { get: function () { return this.keyCodeVal } })
    e.keyCodeVal = keycode
    document.dispatchEvent(e)
}
