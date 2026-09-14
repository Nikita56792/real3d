import Accelerate
import CoreML
import CoreVideo
import Foundation
import QuartzCore
import os

/// State of the metric scale calibration (relative main model + metric calibrator model).
public struct CalibrationStats: Sendable {
    public var a: Float = 1
    public var b: Float = 0
    public var valid = false
    public var residual: Float = 0      // median |1/Z error| of inliers, 1/m
    public var inlierFraction: Float = 0
    public var lastMs: Double = 0
    public var hz: Double = 0
    public var count = 0
    public init() {}
}

/// Owns the capture, the depth engine, the renderer and the display-link driven render loop.
public final class Pipeline: @unchecked Sendable {
    public let renderer: StereoRenderer
    public let capture: ScreenCapture
    private let log = Logger(subsystem: "real3d", category: "pipeline")

    private let inferenceQueue = DispatchQueue(label: "real3d.inference", qos: .userInteractive)
    private let lock = NSLock()
    private var engine: DepthEngine?
    private var inferenceBusy = false
    private var pendingSlot: ModelInputSlot?
    private var inferenceEnabled = true

    private var displayLink: CVDisplayLink?
    private var currentDisplayID: CGDirectDisplayID = CGMainDisplayID()

    private struct LayerEntry {
        let layer: CAMetalLayer
        var mode: PresentMode
        let isVisible: () -> Bool
    }
    private var layers: [ObjectIdentifier: LayerEntry] = [:]

    public private(set) var modelInfo: DepthModelInfo?
    public var onError: ((String) -> Void)?

    // Metric calibrator: a (slow) metric model run at a few Hz to fit 1/Z = a * r + b for the fast model.
    private let calibQueue = DispatchQueue(label: "real3d.calibrate", qos: .utility)
    private var calibrator: DepthEngine?
    public private(set) var calibratorInfo: DepthModelInfo?
    private var calibBusy = false
    private var calibEnabled = false
    private var calibInterval: Double = 0.25
    private var lastCalibStart: Double = 0
    private var calibInput: CVPixelBuffer?
    private var calibRawCopy: [Float] = []
    private var calibRawSize = (0, 0)
    private var _calibStats = CalibrationStats()
    private var calibTimes: [Double] = []
    public var calibrationStats: CalibrationStats { lock.withLock { _calibStats } }

    public init() throws {
        renderer = try StereoRenderer()
        capture = ScreenCapture()
        capture.onFrame = { [weak self] frame in
            self?.renderer.submitSource(frame.pixelBuffer, contentRect: frame.contentRect)
        }
        renderer.onModelInputReady = { [weak self] slot in self?.enqueue(slot) }
    }

    // MARK: Model

    /// Loads a model on the inference queue; the renderer is reconfigured for its I/O.
    public func loadModel(url: URL, computeUnits: MLComputeUnits, outputIsDepth: Bool) async throws -> DepthModelInfo {
        try await withCheckedThrowingContinuation { cont in
            inferenceQueue.async { [self] in
                do {
                    let e = try DepthEngine(modelURL: url, computeUnits: computeUnits)
                    e.outputIsDepth = outputIsDepth
                    try renderer.configureModel(inputWidth: e.info.inputWidth, inputHeight: e.info.inputHeight,
                                                inputFormat: e.preferredInputPixelFormat,
                                                outputWidth: e.info.outputWidth, outputHeight: e.info.outputHeight)
                    lock.withLock {
                        engine = e
                        modelInfo = e.info
                        pendingSlot = nil
                    }
                    // Warm up so the first real frame doesn't stall.
                    if let pb = try? DepthEngine.makePixelBuffer(width: e.info.inputWidth, height: e.info.inputHeight, format: e.preferredInputPixelFormat) {
                        _ = try? e.predict(pb)
                    }
                    cont.resume(returning: e.info)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    public func setOutputIsDepth(_ v: Bool) {
        lock.withLock { engine?.outputIsDepth = v }
        renderer.resetTemporal()
    }

    public func setInferenceEnabled(_ v: Bool) { lock.withLock { inferenceEnabled = v } }

    // MARK: Metric calibration

    /// Loads the metric calibrator model on its own queue.
    public func loadCalibrator(url: URL, computeUnits: MLComputeUnits) async throws -> DepthModelInfo {
        try await withCheckedThrowingContinuation { cont in
            calibQueue.async { [self] in
                do {
                    let e = try DepthEngine(modelURL: url, computeUnits: computeUnits)
                    e.outputIsDepth = true
                    let input = try DepthEngine.makePixelBuffer(width: e.info.inputWidth, height: e.info.inputHeight, format: e.preferredInputPixelFormat)
                    _ = try? e.predict(input) // warm up
                    lock.withLock {
                        calibrator = e
                        calibratorInfo = e.info
                        calibInput = input
                        _calibStats = CalibrationStats()
                        calibTimes = []
                    }
                    renderer.setCalibration(a: 1, b: 0, valid: false)
                    cont.resume(returning: e.info)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    public func unloadCalibrator() {
        calibQueue.async { [self] in
            lock.withLock { calibrator = nil; calibratorInfo = nil; calibInput = nil; _calibStats = CalibrationStats() }
            renderer.setCalibration(a: 1, b: 0, valid: false)
        }
    }

    /// Enables periodic calibration (metric mode with a relative main model).
    public func setCalibration(enabled: Bool, hz: Double) {
        lock.withLock { calibEnabled = enabled; calibInterval = 1.0 / max(hz, 0.2) }
        if !enabled { renderer.setCalibration(a: 1, b: 0, valid: false); lock.withLock { _calibStats.valid = false } }
    }

    /// Called on the inference queue right after the fast model produced `r` for `slot`.
    private func maybeCalibrate(slot: ModelInputSlot, result r: DepthResult, engine e: DepthEngine) {
        lock.lock()
        let now = CACurrentMediaTime()
        guard calibEnabled, let c = calibrator, let cin = calibInput, !calibBusy, now - lastCalibStart >= calibInterval else { lock.unlock(); return }
        calibBusy = true
        lastCalibStart = now
        lock.unlock()
        // Snapshot the fast model's input (rescaled to the calibrator size) and its raw output.
        Self.scale(slot.pixelBuffer, into: cin)
        let raw = Self.readDepthMap(r.pixelBuffer)
        let rawInvert = e.outputIsDepth
        let rawW = CVPixelBufferGetWidth(r.pixelBuffer), rawH = CVPixelBufferGetHeight(r.pixelBuffer)
        calibQueue.async { [self] in
            let t0 = CACurrentMediaTime()
            defer { lock.withLock { calibBusy = false } }
            guard let zr = try? c.predict(cin) else { return }
            let z = Self.readDepthMap(zr.pixelBuffer)
            let zW = CVPixelBufferGetWidth(zr.pixelBuffer), zH = CVPixelBufferGetHeight(zr.pixelBuffer)
            guard let fit = Self.fitAffine(raw: raw, rawW: rawW, rawH: rawH, rawInvert: rawInvert, z: z, zW: zW, zH: zH) else { return }
            let ms = (CACurrentMediaTime() - t0) * 1000
            lock.lock()
            var st = _calibStats
            // Snap on large changes (scene cut), otherwise smooth.
            let rel = st.valid ? abs(fit.a - st.a) / max(abs(st.a), 1e-6) : 1
            let alpha: Float = (!st.valid || rel > 0.5) ? 1 : 0.35
            st.a += (fit.a - st.a) * alpha
            st.b += (fit.b - st.b) * alpha
            st.valid = true
            st.residual = fit.residual; st.inlierFraction = fit.inliers; st.lastMs = ms; st.count += 1
            calibTimes.append(CACurrentMediaTime()); calibTimes.removeAll { $0 < CACurrentMediaTime() - 3 }
            st.hz = Double(calibTimes.count) / 3.0
            _calibStats = st
            lock.unlock()
            renderer.setCalibration(a: st.a, b: st.b, valid: true)
        }
    }

    /// Rescales a BGRA pixel buffer into another (vImage, high quality).
    public static func scale(_ src: CVPixelBuffer, into dst: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(src, .readOnly); CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(src, .readOnly); CVPixelBufferUnlockBaseAddress(dst, []) }
        var sb = vImage_Buffer(data: CVPixelBufferGetBaseAddress(src), height: vImagePixelCount(CVPixelBufferGetHeight(src)),
                               width: vImagePixelCount(CVPixelBufferGetWidth(src)), rowBytes: CVPixelBufferGetBytesPerRow(src))
        var db = vImage_Buffer(data: CVPixelBufferGetBaseAddress(dst), height: vImagePixelCount(CVPixelBufferGetHeight(dst)),
                               width: vImagePixelCount(CVPixelBufferGetWidth(dst)), rowBytes: CVPixelBufferGetBytesPerRow(dst))
        vImageScale_ARGB8888(&sb, &db, nil, vImage_Flags(kvImageHighQualityResampling))
    }

    /// Reads a single-channel fp16/fp32 pixel buffer into a dense Float array (row-major).
    public static func readDepthMap(_ pb: CVPixelBuffer) -> [Float] {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb), stride = CVPixelBufferGetBytesPerRow(pb)
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return [] }
        var out = [Float](repeating: 0, count: w * h)
        let isHalf = CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_OneComponent16Half
        out.withUnsafeMutableBufferPointer { o in
            for y in 0..<h {
                let row = base + y * stride
                if isHalf {
                    let r = row.assumingMemoryBound(to: Float16.self)
                    for x in 0..<w { o[y * w + x] = Float(r[x]) }
                } else {
                    memcpy(o.baseAddress! + y * w, row, w * 4)
                }
            }
        }
        return out
    }

    public struct AffineFit { public var a: Float; public var b: Float; public var residual: Float; public var inliers: Float }

    /// Robust least squares 1/Z = a * r + b on a coarse grid, with two rounds of MAD-based outlier rejection.
    public static func fitAffine(raw: [Float], rawW: Int, rawH: Int, rawInvert: Bool, z: [Float], zW: Int, zH: Int) -> AffineFit? {
        guard rawW > 0, rawH > 0, zW > 0, zH > 0, raw.count == rawW * rawH, z.count == zW * zH else { return nil }
        let gx = 128, gy = 72
        var xs: [Float] = [], ys: [Float] = []
        xs.reserveCapacity(gx * gy); ys.reserveCapacity(gx * gy)
        for j in 0..<gy {
            let v = (Float(j) + 0.5) / Float(gy)
            for i in 0..<gx {
                let u = (Float(i) + 0.5) / Float(gx)
                var r = raw[min(Int(v * Float(rawH)), rawH - 1) * rawW + min(Int(u * Float(rawW)), rawW - 1)]
                let d = z[min(Int(v * Float(zH)), zH - 1) * zW + min(Int(u * Float(zW)), zW - 1)]
                guard r.isFinite, d.isFinite, d > 0.05, d < 2000 else { continue }
                if rawInvert { r = 1 / max(r, 1e-4) }
                xs.append(r); ys.append(1 / d)
            }
        }
        guard xs.count >= 200 else { return nil }
        var mask = [Bool](repeating: true, count: xs.count)
        var a: Float = 0, b: Float = 0, residual: Float = 0, inl: Float = 0
        for round in 0..<3 {
            var n: Float = 0, sx: Float = 0, sy: Float = 0, sxx: Float = 0, sxy: Float = 0
            for i in 0..<xs.count where mask[i] { n += 1; sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i] }
            let det = n * sxx - sx * sx
            guard n >= 50, abs(det) > 1e-9 else { return nil }
            a = (n * sxy - sx * sy) / det
            b = (sy - a * sx) / n
            if round == 2 { inl = n / Float(xs.count); break }
            var res = [Float](); res.reserveCapacity(Int(n))
            for i in 0..<xs.count where mask[i] { res.append(abs(ys[i] - (a * xs[i] + b))) }
            res.sort()
            let mad = max(res[res.count / 2], 1e-5)
            residual = mad
            let thr = 3.0 * 1.4826 * mad
            for i in 0..<xs.count where mask[i] { mask[i] = abs(ys[i] - (a * xs[i] + b)) <= thr }
        }
        guard a > 0 else { return nil }
        return AffineFit(a: a, b: b, residual: residual, inliers: inl)
    }

    private func enqueue(_ slot: ModelInputSlot) {
        lock.lock()
        guard engine != nil, inferenceEnabled else { lock.unlock(); renderer.releaseSlot(slot); return }
        if inferenceBusy {
            if let old = pendingSlot { renderer.releaseSlot(old) }
            pendingSlot = slot
            lock.unlock()
            return
        }
        inferenceBusy = true
        lock.unlock()
        inferenceQueue.async { [self] in runInference(slot) }
    }

    private func runInference(_ startSlot: ModelInputSlot) {
        var slot: ModelInputSlot? = startSlot
        while let s = slot {
            lock.lock(); let e = engine; lock.unlock()
            if let e, CVPixelBufferGetWidth(s.pixelBuffer) == e.info.inputWidth, CVPixelBufferGetHeight(s.pixelBuffer) == e.info.inputHeight {
                do {
                    let r = try e.predict(s.pixelBuffer)
                    r.tag = s.index
                    renderer.submitDepth(r)
                    maybeCalibrate(slot: s, result: r, engine: e)
                } catch {
                    log.error("inference failed: \(error.localizedDescription)")
                }
            }
            renderer.releaseSlot(s)
            lock.lock()
            slot = pendingSlot
            pendingSlot = nil
            if slot == nil { inferenceBusy = false }
            lock.unlock()
        }
    }

    // MARK: Layers

    public func registerLayer(_ layer: CAMetalLayer, mode: PresentMode, isVisible: @escaping () -> Bool) {
        layer.device = renderer.device
        layer.pixelFormat = .bgra8Unorm
        layer.framebufferOnly = true
        layer.allowsNextDrawableTimeout = true
        layer.maximumDrawableCount = 3
        lock.withLock { layers[ObjectIdentifier(layer)] = LayerEntry(layer: layer, mode: mode, isVisible: isVisible) }
    }

    public func setMode(_ mode: PresentMode, for layer: CAMetalLayer) {
        lock.withLock { layers[ObjectIdentifier(layer)]?.mode = mode }
    }

    public func unregisterLayer(_ layer: CAMetalLayer) {
        lock.withLock { layers[ObjectIdentifier(layer)] = nil }
    }

    // MARK: Render loop

    public func startRenderLoop() {
        guard displayLink == nil else { return }
        var link: CVDisplayLink?
        CVDisplayLinkCreateWithActiveCGDisplays(&link)
        guard let link else { return }
        CVDisplayLinkSetOutputHandler(link) { [weak self] _, _, _, _, _ in
            self?.tick()
            return kCVReturnSuccess
        }
        CVDisplayLinkSetCurrentCGDisplay(link, currentDisplayID)
        CVDisplayLinkStart(link)
        displayLink = link
    }

    /// Re-targets the display link at the display that hosts the output window.
    public func setRenderDisplay(_ id: CGDirectDisplayID) {
        currentDisplayID = id
        if let displayLink { CVDisplayLinkSetCurrentCGDisplay(displayLink, id) }
    }

    public func stopRenderLoop() {
        if let displayLink { CVDisplayLinkStop(displayLink) }
        displayLink = nil
    }

    private func tick() {
        let visible: [(CAMetalLayer, PresentMode)] = lock.withLock {
            layers.values.compactMap { $0.isVisible() ? ($0.layer, $0.mode) : nil }
        }
        renderer.renderFrame(layers: visible)
    }
}
