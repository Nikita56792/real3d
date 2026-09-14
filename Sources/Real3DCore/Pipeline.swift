import CoreML
import CoreVideo
import Foundation
import QuartzCore
import os

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
