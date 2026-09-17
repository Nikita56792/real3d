import CoreVideo
import Foundation
import Metal
import MetalFX
import QuartzCore
import os

// MARK: - Parameter structs (layout must match Shaders.metal)

struct CropParams {
    var uvOrigin: SIMD2<Float>
    var uvSize: SIMD2<Float>
    var outWidth: UInt32
    var outHeight: UInt32
    var supersample: UInt32
    var _pad: UInt32 = 0
    var validMin = SIMD2<Float>(-1e4, -1e4)
    var validMax = SIMD2<Float>(1e4, 1e4)
}

struct DepthParams {
    var lo: Float
    var hi: Float
    var alpha: Float
    var invert: UInt32
    var hasPrev: UInt32
    var width: UInt32
    var height: UInt32
    var metric: UInt32 = 0
    var calibA: Float = 1
    var calibB: Float = 0
    var _pad0: Float = 0, _pad1: Float = 0
}

struct UpsampleParams {
    var outWidth: UInt32
    var outHeight: UInt32
    var mode: UInt32
    var radius: Int32
    var sigmaSpatial: Float
    var sigmaRange: Float
    var strength: Float
    var convergence: Float
    var gamma: Float
    var metric: UInt32 = 0
    var metricA: Float = 0
    var metricB: Float = 0
    var worldScale: Float = 1
    var maxCrossed: Float = 60
    var contentMin = SIMD2<Float>(-1, -1)
    var contentMax = SIMD2<Float>(2, 2)
}

struct StereoParams {
    var width: UInt32
    var height: UInt32
    var maxDisp: Float
    var stretchThresh: Float
    var swapEyes: UInt32
    var _pad0: UInt32 = 0, _pad1: UInt32 = 0, _pad2: UInt32 = 0
}

struct PresentParams {
    var scale: SIMD2<Float>
    var uvOrigin: SIMD2<Float>
    var uvSize: SIMD2<Float>
    var mode: UInt32
    var _pad: UInt32 = 0
}

// MARK: - Public settings

public enum PresentMode: Int, CaseIterable, Identifiable, Sendable {
    case mono = 0, fullSBS = 1, halfSBS = 2, anaglyph = 3, depth = 4, source = 5, leftEye = 6, rightEye = 7
    public var id: Int { rawValue }
    public var title: String {
        switch self {
        case .mono: return "2D (mono)"
        case .fullSBS: return "3D Full SBS (3840×1080)"
        case .halfSBS: return "3D Half SBS (1920×1080)"
        case .anaglyph: return "Anaglyph (red/cyan test)"
        case .depth: return "Disparity map"
        case .source: return "Source"
        case .leftEye: return "Left eye"
        case .rightEye: return "Right eye"
        }
    }
}

public enum UpsampleMode: Int, CaseIterable, Identifiable, Sendable {
    case bilinear = 0, jointBilateral = 1
    public var id: Int { rawValue }
    public var title: String { self == .bilinear ? "Bilinear" : "Edge-aware (joint bilateral)" }
}

public enum DepthScaleMode: Int, CaseIterable, Identifiable, Sendable {
    case artistic = 0, metric = 1
    public var id: Int { rawValue }
    public var title: String { self == .artistic ? "Artistic (relative depth)" : "Real scale (metric)" }
}

/// Viewer geometry for orthostereo rendering (virtual screen of the glasses).
public struct ViewerGeometry: Equatable, Sendable {
    public var ipdMM: Float = 63
    public var screenDistanceM: Float = 4.0
    public var screenWidthM: Float = 3.25          // XREAL One: 147" 16:9 at 4 m
    public init() {}
    public init(ipdMM: Float, screenDistanceM: Float, screenWidthM: Float) {
        self.ipdMM = ipdMM; self.screenDistanceM = screenDistanceM; self.screenWidthM = screenWidthM
    }
    /// Disparity (px) of a point at infinity for a per-eye width of `px` pixels.
    public func infinityDisparity(px: Int) -> Float { ipdMM / 1000 / (screenWidthM / Float(px)) }
}

public struct RenderSettings: Equatable, Sendable {
    /// Per-eye output size.
    public var outputWidth: Int = 1920
    public var outputHeight: Int = 1080
    public var depthScaleMode: DepthScaleMode = .artistic
    public var viewer = ViewerGeometry()
    /// Multiplies scene depth in metric mode (1 = as estimated).
    public var worldScale: Float = 1.0
    /// Auto world scale: enlarge the scene (never shrink below real scale) so the nearest content sits
    /// `popBudgetPx` in front of the screen plane.
    public var autoWorldScale: Bool = true
    public var popBudgetPx: Float = 12
    /// Clamp for crossed (pop-out) disparity in metric mode, px.
    public var maxCrossedPx: Float = 60
    /// Maximum disparity in output pixels (interocular shift for the nearest point).
    public var strength: Float = 40
    /// Normalized depth (0 far … 1 near) placed at the screen plane.
    public var convergence: Float = 0.55
    /// Depth curve exponent (1 = linear).
    public var gamma: Float = 1.0
    /// Temporal smoothing: weight of the new depth frame (1 = off).
    public var temporalAlpha: Float = 0.5
    public var upsampleMode: UpsampleMode = .jointBilateral
    public var jbuRadius: Int = 2
    public var jbuSigmaRange: Float = 0.08
    /// Mapped-segment length above which a gap is treated as a disocclusion.
    public var stretchThreshold: Float = 2.5
    public var swapEyes: Bool = false
    /// Model outputs depth (larger = farther) instead of inverse depth.
    public var invertDepth: Bool = false
    /// Model outputs metric depth in meters (implies invertDepth); no calibration needed.
    public var modelIsMetric: Bool = false
    public var useMetalFX: Bool = true
    public var showCursor: Bool = true
    public init() {}
}

public struct RenderStats: Sendable {
    public var renderFPS: Double = 0
    public var captureFPS: Double = 0
    public var depthFPS: Double = 0
    public var gpuMs: Double = 0
    public var inferenceMs: Double = 0
    public var sourceSize: CGSize = .zero
    public var cropPixelSize: CGSize = .zero
    public var usingMetalFX: Bool = false
    public var depthLo: Float = 0
    public var depthHi: Float = 0
    public var metricActive: Bool = false
    /// Effective world scale in metric mode (auto or manual).
    public var worldScale: Float = 1
    /// Nearest content depth (m) as estimated this frame, metric mode.
    public var nearestM: Float = 0
    public init() {}
}

public enum RendererError: Error, CustomStringConvertible {
    case noDevice, shaderSource, function(String), textureCache
    public var description: String {
        switch self {
        case .noDevice: return "No Metal device"
        case .shaderSource: return "Shaders.metal resource missing"
        case .function(let n): return "Missing shader function \(n)"
        case .textureCache: return "CVMetalTextureCache creation failed"
        }
    }
}

/// A model-input slot: an IOSurface-backed pixel buffer the GPU writes into and Core ML reads from.
public final class ModelInputSlot: @unchecked Sendable {
    public let index: Int
    public let pixelBuffer: CVPixelBuffer
    let texture: MTLTexture
    var busy = false
    /// Region of the model input that holds picture (normalized); the rest are letterbox bars.
    public internal(set) var contentRect = CGRect(x: 0, y: 0, width: 1, height: 1)
    init(index: Int, pixelBuffer: CVPixelBuffer, texture: MTLTexture) {
        self.index = index; self.pixelBuffer = pixelBuffer; self.texture = texture
    }
}

/// Metal pipeline: capture texture → 16:9 crop at output resolution → model input → depth →
/// disparity → stereo pair → presentation on any number of CAMetalLayers.
public final class StereoRenderer: @unchecked Sendable {
    public let device: MTLDevice
    private let queue: MTLCommandQueue
    private let cropPSO: MTLComputePipelineState
    private let depthPSO: MTLComputePipelineState
    private let upsamplePSO: MTLComputePipelineState
    private let stereoPSO: MTLComputePipelineState
    private let presentPSO: MTLRenderPipelineState
    private var textureCache: CVMetalTextureCache
    private let log = Logger(subsystem: "real3d", category: "render")

    private let lock = NSLock()
    private var _settings = RenderSettings()
    private var _crop = CGRect(x: 0, y: 0, width: 1, height: 1)
    private var _latestSource: CVPixelBuffer?
    private var _sourceRect = CGRect(x: 0, y: 0, width: 1, height: 1) // valid region, normalized
    private var _sourceDirty = false
    private var _latestDepth: DepthResult?
    private var _depthDirty = false
    private var _paramsDirty = true
    private var _stats = RenderStats()
    /// Affine calibration of the model's inverse depth to metric 1/Z (metric mode, relative models).
    private var _calibA: Float = 1, _calibB: Float = 0, _calibValid = false

    /// Sets 1/Z = a * raw + b for the current model (raw = model output after optional inversion).
    public func setCalibration(a: Float, b: Float, valid: Bool) {
        lock.withLock { _calibA = a; _calibB = b; _calibValid = valid; _paramsDirty = true }
    }

    public var settings: RenderSettings {
        get { lock.withLock { _settings } }
        set { lock.withLock { if _settings != newValue { _settings = newValue; _paramsDirty = true } } }
    }
    /// Crop rectangle in normalized source coordinates (origin top-left).
    public var crop: CGRect {
        get { lock.withLock { _crop } }
        set { lock.withLock { if _crop != newValue { _crop = newValue; _paramsDirty = true } } }
    }
    public var stats: RenderStats { lock.withLock { _stats } }

    // Model I/O configuration
    private var modelInputWidth = 518, modelInputHeight = 392
    private var modelInputFormat: OSType = kCVPixelFormatType_32BGRA
    private var slots: [ModelInputSlot] = []
    /// Called on the render thread when a model input has been written by the GPU.
    public var onModelInputReady: ((ModelInputSlot) -> Void)?

    // Textures
    private var monoTex: MTLTexture!
    private var cropNativeTex: MTLTexture?
    private var depthNormA: MTLTexture!
    private var depthNormB: MTLTexture!
    private var depthPing = false
    private var depthHasPrev = false
    private var lastMetricMode = false
    private var autoScale: Float = 1
    private var nearestM: Float = 0
    private var dispTex: MTLTexture!
    private var sbsTex: MTLTexture!
    private var sourceTexRef: CVMetalTexture?
    private var sourceTexture: MTLTexture?
    private var spatialScaler: MTLFXSpatialScaler?
    private var scalerInputSize = (0, 0)
    private var scalerOutputSize = (0, 0)

    // Counters
    private var renderFrames = 0, captureFrames = 0, depthFrames = 0
    private var statsWindowStart = CACurrentMediaTime()
    private var gpuMsAccum = 0.0, gpuSamples = 0
    private var lastInferenceMs = 0.0
    private var depthLo: Float = 0, depthHi: Float = 1

    public init(device: MTLDevice? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else { throw RendererError.noDevice }
        self.device = device
        guard let queue = device.makeCommandQueue() else { throw RendererError.noDevice }
        self.queue = queue
        let source = ShaderSource.metal
        let opts = MTLCompileOptions()
        if #available(macOS 15.0, *) { opts.mathMode = .fast } else { opts.fastMathEnabled = true }
        let library = try device.makeLibrary(source: source, options: opts)
        func compute(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else { throw RendererError.function(name) }
            return try device.makeComputePipelineState(function: f)
        }
        cropPSO = try compute("cropResample")
        depthPSO = try compute("depthProcess")
        upsamplePSO = try compute("depthUpsample")
        stereoPSO = try compute("stereoSynth")
        let rp = MTLRenderPipelineDescriptor()
        rp.vertexFunction = library.makeFunction(name: "presentVertex")
        rp.fragmentFunction = library.makeFunction(name: "presentFragment")
        rp.colorAttachments[0].pixelFormat = .bgra8Unorm
        presentPSO = try device.makeRenderPipelineState(descriptor: rp)
        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(nil, nil, device, nil, &cache) == kCVReturnSuccess, let cache else {
            throw RendererError.textureCache
        }
        textureCache = cache
        try allocateOutputTextures()
        try configureModel(inputWidth: 518, inputHeight: 392, inputFormat: kCVPixelFormatType_32BGRA, outputWidth: 518, outputHeight: 392)
    }

    // MARK: Configuration

    private func makeTexture(_ format: MTLPixelFormat, _ w: Int, _ h: Int, usage: MTLTextureUsage = [.shaderRead, .shaderWrite, .renderTarget]) -> MTLTexture? {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: max(w, 1), height: max(h, 1), mipmapped: false)
        d.usage = usage
        d.storageMode = .private
        return device.makeTexture(descriptor: d)
    }

    private func allocateOutputTextures() throws {
        let s = settings
        monoTex = makeTexture(.bgra8Unorm, s.outputWidth, s.outputHeight)
        dispTex = makeTexture(.r16Float, s.outputWidth, s.outputHeight)
        sbsTex = makeTexture(.bgra8Unorm, s.outputWidth * 2, s.outputHeight)
        spatialScaler = nil
        cropNativeTex = nil
    }

    /// (Re)creates model input slots and depth textures for a model's I/O contract.
    public func configureModel(inputWidth: Int, inputHeight: Int, inputFormat: OSType, outputWidth: Int, outputHeight: Int) throws {
        lock.lock(); defer { lock.unlock() }
        modelInputWidth = inputWidth; modelInputHeight = inputHeight; modelInputFormat = inputFormat
        slots = []
        for i in 0..<3 {
            let pb = try DepthEngine.makePixelBuffer(width: inputWidth, height: inputHeight, format: inputFormat)
            guard let tex = metalTexture(for: pb, format: .bgra8Unorm)?.1 else { throw RendererError.textureCache }
            slots.append(ModelInputSlot(index: i, pixelBuffer: pb, texture: tex))
        }
        depthNormA = makeTexture(.r16Float, outputWidth, outputHeight)
        depthNormB = makeTexture(.r16Float, outputWidth, outputHeight)
        depthHasPrev = false
        _latestDepth = nil
        _paramsDirty = true
    }

    public func resetTemporal() { lock.withLock { depthHasPrev = false } }

    // MARK: Inputs (any thread)

    /// `contentRect` is the valid region of the buffer in pixels (window captures can be letterboxed).
    public func submitSource(_ pb: CVPixelBuffer, contentRect: CGRect? = nil) {
        let w = CGFloat(CVPixelBufferGetWidth(pb)), h = CGFloat(CVPixelBufferGetHeight(pb))
        let r = contentRect ?? CGRect(x: 0, y: 0, width: w, height: h)
        lock.withLock {
            _latestSource = pb
            _sourceRect = CGRect(x: r.minX / w, y: r.minY / h, width: r.width / w, height: r.height / h)
            _sourceDirty = true
            captureFrames += 1
        }
    }

    public func submitDepth(_ r: DepthResult) {
        lock.withLock {
            _latestDepth = r
            _depthDirty = true
            depthFrames += 1
            lastInferenceMs = r.inferenceMs
        }
    }

    public func releaseSlot(_ slot: ModelInputSlot) {
        lock.withLock { slot.busy = false }
    }

    // MARK: Rendering (render thread)

    private func metalTexture(for pb: CVPixelBuffer, format: MTLPixelFormat) -> (CVMetalTexture, MTLTexture)? {
        var ref: CVMetalTexture?
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let st = CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, pb, nil, format, w, h, 0, &ref)
        guard st == kCVReturnSuccess, let ref, let tex = CVMetalTextureGetTexture(ref) else { return nil }
        return (ref, tex)
    }

    private func depthFormat(for pb: CVPixelBuffer) -> MTLPixelFormat {
        CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_OneComponent16Half ? .r16Float : .r32Float
    }

    private func dispatch(_ enc: MTLComputeCommandEncoder, _ pso: MTLComputePipelineState, width: Int, height: Int, tg: MTLSize = MTLSize(width: 16, height: 16, depth: 1)) {
        enc.setComputePipelineState(pso)
        let grid = MTLSize(width: (width + tg.width - 1) / tg.width, height: (height + tg.height - 1) / tg.height, depth: 1)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
    }

    /// Runs the pipeline for one display refresh and presents into each layer.
    /// Returns false if nothing was drawn.
    @discardableResult
    public func renderFrame(layers: [(CAMetalLayer, PresentMode)]) -> Bool {
        // Snapshot state under the lock.
        lock.lock()
        let s = _settings
        let crop = _crop
        let source = _latestSource
        let srcRect = _sourceRect
        let sourceDirty = _sourceDirty; _sourceDirty = false
        let depth = _latestDepth
        let depthDirty = _depthDirty; _depthDirty = false
        let paramsDirty = _paramsDirty; _paramsDirty = false
        let calibA = _calibA, calibB = _calibB, calibValid = _calibValid
        var freeSlot: ModelInputSlot? = nil
        if sourceDirty, let fs = slots.first(where: { !$0.busy }) { fs.busy = true; freeSlot = fs }
        let guideSlot = (depth.map { $0.tag } ?? -1) >= 0 && (depth?.tag ?? 0) < slots.count ? slots[depth!.tag] : nil
        lock.unlock()

        if s.outputWidth != monoTex.width || s.outputHeight != monoTex.height {
            try? allocateOutputTextures()
        }
        guard let source else {
            if let freeSlot { releaseSlot(freeSlot) }
            return false
        }
        guard let cb = queue.makeCommandBuffer() else {
            if let freeSlot { releaseSlot(freeSlot) }
            return false
        }
        var retain: [Any] = []

        // Source texture (zero-copy view of the capture IOSurface).
        if sourceDirty || sourceTexture == nil {
            if let (ref, tex) = metalTexture(for: source, format: .bgra8Unorm) { sourceTexRef = ref; sourceTexture = tex }
        }
        guard let srcTex = sourceTexture else {
            if let freeSlot { releaseSlot(freeSlot) }
            return false
        }
        retain.append(source)
        if let sourceTexRef { retain.append(sourceTexRef) }

        let srcW = Float(srcTex.width) * Float(srcRect.width), srcH = Float(srcTex.height) * Float(srcRect.height)
        let cropPxW = Float(crop.width) * srcW, cropPxH = Float(crop.height) * srcH
        // Crop in normalized coordinates of the full buffer.
        let realUV = SIMD2<Float>(Float(srcRect.minX + crop.minX * srcRect.width), Float(srcRect.minY + crop.minY * srcRect.height))
        let realUVSize = SIMD2<Float>(Float(crop.width * srcRect.width), Float(crop.height * srcRect.height))
        // A crop that isn't 16:9 is centered in a 16:9 frame with black bars (letterbox / pillarbox).
        let outAspect = Float(s.outputWidth) / Float(s.outputHeight)
        let cropAspect = cropPxW / max(cropPxH, 1)
        let letterboxed = abs(cropAspect / outAspect - 1) > 0.005
        var effPxW = cropPxW, effPxH = cropPxH
        if letterboxed { if cropAspect > outAspect { effPxH = cropPxW / outAspect } else { effPxW = cropPxH * outAspect } }
        let cropUVSize = letterboxed ? realUVSize * SIMD2(effPxW / max(cropPxW, 1), effPxH / max(cropPxH, 1)) : realUVSize
        let cropUV = letterboxed ? realUV + (realUVSize - cropUVSize) * 0.5 : realUV
        var validMin = SIMD2<Float>(-1e4, -1e4), validMax = SIMD2<Float>(1e4, 1e4)
        var contentMin = SIMD2<Float>(-1, -1), contentMax = SIMD2<Float>(2, 2)
        var contentRect = CGRect(x: 0, y: 0, width: 1, height: 1)
        if letterboxed {
            validMin = realUV; validMax = realUV + realUVSize
            contentMin = (realUV - cropUV) / cropUVSize
            contentMax = (realUV + realUVSize - cropUV) / cropUVSize
            contentRect = CGRect(x: CGFloat(contentMin.x), y: CGFloat(contentMin.y),
                                 width: CGFloat(contentMax.x - contentMin.x), height: CGFloat(contentMax.y - contentMin.y))
        }
        let upscaling = s.useMetalFX && effPxW < Float(s.outputWidth) && effPxH < Float(s.outputHeight) && effPxW >= 64 && effPxH >= 36
        var usedMetalFX = false

        // 1. New depth map → normalize + EMA; keep a copy of the model input that produced it as the JBU guide.
        if depthDirty, let depth, let (ref, rawTex) = metalTexture(for: depth.pixelBuffer, format: depthFormat(for: depth.pixelBuffer)) {
            retain.append(ref); retain.append(depth.pixelBuffer)
            if rawTex.width != depthNormA.width || rawTex.height != depthNormA.height {
                depthNormA = makeTexture(.r16Float, rawTex.width, rawTex.height)
                depthNormB = makeTexture(.r16Float, rawTex.width, rawTex.height)
                depthHasPrev = false
            }
            if depthHasPrev {
                depthLo += (depth.lo - depthLo) * s.temporalAlpha
                depthHi += (depth.hi - depthHi) * s.temporalAlpha
            } else {
                depthLo = depth.lo; depthHi = depth.hi
            }
            let prev = depthPing ? depthNormA! : depthNormB!
            let out = depthPing ? depthNormB! : depthNormA!
            let metric = s.depthScaleMode == .metric && (s.modelIsMetric || calibValid)
            if metric != lastMetricMode { depthHasPrev = false; lastMetricMode = metric; autoScale = 1 }
            if metric {
                // Nearest content (98th percentile of 1/Z) drives the automatic world scale.
                let invZnear = max(s.modelIsMetric ? depthHi : calibA * depthHi + calibB, 1e-4)
                nearestM = 1 / invZnear
                let pitch = s.viewer.screenWidthM / Float(s.outputWidth), ipd = s.viewer.ipdMM / 1000
                let A = ipd / pitch, B = ipd * s.viewer.screenDistanceM / pitch
                let target = max(1, B * invZnear / max(A + s.popBudgetPx, 1))
                autoScale += (target - autoScale) * (depthHasPrev ? 0.15 : 1)
            }
            var p = DepthParams(lo: depthLo, hi: depthHi, alpha: s.temporalAlpha, invert: s.invertDepth ? 1 : 0,
                                hasPrev: depthHasPrev ? 1 : 0, width: UInt32(rawTex.width), height: UInt32(rawTex.height),
                                metric: metric ? 1 : 0, calibA: s.modelIsMetric ? 1 : calibA, calibB: s.modelIsMetric ? 0 : calibB)
            let enc = cb.makeComputeCommandEncoder()!
            enc.setTexture(rawTex, index: 0); enc.setTexture(prev, index: 1); enc.setTexture(out, index: 2)
            enc.setBytes(&p, length: MemoryLayout<DepthParams>.stride, index: 0)
            dispatch(enc, depthPSO, width: rawTex.width, height: rawTex.height)
            enc.endEncoding()
            depthPing.toggle(); depthHasPrev = true
            if let guideSlot {
                if guideTex == nil || guideTex!.width != guideSlot.texture.width || guideTex!.height != guideSlot.texture.height {
                    guideTex = makeTexture(.bgra8Unorm, guideSlot.texture.width, guideSlot.texture.height)
                }
                let blit = cb.makeBlitCommandEncoder()!
                blit.copy(from: guideSlot.texture, to: guideTex!)
                blit.endEncoding()
            }
        }

        // 2. Crop → mono at output resolution (MetalFX spatial upscale when the crop is smaller than the output).
        if sourceDirty || paramsDirty {
            let enc = cb.makeComputeCommandEncoder()!
            if upscaling, let scaler = spatialScaler(inW: Int(effPxW.rounded()), inH: Int(effPxH.rounded())), let native = cropNativeTex {
                var p = CropParams(uvOrigin: cropUV, uvSize: cropUVSize,
                                   outWidth: UInt32(native.width), outHeight: UInt32(native.height), supersample: 1,
                                   validMin: validMin, validMax: validMax)
                enc.setTexture(srcTex, index: 0); enc.setTexture(native, index: 1)
                enc.setBytes(&p, length: MemoryLayout<CropParams>.stride, index: 0)
                dispatch(enc, cropPSO, width: native.width, height: native.height)
                enc.endEncoding()
                scaler.colorTexture = native
                scaler.outputTexture = monoTex
                scaler.encode(commandBuffer: cb)
                usedMetalFX = true
            } else {
                let ss = min(4, max(1, Int((effPxW / Float(s.outputWidth)).rounded(.up))))
                var p = CropParams(uvOrigin: cropUV, uvSize: cropUVSize,
                                   outWidth: UInt32(s.outputWidth), outHeight: UInt32(s.outputHeight), supersample: UInt32(ss),
                                   validMin: validMin, validMax: validMax)
                enc.setTexture(srcTex, index: 0); enc.setTexture(monoTex, index: 1)
                enc.setBytes(&p, length: MemoryLayout<CropParams>.stride, index: 0)
                dispatch(enc, cropPSO, width: s.outputWidth, height: s.outputHeight)
                enc.endEncoding()
            }
        }

        // 3. Model input from the mono frame (only for new capture frames, when a slot is free).
        if let slot = freeSlot {
            let enc = cb.makeComputeCommandEncoder()!
            let ss = min(4, max(1, Int((Float(s.outputWidth) / Float(modelInputWidth)).rounded(.up))))
            var p = CropParams(uvOrigin: SIMD2(0, 0), uvSize: SIMD2(1, 1), outWidth: UInt32(modelInputWidth), outHeight: UInt32(modelInputHeight), supersample: UInt32(ss))
            slot.contentRect = contentRect
            enc.setTexture(monoTex, index: 0); enc.setTexture(slot.texture, index: 1)
            enc.setBytes(&p, length: MemoryLayout<CropParams>.stride, index: 0)
            dispatch(enc, cropPSO, width: modelInputWidth, height: modelInputHeight)
            enc.endEncoding()
            retain.append(slot.pixelBuffer)
            cb.addCompletedHandler { [weak self] _ in self?.onModelInputReady?(slot) }
        }

        // 4. Depth → disparity at output resolution, then stereo synthesis.
        if sourceDirty || depthDirty || paramsDirty {
            if depthHasPrev {
                let cur = depthPing ? depthNormA! : depthNormB!
                let enc = cb.makeComputeCommandEncoder()!
                let pitch = s.viewer.screenWidthM / Float(s.outputWidth)
                let ipd = s.viewer.ipdMM / 1000
                var up = UpsampleParams(outWidth: UInt32(s.outputWidth), outHeight: UInt32(s.outputHeight),
                                        mode: UInt32(guideTex == nil ? 0 : s.upsampleMode.rawValue),
                                        radius: Int32(s.jbuRadius), sigmaSpatial: 1.0, sigmaRange: s.jbuSigmaRange,
                                        strength: s.strength, convergence: s.convergence, gamma: s.gamma,
                                        metric: lastMetricMode ? 1 : 0, metricA: ipd / pitch, metricB: ipd * s.viewer.screenDistanceM / pitch,
                                        worldScale: s.autoWorldScale ? autoScale : s.worldScale, maxCrossed: s.maxCrossedPx,
                                        contentMin: contentMin, contentMax: contentMax)
                enc.setTexture(cur, index: 0); enc.setTexture(guideTex ?? cur, index: 1); enc.setTexture(monoTex, index: 2); enc.setTexture(dispTex, index: 3)
                enc.setBytes(&up, length: MemoryLayout<UpsampleParams>.stride, index: 0)
                dispatch(enc, upsamplePSO, width: s.outputWidth, height: s.outputHeight)

                let searchRange: Float = lastMetricMode ? max(s.maxCrossedPx, ipd / pitch) : abs(s.strength)
                var st = StereoParams(width: UInt32(s.outputWidth), height: UInt32(s.outputHeight), maxDisp: max(searchRange, 1),
                                      stretchThresh: s.stretchThreshold, swapEyes: s.swapEyes ? 1 : 0)
                enc.setTexture(monoTex, index: 0); enc.setTexture(dispTex, index: 1); enc.setTexture(sbsTex, index: 2)
                enc.setBytes(&st, length: MemoryLayout<StereoParams>.stride, index: 0)
                dispatch(enc, stereoPSO, width: s.outputWidth * 2, height: s.outputHeight, tg: MTLSize(width: 64, height: 4, depth: 1))
                enc.endEncoding()
            } else {
                // No depth yet: mono in both eyes.
                let blit = cb.makeBlitCommandEncoder()!
                let size = MTLSize(width: monoTex.width, height: monoTex.height, depth: 1)
                blit.copy(from: monoTex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: size,
                          to: sbsTex, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
                blit.copy(from: monoTex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: size,
                          to: sbsTex, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin(x: monoTex.width, y: 0, z: 0))
                blit.endEncoding()
            }
        }

        // 5. Present into each visible layer.
        var drawables: [CAMetalDrawable] = []
        for (layer, mode) in layers {
            guard layer.drawableSize.width > 1, layer.drawableSize.height > 1, let drawable = layer.nextDrawable() else { continue }
            let rpd = MTLRenderPassDescriptor()
            rpd.colorAttachments[0].texture = drawable.texture
            rpd.colorAttachments[0].loadAction = .clear
            rpd.colorAttachments[0].storeAction = .store
            rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            guard let enc = cb.makeRenderCommandEncoder(descriptor: rpd) else { continue }
            enc.setRenderPipelineState(presentPSO)
            let tex: MTLTexture
            let contentAspect: Float
            switch mode {
            case .source: tex = srcTex; contentAspect = srcW / srcH
            case .mono: tex = monoTex; contentAspect = Float(s.outputWidth) / Float(s.outputHeight)
            case .fullSBS: tex = sbsTex; contentAspect = Float(s.outputWidth * 2) / Float(s.outputHeight)
            case .halfSBS, .anaglyph, .depth, .leftEye, .rightEye: tex = sbsTex; contentAspect = Float(s.outputWidth) / Float(s.outputHeight)
            }
            let viewAspect = Float(drawable.texture.width) / Float(drawable.texture.height)
            let scale: SIMD2<Float> = contentAspect > viewAspect ? SIMD2(1, viewAspect / contentAspect) : SIMD2(contentAspect / viewAspect, 1)
            var pp = PresentParams(scale: scale, uvOrigin: SIMD2(0, 0), uvSize: SIMD2(1, 1), mode: UInt32(mode.rawValue))
            if mode == .source {
                pp.uvOrigin = SIMD2(Float(srcRect.minX), Float(srcRect.minY))
                pp.uvSize = SIMD2(Float(srcRect.width), Float(srcRect.height))
            }
            enc.setVertexBytes(&pp, length: MemoryLayout<PresentParams>.stride, index: 0)
            enc.setFragmentBytes(&pp, length: MemoryLayout<PresentParams>.stride, index: 0)
            enc.setFragmentTexture(tex, index: 0)
            enc.setFragmentTexture(dispTex, index: 1)
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            enc.endEncoding()
            drawables.append(drawable)
        }
        for d in drawables { cb.present(d) }
        let retained = retain
        cb.addCompletedHandler { [weak self] cb in
            _ = retained
            guard let self else { return }
            let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            self.lock.withLock { self.gpuMsAccum += ms; self.gpuSamples += 1 }
        }
        cb.commit()
        updateStats(sourceSize: CGSize(width: Double(srcW), height: Double(srcH)), cropPx: CGSize(width: Double(cropPxW), height: Double(cropPxH)), metalFX: usedMetalFX)
        return true
    }

    private var guideTex: MTLTexture?

    /// Synchronously reads back a pipeline texture as BGRA bytes (debug / snapshot). Call after renderFrame.
    public enum SnapshotKind { case sbs, mono, anaglyph, disparity }
    public func snapshot(_ kind: SnapshotKind) -> (width: Int, height: Int, bgra: [UInt8])? {
        let s = settings
        let w = kind == .sbs ? s.outputWidth * 2 : s.outputWidth, h = s.outputHeight
        guard let target = makeTexture(.bgra8Unorm, w, h), let cb = queue.makeCommandBuffer() else { return nil }
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = target
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].storeAction = .store
        guard let enc = cb.makeRenderCommandEncoder(descriptor: rpd) else { return nil }
        enc.setRenderPipelineState(presentPSO)
        let mode: PresentMode = kind == .sbs ? .fullSBS : kind == .mono ? .mono : kind == .anaglyph ? .anaglyph : .depth
        var pp = PresentParams(scale: SIMD2(1, 1), uvOrigin: SIMD2(0, 0), uvSize: SIMD2(1, 1), mode: UInt32(mode.rawValue))
        enc.setVertexBytes(&pp, length: MemoryLayout<PresentParams>.stride, index: 0)
        enc.setFragmentBytes(&pp, length: MemoryLayout<PresentParams>.stride, index: 0)
        enc.setFragmentTexture(kind == .mono ? monoTex : sbsTex, index: 0)
        enc.setFragmentTexture(dispTex, index: 1)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
        guard let buf = device.makeBuffer(length: w * h * 4, options: .storageModeShared), let blit = cb.makeBlitCommandEncoder() else { return nil }
        blit.copy(from: target, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: MTLSize(width: w, height: h, depth: 1),
                  to: buf, destinationOffset: 0, destinationBytesPerRow: w * 4, destinationBytesPerImage: w * h * 4)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let bytes = [UInt8](UnsafeBufferPointer(start: buf.contents().assumingMemoryBound(to: UInt8.self), count: w * h * 4))
        return (w, h, bytes)
    }

    private func spatialScaler(inW: Int, inH: Int) -> MTLFXSpatialScaler? {
        let s = settings
        if let sc = spatialScaler, scalerInputSize == (inW, inH), scalerOutputSize == (s.outputWidth, s.outputHeight) { return sc }
        guard MTLFXSpatialScalerDescriptor.supportsDevice(device) else { return nil }
        let d = MTLFXSpatialScalerDescriptor()
        d.inputWidth = inW; d.inputHeight = inH
        d.outputWidth = s.outputWidth; d.outputHeight = s.outputHeight
        d.colorTextureFormat = .bgra8Unorm
        d.outputTextureFormat = .bgra8Unorm
        d.colorProcessingMode = .perceptual
        guard let sc = d.makeSpatialScaler(device: device) else { return nil }
        cropNativeTex = makeTexture(.bgra8Unorm, inW, inH)
        spatialScaler = sc
        scalerInputSize = (inW, inH); scalerOutputSize = (s.outputWidth, s.outputHeight)
        return sc
    }

    private func updateStats(sourceSize: CGSize, cropPx: CGSize, metalFX: Bool) {
        lock.withLock {
            renderFrames += 1
            let now = CACurrentMediaTime()
            let dt = now - statsWindowStart
            _stats.sourceSize = sourceSize
            _stats.cropPixelSize = cropPx
            _stats.usingMetalFX = metalFX
            _stats.inferenceMs = lastInferenceMs
            _stats.depthLo = depthLo; _stats.depthHi = depthHi
            _stats.metricActive = lastMetricMode
            _stats.worldScale = _settings.autoWorldScale ? autoScale : _settings.worldScale
            _stats.nearestM = nearestM
            if dt >= 0.5 {
                _stats.renderFPS = Double(renderFrames) / dt
                _stats.captureFPS = Double(captureFrames) / dt
                _stats.depthFPS = Double(depthFrames) / dt
                _stats.gpuMs = gpuSamples > 0 ? gpuMsAccum / Double(gpuSamples) : 0
                renderFrames = 0; captureFrames = 0; depthFrames = 0; gpuMsAccum = 0; gpuSamples = 0
                statsWindowStart = now
            }
        }
    }
}
