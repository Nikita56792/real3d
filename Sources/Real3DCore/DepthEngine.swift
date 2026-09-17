import CoreML
import QuartzCore
import CoreVideo
import Foundation
import IOSurface
import Metal
import os

/// Describes the I/O contract of a loaded Core ML depth model.
public struct DepthModelInfo: Sendable {
    public var name: String
    public var inputName: String
    public var inputWidth: Int
    public var inputHeight: Int
    /// `true` if the model takes an image input (CVPixelBuffer); `false` for a [1,3,H,W] multiarray.
    public var inputIsImage: Bool
    public var inputPixelFormat: OSType
    public var inputArrayType: MLMultiArrayDataType
    public var outputName: String
    public var outputIsImage: Bool
    public var outputWidth: Int
    public var outputHeight: Int
    public var outputPixelFormat: OSType
    public var outputArrayType: MLMultiArrayDataType
    public var outputShape: [Int]
    public var allOutputs: [String] = []
    public var description: String {
        let inKind = inputIsImage ? "image(\(fourcc(inputPixelFormat)))" : "array(\(inputArrayType))"
        let outKind = outputIsImage ? "image(\(fourcc(outputPixelFormat)))" : "array(\(outputArrayType) \(outputShape))"
        return "\(name): in '\(inputName)' \(inputWidth)x\(inputHeight) \(inKind) -> out '\(outputName)' \(outputWidth)x\(outputHeight) \(outKind)"
    }
}

func fourcc(_ t: OSType) -> String {
    let c = [24, 16, 8, 0].map { Character(UnicodeScalar(UInt8((t >> UInt32($0)) & 0xff))) }
    return String(c)
}

/// One inference result. `pixelBuffer` is a single-channel float16/float32 IOSurface-backed buffer
/// of size outputWidth x outputHeight containing the raw model output (relative inverse depth).
public final class DepthResult: @unchecked Sendable {
    public let pixelBuffer: CVPixelBuffer
    /// Robust low/high percentiles of the raw output for normalization.
    public let lo: Float
    public let hi: Float
    public let inferenceMs: Double
    public let sequence: UInt64
    /// Caller-defined tag (the renderer uses it to find the model-input slot that produced this result).
    public var tag: Int = -1
    init(pixelBuffer: CVPixelBuffer, lo: Float, hi: Float, inferenceMs: Double, sequence: UInt64) {
        self.pixelBuffer = pixelBuffer
        self.lo = lo
        self.hi = hi
        self.inferenceMs = inferenceMs
        self.sequence = sequence
    }
}

public enum DepthEngineError: Error, CustomStringConvertible {
    case noImageInput, noOutput, unsupportedInput(String), unsupportedOutput(String), allocation(String)
    public var description: String {
        switch self {
        case .noImageInput: return "Model has no usable input"
        case .noOutput: return "Model has no usable output"
        case .unsupportedInput(let s): return "Unsupported input: \(s)"
        case .unsupportedOutput(let s): return "Unsupported output: \(s)"
        case .allocation(let s): return "Allocation failed: \(s)"
        }
    }
}

/// Generic Core ML monocular-depth runner. Loads any model with a single image-like input and a
/// single 2D output (image or [1,1,H,W]/[1,H,W]/[H,W] multiarray). Outputs are delivered as
/// IOSurface-backed CVPixelBuffers so the renderer can read them through CVMetalTextureCache
/// with zero copies.
public final class DepthEngine: @unchecked Sendable {
    public let model: MLModel
    public let info: DepthModelInfo
    public let computeUnits: MLComputeUnits

    private let log = Logger(subsystem: "real3d", category: "depth")
    private var sequence: UInt64 = 0
    /// Set to true for models that output depth (larger = farther); statistics are then taken on 1/v.
    public var outputIsDepth = false

    // Output backings: pool of pixel buffers (image outputs) or a multiarray + staging pixel buffers.
    private var outputPool: [CVPixelBuffer] = []
    private var outputPoolIndex = 0
    private var outputArray: MLMultiArray?

    // Multiarray input support: planar float buffer + normalization.
    private var inputArray: MLMultiArray?
    private let imagenetMean: [Float] = [0.485, 0.456, 0.406]
    private let imagenetStd: [Float] = [0.229, 0.224, 0.225]

    /// Compiles (if needed) and loads a model from a .mlpackage / .mlmodel / .mlmodelc URL.
    /// `fastPrediction` asks Core ML to specialize the graph for latency (longer first load).
    public init(modelURL: URL, computeUnits: MLComputeUnits = .all, cacheDirectory: URL? = nil,
                fastPrediction: Bool = false, lowPrecisionAccumulation: Bool = false) throws {
        let compiledURL = try Self.compiledModelURL(for: modelURL, cacheDirectory: cacheDirectory)
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        if lowPrecisionAccumulation { config.allowLowPrecisionAccumulationOnGPU = true }
        if fastPrediction, #available(macOS 15.0, *) {
            var hints = MLOptimizationHints()
            hints.reshapeFrequency = .infrequent
            hints.specializationStrategy = .fastPrediction
            config.optimizationHints = hints
        }
        self.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: compiledURL, configuration: config)
        self.info = try Self.inspect(model: model, name: modelURL.deletingPathExtension().lastPathComponent)
        try allocateBackings()
    }

    /// Median latency (ms) of `runs` predictions on a blank input after `warmup` predictions.
    public static func measureLatency(modelURL: URL, computeUnits: MLComputeUnits, warmup: Int = 3, runs: Int = 7) throws -> Double {
        let e = try DepthEngine(modelURL: modelURL, computeUnits: computeUnits)
        let input = try makePixelBuffer(width: e.info.inputWidth, height: e.info.inputHeight, format: e.preferredInputPixelFormat)
        for _ in 0..<warmup { _ = try e.predict(input) }
        var t: [Double] = []
        for _ in 0..<runs { t.append(try e.predict(input).inferenceMs) }
        return t.sorted()[t.count / 2]
    }

    /// Returns a compiled .mlmodelc URL, compiling into the cache directory if the source is a package.
    public static func compiledModelURL(for url: URL, cacheDirectory: URL?) throws -> URL {
        if url.pathExtension == "mlmodelc" { return url }
        let cacheDir = cacheDirectory ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Real3D/CompiledModels", isDirectory: true)
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        // Cache key: name + modification date + size of the weights so a replaced model recompiles.
        let attrs = (try? FileManager.default.attributesOfItem(atPath: url.path)) ?? [:]
        let mod = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        var weightSize: UInt64 = 0
        if let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) {
            for case let f as URL in e {
                weightSize &+= UInt64((try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            }
        }
        let key = "\(url.deletingPathExtension().lastPathComponent)-\(Int(mod))-\(weightSize).mlmodelc"
        let dest = cacheDir.appendingPathComponent(key)
        if FileManager.default.fileExists(atPath: dest.path) { return dest }
        let tmp = try MLModel.compileModel(at: url)
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.moveItem(at: tmp, to: dest)
        return dest
    }

    static func inspect(model: MLModel, name: String) throws -> DepthModelInfo {
        let desc = model.modelDescription
        // Input: prefer image; else a [1,3,H,W] / [3,H,W] multiarray.
        var inName = "", inW = 0, inH = 0, inIsImage = false
        var inFmt: OSType = kCVPixelFormatType_32BGRA
        var inArr: MLMultiArrayDataType = .float32
        for (n, d) in desc.inputDescriptionsByName {
            if d.type == .image, let c = d.imageConstraint {
                inName = n; inW = c.pixelsWide; inH = c.pixelsHigh; inIsImage = true; inFmt = c.pixelFormatType
                break
            }
        }
        if inName.isEmpty {
            for (n, d) in desc.inputDescriptionsByName {
                if d.type == .multiArray, let c = d.multiArrayConstraint {
                    let shape = c.shape.map { $0.intValue }
                    guard shape.count >= 3, shape[shape.count - 3] == 3 else { continue }
                    inName = n; inW = shape[shape.count - 1]; inH = shape[shape.count - 2]; inArr = c.dataType
                    break
                }
            }
        }
        guard !inName.isEmpty else { throw DepthEngineError.noImageInput }

        // Output: prefer image; else a multiarray whose last two dims are H,W.
        var outName = "", outW = 0, outH = 0, outIsImage = false
        var outFmt: OSType = kCVPixelFormatType_OneComponent16Half
        var outArr: MLMultiArrayDataType = .float16
        var outShape: [Int] = []
        for (n, d) in desc.outputDescriptionsByName {
            if d.type == .image, let c = d.imageConstraint {
                outName = n; outW = c.pixelsWide; outH = c.pixelsHigh; outIsImage = true; outFmt = c.pixelFormatType
                break
            }
        }
        if outName.isEmpty {
            // Pick the single-channel multiarray output with the largest H*W, preferring names that
            // look like depth over auxiliary outputs (confidence, mask, ...).
            var bestScore = -1
            for (n, d) in desc.outputDescriptionsByName {
                if d.type == .multiArray, let c = d.multiArrayConstraint {
                    let shape = c.shape.map { $0.intValue }
                    guard shape.count >= 2 else { continue }
                    let extra = shape.dropLast(2)
                    guard extra.allSatisfy({ $0 == 1 }) else { continue }
                    let hw = shape[shape.count - 1] * shape[shape.count - 2]
                    let ln = n.lowercased()
                    var score = hw
                    if ln.contains("depth") || ln.contains("disp") || ln.contains("inv") { score += 1 << 30 }
                    if ln.contains("conf") || ln.contains("mask") || ln.contains("sigma") { score -= 1 << 29 }
                    if score > bestScore {
                        bestScore = score
                        outName = n; outW = shape[shape.count - 1]; outH = shape[shape.count - 2]
                        outArr = c.dataType; outShape = shape
                    }
                }
            }
        }
        guard !outName.isEmpty else { throw DepthEngineError.noOutput }
        let allOutputs = desc.outputDescriptionsByName.map { n, d -> String in
            if d.type == .image, let c = d.imageConstraint { return "\(n): image \(c.pixelsWide)x\(c.pixelsHigh)" }
            if d.type == .multiArray, let c = d.multiArrayConstraint { return "\(n): array \(c.shape.map { $0.intValue })" }
            return "\(n): \(d.type)"
        }.sorted()
        var mi = DepthModelInfo(
            name: name, inputName: inName, inputWidth: inW, inputHeight: inH, inputIsImage: inIsImage,
            inputPixelFormat: inFmt, inputArrayType: inArr, outputName: outName, outputIsImage: outIsImage,
            outputWidth: outW, outputHeight: outH, outputPixelFormat: outFmt, outputArrayType: outArr,
            outputShape: outShape)
        mi.allOutputs = allOutputs
        return mi
    }

    private func allocateBackings() throws {
        let fmt = info.outputIsImage ? info.outputPixelFormat : kCVPixelFormatType_OneComponent32Float
        for _ in 0..<4 {
            outputPool.append(try Self.makePixelBuffer(width: info.outputWidth, height: info.outputHeight, format: fmt))
        }
        if !info.outputIsImage {
            outputArray = try MLMultiArray(shape: info.outputShape.map { NSNumber(value: $0) }, dataType: info.outputArrayType)
        }
        if !info.inputIsImage {
            inputArray = try MLMultiArray(shape: [1, 3, NSNumber(value: info.inputHeight), NSNumber(value: info.inputWidth)],
                                          dataType: info.inputArrayType)
        }
    }

    /// Creates an IOSurface-backed, Metal-compatible pixel buffer.
    public static func makePixelBuffer(width: Int, height: Int, format: OSType) throws -> CVPixelBuffer {
        let attrs: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferPixelFormatTypeKey: format,
        ]
        var pb: CVPixelBuffer?
        let st = CVPixelBufferCreate(kCFAllocatorDefault, width, height, format, attrs as CFDictionary, &pb)
        guard st == kCVReturnSuccess, let pb else { throw DepthEngineError.allocation("CVPixelBufferCreate \(st)") }
        return pb
    }

    /// The pixel format the caller must use for the model-input pixel buffer.
    public var preferredInputPixelFormat: OSType { info.inputIsImage ? info.inputPixelFormat : kCVPixelFormatType_32BGRA }

    /// Runs one inference on a BGRA/RGB pixel buffer of exactly inputWidth x inputHeight.
    /// Thread-safe only if called serially (one in flight).
    /// `statsRect` (normalized) limits the depth statistics to the picture area, excluding letterbox bars.
    public func predict(_ input: CVPixelBuffer, statsRect: CGRect? = nil) throws -> DepthResult {
        let t0 = CACurrentMediaTime()
        let provider: MLFeatureProvider
        if info.inputIsImage {
            provider = try MLDictionaryFeatureProvider(dictionary: [info.inputName: MLFeatureValue(pixelBuffer: input)])
        } else {
            fillInputArray(from: input)
            provider = try MLDictionaryFeatureProvider(dictionary: [info.inputName: MLFeatureValue(multiArray: inputArray!)])
        }
        let opts = MLPredictionOptions()
        let outPB = outputPool[outputPoolIndex]
        outputPoolIndex = (outputPoolIndex + 1) % outputPool.count
        if info.outputIsImage {
            opts.outputBackings = [info.outputName: outPB]
        } else if let arr = outputArray {
            opts.outputBackings = [info.outputName: arr]
        }
        let out = try model.prediction(from: provider, options: opts)
        guard let fv = out.featureValue(for: info.outputName) else { throw DepthEngineError.noOutput }

        var result = outPB
        if info.outputIsImage {
            if let pb = fv.imageBufferValue, pb !== outPB {
                // Backing was not honored; copy.
                copyPixels(from: pb, to: outPB)
            }
        } else if let arr = fv.multiArrayValue {
            copyArray(arr, to: outPB)
        } else {
            throw DepthEngineError.unsupportedOutput("no image/array value")
        }
        result = outPB
        let (lo, hi) = Self.robustRange(of: result, invert: outputIsDepth, rect: statsRect)
        let ms = (CACurrentMediaTime() - t0) * 1000
        sequence &+= 1
        return DepthResult(pixelBuffer: result, lo: lo, hi: hi, inferenceMs: ms, sequence: sequence)
    }

    // MARK: - Helpers

    private func fillInputArray(from pb: CVPixelBuffer) {
        guard let arr = inputArray else { return }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let w = info.inputWidth, h = info.inputHeight
        let stride = CVPixelBufferGetBytesPerRow(pb)
        guard let base = CVPixelBufferGetBaseAddress(pb)?.assumingMemoryBound(to: UInt8.self) else { return }
        let plane = w * h
        let inv255: Float = 1.0 / 255.0
        // BGRA -> planar RGB, normalized with ImageNet statistics.
        if info.inputArrayType == .float32 {
            let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
            for y in 0..<h {
                let row = base + y * stride
                for x in 0..<w {
                    let px = row + x * 4
                    let i = y * w + x
                    p[i] = (Float(px[2]) * inv255 - imagenetMean[0]) / imagenetStd[0]
                    p[plane + i] = (Float(px[1]) * inv255 - imagenetMean[1]) / imagenetStd[1]
                    p[2 * plane + i] = (Float(px[0]) * inv255 - imagenetMean[2]) / imagenetStd[2]
                }
            }
        } else {
            let p = arr.dataPointer.assumingMemoryBound(to: Float16.self)
            for y in 0..<h {
                let row = base + y * stride
                for x in 0..<w {
                    let px = row + x * 4
                    let i = y * w + x
                    p[i] = Float16((Float(px[2]) * inv255 - imagenetMean[0]) / imagenetStd[0])
                    p[plane + i] = Float16((Float(px[1]) * inv255 - imagenetMean[1]) / imagenetStd[1])
                    p[2 * plane + i] = Float16((Float(px[0]) * inv255 - imagenetMean[2]) / imagenetStd[2])
                }
            }
        }
    }

    private func copyPixels(from src: CVPixelBuffer, to dst: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(src, .readOnly); CVPixelBufferUnlockBaseAddress(dst, []) }
        guard let s = CVPixelBufferGetBaseAddress(src), let d = CVPixelBufferGetBaseAddress(dst) else { return }
        let sStride = CVPixelBufferGetBytesPerRow(src), dStride = CVPixelBufferGetBytesPerRow(dst)
        let rowBytes = min(sStride, dStride)
        for y in 0..<min(CVPixelBufferGetHeight(src), CVPixelBufferGetHeight(dst)) {
            memcpy(d + y * dStride, s + y * sStride, rowBytes)
        }
    }

    private func copyArray(_ arr: MLMultiArray, to dst: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(dst, []) }
        guard let d = CVPixelBufferGetBaseAddress(dst)?.assumingMemoryBound(to: Float.self) else { return }
        let w = info.outputWidth, h = info.outputHeight
        let dStride = CVPixelBufferGetBytesPerRow(dst) / 4
        let strides = arr.strides.map { $0.intValue }
        let rowStride = strides[strides.count - 2], colStride = strides[strides.count - 1]
        switch arr.dataType {
        case .float32:
            let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
            for y in 0..<h {
                let s = p + y * rowStride
                let o = d + y * dStride
                if colStride == 1 { memcpy(o, s, w * 4) } else { for x in 0..<w { o[x] = s[x * colStride] } }
            }
        case .float16:
            let p = arr.dataPointer.assumingMemoryBound(to: Float16.self)
            for y in 0..<h {
                let s = p + y * rowStride
                let o = d + y * dStride
                for x in 0..<w { o[x] = Float(s[x * colStride]) }
            }
        case .double:
            let p = arr.dataPointer.assumingMemoryBound(to: Double.self)
            for y in 0..<h {
                let s = p + y * rowStride
                let o = d + y * dStride
                for x in 0..<w { o[x] = Float(s[x * colStride]) }
            }
        default:
            break
        }
    }

    /// Subsampled 2%/98% percentiles of a single-channel float16/float32 pixel buffer.
    public static func robustRange(of pb: CVPixelBuffer, loPct: Float = 0.02, hiPct: Float = 0.98, invert: Bool = false, rect: CGRect? = nil) -> (Float, Float) {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return (0, 1) }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let stride = CVPixelBufferGetBytesPerRow(pb)
        let isHalf = CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_OneComponent16Half
        // Sampled region (whole map unless a picture rect excludes letterbox bars).
        var x0 = 0, x1 = w, y0 = 0, y1 = h
        if let rect, rect != CGRect(x: 0, y: 0, width: 1, height: 1) {
            x0 = min(max(Int((rect.minX * CGFloat(w)).rounded(.up)), 0), w - 1)
            x1 = min(max(Int((rect.maxX * CGFloat(w)).rounded(.down)), x0 + 1), w)
            y0 = min(max(Int((rect.minY * CGFloat(h)).rounded(.up)), 0), h - 1)
            y1 = min(max(Int((rect.maxY * CGFloat(h)).rounded(.down)), y0 + 1), h)
        }
        let step = max(1, Int((Float((x1 - x0) * (y1 - y0)) / 16000).squareRoot()))
        var samples: [Float] = []
        samples.reserveCapacity(((x1 - x0) / step + 1) * ((y1 - y0) / step + 1))
        var y = y0
        while y < y1 {
            let row = base + y * stride
            var x = x0
            if isHalf {
                let r = row.assumingMemoryBound(to: Float16.self)
                while x < x1 { let v = Float(r[x]); if v.isFinite { samples.append(invert ? 1 / max(v, 1e-4) : v) }; x += step }
            } else {
                let r = row.assumingMemoryBound(to: Float.self)
                while x < x1 { let v = r[x]; if v.isFinite { samples.append(invert ? 1 / max(v, 1e-4) : v) }; x += step }
            }
            y += step
        }
        guard samples.count > 16 else { return (0, 1) }
        var mn = Float.greatestFiniteMagnitude, mx = -Float.greatestFiniteMagnitude
        for v in samples { mn = min(mn, v); mx = max(mx, v) }
        if mx - mn < 1e-6 { return (mn, mn + 1e-6) }
        // 512-bin histogram → percentiles.
        let bins = 512
        var hist = [Int](repeating: 0, count: bins)
        let scale = Float(bins - 1) / (mx - mn)
        for v in samples { hist[Int((v - mn) * scale)] += 1 }
        let n = samples.count
        var acc = 0, loBin = 0, hiBin = bins - 1
        var foundLo = false
        for i in 0..<bins {
            acc += hist[i]
            if !foundLo && Float(acc) >= loPct * Float(n) { loBin = i; foundLo = true }
            if Float(acc) >= hiPct * Float(n) { hiBin = i; break }
        }
        let lo = mn + Float(loBin) / scale
        let hi = mn + Float(hiBin + 1) / scale
        return (lo, max(hi, lo + 1e-6))
    }
}
