// Standalone Core ML stage / pipeline benchmark for split (encoder -> head) models.
// Build:  swiftc -O convert/pipeline_bench.swift -o <somewhere>/pipeline_bench
// Usage:
//   pipeline_bench stage <model> <gpu|ane|all|cpu> [iters]
//   pipeline_bench concurrent <modelA> <unitsA> <modelB> <unitsB> [seconds]      (both flat-out, independent)
//   pipeline_bench pipeline <enc> <encUnits> <head> <headUnits> [seconds] [--pace fps] [--latest]
//       2-stage pipeline, encoder thread -> 1-slot queue -> head thread, encoder outputs fed directly to head.
//       default: flat-out (encoder blocks until head takes its result). --pace 60: frames "arrive" at 60 Hz,
//       encoder always takes the newest frame, latency is measured from frame arrival. --latest: encoder never
//       blocks; a result the head has not taken yet is replaced (dropped).
//   pipeline_bench full <model> <units> [seconds] [--pace fps]      (single model, same accounting as pipeline)
//   pipeline_bench chain <enc> <encUnits> <head> <headUnits> <image.png> <out.f16>
//       runs one real image through enc -> head exactly like the app would (stage-A provider passed straight to
//       stage B) and writes the raw float16 depth (row-major, W x H) for comparison against PyTorch.
// Extra flags for pipeline/full: --gpuload <ms>  a 60 Hz Metal compute job using ~<ms> ms of GPU time per frame
//   (stand-in for the app's stereo renderer); --window <s>  print fps / stage medians / thermal state every <s> s.
import CoreGraphics
import CoreML
import ImageIO
import CoreVideo
import Foundation
import Metal
import QuartzCore

setvbuf(stdout, nil, _IOLBF, 0)
let args = Array(CommandLine.arguments.dropFirst())
func unitsOf(_ s: String) -> MLComputeUnits {
    switch s { case "ane": return .cpuAndNeuralEngine; case "all": return .all; case "cpu": return .cpuOnly; default: return .cpuAndGPU }
}
let cacheDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("pipeline_bench_cache")
try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

func loadModel(_ path: String, _ units: String) throws -> MLModel {
    let url = URL(fileURLWithPath: path)
    let attrs = try FileManager.default.attributesOfItem(atPath: url.appendingPathComponent("Data/com.apple.CoreML/model.mlmodel").path)
    let mtime = Int((attrs[.modificationDate] as! Date).timeIntervalSince1970)
    let compiled = cacheDir.appendingPathComponent("\(url.deletingPathExtension().lastPathComponent)-\(mtime).mlmodelc")
    if !FileManager.default.fileExists(atPath: compiled.path) {
        let tmp = try MLModel.compileModel(at: url)
        try? FileManager.default.removeItem(at: compiled)
        try FileManager.default.moveItem(at: tmp, to: compiled)
    }
    let cfg = MLModelConfiguration()
    cfg.computeUnits = unitsOf(units)
    let t = CACurrentMediaTime()
    let m = try MLModel(contentsOf: compiled, configuration: cfg)
    print(String(format: "  loaded %@ on %@ in %.2fs", url.lastPathComponent, units, CACurrentMediaTime() - t))
    return m
}

func makeInputs(_ m: MLModel) throws -> MLFeatureProvider {
    var dict: [String: MLFeatureValue] = [:]
    for (name, d) in m.modelDescription.inputDescriptionsByName {
        if let ic = d.imageConstraint {
            var pb: CVPixelBuffer?
            CVPixelBufferCreate(nil, ic.pixelsWide, ic.pixelsHigh, kCVPixelFormatType_32BGRA,
                                [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pb)
            CVPixelBufferLockBaseAddress(pb!, [])
            let p = CVPixelBufferGetBaseAddress(pb!)!.assumingMemoryBound(to: UInt8.self)
            let bpr = CVPixelBufferGetBytesPerRow(pb!)
            for y in 0..<ic.pixelsHigh { for x in 0..<ic.pixelsWide {
                let o = y * bpr + x * 4
                p[o] = UInt8((x * 7 + y) & 255); p[o + 1] = UInt8((y * 5) & 255); p[o + 2] = UInt8((x ^ y) & 255); p[o + 3] = 255
            } }
            CVPixelBufferUnlockBaseAddress(pb!, [])
            dict[name] = MLFeatureValue(pixelBuffer: pb!)
        } else if let mc = d.multiArrayConstraint {
            let a = try MLMultiArray(shape: mc.shape, dataType: .float16)
            let n = a.count
            a.withUnsafeMutableBytes { raw, _ in
                let f = raw.bindMemory(to: Float16.self)
                var s: UInt32 = 12345
                for i in 0..<n { s = s &* 1664525 &+ 1013904223; f[i] = Float16(Float(Int(s >> 16) % 2000 - 1000) / 500) }
            }
            dict[name] = MLFeatureValue(multiArray: a)
        }
    }
    return try MLDictionaryFeatureProvider(dictionary: dict)
}

func stats(_ xs: [Double]) -> String {
    guard !xs.isEmpty else { return "n/a" }
    let s = xs.sorted()
    return String(format: "med %.1f p90 %.1f max %.1f ms", s[s.count / 2], s[min(s.count - 1, Int(Double(s.count) * 0.9))], s.last!)
}
func now() -> Double { CACurrentMediaTime() }

func thermal() -> String {
    switch ProcessInfo.processInfo.thermalState { case .nominal: return "nominal"; case .fair: return "fair"; case .serious: return "serious"; default: return "critical" }
}

/// 60 Hz GPU load: a compute kernel over a 3840x1080 RGBA16F texture, iteration count calibrated to ~targetMs.
func startGPULoad(targetMs: Double, until end: Double) -> (() -> [Double]) {
    let dev = MTLCreateSystemDefaultDevice()!, q = dev.makeCommandQueue()!
    let src = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void burn(texture2d<half, access::read_write> t [[texture(0)]], constant uint &iters [[buffer(0)]],
                     uint2 gid [[thread_position_in_grid]]) {
        half4 c = t.read(gid); float a = float(gid.x) * 0.001, b = float(gid.y) * 0.001;
        for (uint i = 0; i < iters; i++) { a = sin(a + b * 1.3) + cos(b - a); b = a * 0.5 + fract(b * 3.7); }
        c.r = half(a); c.g = half(b); t.write(c, gid);
    }
    """
    let lib = try! dev.makeLibrary(source: src, options: nil)
    let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "burn")!)
    let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 3840, height: 1080, mipmapped: false)
    td.usage = [.shaderRead, .shaderWrite]; td.storageMode = .private
    let tex = dev.makeTexture(descriptor: td)!
    func run(_ iters: UInt32) -> Double {
        var it = iters
        let cb = q.makeCommandBuffer()!, enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso); enc.setTexture(tex, index: 0); enc.setBytes(&it, length: 4, index: 0)
        enc.dispatchThreads(MTLSize(width: 3840, height: 1080, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        return (cb.gpuEndTime - cb.gpuStartTime) * 1000
    }
    var iters: UInt32 = 1
    for _ in 0..<3 { _ = run(iters) }
    while iters < 4096 { let t = (0..<3).map { _ in run(iters) }.min()!; if t >= targetMs { break }; iters = max(iters + 1, UInt32(Double(iters) * min(2, targetMs / max(t, 0.05)))) }
    print(String(format: "  gpuload: %u iters/pixel ~ %.2f ms GPU per frame at 60 Hz", iters, (0..<5).map { _ in run(iters) }.sorted()[2]))
    var samples: [Double] = []; let lock = NSLock()
    Thread.detachNewThread {
        Thread.current.qualityOfService = .userInteractive
        let t0 = now(); var k = 0
        while now() < end {
            let d = run(iters); lock.withLock { samples.append(d) }
            k += 1; Thread.sleep(forTimeInterval: max(0, t0 + Double(k) / 60 - now()))
        }
    }
    return { lock.withLock { samples } }
}

final class Slot<T> {   // 1-element blocking queue
    let cond = NSCondition(); var item: T?; var closed = false
    func put(_ x: T, replace: Bool) -> Bool {   // returns false if an older item was dropped
        cond.lock(); defer { cond.unlock() }
        var dropped = false
        if replace { dropped = item != nil } else { while item != nil && !closed { cond.wait() } }
        item = x; cond.broadcast(); return !dropped
    }
    func take() -> T? {
        cond.lock(); defer { cond.unlock() }
        while item == nil && !closed { cond.wait() }
        let x = item; item = nil; cond.broadcast(); return x
    }
    func close() { cond.lock(); closed = true; cond.broadcast(); cond.unlock() }
}

func argValue(_ key: String) -> Double? {
    if let i = args.firstIndex(of: key), i + 1 < args.count { return Double(args[i + 1]) }
    return nil
}
let positional: [String] = {
    var out: [String] = []; var i = 0
    while i < args.count { if ["--pace", "--gpuload", "--window"].contains(args[i]) { i += 2; continue }; if args[i].hasPrefix("--") { i += 1; continue }; out.append(args[i]); i += 1 }
    return out
}()

let mode = positional.first ?? ""
switch mode {
case "stage":
    let m = try loadModel(positional[1], positional[2])
    let iters = positional.count > 3 ? Int(positional[3])! : 40
    let inp = try makeInputs(m)
    for _ in 0..<5 { _ = try m.prediction(from: inp) }
    var lat: [Double] = []
    for _ in 0..<iters { let t = now(); _ = try m.prediction(from: inp); lat.append((now() - t) * 1000) }
    print("stage \(URL(fileURLWithPath: positional[1]).lastPathComponent) [\(positional[2])]: \(stats(lat))")

case "concurrent":
    let a = try loadModel(positional[1], positional[2]), b = try loadModel(positional[3], positional[4])
    let secs = positional.count > 5 ? Double(positional[5])! : 8
    let ia = try makeInputs(a), ib = try makeInputs(b)
    for _ in 0..<3 { _ = try a.prediction(from: ia); _ = try b.prediction(from: ib) }
    var lat = [[Double]](repeating: [], count: 2)
    let lock = NSLock(), g = DispatchGroup(), end = now() + secs
    for (k, (m, inp)) in [(a, ia), (b, ib)].enumerated() {
        g.enter()
        Thread.detachNewThread {
            Thread.current.qualityOfService = .userInteractive
            while now() < end { let t = now(); _ = try? m.prediction(from: inp); let d = (now() - t) * 1000; lock.withLock { lat[k].append(d) } }
            g.leave()
        }
    }
    g.wait()
    for k in 0..<2 {
        print(String(format: "concurrent %@ [%@]: %.1f fps  %@", URL(fileURLWithPath: positional[1 + 2 * k]).lastPathComponent,
                     positional[2 + 2 * k], Double(lat[k].count) / secs, stats(lat[k])))
    }

case "pipeline", "full":
    let isFull = mode == "full"
    let enc = try loadModel(positional[1], positional[2])
    let head: MLModel? = isFull ? nil : try loadModel(positional[3], positional[4])
    let secsIdx = isFull ? 3 : 5
    let secs = positional.count > secsIdx ? Double(positional[secsIdx])! : 10
    let pace = argValue("--pace"), latest = args.contains("--latest")
    let inp = try makeInputs(enc)
    for _ in 0..<3 { let f = try enc.prediction(from: inp); if let h = head { _ = try h.prediction(from: f) } }
    print("  thermal state at start: \(thermal())")
    let loadSamples: (() -> [Double])? = argValue("--gpuload").map { startGPULoad(targetMs: $0, until: now() + secs + 30) }
    let window = argValue("--window")
    struct Item { let arrival: Double; let encStart: Double; let encMs: Double; let feats: MLFeatureProvider }
    let slot = Slot<Item>()
    let lock = NSLock()
    var encLat: [Double] = [], headLat: [Double] = [], e2e: [Double] = [], e2eFromStart: [Double] = [], waitMs: [Double] = []
    var done = 0, dropped = 0
    let t0 = now(), end = t0 + secs
    let g = DispatchGroup()
        if let w = window {
        Thread.detachNewThread {
            var k = 1, lastDone = 0, lastEnc = 0, lastHead = 0
            while now() < end {
                Thread.sleep(forTimeInterval: max(0, t0 + Double(k) * w - now()))
                let (d, e, h) = lock.withLock { (done, Array(encLat[lastEnc...]), Array(headLat[lastHead...])) }
                lock.withLock { lastEnc = encLat.count; lastHead = headLat.count }
                print(String(format: "   [%3.0fs] %.1f fps | stage A %@ | stage B %@ | thermal %@", Double(k) * w, Double(d - lastDone) / w,
                             stats(e), head == nil ? "-" : stats(h), thermal()))
                lastDone = d; k += 1
            }
        }
    }
    g.enter()
    Thread.detachNewThread {   // encoder / single model
        Thread.current.qualityOfService = .userInteractive
        var lastFrame = -1
        while now() < end {
            var arrival = now()
            if let fps = pace {
                var k = Int((now() - t0) * fps)
                if k <= lastFrame {   // wait for the next frame to arrive
                    let next = t0 + Double(lastFrame + 1) / fps
                    Thread.sleep(forTimeInterval: max(0, next - now()))
                    k = lastFrame + 1
                }
                lastFrame = k; arrival = t0 + Double(k) / fps
            }
            let s = now()
            guard let f = try? enc.prediction(from: inp) else { continue }
            let e = now()
            if isFull {
                lock.withLock { encLat.append((e - s) * 1000); e2e.append((e - arrival) * 1000); e2eFromStart.append((e - s) * 1000); done += 1 }
            } else {
                lock.withLock { encLat.append((e - s) * 1000) }
                if !slot.put(Item(arrival: arrival, encStart: s, encMs: (e - s) * 1000, feats: f), replace: latest) { lock.withLock { dropped += 1 } }
            }
        }
        slot.close(); g.leave()
    }
    if let h = head {
        g.enter()
        Thread.detachNewThread {
            Thread.current.qualityOfService = .userInteractive
            while let it = slot.take() {
                let s = now()
                _ = try? h.prediction(from: it.feats)
                let e = now()
                lock.withLock {
                    headLat.append((e - s) * 1000); e2e.append((e - it.arrival) * 1000); e2eFromStart.append((e - it.encStart) * 1000)
                    waitMs.append((s - it.encStart) * 1000 - it.encMs); done += 1
                }
            }
            g.leave()
        }
    }
    g.wait()
    let wall = now() - t0
    let label = isFull ? "full \(URL(fileURLWithPath: positional[1]).lastPathComponent) [\(positional[2])]"
        : "pipeline \(URL(fileURLWithPath: positional[1]).lastPathComponent) [\(positional[2])] -> \(URL(fileURLWithPath: positional[3]).lastPathComponent) [\(positional[4])]"
    print("\(label)\(pace.map { String(format: " pace %.0f", $0) } ?? " flat-out")\(latest ? " latest-wins" : ""):")
    print(String(format: "  OUTPUT %.1f fps (%d frames, %d dropped after encode)", Double(done) / wall, done, dropped))
    print("  encoder/model: \(stats(encLat))")
    if head != nil { print("  head:          \(stats(headLat))"); print("  queue wait:    \(stats(waitMs))") }
    print("  latency encode-start -> depth: \(stats(e2eFromStart))")
    if pace != nil { print("  latency frame-arrival -> depth: \(stats(e2e))") }
    if let ls = loadSamples { print("  gpuload frames: \(stats(ls())) (\(ls().count) frames)") }
    print("  thermal state at end: \(thermal())")

case "chain":
    let enc = try loadModel(positional[1], positional[2]), head = try loadModel(positional[3], positional[4])
    let ic = enc.modelDescription.inputDescriptionsByName["image"]!.imageConstraint!
    let W = ic.pixelsWide, H = ic.pixelsHigh
    let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: positional[5]) as CFURL, nil)!
    let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)!
    var pb: CVPixelBuffer?
    CVPixelBufferCreate(nil, W, H, kCVPixelFormatType_32BGRA, [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pb)
    CVPixelBufferLockBaseAddress(pb!, [])
    let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb!), width: W, height: H, bitsPerComponent: 8,
                        bytesPerRow: CVPixelBufferGetBytesPerRow(pb!), space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue)!
    ctx.interpolationQuality = .high
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: W, height: H))
    CVPixelBufferUnlockBaseAddress(pb!, [])
    let feats = try enc.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(pixelBuffer: pb!)]))
    for n in feats.featureNames.sorted() { if let a = feats.featureValue(for: n)?.multiArrayValue { print("  stage A output \(n): \(a.shape) \(a.dataType == .float16 ? "float16" : "\(a.dataType.rawValue)")") } }
    let out = try head.prediction(from: feats)
    let d = out.featureValue(for: "depth")!.imageBufferValue!
    CVPixelBufferLockBaseAddress(d, .readOnly)
    let dw = CVPixelBufferGetWidth(d), dh = CVPixelBufferGetHeight(d), bpr = CVPixelBufferGetBytesPerRow(d)
    let base = CVPixelBufferGetBaseAddress(d)!.assumingMemoryBound(to: Float16.self)
    var data = Data(capacity: dw * dh * 2); var lo = Float.infinity, hi = -Float.infinity
    for y in 0..<dh { for x in 0..<dw { var v = base[y * (bpr / 2) + x]; lo = min(lo, Float(v)); hi = max(hi, Float(v)); withUnsafeBytes(of: &v) { data.append(contentsOf: $0) } } }
    CVPixelBufferUnlockBaseAddress(d, .readOnly)
    try data.write(to: URL(fileURLWithPath: positional[6]))
    print(String(format: "  depth %dx%d format %@ range [%.3f, %.3f] -> %@", dw, dh, CVPixelBufferGetPixelFormatType(d) == kCVPixelFormatType_OneComponent16Half ? "L00h" : "other", lo, hi, positional[6]))

default:
    print("usage: see header of pipeline_bench.swift")
}
