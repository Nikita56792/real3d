import CoreGraphics
import CoreML
import CoreVideo
import Foundation
import ImageIO
import QuartzCore
import Real3DCore
import UniformTypeIdentifiers

// Usage: real3d-bench <model> [--iters N] [--units all,ane,gpu,cpu] [--image in.png] [--dump out.png]
var args = Array(CommandLine.arguments.dropFirst())
if args.first == "--shaders" {
    do {
        let t = Date()
        _ = try StereoRenderer()
        print("Shaders compiled OK in \(String(format: "%.0f", Date().timeIntervalSince(t) * 1000)) ms")
        exit(0)
    } catch {
        print("Shader compile FAILED: \(error)")
        exit(2)
    }
}
func writePNG(width: Int, height: Int, bgra: [UInt8], to path: String) {
    var data = bgra
    let cs = CGColorSpaceCreateDeviceRGB()
    let info = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
    let ctx = CGContext(data: &data, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: cs, bitmapInfo: info)!
    let img = ctx.makeImage()!
    let dst = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dst, img, nil)
    CGImageDestinationFinalize(dst)
    print("  wrote \(path) (\(width)x\(height))")
}

// --stereo <image> <model> <outprefix> [strength] [convergence] [--calib <metricModel>] [--metric [worldScale]]
if args.first == "--stereo", args.count >= 4 {
    let imgURL = args[1], modelPath = args[2], prefix = args[3]
    let strength = args.count > 4 ? Float(args[4]) ?? 40 : 40
    let conv = args.count > 5 ? Float(args[5]) ?? 0.55 : 0.55
    var calibPath: String? = nil
    var metric = false
    var worldScale: Float = 1
    if let i = args.firstIndex(of: "--calib"), i + 1 < args.count { calibPath = args[i + 1] }
    if let i = args.firstIndex(of: "--metric") { metric = true; if i + 1 < args.count, let w = Float(args[i + 1]) { worldScale = w } }
    guard let cg = loadImage(imgURL) else { print("cannot load image"); exit(1) }
    let renderer = try StereoRenderer()
    let engine = try DepthEngine(modelURL: URL(fileURLWithPath: modelPath), computeUnits: .cpuAndGPU)
    let lower = modelPath.lowercased()
    engine.outputIsDepth = lower.contains("v3") || lower.contains("da3")
    try renderer.configureModel(inputWidth: engine.info.inputWidth, inputHeight: engine.info.inputHeight, inputFormat: engine.preferredInputPixelFormat,
                                outputWidth: engine.info.outputWidth, outputHeight: engine.info.outputHeight)
    let mainIsMetric = lower.contains("metric")
    if mainIsMetric { engine.outputIsDepth = true }
    var st = renderer.settings
    st.strength = strength; st.convergence = conv; st.invertDepth = engine.outputIsDepth; st.temporalAlpha = 1
    st.modelIsMetric = mainIsMetric
    st.depthScaleMode = metric ? .metric : .artistic
    st.worldScale = worldScale
    renderer.settings = st
    // Source = the image as a BGRA pixel buffer at its native size.
    let src = try makeTestInput(width: cg.width, height: cg.height, format: kCVPixelFormatType_32BGRA, image: cg)
    // Largest centered 16:9 crop.
    let W = CGFloat(cg.width), H = CGFloat(cg.height)
    var cw = W, ch = W * 9 / 16
    if ch > H { ch = H; cw = H * 16 / 9 }
    renderer.crop = CGRect(x: (W - cw) / 2 / W, y: (H - ch) / 2 / H, width: cw / W, height: ch / H)
    let sem = DispatchSemaphore(value: 0)
    var readySlot: ModelInputSlot?
    renderer.onModelInputReady = { slot in readySlot = slot; sem.signal() }
    renderer.submitSource(src)
    renderer.renderFrame(layers: [])            // crop + model input
    sem.wait()
    let t0 = CACurrentMediaTime()
    let r = try engine.predict(readySlot!.pixelBuffer)
    print(String(format: "  inference %.1f ms, range [%.3f, %.3f]", (CACurrentMediaTime() - t0) * 1000, r.lo, r.hi))
    r.tag = readySlot!.index
    renderer.submitDepth(r)
    if let calibPath {
        let c = try DepthEngine(modelURL: URL(fileURLWithPath: calibPath), computeUnits: .all)
        c.outputIsDepth = true
        let cin = try DepthEngine.makePixelBuffer(width: c.info.inputWidth, height: c.info.inputHeight, format: c.preferredInputPixelFormat)
        Pipeline.scale(readySlot!.pixelBuffer, into: cin)
        let tc = CACurrentMediaTime()
        let zr = try c.predict(cin)
        let raw = Pipeline.readDepthMap(r.pixelBuffer), z = Pipeline.readDepthMap(zr.pixelBuffer)
        if let fit = Pipeline.fitAffine(raw: raw, rawW: engine.info.outputWidth, rawH: engine.info.outputHeight, rawInvert: engine.outputIsDepth,
                                        z: z, zW: c.info.outputWidth, zH: c.info.outputHeight) {
            print(String(format: "  calibrator %@: %.1f ms, Z range [%.2f, %.2f] m -> 1/Z = %.4f*r + %.4f (inliers %.0f%%, MAD %.4f)",
                         c.info.name as NSString, (CACurrentMediaTime() - tc) * 1000, 1 / zr.hi, 1 / zr.lo, fit.a, fit.b, fit.inliers * 100, fit.residual))
            renderer.setCalibration(a: fit.a, b: fit.b, valid: true)
        } else {
            print("  calibration FAILED")
        }
    }
    if metric {
        let inf = st.viewer.infinityDisparity(px: st.outputWidth)
        print(String(format: "  metric mode: infinity = %.1f px, screen at %.1f m, world scale %.2f", inf, st.viewer.screenDistanceM, worldScale))
    }
    renderer.releaseSlot(readySlot!)
    let t1 = CACurrentMediaTime()
    renderer.renderFrame(layers: [])            // depth process + stereo
    for _ in 0..<20 { renderer.submitSource(src); renderer.renderFrame(layers: []) }
    if let s = renderer.snapshot(.sbs) { writePNG(width: s.width, height: s.height, bgra: s.bgra, to: prefix + "_sbs.png") }
    print(String(format: "  20 frames: %.2f ms/frame (CPU wall incl. sync)", (CACurrentMediaTime() - t1) * 1000 / 21))
    if let s = renderer.snapshot(.anaglyph) { writePNG(width: s.width, height: s.height, bgra: s.bgra, to: prefix + "_anaglyph.png") }
    if let s = renderer.snapshot(.disparity) { writePNG(width: s.width, height: s.height, bgra: s.bgra, to: prefix + "_disp.png") }
    if let s = renderer.snapshot(.mono) { writePNG(width: s.width, height: s.height, bgra: s.bgra, to: prefix + "_mono.png") }
    // GPU timing: run ~1.5 s of frames with a fresh source each time (worst case: full pipeline every frame).
    let tEnd = CACurrentMediaTime() + 1.5
    var n = 0
    while CACurrentMediaTime() < tEnd { renderer.submitSource(src); renderer.renderFrame(layers: []); n += 1; usleep(2000) }
    usleep(100_000)
    let stats = renderer.stats
    print(String(format: "  GPU %.2f ms/frame (full pipeline, %d frames, MetalFX=%@)", stats.gpuMs, n, stats.usingMetalFX ? "on" : "off"))
    // Same with the bilinear crop path (no MetalFX) and larger source to exercise downscale.
    var st2 = renderer.settings; st2.useMetalFX = false; renderer.settings = st2
    let tEnd2 = CACurrentMediaTime() + 1.5
    while CACurrentMediaTime() < tEnd2 { renderer.submitSource(src); renderer.renderFrame(layers: []); usleep(2000) }
    usleep(100_000)
    print(String(format: "  GPU %.2f ms/frame (no MetalFX)", renderer.stats.gpuMs))
    exit(0)
}
guard !args.isEmpty else {
    print("usage: real3d-bench <model> [--iters N] [--units all,ane,gpu,cpu] [--image in.png] [--dump out.png]")
    exit(1)
}
let modelURL = URL(fileURLWithPath: args.removeFirst())
var iterations = 60
var unitNames = ["all", "ane", "gpu"]
var imagePath: String?
var dumpPath: String?
var i = 0
while i < args.count {
    switch args[i] {
    case "--iters": iterations = Int(args[i + 1]) ?? 60; i += 2
    case "--units": unitNames = args[i + 1].split(separator: ",").map(String.init); i += 2
    case "--image": imagePath = args[i + 1]; i += 2
    case "--dump": dumpPath = args[i + 1]; i += 2
    default: i += 1
    }
}
let allUnits: [String: MLComputeUnits] = ["all": .all, "ane": .cpuAndNeuralEngine, "gpu": .cpuAndGPU, "cpu": .cpuOnly]
let units = unitNames.compactMap { n in allUnits[n].map { (n, $0) } }

func makeTestInput(width: Int, height: Int, format: OSType, image: CGImage?) throws -> CVPixelBuffer {
    let pb = try DepthEngine.makePixelBuffer(width: width, height: height, format: format)
    CVPixelBufferLockBaseAddress(pb, [])
    defer { CVPixelBufferUnlockBaseAddress(pb, []) }
    let base = CVPixelBufferGetBaseAddress(pb)!
    let stride = CVPixelBufferGetBytesPerRow(pb)
    if let image {
        // Draw the image resized into the BGRA buffer via CoreGraphics.
        let cs = CGColorSpaceCreateDeviceRGB()
        let bmp: CGBitmapInfo = format == kCVPixelFormatType_32BGRA
            ? [CGBitmapInfo.byteOrder32Little, CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)]
            : [CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue)]
        let ctx = CGContext(data: base, width: width, height: height, bitsPerComponent: 8, bytesPerRow: stride, space: cs, bitmapInfo: bmp.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pb
    }
    let b = base.assumingMemoryBound(to: UInt8.self)
    for y in 0..<height {
        for x in 0..<width {
            let p = b + y * stride + x * 4
            var r = UInt8(255 * x / width), g = UInt8(255 * y / height), bl: UInt8 = 128
            if x > width / 4 && x < width / 2 && y > height / 3 && y < height * 2 / 3 { r = 220; g = 40; bl = 40 }
            if x > width * 5 / 8 && x < width * 7 / 8 && y > height / 5 && y < height / 2 { r = 40; g = 200; bl = 60 }
            if format == kCVPixelFormatType_32BGRA { p[0] = bl; p[1] = g; p[2] = r; p[3] = 255 } else { p[0] = 255; p[1] = r; p[2] = g; p[3] = bl }
        }
    }
    return pb
}

func loadImage(_ path: String) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}

func dumpDepth(_ r: DepthResult, to path: String) {
    let pb = r.pixelBuffer
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb), stride = CVPixelBufferGetBytesPerRow(pb)
    let base = CVPixelBufferGetBaseAddress(pb)!
    let isHalf = CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_OneComponent16Half
    var gray = [UInt8](repeating: 0, count: w * h)
    let range = max(r.hi - r.lo, 1e-6)
    for y in 0..<h {
        for x in 0..<w {
            let v: Float = isHalf ? Float((base + y * stride).assumingMemoryBound(to: Float16.self)[x])
                                  : (base + y * stride).assumingMemoryBound(to: Float.self)[x]
            let n = min(max((v - r.lo) / range, 0), 1)
            gray[y * w + x] = UInt8(n * 255)
        }
    }
    let cs = CGColorSpaceCreateDeviceGray()
    let ctx = CGContext(data: &gray, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w, space: cs, bitmapInfo: CGImageAlphaInfo.none.rawValue)!
    let img = ctx.makeImage()!
    let dst = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dst, img, nil)
    CGImageDestinationFinalize(dst)
    print("  dumped \(path) (\(w)x\(h), raw range [\(r.lo), \(r.hi)]) — white = higher raw value")
}

print("Model: \(modelURL.path)")
let t0 = Date()
let compiled = try DepthEngine.compiledModelURL(for: modelURL, cacheDirectory: nil)
print("Compiled: \(compiled.lastPathComponent) (\(String(format: "%.2f", Date().timeIntervalSince(t0)))s)")
let image = imagePath.flatMap(loadImage)
if imagePath != nil && image == nil { print("WARNING: could not load image \(imagePath!)") }

for (idx, (name, cu)) in units.enumerated() {
    do {
        let tl = Date()
        let engine = try DepthEngine(modelURL: modelURL, computeUnits: cu)
        let loadS = Date().timeIntervalSince(tl)
        if idx == 0 {
            print(engine.info.description)
            print("  outputs: \(engine.info.allOutputs.joined(separator: " | "))")
        }
        let input = try makeTestInput(width: engine.info.inputWidth, height: engine.info.inputHeight, format: engine.preferredInputPixelFormat, image: image)
        for _ in 0..<5 { _ = try engine.predict(input) }
        var times: [Double] = []
        let tb = CACurrentMediaTime()
        for _ in 0..<iterations { times.append(try engine.predict(input).inferenceMs) }
        let wall = (CACurrentMediaTime() - tb) * 1000 / Double(iterations)
        times.sort()
        let med = times[times.count / 2], p90 = times[Int(Double(times.count) * 0.9)], mn = times[0]
        let last = try engine.predict(input)
        print(String(format: "  %-4@ load %.2fs | median %.2f ms | p90 %.2f ms | min %.2f ms | wall %.2f ms -> %.1f fps | range [%.3f, %.3f]",
                     name as NSString, loadS, med, p90, mn, wall, 1000 / wall, last.lo, last.hi))
        if let dumpPath, idx == 0 { dumpDepth(last, to: dumpPath) }
    } catch {
        print("  \(name): FAILED: \(error)")
    }
}
