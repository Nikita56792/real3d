import AppKit
import Combine
import CoreML
import Real3DCore
import SwiftUI

struct ModelEntry: Identifiable, Hashable {
    let id: String
    let name: String
    let url: URL
    let outputIsDepth: Bool
    /// Outputs depth in meters (metric model).
    let isMetric: Bool
    let isBundled: Bool
}

enum ViewerPreset: String, CaseIterable, Identifiable {
    case xrealOne = "XREAL One (147\" @ 4 m)"
    case xrealOnePro = "XREAL One Pro (171\" @ 4 m)"
    case xrealAir2 = "XREAL Air 2 (130\" @ 4 m)"
    case monitor27 = "27\" monitor @ 0.7 m"
    case custom = "Custom"
    var id: String { rawValue }
    var geometry: ViewerGeometry? {
        switch self {
        case .xrealOne: return ViewerGeometry(ipdMM: 63, screenDistanceM: 4.0, screenWidthM: 3.25)
        case .xrealOnePro: return ViewerGeometry(ipdMM: 63, screenDistanceM: 4.0, screenWidthM: 3.79)
        case .xrealAir2: return ViewerGeometry(ipdMM: 63, screenDistanceM: 4.0, screenWidthM: 2.88)
        case .monitor27: return ViewerGeometry(ipdMM: 63, screenDistanceM: 0.7, screenWidthM: 0.60)
        case .custom: return nil
        }
    }
}

enum ComputeChoice: String, CaseIterable, Identifiable {
    case all = "Auto (all)", gpu = "GPU", ane = "Neural Engine", cpu = "CPU"
    var id: String { rawValue }
    var units: MLComputeUnits {
        switch self {
        case .all: return .all
        case .gpu: return .cpuAndGPU
        case .ane: return .cpuAndNeuralEngine
        case .cpu: return .cpuOnly
        }
    }
}

struct PersistedState: Codable {
    var settings: SettingsCodable = .init()
    var crops: [String: [Double]] = [:]      // source id → [x, y, w, h] normalized
    var selectedModelID: String?
    var calibratorModelID: String?
    var calibratorCompute: String = ComputeChoice.ane.rawValue
    var calibrationHz: Double = 4
    var viewerPreset: String = ViewerPreset.xrealOne.rawValue
    var compute: String = ComputeChoice.gpu.rawValue
    var outputMode: Int = PresentMode.fullSBS.rawValue
    var previewMode: Int = PresentMode.anaglyph.rawValue
    var outputScreenName: String?
}

struct SettingsCodable: Codable {
    var strength: Float = 40, convergence: Float = 0.55, gamma: Float = 1, temporalAlpha: Float = 0.5
    var upsample: Int = 1, jbuRadius: Int = 2, jbuSigmaRange: Float = 0.08, stretch: Float = 2.5
    var swapEyes = false, useMetalFX = true, showCursor = true
    var outputWidth = 1920, outputHeight = 1080
    var depthScaleMode: Int = 0
    var ipdMM: Float = 63, screenDistanceM: Float = 4, screenWidthM: Float = 3.25
    var worldScale: Float = 1, maxCrossedPx: Float = 60
    var autoWorldScale = true, popBudgetPx: Float = 12
    func apply(to s: inout RenderSettings) {
        s.strength = strength; s.convergence = convergence; s.gamma = gamma; s.temporalAlpha = temporalAlpha
        s.upsampleMode = UpsampleMode(rawValue: upsample) ?? .jointBilateral
        s.jbuRadius = jbuRadius; s.jbuSigmaRange = jbuSigmaRange; s.stretchThreshold = stretch
        s.swapEyes = swapEyes; s.useMetalFX = useMetalFX; s.showCursor = showCursor
        s.outputWidth = outputWidth; s.outputHeight = outputHeight
        s.depthScaleMode = DepthScaleMode(rawValue: depthScaleMode) ?? .artistic
        s.viewer = ViewerGeometry(ipdMM: ipdMM, screenDistanceM: screenDistanceM, screenWidthM: screenWidthM)
        s.worldScale = worldScale; s.maxCrossedPx = maxCrossedPx
        s.autoWorldScale = autoWorldScale; s.popBudgetPx = popBudgetPx
    }
    init() {}
    init(_ s: RenderSettings) {
        strength = s.strength; convergence = s.convergence; gamma = s.gamma; temporalAlpha = s.temporalAlpha
        upsample = s.upsampleMode.rawValue; jbuRadius = s.jbuRadius; jbuSigmaRange = s.jbuSigmaRange; stretch = s.stretchThreshold
        swapEyes = s.swapEyes; useMetalFX = s.useMetalFX; showCursor = s.showCursor
        outputWidth = s.outputWidth; outputHeight = s.outputHeight
        depthScaleMode = s.depthScaleMode.rawValue
        ipdMM = s.viewer.ipdMM; screenDistanceM = s.viewer.screenDistanceM; screenWidthM = s.viewer.screenWidthM
        worldScale = s.worldScale; maxCrossedPx = s.maxCrossedPx
        autoWorldScale = s.autoWorldScale; popBudgetPx = s.popBudgetPx
    }
}

@MainActor
final class AppModel: ObservableObject {
    let pipeline: Pipeline

    @Published var sources: [CaptureSource] = []
    @Published var selectedSourceID: String?
    @Published var isCapturing = false
    @Published var sourcePixelSize: CGSize = .zero
    /// Crop in normalized source coordinates, always 16:9 in pixel space.
    @Published var crop: CGRect = CGRect(x: 0, y: 0, width: 1, height: 1)
    @Published var settings = RenderSettings() {
        didSet {
            pipeline.renderer.settings = settings
            if oldValue.showCursor != settings.showCursor { pipeline.capture.updateCursor(settings.showCursor) }
            if oldValue.invertDepth != settings.invertDepth { pipeline.setOutputIsDepth(settings.invertDepth) }
            if oldValue.depthScaleMode != settings.depthScaleMode || oldValue.modelIsMetric != settings.modelIsMetric { updateCalibrationState() }
            schedulePersist()
        }
    }
    @Published var calibratorModelID: String? { didSet { schedulePersist() } }
    @Published var calibratorCompute: ComputeChoice = .ane { didSet { schedulePersist() } }
    @Published var calibrationHz: Double = 4 { didSet { updateCalibrationState(); schedulePersist() } }
    @Published var viewerPreset: ViewerPreset = .xrealOne {
        didSet {
            if let g = viewerPreset.geometry { settings.viewer = g }
            schedulePersist()
        }
    }
    @Published var calibratorInfo: DepthModelInfo?
    @Published var calibratorLoading = false
    @Published var calibStats = CalibrationStats()
    var metricModels: [ModelEntry] { models.filter(\.isMetric) }
    @Published var outputMode: PresentMode = .fullSBS { didSet { output.mode = outputMode; schedulePersist() } }
    @Published var previewMode: PresentMode = .anaglyph { didSet { schedulePersist() } }
    @Published var models: [ModelEntry] = []
    @Published var selectedModelID: String? { didSet { schedulePersist() } }
    @Published var compute: ComputeChoice = .gpu { didSet { schedulePersist() } }
    @Published var modelInfo: DepthModelInfo?
    @Published var modelLoading = false
    @Published var stats = RenderStats()
    @Published var status = "Idle"
    @Published var errorMessage: String?
    @Published var screens: [NSScreen] = NSScreen.screens
    @Published var selectedScreenIndex = 0 { didSet { schedulePersist() } }
    @Published var outputFullscreen = false

    let output: OutputWindowController
    private var statsTimer: Timer?
    private var persistWork: DispatchWorkItem?
    private var state = PersistedState()
    private var cancellables = Set<AnyCancellable>()
    private var lastSourceSize = CGSize.zero

    init() throws {
        pipeline = try Pipeline()
        output = OutputWindowController(pipeline: pipeline)
        output.onFullscreenChanged = { [weak self] fs in self?.outputFullscreen = fs }
        loadPersisted()
        models = ModelCatalog.scan()
        if selectedModelID == nil || !models.contains(where: { $0.id == selectedModelID }) {
            selectedModelID = models.first(where: { $0.name.contains("Balanced") && !$0.isMetric })?.id
                ?? models.first(where: { $0.name.contains("DAv2") || $0.name.contains("V2") })?.id ?? models.first?.id
        }
        if calibratorModelID == nil || !models.contains(where: { $0.id == calibratorModelID }) {
            calibratorModelID = models.first(where: { $0.name.contains("DA3-Metric") })?.id
        }
        pipeline.renderer.settings = settings
        output.mode = outputMode
        pipeline.startRenderLoop()
        statsTimer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.pollStats() }
        }
        NotificationCenter.default.publisher(for: NSApplication.didChangeScreenParametersNotification)
            .sink { [weak self] _ in self?.screens = NSScreen.screens }
            .store(in: &cancellables)
        pipeline.capture.onStopped = { [weak self] err in
            Task { @MainActor in
                self?.isCapturing = false
                self?.status = "Capture stopped" + (err.map { ": \($0.localizedDescription)" } ?? "")
            }
        }
        Task { await refreshSources() }
        Task { await loadSelectedModel(); await loadSelectedCalibrator() }
    }

    // MARK: Sources

    func refreshSources() async {
        do {
            let list = try await ScreenCapture.enumerateSources()
            sources = list
            FileLog.log("sources: \(list.count) (\(list.filter(\.isDisplay).count) displays)")
            if selectedSourceID == nil || !list.contains(where: { $0.id == selectedSourceID }) { selectedSourceID = list.first?.id }
            errorMessage = nil
        } catch {
            FileLog.log("enumerate failed: \(error)")
            errorMessage = "\(error)"
        }
    }

    func startCapture() async {
        guard let id = selectedSourceID, let src = sources.first(where: { $0.id == id }) else { return }
        do {
            status = "Starting capture…"
            lastSourceSize = .zero
            try await pipeline.capture.start(source: src, fps: 60, showCursor: settings.showCursor)
            pipeline.renderer.resetTemporal()
            isCapturing = true
            status = "Capturing \(src.title)"
            errorMessage = nil
            if let saved = state.crops[id], saved.count == 4 {
                crop = CGRect(x: saved[0], y: saved[1], width: saved[2], height: saved[3])
                pipeline.renderer.crop = crop
            } else {
                crop = CGRect(x: 0, y: 0, width: 1, height: 1)
            }
        } catch {
            FileLog.log("capture start failed: \(error)")
            errorMessage = "\(error)"
            status = "Capture failed"
        }
    }

    func stopCapture() async {
        await pipeline.capture.stop()
        isCapturing = false
        status = "Idle"
    }

    // MARK: Crop

    /// Largest centered 16:9 rectangle of the source.
    func fitCrop() {
        guard sourcePixelSize.width > 0 else { return }
        let W = sourcePixelSize.width, H = sourcePixelSize.height
        var w = W, h = W * 9 / 16
        if h > H { h = H; w = H * 16 / 9 }
        setCropPixels(CGRect(x: (W - w) / 2, y: (H - h) / 2, width: w, height: h))
    }

    /// Crop of exactly w×(w*9/16) source pixels, centered (or at the same center as the current crop).
    func presetCrop(width: CGFloat) {
        guard sourcePixelSize.width > 0 else { return }
        let w = min(width, sourcePixelSize.width, sourcePixelSize.height * 16 / 9)
        let h = w * 9 / 16
        let cur = cropPixels
        let cx = cur.width > 0 ? cur.midX : sourcePixelSize.width / 2
        let cy = cur.height > 0 ? cur.midY : sourcePixelSize.height / 2
        setCropPixels(CGRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h))
    }

    var cropPixels: CGRect {
        CGRect(x: crop.minX * sourcePixelSize.width, y: crop.minY * sourcePixelSize.height,
               width: crop.width * sourcePixelSize.width, height: crop.height * sourcePixelSize.height)
    }

    /// Clamps to the source, keeps 16:9, and pushes to the renderer.
    func setCropPixels(_ r: CGRect) {
        guard sourcePixelSize.width > 0, sourcePixelSize.height > 0 else { return }
        let W = sourcePixelSize.width, H = sourcePixelSize.height
        var w = max(min(r.width, W, H * 16 / 9), 64)
        var h = w * 9 / 16
        if h > H { h = H; w = h * 16 / 9 }
        let x = min(max(r.minX, 0), W - w)
        let y = min(max(r.minY, 0), H - h)
        crop = CGRect(x: x / W, y: y / H, width: w / W, height: h / H)
        pipeline.renderer.crop = crop
        if let id = selectedSourceID { state.crops[id] = [crop.minX, crop.minY, crop.width, crop.height] }
        schedulePersist()
    }

    // MARK: Models

    func loadSelectedModel() async {
        guard let id = selectedModelID, let m = models.first(where: { $0.id == id }) else { return }
        modelLoading = true
        status = "Loading model \(m.name)…"
        do {
            let info = try await pipeline.loadModel(url: m.url, computeUnits: compute.units, outputIsDepth: m.outputIsDepth)
            modelInfo = info
            FileLog.log("model loaded: \(info.description) on \(compute.rawValue) metric=\(m.isMetric)")
            var s = settings
            s.invertDepth = m.outputIsDepth || m.isMetric
            s.modelIsMetric = m.isMetric
            settings = s
            pipeline.setOutputIsDepth(s.invertDepth)
            updateCalibrationState()
            status = isCapturing ? "Capturing" : "Model ready"
            errorMessage = nil
        } catch {
            FileLog.log("model load failed: \(error)")
            errorMessage = "Model load failed: \(error)"
            status = "Model error"
        }
        modelLoading = false
    }

    func loadSelectedCalibrator() async {
        guard let id = calibratorModelID, let m = models.first(where: { $0.id == id }) else {
            pipeline.unloadCalibrator(); calibratorInfo = nil; updateCalibrationState(); return
        }
        calibratorLoading = true
        do {
            let info = try await pipeline.loadCalibrator(url: m.url, computeUnits: calibratorCompute.units)
            calibratorInfo = info
            FileLog.log("calibrator loaded: \(info.description) on \(calibratorCompute.rawValue)")
            errorMessage = nil
        } catch {
            FileLog.log("calibrator load failed: \(error)")
            errorMessage = "Calibrator load failed: \(error)"
            calibratorInfo = nil
        }
        calibratorLoading = false
        updateCalibrationState()
    }

    /// Calibration runs only in metric mode with a relative main model and a loaded calibrator.
    private func updateCalibrationState() {
        let enabled = settings.depthScaleMode == .metric && !settings.modelIsMetric && calibratorInfo != nil
        pipeline.setCalibration(enabled: enabled, hz: calibrationHz)
        pipeline.renderer.resetTemporal()
    }

    func importModel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.init(filenameExtension: "mlpackage")!, .init(filenameExtension: "mlmodelc")!, .init(filenameExtension: "mlmodel")!]
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Choose a Core ML depth model (.mlpackage)"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let dest = try ModelCatalog.importModel(from: url)
            models = ModelCatalog.scan()
            selectedModelID = models.first(where: { $0.url == dest })?.id
            Task { await loadSelectedModel() }
        } catch {
            errorMessage = "Import failed: \(error)"
        }
    }

    // MARK: Output window

    func showOutput() {
        output.show()
        selectedScreenIndex = min(selectedScreenIndex, max(screens.count - 1, 0))
    }

    func toggleFullscreen() {
        guard selectedScreenIndex < screens.count else { return }
        let screen = screens[selectedScreenIndex]
        if outputFullscreen { output.exitFullscreen() } else { output.enterFullscreen(on: screen) }
        if let id = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID {
            pipeline.setRenderDisplay(id)
        }
    }

    // MARK: Stats

    private func pollStats() {
        let s = pipeline.renderer.stats
        stats = s
        calibStats = pipeline.calibrationStats
        if s.sourceSize != lastSourceSize, s.sourceSize.width > 0 {
            lastSourceSize = s.sourceSize
            sourcePixelSize = s.sourceSize
            if crop == CGRect(x: 0, y: 0, width: 1, height: 1) { fitCrop() } else { setCropPixels(cropPixels) }
        }
    }

    // MARK: Persistence

    private static let key = "real3d.state.v1"

    private func loadPersisted() {
        if let data = UserDefaults.standard.data(forKey: Self.key), let st = try? JSONDecoder().decode(PersistedState.self, from: data) {
            state = st
            var s = RenderSettings()
            st.settings.apply(to: &s)
            settings = s
            selectedModelID = st.selectedModelID
            calibratorModelID = st.calibratorModelID
            calibratorCompute = ComputeChoice(rawValue: st.calibratorCompute) ?? .ane
            calibrationHz = st.calibrationHz
            viewerPreset = ViewerPreset(rawValue: st.viewerPreset) ?? .custom
            compute = ComputeChoice(rawValue: st.compute) ?? .gpu
            outputMode = PresentMode(rawValue: st.outputMode) ?? .fullSBS
            previewMode = PresentMode(rawValue: st.previewMode) ?? .anaglyph
            if let name = st.outputScreenName, let idx = NSScreen.screens.firstIndex(where: { $0.localizedName == name }) { selectedScreenIndex = idx }
        }
    }

    private func schedulePersist() {
        persistWork?.cancel()
        let w = DispatchWorkItem { [weak self] in self?.persist() }
        persistWork = w
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5, execute: w)
    }

    private func persist() {
        state.settings = SettingsCodable(settings)
        state.selectedModelID = selectedModelID
        state.calibratorModelID = calibratorModelID
        state.calibratorCompute = calibratorCompute.rawValue
        state.calibrationHz = calibrationHz
        state.viewerPreset = viewerPreset.rawValue
        state.compute = compute.rawValue
        state.outputMode = outputMode.rawValue
        state.previewMode = previewMode.rawValue
        state.outputScreenName = selectedScreenIndex < screens.count ? screens[selectedScreenIndex].localizedName : nil
        if let data = try? JSONEncoder().encode(state) { UserDefaults.standard.set(data, forKey: Self.key) }
    }
}

enum ModelCatalog {
    static var userDir: URL {
        let d = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("Real3D/Models", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    static func scan() -> [ModelEntry] {
        var dirs: [(URL, Bool)] = []
        if let r = Bundle.main.resourceURL?.appendingPathComponent("Models", isDirectory: true) { dirs.append((r, true)) }
        if let env = ProcessInfo.processInfo.environment["REAL3D_MODELS_DIR"] { dirs.append((URL(fileURLWithPath: env), true)) }
        dirs.append((userDir, false))
        var out: [ModelEntry] = []
        for (dir, bundled) in dirs {
            guard let e = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]) else { continue }
            for case let u as URL in e {
                let ext = u.pathExtension
                guard ext == "mlpackage" || ext == "mlmodelc" || ext == "mlmodel" else { continue }
                e.skipDescendants()
                let name = u.deletingPathExtension().lastPathComponent
                let lower = name.lowercased()
                let isMetric = lower.contains("metric")
                let isDepth = lower.contains("v3") || lower.contains("da3") || lower.contains("depthpro") || isMetric
                out.append(ModelEntry(id: u.path, name: name, url: u, outputIsDepth: isDepth, isMetric: isMetric, isBundled: bundled))
            }
        }
        return out.sorted { $0.name < $1.name }
    }

    static func importModel(from url: URL) throws -> URL {
        let dest = userDir.appendingPathComponent(url.lastPathComponent)
        if FileManager.default.fileExists(atPath: dest.path) { try FileManager.default.removeItem(at: dest) }
        try FileManager.default.copyItem(at: url, to: dest)
        return dest
    }
}
