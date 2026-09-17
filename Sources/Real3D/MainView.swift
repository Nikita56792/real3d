import Real3DCore
import SwiftUI

struct MainView: View {
    @ObservedObject var model: AppModel

    var body: some View {
        HSplitView {
            previews
                .frame(minWidth: 600, idealWidth: 900, maxWidth: .infinity, maxHeight: .infinity)
            controls
                .frame(minWidth: 330, idealWidth: 360, maxWidth: 420)
        }
        .frame(minWidth: 960, minHeight: 640)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button {
                    Task { if model.isCapturing { await model.stopCapture() } else { await model.startCapture() } }
                } label: {
                    Label(model.isCapturing ? "Stop" : "Start", systemImage: model.isCapturing ? "stop.fill" : "play.fill")
                }
                .disabled(model.selectedSourceID == nil)
                .keyboardShortcut("r", modifiers: .command)

                Button { model.showOutput() } label: { Label("Output window", systemImage: "rectangle.on.rectangle") }
                    .keyboardShortcut("o", modifiers: .command)

                Button { model.showOutput(); model.toggleFullscreen() } label: {
                    Label(model.outputFullscreen ? "Exit fullscreen" : "Fullscreen on display", systemImage: model.outputFullscreen ? "arrow.down.right.and.arrow.up.left" : "arrow.up.left.and.arrow.down.right")
                }
                .keyboardShortcut("f", modifiers: .command)
            }
        }
    }

    // MARK: Previews

    private var previews: some View {
        VStack(spacing: 8) {
            ZStack {
                MetalView(pipeline: model.pipeline, mode: .source)
                if model.isCapturing {
                    CropEditor(model: model)
                } else if model.errorMessage?.contains("Screen Recording") == true {
                    VStack(spacing: 8) {
                        Image(systemName: "rectangle.dashed.badge.record").font(.largeTitle)
                        Text("Screen Recording permission needed").font(.headline)
                        Text("System Settings → Privacy & Security → Screen & System Audio Recording → enable Real3D").font(.caption)
                        Button("Open System Settings") {
                            NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")!)
                        }
                    }
                    .foregroundStyle(.secondary)
                } else {
                    Text("Select a display or window and press Start")
                        .foregroundStyle(.secondary)
                }
            }
            .background(Color.black)
            .clipShape(RoundedRectangle(cornerRadius: 6))

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text("Crop").font(.headline).fixedSize()
                    Picker("", selection: Binding(get: { model.cropFree }, set: { model.setCropFree($0) })) {
                        Text("16:9").tag(false)
                        Text("Free").tag(true)
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .frame(width: 96)
                    .help("Free: any aspect; the output is letterboxed/pillarboxed to 16:9 with black bars on the screen plane.")
                    Button("Fit") { model.fitCrop() }.fixedSize()
                    Button("Full screen") { model.fullSourceCrop() }.fixedSize()
                        .help("Whole source in free mode, scaled into 16:9 with bars")
                    Button("1920") { model.presetCrop(width: 1920) }.fixedSize()
                    Button("2560") { model.presetCrop(width: 2560) }.fixedSize()
                    Button("3840") { model.presetCrop(width: 3840) }.fixedSize()
                    Spacer(minLength: 0)
                }
                HStack(spacing: 6) {
                    cropFields
                    if let bars = model.cropBars {
                        Text("→ 16:9 \(bars)").font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer(minLength: 0)
                }
            }
            .controlSize(.small)
            .disabled(!model.isCapturing)
            .help("Drag the rectangle to move, drag its corners (and edges in Free mode) to resize. Width presets are in source pixels.")

            ZStack {
                MetalView(pipeline: model.pipeline, mode: model.previewMode)
                VStack {
                    HStack {
                        Picker("", selection: $model.previewMode) {
                            ForEach([PresentMode.anaglyph, .fullSBS, .halfSBS, .leftEye, .rightEye, .depth, .mono], id: \.self) { m in
                                Text(m.title).tag(m)
                            }
                        }
                        .frame(width: 230)
                        .padding(6)
                        Spacer()
                        statsBadge.padding(6)
                    }
                    Spacer()
                }
            }
            .frame(height: 240)
            .background(Color.black)
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
        .padding(10)
    }

    private var cropFields: some View {
        HStack(spacing: 4) {
            let px = model.cropPixels
            numberField("X", value: px.minX) { model.setCropPixels(CGRect(x: $0, y: px.minY, width: px.width, height: px.height)) }
            numberField("Y", value: px.minY) { model.setCropPixels(CGRect(x: px.minX, y: $0, width: px.width, height: px.height)) }
            numberField("W", value: px.width) { model.setCropPixels(CGRect(x: px.minX, y: px.minY, width: $0, height: model.cropFree ? px.height : $0 * 9 / 16)) }
            if model.cropFree {
                numberField("H", value: px.height) { model.setCropPixels(CGRect(x: px.minX, y: px.minY, width: px.width, height: $0)) }
            } else {
                Text("H \(Int(px.height.rounded()))").font(.caption.monospacedDigit()).foregroundStyle(.secondary).frame(width: 60, alignment: .leading)
            }
        }
    }

    private func numberField(_ label: String, value: CGFloat, commit: @escaping (CGFloat) -> Void) -> some View {
        HStack(spacing: 2) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            TextField("", value: Binding(get: { Int(value.rounded()) }, set: { commit(CGFloat($0)) }), format: .number)
                .textFieldStyle(.roundedBorder)
                .font(.caption.monospacedDigit())
                .frame(width: 58)
        }
    }

    private var statsBadge: some View {
        let s = model.stats
        return VStack(alignment: .trailing, spacing: 1) {
            Text(String(format: "render %.0f fps · gpu %.1f ms", s.renderFPS, s.gpuMs))
            Text(String(format: "depth %.0f fps · %.1f ms", s.depthFPS, s.inferenceMs))
            Text(String(format: "capture %.0f fps%@", s.captureFPS, s.usingMetalFX ? " · MetalFX" : ""))
            if model.thermalState == .serious || model.thermalState == .critical {
                Text("Mac is hot — GPU throttling, depth fps may drop").foregroundStyle(.orange)
            }
        }
        .font(.caption2.monospacedDigit())
        .padding(6)
        .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 5))
        .foregroundStyle(.white)
    }

    // MARK: Controls

    private var controls: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                GroupBox("Source") {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Picker("", selection: $model.selectedSourceID) {
                                Section("Displays") {
                                    ForEach(model.sources.filter(\.isDisplay)) { s in Text("\(s.title) (\(s.subtitle))").tag(Optional(s.id)) }
                                }
                                Section("Windows") {
                                    ForEach(model.sources.filter { !$0.isDisplay }) { s in Text("\(s.title) — \(s.subtitle)").tag(Optional(s.id)) }
                                }
                            }
                            .labelsHidden()
                            Button { Task { await model.refreshSources() } } label: { Image(systemName: "arrow.clockwise") }
                        }
                        Toggle("Show cursor", isOn: $model.settings.showCursor)
                        Text(model.status).font(.caption).foregroundStyle(.secondary)
                        if model.sourcePixelSize.width > 0 {
                            Text("Source \(Int(model.sourcePixelSize.width))×\(Int(model.sourcePixelSize.height)) px").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }

                GroupBox("Output") {
                    VStack(alignment: .leading, spacing: 6) {
                        Picker("Format", selection: $model.outputMode) {
                            ForEach([PresentMode.fullSBS, .halfSBS, .anaglyph, .mono, .leftEye, .rightEye, .depth], id: \.self) { m in Text(m.title).tag(m) }
                        }
                        Picker("Display", selection: $model.selectedScreenIndex) {
                            ForEach(Array(model.screens.enumerated()), id: \.offset) { i, s in
                                Text("\(s.localizedName) \(Int(s.frame.width * s.backingScaleFactor))×\(Int(s.frame.height * s.backingScaleFactor))").tag(i)
                            }
                        }
                        Picker("Resolution", selection: Binding(get: { model.settings.outputHeight }, set: { h in
                            model.settings.outputHeight = h; model.settings.outputWidth = h * 16 / 9 })) {
                            Text("1280×720 per eye").tag(720)
                            Text("1920×1080 per eye").tag(1080)
                            Text("2560×1440 per eye").tag(1440)
                        }
                        Toggle("Swap eyes", isOn: $model.settings.swapEyes)
                        Toggle("MetalFX upscale when the crop is smaller than the output", isOn: $model.settings.useMetalFX)
                        Text("Put the output window on the glasses' display and press ⌘F. Esc exits fullscreen.").font(.caption).foregroundStyle(.secondary)
                    }
                }

                GroupBox("Depth model") {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Picker("", selection: $model.selectedModelID) {
                                ForEach(model.models) { m in Text(m.name).tag(Optional(m.id)) }
                            }
                            .labelsHidden()
                            .onChange(of: model.selectedModelID) { _, _ in Task { await model.loadSelectedModel() } }
                            Button { model.importModel() } label: { Image(systemName: "plus") }.help("Import a Core ML depth model")
                        }
                        Picker("Compute", selection: $model.compute) {
                            ForEach(ComputeChoice.allCases) { c in Text(c.rawValue).tag(c) }
                        }
                        .onChange(of: model.compute) { _, _ in Task { await model.loadSelectedModel() } }
                        .disabled(model.selectedModelIsSplit)
                        if model.selectedModelIsSplit || model.compute == .all, !model.computeNote.isEmpty {
                            Text(model.computeNote).font(.caption).foregroundStyle(.secondary)
                        }
                        Toggle("Model outputs depth (not inverse depth)", isOn: $model.settings.invertDepth)
                        if model.modelLoading { ProgressView().controlSize(.small) }
                        if let info = model.modelInfo {
                            Text("Input \(info.inputWidth)×\(info.inputHeight) → depth \(info.outputWidth)×\(info.outputHeight)").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }

                GroupBox("Depth scale") {
                    VStack(alignment: .leading, spacing: 8) {
                        Picker("Mode", selection: $model.settings.depthScaleMode) {
                            ForEach(DepthScaleMode.allCases) { m in Text(m.title).tag(m) }
                        }
                        if model.settings.depthScaleMode == .artistic {
                            slider("Depth strength", value: $model.settings.strength, in: 0...120, format: "%.0f px")
                            slider("Convergence (screen plane)", value: $model.settings.convergence, in: 0...1, format: "%.2f")
                            slider("Depth curve", value: $model.settings.gamma, in: 0.4...2.5, format: "%.2f")
                        } else {
                            metricControls
                        }
                    }
                }

                GroupBox("3D") {
                    VStack(alignment: .leading, spacing: 8) {
                        slider("Temporal smoothing", value: Binding(get: { 1 - model.settings.temporalAlpha }, set: { model.settings.temporalAlpha = 1 - $0 }), in: 0...0.9, format: "%.2f")
                        Picker("Depth upsampling", selection: $model.settings.upsampleMode) {
                            ForEach(UpsampleMode.allCases) { m in Text(m.title).tag(m) }
                        }
                        if model.settings.upsampleMode == .jointBilateral {
                            slider("Edge sensitivity", value: $model.settings.jbuSigmaRange, in: 0.02...0.3, format: "%.2f")
                        }
                        slider("Hole-fill threshold", value: $model.settings.stretchThreshold, in: 1.2...8, format: "%.1f px")
                        Button("Reset 3D settings") {
                            let keep = model.settings
                            var s = RenderSettings()
                            s.outputWidth = keep.outputWidth; s.outputHeight = keep.outputHeight
                            s.swapEyes = keep.swapEyes; s.useMetalFX = keep.useMetalFX; s.showCursor = keep.showCursor
                            s.invertDepth = keep.invertDepth; s.modelIsMetric = keep.modelIsMetric
                            s.depthScaleMode = keep.depthScaleMode; s.viewer = keep.viewer
                            model.settings = s
                        }
                    }
                }

                if let err = model.errorMessage {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(err).font(.caption).foregroundStyle(.red).textSelection(.enabled)
                        if err.contains("Screen Recording") {
                            HStack {
                                Button("Open System Settings") {
                                    NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")!)
                                }
                                Button("Retry") { Task { await model.refreshSources() } }
                            }
                            Text("Enable Real3D under Screen & System Audio Recording, then press Retry (or relaunch the app).").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            .padding(10)
        }
    }

    private var metricControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Viewer", selection: $model.viewerPreset) {
                ForEach(ViewerPreset.allCases) { p in Text(p.rawValue).tag(p) }
            }
            HStack(spacing: 6) {
                floatField("IPD mm", value: $model.settings.viewer.ipdMM, format: "%.0f")
                floatField("Distance m", value: $model.settings.viewer.screenDistanceM, format: "%.2f")
                floatField("Width m", value: $model.settings.viewer.screenWidthM, format: "%.2f")
            }
            .onChange(of: model.settings.viewer) { _, v in
                if let g = model.viewerPreset.geometry, g != v { model.viewerPreset = .custom }
            }
            let inf = model.settings.viewer.infinityDisparity(px: model.settings.outputWidth)
            Text(String(format: "Infinity = %.1f px behind the screen · screen plane at %.1f m", inf, model.settings.viewer.screenDistanceM))
                .font(.caption).foregroundStyle(.secondary)
            Toggle("Auto world scale (never smaller than real, enlarge to fit the screen plane)", isOn: $model.settings.autoWorldScale)
            if model.settings.autoWorldScale {
                slider("Pop-out budget for the nearest object", value: $model.settings.popBudgetPx, in: 0...40, format: "%.0f px")
                Text(String(format: "Scene ×%.2f · nearest %.1f m", model.stats.worldScale, model.stats.nearestM))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            } else {
                slider("World scale", value: Binding(get: { log2(model.settings.worldScale) }, set: { model.settings.worldScale = pow(2, $0) }),
                       in: -2...3, format: "×%.2f", display: model.settings.worldScale)
            }
            slider("Max pop-out", value: $model.settings.maxCrossedPx, in: 0...120, format: "%.0f px")

            Divider()
            if model.settings.modelIsMetric {
                Label("Main model outputs meters — no calibration needed", systemImage: "checkmark.circle").font(.caption).foregroundStyle(.green)
            } else {
                Text("Metric calibrator (auto indoor/outdoor)").font(.caption).foregroundStyle(.secondary)
                HStack {
                    Picker("", selection: $model.calibratorModelID) {
                        Text("None").tag(String?.none)
                        ForEach(model.metricModels) { m in Text(m.name).tag(Optional(m.id)) }
                    }
                    .labelsHidden()
                    .onChange(of: model.calibratorModelID) { _, _ in Task { await model.loadSelectedCalibrator() } }
                    if model.calibratorLoading { ProgressView().controlSize(.small) }
                }
                Picker("Compute", selection: $model.calibratorCompute) {
                    ForEach(ComputeChoice.allCases.filter { $0 != .all }) { c in Text(c.rawValue).tag(c) }
                }
                .onChange(of: model.calibratorCompute) { _, _ in Task { await model.loadSelectedCalibrator() } }
                slider("Calibration rate", value: Binding(get: { Float(model.calibrationHz) }, set: { model.calibrationHz = Double($0) }), in: 0.5...10, format: "%.1f Hz")
                let c = model.calibStats
                if model.calibratorInfo == nil {
                    Text("Select a metric model (e.g. DA3-Metric) to calibrate the fast model's scale.").font(.caption).foregroundStyle(.orange)
                } else if c.valid {
                    Text(String(format: "1/Z = %.3f·r + %.3f · fit %.0f%% inliers · %.1f Hz · %.0f ms", c.a, c.b, c.inlierFraction * 100, c.hz, c.lastMs))
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                } else {
                    Text("Waiting for the first calibration…").font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }

    private func floatField(_ label: String, value: Binding<Float>, format: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            TextField("", value: value, format: .number.precision(.fractionLength(0...2)))
                .textFieldStyle(.roundedBorder)
                .font(.caption.monospacedDigit())
                .frame(width: 70)
        }
    }

    private func slider(_ title: String, value: Binding<Float>, in range: ClosedRange<Float>, format: String, display: Float? = nil) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title).font(.caption)
                Spacer()
                Text(String(format: format, display ?? value.wrappedValue)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Slider(value: value, in: range)
        }
    }
}
