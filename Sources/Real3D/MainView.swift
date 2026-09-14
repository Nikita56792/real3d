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

            HStack(spacing: 6) {
                Text("Crop 16:9").font(.headline).fixedSize()
                Button("Fit") { model.fitCrop() }.fixedSize()
                Button("1920") { model.presetCrop(width: 1920) }.fixedSize()
                Button("2560") { model.presetCrop(width: 2560) }.fixedSize()
                Button("3840") { model.presetCrop(width: 3840) }.fixedSize()
                Spacer(minLength: 4)
                cropFields
            }
            .controlSize(.small)
            .disabled(!model.isCapturing)
            .help("Drag the rectangle to move, drag its corners to resize. Width presets are in source pixels.")

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
            numberField("W", value: px.width) { model.setCropPixels(CGRect(x: px.minX, y: px.minY, width: $0, height: $0 * 9 / 16)) }
            Text("H \(Int(px.height.rounded()))").font(.caption.monospacedDigit()).foregroundStyle(.secondary).frame(width: 60, alignment: .leading)
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
                        Toggle("Model outputs depth (not inverse depth)", isOn: $model.settings.invertDepth)
                        if model.modelLoading { ProgressView().controlSize(.small) }
                        if let info = model.modelInfo {
                            Text("Input \(info.inputWidth)×\(info.inputHeight) → depth \(info.outputWidth)×\(info.outputHeight)").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }

                GroupBox("3D") {
                    VStack(alignment: .leading, spacing: 8) {
                        slider("Depth strength", value: $model.settings.strength, in: 0...120, format: "%.0f px")
                        slider("Convergence (screen plane)", value: $model.settings.convergence, in: 0...1, format: "%.2f")
                        slider("Depth curve", value: $model.settings.gamma, in: 0.4...2.5, format: "%.2f")
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
                            s.swapEyes = keep.swapEyes; s.useMetalFX = keep.useMetalFX; s.showCursor = keep.showCursor; s.invertDepth = keep.invertDepth
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

    private func slider(_ title: String, value: Binding<Float>, in range: ClosedRange<Float>, format: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title).font(.caption)
                Spacer()
                Text(String(format: format, value.wrappedValue)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Slider(value: value, in: range)
        }
    }
}
