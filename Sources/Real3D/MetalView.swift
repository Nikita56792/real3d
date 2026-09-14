import AppKit
import Real3DCore
import SwiftUI

/// NSView backed by a CAMetalLayer that the pipeline renders into.
final class MetalLayerView: NSView {
    let metalLayer = CAMetalLayer()
    weak var pipeline: Pipeline?
    var mode: PresentMode = .source {
        didSet { pipeline?.setMode(mode, for: metalLayer) }
    }

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer = metalLayer
        metalLayer.backgroundColor = NSColor.black.cgColor
        metalLayer.contentsGravity = .resizeAspect
        metalLayer.isOpaque = true
    }
    required init?(coder: NSCoder) { fatalError() }

    override var wantsUpdateLayer: Bool { true }
    override func updateLayer() {}

    func attach(to pipeline: Pipeline, mode: PresentMode) {
        self.pipeline = pipeline
        self.mode = mode
        pipeline.registerLayer(metalLayer, mode: mode) { [weak self] in
            guard let self, let w = self.window else { return false }
            return w.isVisible && !w.isMiniaturized && w.occlusionState.contains(.visible) && self.bounds.width > 1
        }
        updateDrawableSize()
    }

    deinit { pipeline?.unregisterLayer(metalLayer) }

    override func layout() {
        super.layout()
        updateDrawableSize()
    }

    override func viewDidChangeBackingProperties() {
        super.viewDidChangeBackingProperties()
        updateDrawableSize()
    }

    private func updateDrawableSize() {
        let scale = window?.backingScaleFactor ?? 2
        metalLayer.contentsScale = scale
        let size = CGSize(width: max(bounds.width * scale, 1), height: max(bounds.height * scale, 1))
        if metalLayer.drawableSize != size { metalLayer.drawableSize = size }
    }
}

struct MetalView: NSViewRepresentable {
    let pipeline: Pipeline
    let mode: PresentMode

    func makeNSView(context: Context) -> MetalLayerView {
        let v = MetalLayerView(frame: .zero)
        v.attach(to: pipeline, mode: mode)
        return v
    }

    func updateNSView(_ nsView: MetalLayerView, context: Context) {
        if nsView.mode != mode { nsView.mode = mode }
    }
}
