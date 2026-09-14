import AppKit
import Real3DCore

/// The window that shows the stereo output; can go borderless-fullscreen on any display (e.g. XREAL).
final class OutputWindowController: NSObject, NSWindowDelegate {
    private let pipeline: Pipeline
    private var window: NSWindow?
    private var view: MetalLayerView?
    private var savedFrame: NSRect = NSRect(x: 200, y: 200, width: 960, height: 270)
    private(set) var isFullscreen = false
    var onFullscreenChanged: ((Bool) -> Void)?
    var mode: PresentMode = .fullSBS {
        didSet {
            view?.mode = mode
            if let w = window, !isFullscreen { w.contentAspectRatio = aspect(for: mode) }
        }
    }

    private func aspect(for mode: PresentMode) -> NSSize {
        mode == .fullSBS ? NSSize(width: 32, height: 9) : NSSize(width: 16, height: 9)
    }

    init(pipeline: Pipeline) {
        self.pipeline = pipeline
        super.init()
    }

    private func makeWindow() {
        let w = KeyCatchingWindow(contentRect: savedFrame, styleMask: [.titled, .closable, .resizable, .miniaturizable], backing: .buffered, defer: false)
        w.title = "Real3D Output"
        w.isReleasedWhenClosed = false
        w.backgroundColor = .black
        w.delegate = self
        w.contentAspectRatio = aspect(for: mode)
        w.onEscape = { [weak self] in self?.exitFullscreen() }
        let v = MetalLayerView(frame: w.contentView!.bounds)
        v.autoresizingMask = [.width, .height]
        v.attach(to: pipeline, mode: mode)
        w.contentView?.addSubview(v)
        window = w
        view = v
    }

    func show() {
        if window == nil { makeWindow() }
        window?.makeKeyAndOrderFront(nil)
    }

    func enterFullscreen(on screen: NSScreen) {
        if window == nil { makeWindow() }
        guard let w = window, !isFullscreen else { return }
        savedFrame = w.frame
        w.styleMask = [.borderless]
        w.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary, .ignoresCycle]
        w.level = .init(rawValue: Int(CGWindowLevelForKey(.mainMenuWindow)) + 1)
        w.contentAspectRatio = NSSize(width: 0, height: 0)
        w.setFrame(screen.frame, display: true)
        w.makeKeyAndOrderFront(nil)
        w.acceptsMouseMovedEvents = true
        NSCursor.setHiddenUntilMouseMoves(true)
        isFullscreen = true
        onFullscreenChanged?(true)
    }

    func exitFullscreen() {
        guard let w = window, isFullscreen else { return }
        w.styleMask = [.titled, .closable, .resizable, .miniaturizable]
        w.level = .normal
        w.collectionBehavior = []
        w.contentAspectRatio = aspect(for: mode)
        w.setFrame(savedFrame, display: true)
        w.title = "Real3D Output"
        isFullscreen = false
        onFullscreenChanged?(false)
    }

    func windowWillClose(_ notification: Notification) {
        if isFullscreen { exitFullscreen() }
    }
}

final class KeyCatchingWindow: NSWindow {
    var onEscape: (() -> Void)?
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
    override func keyDown(with event: NSEvent) {
        if event.keyCode == 53 { onEscape?() } else { super.keyDown(with: event) }
    }
    override func cancelOperation(_ sender: Any?) { onEscape?() }
}
