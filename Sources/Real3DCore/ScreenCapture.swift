import AppKit
import CoreMedia
import Foundation
import ScreenCaptureKit
import os

/// A capturable display or window.
public struct CaptureSource: Identifiable, Hashable, Sendable {
    public enum Kind: Hashable, Sendable {
        case display(CGDirectDisplayID)
        case window(CGWindowID)
    }
    public let id: String
    public let kind: Kind
    public let title: String
    public let subtitle: String
    /// Native pixel size.
    public let pixelSize: CGSize
    public var isDisplay: Bool { if case .display = kind { return true } else { return false } }
}

/// One delivered frame: the pixel buffer plus the sub-rectangle (in pixels) that holds real content
/// (window captures can be letterboxed inside the configured buffer).
public struct CaptureFrame: @unchecked Sendable {
    public let pixelBuffer: CVPixelBuffer
    public let contentRect: CGRect
}

public enum CaptureError: Error, CustomStringConvertible {
    case notFound, permission
    public var description: String {
        switch self {
        case .notFound: return "Capture source no longer exists"
        case .permission: return "Screen Recording permission is required (System Settings → Privacy & Security → Screen Recording)"
        }
    }
}

/// ScreenCaptureKit wrapper delivering BGRA IOSurface frames at native resolution.
public final class ScreenCapture: NSObject, SCStreamOutput, SCStreamDelegate, @unchecked Sendable {
    private let log = Logger(subsystem: "real3d", category: "capture")
    private var stream: SCStream?
    private let outputQueue = DispatchQueue(label: "real3d.capture", qos: .userInteractive)
    private var configuredSize = CGSize.zero
    public var onFrame: ((CaptureFrame) -> Void)?
    public var onStopped: ((Error?) -> Void)?
    public private(set) var currentSource: CaptureSource?

    public override init() { super.init() }

    /// Lists displays and windows (including full-screen apps living in other Spaces).
    /// Triggers the Screen Recording permission prompt on first use.
    public static func enumerateSources() async throws -> [CaptureSource] {
        let content: SCShareableContent
        do {
            // onScreenWindowsOnly:false so full-screen apps in other Spaces are listed too.
            content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: false)
        } catch {
            FileLog.log("SCShareableContent failed: \(error)")
            throw CaptureError.permission
        }
        var out: [CaptureSource] = []
        func screen(for id: CGDirectDisplayID) -> NSScreen? {
            NSScreen.screens.first { ($0.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID) == id }
        }
        for d in content.displays {
            let scale = screen(for: d.displayID)?.backingScaleFactor ?? 2
            let px = CGSize(width: CGFloat(d.width) * scale, height: CGFloat(d.height) * scale)
            let name = screen(for: d.displayID)?.localizedName ?? "Display \(d.displayID)"
            out.append(CaptureSource(id: "display-\(d.displayID)", kind: .display(d.displayID), title: name,
                                     subtitle: "\(Int(px.width))×\(Int(px.height))", pixelSize: px))
        }
        let myPID = ProcessInfo.processInfo.processIdentifier
        let apps = Dictionary(uniqueKeysWithValues: content.applications.map { ($0.processID, $0) })
        let displaySizes = Set(content.displays.map { "\(Int($0.width))x\(Int($0.height))" })
        func isFullscreen(_ w: SCWindow) -> Bool { displaySizes.contains("\(Int(w.frame.width))x\(Int(w.frame.height))") }
        let systemBundles: Set<String> = ["com.apple.loginwindow", "com.apple.coreautha", "com.apple.controlcenter",
                                          "com.apple.notificationcenterui", "com.apple.dock", "com.apple.WindowManager",
                                          "com.apple.Spotlight", "com.apple.systemuiserver", "com.apple.ScreenContinuity",
                                          "com.apple.TextInputMenuAgent", "com.apple.universalcontrol", "com.apple.wallpaper.agent"]
        var windows = content.windows.filter { w in
            guard let app = w.owningApplication, app.processID != myPID else { return false }
            guard w.windowLayer == 0 else { return false }
            if let a = apps[app.processID], a.bundleIdentifier.isEmpty || systemBundles.contains(a.bundleIdentifier) { return false }
            let title = w.title ?? ""
            if title.isEmpty && app.applicationName.isEmpty { return false }
            // Popups, toolbars, palettes: too small or bar-shaped.
            let fw = w.frame.width, fh = w.frame.height
            guard fw >= 320, fh >= 240, fw / fh < 5 else { return false }
            if title.hasPrefix("AutoFill") || title.hasPrefix("Autofill") { return false }
            if !w.isOnScreen {
                // From other Spaces keep only full-screen apps and real large windows;
                // hidden apps park a 500×500 placeholder window that captures nothing useful.
                if isFullscreen(w) { return true }
                if Int(fw) == 500 && Int(fh) == 500 { return false }
                return fw >= 800 && fh >= 500
            }
            return true
        }
        // Full-screen windows first, then on-screen, then other Spaces; stable by app name.
        windows.sort { a, b in
            let fa = isFullscreen(a), fb = isFullscreen(b)
            if fa != fb { return fa }
            if a.isOnScreen != b.isOnScreen { return a.isOnScreen }
            let na = a.owningApplication?.applicationName ?? "", nb = b.owningApplication?.applicationName ?? ""
            return na == nb ? (a.title ?? "") < (b.title ?? "") : na < nb
        }
        for w in windows {
            let scale = NSScreen.screens.first { $0.frame.intersects(w.frame) }?.backingScaleFactor ?? 2
            let app = w.owningApplication?.applicationName ?? ""
            var title = (w.title?.isEmpty == false) ? w.title! : app
            if title.count > 60 { title = String(title.prefix(57)) + "…" }
            var tag = ""
            if isFullscreen(w) { tag = "full screen · " } else if !w.isOnScreen { tag = "other space · " }
            out.append(CaptureSource(id: "window-\(w.windowID)", kind: .window(w.windowID), title: title,
                                     subtitle: "\(tag)\(app) — \(Int(w.frame.width))×\(Int(w.frame.height)) pt",
                                     pixelSize: CGSize(width: w.frame.width * scale, height: w.frame.height * scale)))
        }
        return out
    }

    /// Starts capturing `source`, excluding this app's own windows.
    public func start(source: CaptureSource, fps: Int = 60, showCursor: Bool = true) async throws {
        await stop()
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        let myPID = ProcessInfo.processInfo.processIdentifier
        let myApps = content.applications.filter { $0.processID == myPID }
        let filter: SCContentFilter
        switch source.kind {
        case .display(let id):
            guard let d = content.displays.first(where: { $0.displayID == id }) else { throw CaptureError.notFound }
            filter = SCContentFilter(display: d, excludingApplications: myApps, exceptingWindows: [])
        case .window(let id):
            guard let w = content.windows.first(where: { $0.windowID == id }) else { throw CaptureError.notFound }
            filter = SCContentFilter(desktopIndependentWindow: w)
        }
        let scale = CGFloat(filter.pointPixelScale)
        let rect = filter.contentRect
        var w = Int((rect.width * scale).rounded()), h = Int((rect.height * scale).rounded())
        w = max(w, 64); h = max(h, 64)
        // ScreenCaptureKit needs even dimensions for BGRA IOSurfaces on some paths.
        w &= ~1; h &= ~1
        configuredSize = CGSize(width: w, height: h)

        let config = SCStreamConfiguration()
        config.width = w
        config.height = h
        config.pixelFormat = kCVPixelFormatType_32BGRA
        config.colorSpaceName = CGColorSpace.sRGB
        config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(max(fps, 1)))
        config.queueDepth = 5
        config.showsCursor = showCursor
        config.scalesToFit = false
        config.capturesAudio = false
        if #available(macOS 14.0, *) {
            config.shouldBeOpaque = true
            config.ignoreShadowsSingleWindow = true
            config.ignoreGlobalClipSingleWindow = true
        }
        if #available(macOS 14.2, *) {
            config.includeChildWindows = true
        }
        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: outputQueue)
        try await stream.startCapture()
        self.stream = stream
        self.currentSource = source
        log.info("capture started \(source.title) \(w)x\(h) @\(fps)")
        FileLog.log("capture started \(source.title) \(w)x\(h) @\(fps) scale=\(scale) rect=\(rect)")
    }

    public func updateCursor(_ show: Bool) {
        guard let stream else { return }
        let cfg = SCStreamConfiguration()
        cfg.showsCursor = show
        // updateConfiguration merges only the properties we set? It replaces; re-apply size.
        cfg.width = Int(configuredSize.width); cfg.height = Int(configuredSize.height)
        cfg.pixelFormat = kCVPixelFormatType_32BGRA
        cfg.colorSpaceName = CGColorSpace.sRGB
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: 60)
        cfg.queueDepth = 5
        cfg.scalesToFit = false
        if #available(macOS 14.0, *) { cfg.shouldBeOpaque = true; cfg.ignoreShadowsSingleWindow = true; cfg.ignoreGlobalClipSingleWindow = true }
        if #available(macOS 14.2, *) { cfg.includeChildWindows = true }
        Task { try? await stream.updateConfiguration(cfg) }
    }

    public func stop() async {
        guard let stream else { return }
        self.stream = nil
        self.currentSource = nil
        try? await stream.stopCapture()
    }

    // MARK: SCStreamOutput

    private var loggedFrames = 0
    private var lastContent = CGRect.zero

    public func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, sampleBuffer.isValid, let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let bufW = CGFloat(CVPixelBufferGetWidth(pb)), bufH = CGFloat(CVPixelBufferGetHeight(pb))
        var content = CGRect(x: 0, y: 0, width: bufW, height: bufH)
        var debug = ""
        if let atts = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
           let a = atts.first {
            if let st = a[.status] as? Int, let status = SCFrameStatus(rawValue: st), status != .complete && status != .idle {
                return
            }
            let sf = (a[.scaleFactor] as? CGFloat) ?? (a[.scaleFactor] as? Double).map { CGFloat($0) } ?? 1
            let cs = (a[.contentScale] as? CGFloat) ?? (a[.contentScale] as? Double).map { CGFloat($0) } ?? 1
            var rect: CGRect?
            if let rd = a[.contentRect] as? NSDictionary, let r = CGRect(dictionaryRepresentation: rd) { rect = r }
            else if let rd = a[.contentRect] as? [String: Any], let r = CGRect(dictionaryRepresentation: rd as CFDictionary) { rect = r }
            if let r = rect {
                // contentRect is in points; the frame holds the content scaled by contentScale, at scaleFactor px/pt.
                var px = CGRect(x: r.origin.x * sf, y: r.origin.y * sf, width: r.width * sf, height: r.height * sf)
                px = px.intersection(content)
                if px.width >= 16, px.height >= 16 { content = px.integral }
            }
            if loggedFrames < 3 || content != lastContent {
                debug = "status=\(a[.status] ?? "-") contentRect=\(rect.map { "\($0)" } ?? "nil") scaleFactor=\(sf) contentScale=\(cs) -> content=\(content)"
            }
        }
        if !debug.isEmpty || content != lastContent {
            FileLog.log("frame \(Int(bufW))x\(Int(bufH)) \(debug)")
            lastContent = content
            loggedFrames += 1
        }
        onFrame?(CaptureFrame(pixelBuffer: pb, contentRect: content))
    }

    public func stream(_ stream: SCStream, didStopWithError error: Error) {
        log.error("capture stopped: \(error.localizedDescription)")
        FileLog.log("capture stopped: \(error)")
        self.stream = nil
        onStopped?(error)
    }
}
