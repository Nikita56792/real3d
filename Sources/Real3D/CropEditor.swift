import SwiftUI

/// Interactive 16:9 crop rectangle drawn over the aspect-fit source preview.
struct CropEditor: View {
    @ObservedObject var model: AppModel

    /// Gesture start rect lives in a reference box: the Command Line Tools SDK ships without the
    /// SwiftUI `@State` macro plugin, so plain property wrappers are used instead.
    @StateObject private var drag = DragBox()
    private var dragStart: CGRect? {
        get { drag.start }
        nonmutating set { drag.start = newValue }
    }

    private enum Handle: CaseIterable { case tl, tr, bl, br, top, bottom, left, right
        var isCorner: Bool { [.tl, .tr, .bl, .br].contains(self) }
    }

    var body: some View {
        GeometryReader { geo in
            let W = model.sourcePixelSize.width, H = model.sourcePixelSize.height
            if W > 0, H > 0 {
                let scale = min(geo.size.width / W, geo.size.height / H)
                let fitted = CGSize(width: W * scale, height: H * scale)
                let offset = CGPoint(x: (geo.size.width - fitted.width) / 2, y: (geo.size.height - fitted.height) / 2)
                let c = model.crop
                let rect = CGRect(x: offset.x + c.minX * fitted.width, y: offset.y + c.minY * fitted.height,
                                  width: c.width * fitted.width, height: c.height * fitted.height)
                ZStack(alignment: .topLeading) {
                    // Dim everything outside the crop.
                    Path { p in
                        p.addRect(CGRect(origin: offset, size: fitted))
                        p.addRect(rect)
                    }
                    .fill(Color.black.opacity(0.55), style: FillStyle(eoFill: true))
                    .allowsHitTesting(false)

                    // Crop body: drag to move.
                    Rectangle()
                        .fill(Color.white.opacity(0.001))
                        .overlay(
                            ZStack {
                                Rectangle().stroke(Color.accentColor, lineWidth: 2)
                                Path { p in
                                    for i in 1..<3 {
                                        p.move(to: CGPoint(x: rect.width * CGFloat(i) / 3, y: 0)); p.addLine(to: CGPoint(x: rect.width * CGFloat(i) / 3, y: rect.height))
                                        p.move(to: CGPoint(x: 0, y: rect.height * CGFloat(i) / 3)); p.addLine(to: CGPoint(x: rect.width, y: rect.height * CGFloat(i) / 3))
                                    }
                                }.stroke(Color.white.opacity(0.35), lineWidth: 1)
                            }
                        )
                        .frame(width: rect.width, height: rect.height)
                        .offset(x: rect.minX, y: rect.minY)
                        .gesture(
                            DragGesture(minimumDistance: 1)
                                .onChanged { g in
                                    if dragStart == nil { dragStart = model.cropPixels }
                                    guard let s = dragStart else { return }
                                    model.setCropPixels(CGRect(x: s.minX + g.translation.width / scale, y: s.minY + g.translation.height / scale,
                                                               width: s.width, height: s.height))
                                }
                                .onEnded { _ in dragStart = nil }
                        )

                    // Handles: corners resize (aspect locked in 16:9 mode, opposite corner anchored);
                    // in free mode corners resize both axes and edge handles resize one axis.
                    ForEach(Array(Handle.allCases.filter { $0.isCorner || model.cropFree }.enumerated()), id: \.offset) { _, h in
                        let pos = handlePosition(h, rect)
                        Group {
                            if h.isCorner {
                                Circle().fill(Color.accentColor).overlay(Circle().stroke(Color.white, lineWidth: 1.5)).frame(width: 14, height: 14)
                            } else {
                                RoundedRectangle(cornerRadius: 2).fill(Color.accentColor).overlay(RoundedRectangle(cornerRadius: 2).stroke(Color.white, lineWidth: 1.5))
                                    .frame(width: (h == .top || h == .bottom) ? 22 : 8, height: (h == .top || h == .bottom) ? 8 : 22)
                            }
                        }
                        .position(pos)
                        .gesture(
                            DragGesture(minimumDistance: 1)
                                .onChanged { g in
                                    if dragStart == nil { dragStart = model.cropPixels }
                                    guard let s = dragStart else { return }
                                    let dx = g.translation.width / scale, dy = g.translation.height / scale
                                    model.setCropPixels(resized(s, handle: h, dx: dx, dy: dy, free: model.cropFree))
                                }
                                .onEnded { _ in dragStart = nil }
                        )
                    }

                    // Size label.
                    Text("\(Int(model.cropPixels.width.rounded()))×\(Int(model.cropPixels.height.rounded())) px" + (model.cropBars.map { " · \($0)" } ?? ""))
                        .font(.caption.monospacedDigit())
                        .padding(4)
                        .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 4))
                        .foregroundStyle(.white)
                        .offset(x: rect.minX + 6, y: rect.minY + 6)
                        .allowsHitTesting(false)
                }
            }
        }
    }

    private func handlePosition(_ h: Handle, _ r: CGRect) -> CGPoint {
        switch h {
        case .tl: return CGPoint(x: r.minX, y: r.minY)
        case .tr: return CGPoint(x: r.maxX, y: r.minY)
        case .bl: return CGPoint(x: r.minX, y: r.maxY)
        case .br: return CGPoint(x: r.maxX, y: r.maxY)
        case .top: return CGPoint(x: r.midX, y: r.minY)
        case .bottom: return CGPoint(x: r.midX, y: r.maxY)
        case .left: return CGPoint(x: r.minX, y: r.midY)
        case .right: return CGPoint(x: r.maxX, y: r.midY)
        }
    }

    /// New crop (source pixels) after dragging `handle` by (dx, dy) from the start rect `s`.
    private func resized(_ s: CGRect, handle h: Handle, dx: CGFloat, dy: CGFloat, free: Bool) -> CGRect {
        if !free {
            let sx: CGFloat = (h == .tr || h == .br) ? 1 : -1
            let sy: CGFloat = (h == .bl || h == .br) ? 1 : -1
            let w = max(64, s.width + (sx * dx + sy * dy * 16 / 9) / 2)
            let hh = w * 9 / 16
            return CGRect(x: sx > 0 ? s.minX : s.maxX - w, y: sy > 0 ? s.minY : s.maxY - hh, width: w, height: hh)
        }
        var minX = s.minX, maxX = s.maxX, minY = s.minY, maxY = s.maxY
        if [.tl, .bl, .left].contains(h) { minX = min(s.minX + dx, s.maxX - 64) }
        if [.tr, .br, .right].contains(h) { maxX = max(s.maxX + dx, s.minX + 64) }
        if [.tl, .tr, .top].contains(h) { minY = min(s.minY + dy, s.maxY - 36) }
        if [.bl, .br, .bottom].contains(h) { maxY = max(s.maxY + dy, s.minY + 36) }
        // Keep the anchored edges fixed when clamping to the source.
        minX = max(minX, 0); minY = max(minY, 0)
        maxX = min(maxX, model.sourcePixelSize.width); maxY = min(maxY, model.sourcePixelSize.height)
        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }
}

final class DragBox: ObservableObject {
    var start: CGRect?
}
