import Foundation

/// Minimal append-only logger to ~/Library/Logs/Real3D.log (os_log privacy hides most payloads).
public enum FileLog {
    private static let url: URL = {
        let d = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask)[0].appendingPathComponent("Logs", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d.appendingPathComponent("Real3D.log")
    }()
    private static let queue = DispatchQueue(label: "real3d.filelog")
    private static let fmt: DateFormatter = { let f = DateFormatter(); f.dateFormat = "HH:mm:ss.SSS"; return f }()

    public static func log(_ message: String) {
        let line = "\(fmt.string(from: Date())) \(message)\n"
        queue.async {
            if let h = try? FileHandle(forWritingTo: url) {
                h.seekToEndOfFile(); h.write(line.data(using: .utf8)!); try? h.close()
            } else {
                try? line.data(using: .utf8)?.write(to: url)
            }
        }
    }
}
