// Compiles every .mlpackage/.mlmodel in <srcdir> into <dstdir>/<name>.mlmodelc
import CoreML
import Foundation

let args = CommandLine.arguments
guard args.count == 3 else { print("usage: compile_models.swift <srcdir> <dstdir>"); exit(1) }
let src = URL(fileURLWithPath: args[1]), dst = URL(fileURLWithPath: args[2])
try FileManager.default.createDirectory(at: dst, withIntermediateDirectories: true)
let items = try FileManager.default.contentsOfDirectory(at: src, includingPropertiesForKeys: nil)
for u in items where ["mlpackage", "mlmodel"].contains(u.pathExtension) {
    let out = dst.appendingPathComponent(u.deletingPathExtension().lastPathComponent + ".mlmodelc")
    if FileManager.default.fileExists(atPath: out.path) { try FileManager.default.removeItem(at: out) }
    let t = Date()
    let tmp = try MLModel.compileModel(at: u)
    try FileManager.default.moveItem(at: tmp, to: out)
    print("compiled \(u.lastPathComponent) -> \(out.lastPathComponent) (\(String(format: "%.1f", Date().timeIntervalSince(t)))s)")
}
