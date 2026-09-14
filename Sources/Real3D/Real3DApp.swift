import AppKit
import SwiftUI

@main
struct Real3DApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        WindowGroup("Real3D") {
            RootView()
        }
        .windowResizability(.contentSize)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}

struct RootView: View {
    @StateObject private var holder = ModelHolder()
    var body: some View {
        Group {
            if let m = holder.model {
                MainView(model: m)
            } else {
                VStack(spacing: 10) {
                    Text("Real3D failed to start").font(.title2)
                    Text(holder.error ?? "").foregroundStyle(.red).textSelection(.enabled)
                }
                .padding(40)
            }
        }
    }
}

@MainActor
final class ModelHolder: ObservableObject {
    @Published var model: AppModel?
    @Published var error: String?
    init() {
        do { model = try AppModel() } catch { self.error = "\(error)" }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
