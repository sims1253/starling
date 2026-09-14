import SwiftUI
import StarlingVoiceCore

@main
struct StarlingVoiceApp: App {
    @StateObject private var model: AppModel

    init() {
        let repository = SessionRepository()
        _model = StateObject(wrappedValue: AppModel(repository: repository))
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
                .preferredColorScheme(.dark)
        }
    }
}
