import StarlingVoiceCore
import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @Binding var endpoint: String
    @Binding var serverModel: String
    @Binding var protocolValue: String
    @Binding var allowLocalHTTP: Bool

    private var validationMessage: String? {
        let configuration = ServerConfiguration(
            endpoint: endpoint,
            model: serverModel,
            apiProtocol: ServerProtocol(rawValue: protocolValue) ?? .openAI,
            allowsInsecureLocalHTTP: allowLocalHTTP
        )
        do {
            _ = try configuration.validatedBaseURL()
            return nil
        } catch {
            return error.localizedDescription
        }
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Server") {
                    TextField("https://starling.local:8181", text: $endpoint)
                        .textInputAutocapitalization(.never)
                        .keyboardType(.URL)
                        .autocorrectionDisabled()
                    TextField("Model", text: $serverModel)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    Picker("API", selection: $protocolValue) {
                        ForEach(ServerProtocol.allCases, id: \.rawValue) { item in
                            Text(item.title).tag(item.rawValue)
                        }
                    }
                }

                Section {
                    Toggle("Allow HTTP on local networks", isOn: $allowLocalHTTP)
                } footer: {
                    Text("HTTPS is recommended. When enabled, HTTP is still limited to loopback, .local, and private network addresses.")
                }

                if let validationMessage {
                    Section {
                        Label(validationMessage, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.orange)
                    }
                }

                Section("Privacy") {
                    Label("Recordings stay in app-private storage", systemImage: "lock")
                    Label("Only a transcription request uploads audio", systemImage: "arrow.up.circle")
                    Label("Deleting a session removes its audio and text", systemImage: "trash")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
