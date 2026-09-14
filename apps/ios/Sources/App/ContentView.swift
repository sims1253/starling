import StarlingVoiceCore
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: AppModel
    @AppStorage("server.endpoint") private var endpoint = "https://starling.local:8181"
    @AppStorage("server.model") private var serverModel = "parakeet"
    @AppStorage("server.protocol") private var protocolValue = ServerProtocol.openAI.rawValue
    @AppStorage("server.allowLocalHTTP") private var allowLocalHTTP = false
    @State private var showsSettings = false

    private var configuration: ServerConfiguration {
        ServerConfiguration(
            endpoint: endpoint,
            model: serverModel,
            apiProtocol: ServerProtocol(rawValue: protocolValue) ?? .openAI,
            allowsInsecureLocalHTTP: allowLocalHTTP
        )
    }

    var body: some View {
        NavigationStack {
            ZStack {
                LinearGradient(
                    colors: [StarlingTheme.charcoal, StarlingTheme.panel],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()

                ScrollView {
                    VStack(spacing: 28) {
                        RecorderCard(configuration: configuration, recorder: model.recorder)
                        HistorySection(configuration: configuration)
                    }
                    .padding(.horizontal, 20)
                    .padding(.bottom, 32)
                }
            }
            .navigationTitle("Starling Voice")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showsSettings = true } label: {
                        Image(systemName: "gearshape")
                    }
                    .accessibilityLabel("Server settings")
                }
            }
            .sheet(isPresented: $showsSettings) {
                SettingsView(
                    endpoint: $endpoint,
                    serverModel: $serverModel,
                    protocolValue: $protocolValue,
                    allowLocalHTTP: $allowLocalHTTP
                )
            }
            .sheet(item: $model.selectedSession) { session in
                SessionDetailView(session: session, configuration: configuration, playback: model.playback)
                    .environmentObject(model)
            }
            .alert("Starling Voice", isPresented: Binding(
                get: { model.errorMessage != nil },
                set: { if !$0 { model.errorMessage = nil } }
            )) {
                Button("OK", role: .cancel) { model.errorMessage = nil }
            } message: {
                Text(model.errorMessage ?? "Unknown error")
            }
        }
        .tint(StarlingTheme.lime)
    }
}

private struct RecorderCard: View {
    @EnvironmentObject private var model: AppModel
    let configuration: ServerConfiguration
    @ObservedObject var recorder: AudioRecorder

    var body: some View {
        VStack(spacing: 22) {
            VStack(spacing: 7) {
                Text(recorder.isRecording ? "Listening" : recorder.isStarting ? "Preparing microphone" : model.isWorking ? "Transcribing" : "Ready when you are")
                    .font(.title2.weight(.semibold))
                Text(statusDetail)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }

            Button {
                Task { await model.toggleRecording(configuration: configuration) }
            } label: {
                ZStack {
                    if recorder.isRecording {
                        Circle()
                            .stroke(StarlingTheme.coral.opacity(0.25), lineWidth: 14)
                            .frame(width: 116, height: 116)
                    }
                    Circle()
                        .fill(recorder.isRecording ? StarlingTheme.coral : StarlingTheme.lime)
                        .frame(width: 88, height: 88)
                        .shadow(color: (recorder.isRecording ? StarlingTheme.coral : StarlingTheme.lime).opacity(0.35), radius: 22)
                    Image(systemName: recorder.isRecording ? "stop.fill" : "waveform")
                        .font(.system(size: 30, weight: .bold))
                        .foregroundStyle(StarlingTheme.charcoal)
                }
            }
            .buttonStyle(.plain)
            .disabled(model.isWorking || recorder.isStarting)
            .accessibilityLabel(recorder.isRecording ? "Stop recording" : "Start recording")

            if recorder.isRecording, let startedAt = recorder.startedAt {
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    Text(Self.elapsed(context.date.timeIntervalSince(startedAt)))
                        .font(.system(.body, design: .monospaced).weight(.medium))
                        .foregroundStyle(.secondary)
                }
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 34)
        .padding(.horizontal, 20)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 28, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .stroke(.white.opacity(0.08), lineWidth: 1)
        }
    }

    private var statusDetail: String {
        if recorder.isRecording { return "Tap stop when you’re finished" }
        if model.isWorking { return "Your recording is saved while the server works" }
        return "Tap Stop to save and send this recording"
    }

    private static func elapsed(_ interval: TimeInterval) -> String {
        let seconds = max(0, Int(interval))
        return String(format: "%02d:%02d", seconds / 60, seconds % 60)
    }
}

private struct HistorySection: View {
    @EnvironmentObject private var model: AppModel
    let configuration: ServerConfiguration

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("History")
                    .font(.headline)
                Spacer()
                Text("\(model.sessions.count)")
                    .foregroundStyle(.secondary)
            }

            if model.sessions.isEmpty {
                ContentUnavailableView(
                    "No recordings yet",
                    systemImage: "waveform.badge.mic",
                    description: Text("Your recordings and raw transcripts will appear here.")
                )
                .frame(maxWidth: .infinity)
                .padding(.vertical, 24)
            } else {
                LazyVStack(spacing: 10) {
                    ForEach(model.sessions) { session in
                        Button { model.selectedSession = session } label: {
                            HistoryRow(session: session)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }
}

private struct HistoryRow: View {
    let session: SessionRecord

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(color)
                .frame(width: 38, height: 38)
                .background(color.opacity(0.13), in: Circle())
            VStack(alignment: .leading, spacing: 4) {
                Text(session.transcript?.text ?? fallback)
                    .font(.body)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                Text(session.createdAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 8)
            Image(systemName: "chevron.right")
                .font(.caption.weight(.bold))
                .foregroundStyle(.tertiary)
        }
        .padding(14)
        .background(.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private var fallback: String {
        switch session.status {
        case .captured: "Saved recording"
        case .transcribing: "Transcribing…"
        case .failed: session.lastError ?? "Transcription failed"
        case .transcribed: "Empty transcript"
        }
    }

    private var icon: String {
        switch session.status {
        case .captured: "waveform"
        case .transcribing: "ellipsis"
        case .transcribed: "checkmark"
        case .failed: "exclamationmark"
        }
    }

    private var color: Color {
        switch session.status {
        case .captured: StarlingTheme.muted
        case .transcribing: StarlingTheme.amber
        case .transcribed: StarlingTheme.lime
        case .failed: StarlingTheme.coral
        }
    }
}
